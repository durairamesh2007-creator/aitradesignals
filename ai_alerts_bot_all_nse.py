import os
import time
import logging
import requests
import pandas as pd
import pandas_ta as ta
import joblib
from flask import Flask, jsonify
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# ============================================
# CONFIGURATION
# ============================================
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CSV_FILE = "nse_stocks.csv"
MODEL_DIR = "models"
FORCE_MARKET_OPEN = True   # 🔥 True = always run, False = only NSE market hours

os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)

app = Flask(__name__)

# ============================================
# TELEGRAM ALERTS
# ============================================
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logging.error(f"Telegram error: {resp.text}")
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")

# ============================================
# MARKET TIME
# ============================================
IST = timedelta(hours=5, minutes=30)

def market_open(now: datetime) -> bool:
    if now.weekday() >= 5:  # Sat/Sun
        return False
    open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
    close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_time <= now <= close_time

# ============================================
# STOCK SYMBOLS
# ============================================
def load_symbols():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            return df['Symbol'].dropna().astype(str).tolist()
        except Exception as e:
            logging.error(f"Failed to load {CSV_FILE}: {e}")
    logging.warning("No CSV found, using defaults.")
    return ["INFY.NSE", "TCS.NSE", "RELIANCE.NSE", "HDFCBANK.NSE"]

symbols = load_symbols()

# ============================================
# DATA FETCHING
# ============================================
def fetch_stock_data(symbol, interval="5min", output_size=300):
    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={interval}&outputsize={output_size}&apikey={API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "values" not in data:
            logging.error(f"API error for {symbol}: {data}")
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        return df
    except Exception as e:
        logging.error(f"Data fetch failed for {symbol}: {e}")
        return None

# ============================================
# INDICATORS
# ============================================
def add_indicators(df):
    df["MA5"] = df["close"].rolling(5).mean()
    df["MA10"] = df["close"].rolling(10).mean()
    df["RSI"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["Signal"] = macd["MACDs_12_26_9"]
    return df.dropna()

# ============================================
# MODEL TRAINING
# ============================================
def train_model(symbol):
    df = fetch_stock_data(symbol)
    if df is None:
        return None
    df = add_indicators(df)
    if len(df) < 50:
        return None

    df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    features = ["MA5", "MA10", "RSI", "MACD", "Signal"]
    X, y = df[features], df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.01],
        "n_estimators": [50, 100],
    }
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=0)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))
    logging.info(f"{symbol} model trained. Accuracy={acc:.2f}")

    joblib.dump(best_model, os.path.join(MODEL_DIR, f"{symbol}.pkl"))
    return best_model

def load_model(symbol):
    model_file = os.path.join(MODEL_DIR, f"{symbol}.pkl")
    if os.path.exists(model_file):
        return joblib.load(model_file)
    return train_model(symbol)

# ============================================
# PREDICTIONS & ALERTS
# ============================================
def process_symbol(symbol):
    df = fetch_stock_data(symbol, output_size=100)
    if df is None:
        return
    df = add_indicators(df)
    if len(df) < 20:
        return

    model = load_model(symbol)
    features = ["MA5", "MA10", "RSI", "MACD", "Signal"]

    latest = df.iloc[-1]
    prev_close = df.iloc[-2]["close"]

    X_live = latest[features].values.reshape(1, -1)
    pred = model.predict(X_live)[0]
    proba = model.predict_proba(X_live)[0][1]

    price_change = ((latest["close"] - prev_close) / prev_close) * 100
    if abs(price_change) < 0.5:
        return

    action = "BUY" if pred == 1 else "SELL"
    msg = (
        f"⚡ {symbol} | {action}\n"
        f"Price={latest['close']:.2f}, Δ={price_change:.2f}%\n"
        f"Conf={proba:.2f}, Time={latest['datetime']}"
    )
    send_telegram_message(msg)
    logging.info(msg)

# ============================================
# MAIN LOOP WITH DAILY RETRAINING
# ============================================
def main():
    logging.info("Starting trading bot...")

    last_trained_date = None

    # Train once at startup
    for sym in symbols:
        try:
            train_model(sym)
        except Exception as e:
            logging.error(f"Training failed for {sym}: {e}")

    # Send initial alerts
    with ThreadPoolExecutor() as executor:
        executor.map(process_symbol, symbols)

    while True:
        now = datetime.utcnow() + IST

        # Daily retraining at 09:15 IST
        if (FORCE_MARKET_OPEN or market_open(now)) and (last_trained_date != now.date()) and now.hour == 9 and now.minute >= 15:
            logging.info("⏳ Daily retraining triggered...")
            for sym in symbols:
                try:
                    train_model(sym)
                except Exception as e:
                    logging.error(f"Retraining failed for {sym}: {e}")
            last_trained_date = now.date()
            logging.info("✅ Daily retraining complete.")

        # Run predictions during market hours
        if FORCE_MARKET_OPEN or market_open(now):
            with ThreadPoolExecutor() as executor:
                executor.map(process_symbol, symbols)
        else:
            logging.info("Market closed; sleeping.")

        time.sleep(300)  # 5 min

# ============================================
# FLASK ENDPOINTS
# ============================================
@app.route("/")
def index():
    return "Trading bot is running."

@app.route("/health")
def health():
    return jsonify({
        "symbols": symbols,
        "models": os.listdir(MODEL_DIR),
        "time": str(datetime.utcnow())
    })

# ============================================
# ENTRYPOINT (for Gunicorn)
# ============================================
if __name__ == "__main__":
    from threading import Thread
    Thread(target=main, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
