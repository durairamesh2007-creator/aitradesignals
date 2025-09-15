import datetime
import time
import requests
import pandas as pd
import numpy as np
import joblib
import os
import threading
import logging
import pandas_ta as ta
from io import StringIO
from flask import Flask
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from apscheduler.schedulers.background import BackgroundScheduler

# =====================
# CONFIGURATION
# =====================
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SYMBOLS_CSV = "nse_symbols.csv"
MODEL_FILE = "model.pkl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# =====================
# Telegram Helper
# =====================
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if not resp.ok:
            logging.error(f"Telegram error: {resp.status_code} {resp.text}")
    except Exception as e:
        logging.error(f"Telegram exception: {e}")

# =====================
# Market Hours Check IST
# =====================
def market_open():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end

# =====================
# Auto update NSE symbols from official EQUITY_L.csv daily at 6 AM IST
# =====================
def update_symbol_list():
    url = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com"
    }
    try:
        with requests.Session() as session:
            session.headers.update(headers)
            session.get("https://www.nseindia.com", timeout=5)
            res = session.get(url, timeout=10)
            res.encoding = 'utf-8'
            df = pd.read_csv(StringIO(res.text))
            df.to_csv(SYMBOLS_CSV, index=False)
            logging.info(f"Updated symbol list with {len(df)} entries.")
    except Exception as e:
        logging.error(f"Error updating symbol list: {e}")
        send_telegram_message(f"⚠️ Error updating symbol list: {e}")

def load_stock_symbols():
    if not os.path.exists(SYMBOLS_CSV):
        logging.warning("Symbol CSV not found, updating now.")
        update_symbol_list()
    try:
        df = pd.read_csv(SYMBOLS_CSV)
        if "SYMBOL" not in df.columns:
            raise ValueError("No SYMBOL column in symbols CSV")
        symbols = [f"{s}.NS" for s in df["SYMBOL"].tolist()]
        logging.info(f"Loaded {len(symbols)} symbols from CSV.")
        return symbols
    except Exception as e:
        logging.error(f"Failed to load symbols: {e}")
        send_telegram_message(f"⚠️ Failed to load symbol list: {e}")
        return ["INFY.NS", "TCS.NS", "RELIANCE.NS"]  # fallback

# =====================
# Fetch 5min intraday data from Twelve Data API with indicators
# =====================
def fetch_stock_data(symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "5min",
        "apikey": TWELVE_DATA_API_KEY,
        "format": "json",
        "outputsize": 100
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if not resp.ok:
            logging.error(f"Twelve Data API error {resp.status_code} for {symbol}: {resp.text}")
            return None
        data = resp.json()
        if "values" not in data:
            logging.warning(f"No 'values' in API response for {symbol}: {data}")
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df["Close"] = pd.to_numeric(df["close"])
        # Add technical indicators
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["RSI"] = ta.rsi(df["Close"], length=14)
        macd = ta.macd(df["Close"])
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return None

# =====================
# Train & Save Model
# =====================
FEATURES = ["Close", "MA5", "MA10", "RSI", "MACD", "MACD_signal"]

def train_model(symbol="INFY.NS"):
    logging.info(f"Training model on {symbol}...")
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        logging.warning("No data to train on!")
        return None, None
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    X = df[FEATURES]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    params = {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
    from sklearn.model_selection import GridSearchCV
    grid = GridSearchCV(model, params, cv=3)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logging.info(f"Model trained with accuracy: {acc:.4f}")
    joblib.dump((best_model, FEATURES), MODEL_FILE)
    logging.info(f"Model saved to {MODEL_FILE}")
    return best_model, FEATURES

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            model, features = joblib.load(MODEL_FILE)
            logging.info("Model loaded from file.")
            return model, features
        except Exception as e:
            logging.error(f"Error loading model, retraining: {e}")
            return train_model()
    else:
        return train_model()

# =====================
# Generate Alerts
# =====================
MIN_PRICE_CHANGE = 0.005  # 0.5%

def generate_alerts(model, features, stocks):
    if not market_open():
        logging.info("Market closed; skipping alerts.")
        return
    for symbol in stocks:
        try:
            df = fetch_stock_data(symbol)
            if df is None or df.empty:
                continue
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            if prev is not None:
                change = abs(latest["Close"] - prev["Close"]) / prev["Close"]
                if change < MIN_PRICE_CHANGE:
                    continue  # skip low moves
            X_live = pd.DataFrame([[latest[feat] for feat in features]], columns=features)
            pred = model.predict(X_live)[0]
            msg = f"📈 BUY: {symbol} at {latest['Close']:.2f}" if pred == 1 else f"📉 SELL: {symbol} at {latest['Close']:.2f}"
            logging.info(msg)
            send_telegram_message(msg)
        except Exception as e:
            logging.error(f"Alert generation error for {symbol}: {e}")
            send_telegram_message(f"⚠️ Alert error {symbol}: {e}")

# =====================
# Flask Server for health check & Render port binding
# =====================
app = Flask(__name__)

@app.route("/")
def home():
    return "🎉 Twelve Data NSE Trading Alert Bot is Running 🎉"

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# =====================
# Main execution
# =====================
if __name__ == "__main__":
    # Schedule daily symbol update at 6 AM IST
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(update_symbol_list, "cron", hour=6, minute=0)
    scheduler.start()

    # Initial symbol load
    STOCKS = load_stock_symbols()

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    model, FEATURES = load_model()
    if model is None:
        logging.error("Model training failed. Exiting.")
        exit(1)
    send_telegram_message("🚀 Bot started successfully with Twelve Data API!")

    # Scheduled alert generation every 5 minutes
    while True:
        if market_open():
            generate_alerts(model, FEATURES, STOCKS)
        else:
            logging.info("Market closed. Sleeping.")
        time.sleep(300)
