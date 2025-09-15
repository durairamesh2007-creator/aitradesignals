import datetime
import time
import requests
import pandas as pd
import numpy as np
import joblib
import os
import threading
from flask import Flask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================
# CONFIG
# =====================
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =====================
# Telegram notification
# =====================
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if not resp.ok:
            print(f"Telegram error: {resp.status_code} {resp.text}")
    except Exception as e:
        print("Telegram exception:", e)

# =====================
# Market hours check IST
# =====================
def market_open():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end

# =====================
# Static NSE symbol list (update or fetch externally)
# =====================
def load_nse_symbols():
    # You can expand or maintain this list regularly or read it from a file.
    return [
        "INFY.NS", "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "KOTAKBANK.NS", "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "AXISBANK.NS"
    ]

STOCKS = load_nse_symbols()

# =====================
# Fetch intraday data from Twelve Data API
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
        resp = requests.get(url, params=params, timeout=10)
        if not resp.ok:
            print(f"Twelve Data API error for {symbol}: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        if "values" not in data:
            print(f"No 'values' field in API response for {symbol}: {data}")
            return None
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df["Close"] = pd.to_numeric(df["close"])
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        return df.dropna()
    except Exception as e:
        print(f"Error fetching data for {symbol}:", e)
        return None

# =====================
# Model training & loading
# =====================
MODEL_FILE = "model.pkl"
FEATURES = ["Close", "MA5", "MA10"]

def train_model(symbol="INFY.NS"):
    print(f"Training model on {symbol} historical data...")
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        print("No data for training!")
        return None, None
    df["Return"] = df["Close"].pct_change()
    df["Target"] = (df["Return"].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    X = df[FEATURES]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.4f}")
    joblib.dump((model, FEATURES), MODEL_FILE)
    print("Model saved to", MODEL_FILE)
    return model, FEATURES

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            print("Failed to load model, retraining.")
            return train_model()
    else:
        return train_model()

# =====================
# Alert generator
# =====================
def generate_alerts(model, features, stocks):
    if not market_open():
        print("Market closed. Waiting...")
        return
    for symbol in stocks:
        try:
            df = fetch_stock_data(symbol)
            if df is None or df.empty:
                continue
            latest = df.iloc[-1]
            X_live = pd.DataFrame([[latest[feat] for feat in features]], columns=features)
            prediction = model.predict(X_live)[0]
            msg = (f"📈 BUY Signal: {symbol} at {latest['Close']:.2f}"
                   if prediction == 1 else
                   f"📉 SELL Signal: {symbol} at {latest['Close']:.2f}")
            print(msg)
            send_telegram_message(msg)
        except Exception as e:
            print(f"Error processing {symbol}:", e)

# =====================
# Flask server for Render port binding
# =====================
app = Flask(__name__)

@app.route("/")
def home():
    return "Twelve Data NSE Trading Alert Bot is Running."

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# =====================
# Main execution
# =====================
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    model, features = load_model()
    send_telegram_message("🚀 Bot started successfully with Twelve Data API!")

    while True:
        if market_open():
            generate_alerts(model, features, STOCKS)
            time.sleep(300)  # 5 minutes
        else:
            print("Market closed. Sleeping 5 minutes...")
            time.sleep(300)
