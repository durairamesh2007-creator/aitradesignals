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
# TELEGRAM SETTINGS
# =====================
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    try:
        response = requests.get(url, params=params, timeout=10)
        if not response.ok:
            print(f"Telegram API returned status {response.status_code}: {response.text}")
    except Exception as e:
        print("Telegram Error:", e)

# =====================
# MARKET HOURS CHECK
# =====================
def market_open():
    now = datetime.datetime.now()
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end

# =====================
# Robust NSE fetch for Nifty 50 stocks
# =====================
def get_all_nse_stocks():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com"
    }
    with requests.Session() as session:
        session.headers.update(headers)
        session.get("https://www.nseindia.com")  # required to set cookies
        res = session.get(url, timeout=10)
        try:
            data = res.json()
        except Exception as e:
            print("Failed to parse JSON from NSE:", e)
            print("Response text:", res.text)
            return []
    if "data" not in data:
        print("NSE API response missing 'data' key:", data)
        send_telegram_message("⚠️ NSE API structure issue: 'data' key missing for NIFTY 50 stocks.")
        return []
    return [stock["symbol"] for stock in data["data"] if "symbol" in stock]

STOCKS = get_all_nse_stocks()

# =====================
# Fetch stock data from NSE API
# =====================
def fetch_stock_data(symbol):
    url = f"https://www.nseindia.com/api/chart-databyindex?index={symbol}-EQ&interval=5minute"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com"
    }
    with requests.Session() as session:
        session.headers.update(headers)
        session.get("https://www.nseindia.com")
        res = session.get(url)
        try:
            data = res.json()
        except Exception as e:
            print(f"Failed to get data for {symbol}:", e)
            return None
    if "grapthData" not in data:  # Note: 'grapthData' is misspelled in NSE API
        print(f"No 'grapthData' for {symbol}: {data.keys()}")
        return None
    try:
        df = pd.DataFrame(data["grapthData"], columns=["timestamp", "Close"])
        df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
        df["MA5"] = df["Close"].rolling(5).mean()
        df["MA10"] = df["Close"].rolling(10).mean()
        return df.dropna()
    except Exception as e:
        print(f"Error processing data for {symbol}:", e)
        return None

# =====================
# Train the Random Forest model
# =====================
MODEL_FILE = "model.pkl"
FEATURES = ["Close", "MA5", "MA10"]

def train_model(symbol="INFY"):
    print(f"Training model on {symbol} historical data...")
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        print("No data for training!")
        return None, None
    df["Return"] = df["Close"].pct_change()
    df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
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
            print("Old model format or load error, retraining...")
            return train_model()
    return train_model()

# =====================
# Generate buy/sell alerts
# =====================
def generate_alerts(model, features):
    if not market_open():
        print("Market closed. Waiting...")
        return
    for symbol in STOCKS:
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
# Minimal Flask server for Render port binding
# =====================
app = Flask(__name__)

@app.route("/")
def home():
    return "Trading Alert Bot is running."

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# =====================
# Main entry point
# =====================
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    if not STOCKS:
        print("No stocks fetched. Please check logs and NSE API accessibility.")

    model, features = load_model()
    send_telegram_message("🚀 Bot started successfully with NSE live data!")

    while True:
        if market_open():
            generate_alerts(model, features)
            time.sleep(300)  # check every 5 minutes
        else:
            print("Market closed. Sleeping 5 minutes...")
            time.sleep(300)
