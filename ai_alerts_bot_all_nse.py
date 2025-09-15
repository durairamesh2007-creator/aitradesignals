import datetime
import time
import requests
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nsepython import nsefetch

# =====================
# TELEGRAM SETTINGS
# =====================
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")  # set in Render dashboard
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.get(url, params=params, timeout=10)
    except Exception as e:
        print("Telegram Error:", e)

# =====================
# MARKET HOURS CHECK
# =====================
def market_open():
    now = datetime.datetime.now()
    if now.weekday() >= 5:  # Saturday(5), Sunday(6)
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end

# =====================
# NSE STOCK LIST
# =====================
def get_all_nse_stocks():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    data = nsefetch(url)
    return [stock["symbol"] for stock in data["data"]]

STOCKS = get_all_nse_stocks()

# =====================
# MODEL TRAINING
# =====================
MODEL_FILE = "model.pkl"
FEATURES = ["Close", "MA5", "MA10"]

def fetch_stock_data(symbol):
    url = f"https://www.nseindia.com/api/chart-databyindex?index={symbol}-EQ&interval=5minute"
    data = nsefetch(url)
    if "grapthData" not in data:
        return None
    df = pd.DataFrame(data["grapthData"], columns=["timestamp", "Close"])
    df["Close"] = df["Close"].astype(float)
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    return df.dropna()

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
    print("Model accuracy:", acc)

    joblib.dump((model, FEATURES), MODEL_FILE)
    print("Model saved to", MODEL_FILE)
    return model, FEATURES

def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except:
            print("Old model format detected, retraining...")
            return train_model()
    else:
        return train_model()

# =====================
# ALERT GENERATOR
# =====================
def generate_alerts(model, FEATURES):
    if not market_open():
        print("Market closed. Waiting...")
        return

    for symbol in STOCKS:
        try:
            df = fetch_stock_data(symbol)
            if df is None or df.empty:
                continue

            latest = df.iloc[-1]
            X_live = pd.DataFrame([[latest[feat] for feat in FEATURES]], columns=FEATURES)

            prediction = model.predict(X_live)[0]

            if prediction == 1:
                msg = f"📈 BUY Signal: {symbol} at {latest['Close']:.2f}"
            else:
                msg = f"📉 SELL Signal: {symbol} at {latest['Close']:.2f}"

            print(msg)
            send_telegram_message(msg)

        except Exception as e:
            print(f"Error processing {symbol}:", e)

# =====================
# MAIN LOOP
# =====================
if __name__ == "__main__":
    model, FEATURES = load_model()
    send_telegram_message("🚀 Bot started successfully with NSE live data!")

    while True:
        if market_open():
            generate_alerts(model, FEATURES)
            time.sleep(300)  # every 5 min
        else:
            print("Market closed. Sleeping 5 minutes...")
            time.sleep(300)
