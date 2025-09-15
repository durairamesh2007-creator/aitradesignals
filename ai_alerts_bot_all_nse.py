import os
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import joblib
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed

TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# =======================================
# Telegram
# =======================================
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("Telegram Error:", e)

# =======================================
# Market Hours
# =======================================
def market_open():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))  # IST
    if now.weekday() >= 5:  # Sat/Sun
        return False
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_start <= now <= market_close

# =======================================
# Features
# =======================================
FEATURES = ["Close", "MA5", "MA10", "RSI"]

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =======================================
# Model
# =======================================
MODEL_FILE = "trading_model.pkl"

def train_model():
    print("Training model...")

    data = yf.download("^NSEI", period="6mo", interval="1d", progress=False)
    df = data.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["RSI"] = compute_rsi(df["Close"])
    df.dropna(inplace=True)

    X = df[FEATURES]
    y = (df["Close"].shift(-1) > df["Close"]).astype(int)
    y = y.loc[X.index]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump((model, FEATURES), MODEL_FILE)
    return model

def load_model():
    if os.path.exists(MODEL_FILE):
        model, features = joblib.load(MODEL_FILE)
        return model
    else:
        return train_model()

# =======================================
# NSE Symbol List
# =======================================
def get_all_nse_symbols():
    url = "https://www1.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    symbols = df["SYMBOL"].tolist()
    return [s + ".NS" for s in symbols]

# =======================================
# Prediction
# =======================================
def predict_for_batch(model, symbols):
    try:
        data = yf.download(" ".join(symbols), period="1d", interval="5m", group_by="ticker", progress=False)
    except Exception as e:
        print("Batch fetch failed:", e)
        return []

    alerts = []
    for sym in symbols:
        try:
            df = data[sym].copy()
            if df.empty or len(df) < 10:
                continue

            df["MA5"] = df["Close"].rolling(5).mean()
            df["MA10"] = df["Close"].rolling(10).mean()
            df["RSI"] = compute_rsi(df["Close"])
            df.fillna(method="bfill", inplace=True)

            latest = df.iloc[-1]
            price = float(latest["Close"])
            X_live = pd.DataFrame([[price, float(latest["MA5"]), float(latest["MA10"]), float(latest["RSI"])]],
                                  columns=FEATURES)
            pred = model.predict(X_live)[0]

            if pred == 1:
                alerts.append(f"📈 BUY Signal: {sym} at {price:.2f}")
            else:
                alerts.append(f"📉 SELL Signal: {sym} at {price:.2f}")

        except Exception as e:
            print(f"Error in {sym}:", e)

    return alerts

# =======================================
# Main
# =======================================
if __name__ == "__main__":
    model = load_model()
    print("🚀 AI NSE Alerts Bot Started...")

    all_symbols = get_all_nse_symbols()

    while True:
        if not market_open():
            print("Market closed ❌ Waiting...")
            time.sleep(600)
            continue

        batch_size = 50
        tasks = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            for i in range(0, len(all_symbols), batch_size):
                batch = all_symbols[i:i + batch_size]
                tasks.append(executor.submit(predict_for_batch, model, batch))

            for future in as_completed(tasks):
                alerts = future.result()
                for msg in alerts:
                    print(msg)
                    send_telegram_message(msg)

        time.sleep(300)  # check every 5 mins
