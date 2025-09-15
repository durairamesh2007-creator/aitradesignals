#!/usr/bin/env python3
"""
Production-ready Twelve Data NSE Trading Alert Bot

Key features:
- concurrent data fetching with ThreadPoolExecutor
- exponential backoff + request session
- optional per-symbol model persisted under models/
- probability-based alert filtering
- rotating file logging + structured messages
- health endpoint + graceful shutdown
"""

import os
import time
import datetime
import threading
import signal
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib

from flask import Flask, jsonify
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -----------------------
# Configuration (env)
# -----------------------
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SYMBOLS_CSV = os.getenv("SYMBOLS_CSV", "nse_symbols.csv")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Behavior flags
USE_PER_SYMBOL_MODEL = os.getenv("USE_PER_SYMBOL_MODEL", "true").lower() in ("1", "true", "yes")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))  # require >= this probability
MIN_PRICE_CHANGE = float(os.getenv("MIN_PRICE_CHANGE", "0.005"))  # 0.5% default
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))  # how many symbols to process per loop iteration
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "6"))  # concurrency for fetching
API_OUTPUTSIZE = int(os.getenv("API_OUTPUTSIZE", "100"))  # Twelve Data outputsize

# Twelve Data specifics
TD_BASE = "https://api.twelvedata.com/time_series"
TD_INTERVAL = "5min"

# Timing
LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", "300"))  # 5 minutes default

# Model training grid (small by default)
GRID_PARAMS = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1],
}

FEATURES = ["Close", "MA5", "MA10", "RSI", "MACD", "MACD_signal"]

# Logging
LOGFILE = os.getenv("LOGFILE", "bot.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Flask port
FLASK_PORT = int(os.getenv("PORT", "10000"))

# -----------------------
# Logging setup
# -----------------------
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("td_alert_bot")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
fmt = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")

sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)

fh = RotatingFileHandler(LOGFILE, maxBytes=5_000_000, backupCount=3)
fh.setFormatter(fmt)
logger.addHandler(fh)

# -----------------------
# Globals & state
# -----------------------
app = Flask(__name__)
session = requests.Session()
session.headers.update({"User-Agent": "TD-NSE-Alert-Bot/1.0"})

stop_event = threading.Event()
last_run_ts: Optional[datetime.datetime] = None
alerts_sent = 0
models_cache: Dict[str, Tuple[object, List[str]]] = {}  # symbol -> (model, features)
loaded_models_count = 0

# -----------------------
# Utilities
# -----------------------
def send_telegram_message(message: str):
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning("Telegram credentials missing; skipping send.")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    params = {"chat_id": CHAT_ID, "text": message}
    try:
        r = session.get(url, params=params, timeout=10)
        if not r.ok:
            logger.error("Telegram error: %s %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("Telegram exception: %s", e)

def market_open(now=None) -> bool:
    if now is None:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end

def load_stock_symbols() -> List[str]:
    try:
        df = pd.read_csv(SYMBOLS_CSV)
        if "SYMBOL" not in df.columns:
            raise ValueError("CSV missing 'SYMBOL' column")
        symbols = [f"{s.strip().upper()}.NSE" for s in df["SYMBOL"].dropna().unique().tolist()]
        logger.info("Loaded %d symbols from CSV", len(symbols))
        return symbols
    except Exception as e:
        logger.exception("Failed to load symbols, using fallback: %s", e)
        return ["INFY.NSE", "TCS.NSE", "RELIANCE.NSE"]

STOCKS = load_stock_symbols()

# -----------------------
# Twelve Data fetch with retries/backoff
# -----------------------
def fetch_stock_data(symbol: str, max_retries=3, backoff=1.0) -> Optional[pd.DataFrame]:
    params = {
        "symbol": symbol,
        "interval": TD_INTERVAL,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "json",
        "outputsize": API_OUTPUTSIZE,
    }
    attempt = 0
    while attempt <= max_retries and not stop_event.is_set():
        attempt += 1
        try:
            resp = session.get(TD_BASE, params=params, timeout=15)
            if not resp.ok:
                logger.warning("TD API non-200 for %s: %s %s", symbol, resp.status_code, resp.text)
                raise RuntimeError(f"status {resp.status_code}")
            data = resp.json()
            if "values" not in data:
                logger.warning("TD API response missing values for %s: %s", symbol, data)
                return None
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["Close"] = pd.to_numeric(df["close"], errors="coerce")
            # indicators
            df["MA5"] = df["Close"].rolling(5).mean()
            df["MA10"] = df["Close"].rolling(10).mean()
            df["RSI"] = ta.rsi(df["Close"], length=14)
            macd = ta.macd(df["Close"])
            df["MACD"] = macd.get("MACD_12_26_9")
            df["MACD_signal"] = macd.get("MACDs_12_26_9")
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.warning("Fetch attempt %d for %s failed: %s", attempt, symbol, e)
            if attempt > max_retries:
                logger.error("Max retries reached for %s; skipping.", symbol)
                return None
            sleep = backoff * (2 ** (attempt - 1))
            time.sleep(sleep)
    return None

# -----------------------
# Model utilities
# -----------------------
def model_path_for(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace(".", "_")
    return MODEL_DIR / f"{safe}.pkl"

def train_model_on_df(df: pd.DataFrame) -> Optional[object]:
    if df is None or df.empty:
        return None
    d = df.copy()
    d["Return"] = d["Close"].pct_change()
    d["Target"] = (d["Return"].shift(-1) > 0).astype(int)
    d.dropna(inplace=True)
    if d.empty or d["Target"].nunique() < 2:
        logger.warning("Not enough variation to train model.")
        return None
    X = d[FEATURES]
    y = d["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0)
    grid = GridSearchCV(model, GRID_PARAMS, cv=3, n_jobs=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    preds = best.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info("Trained model (acc=%.4f) with params: %s", acc, grid.best_params_)
    return best

def load_or_train_model_for_symbol(symbol: str) -> Optional[Tuple[object, List[str]]]:
    global loaded_models_count
    # check cache
    if symbol in models_cache:
        return models_cache[symbol]
    p = model_path_for(symbol)
    if p.exists():
        try:
            model, features = joblib.load(p)
            models_cache[symbol] = (model, features)
            loaded_models_count += 1
            logger.info("Loaded model from disk for %s", symbol)
            return models_cache[symbol]
        except Exception as e:
            logger.exception("Failed to load model file for %s: %s", symbol, e)
            # fall through to retrain
    # Train lazily on latest data for this symbol
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        logger.warning("Cannot fetch data to train model for %s", symbol)
        return None
    model = train_model_on_df(df)
    if model is None:
        return None
    joblib.dump((model, FEATURES), p)
    models_cache[symbol] = (model, FEATURES)
    loaded_models_count += 1
    logger.info("Trained & persisted model for %s", symbol)
    return models_cache[symbol]

def load_global_model(sample_symbols: List[str]) -> Optional[Tuple[object, List[str]]]:
    """
    Build a single global model by concatenating recent data across multiple symbols.
    This only runs if per-symbol models are disabled.
    """
    dfs = []
    for s in sample_symbols:
        df = fetch_stock_data(s)
        if df is not None and not df.empty:
            d = df.copy()
            d["SYMBOL"] = s
            dfs.append(d)
        if len(dfs) >= 10:
            break
    if not dfs:
        logger.warning("No data available to train global model.")
        return None
    big = pd.concat(dfs, ignore_index=True)
    # optionally shuffle or keep time ordering across each symbol - we'll group by symbol and train on combined
    return (train_model_on_df(big), FEATURES) if not big.empty else None

# -----------------------
# Generate alerts
# -----------------------
def analyze_and_alert_for_symbol(symbol: str, global_model_tuple=None) -> Optional[str]:
    """
    fetch data -> check price change filter -> predict -> send if probability > threshold.
    returns message sent or None
    """
    global alerts_sent
    df = fetch_stock_data(symbol)
    if df is None or df.empty:
        return None
    latest = df.iloc[-1]
    if len(df) > 1:
        prev = df.iloc[-2]
        change = abs(latest["Close"] - prev["Close"]) / prev["Close"]
        if change < MIN_PRICE_CHANGE:
            logger.debug("Skip %s: price change %.4f < min %.4f", symbol, change, MIN_PRICE_CHANGE)
            return None

    if USE_PER_SYMBOL_MODEL:
        model_tuple = load_or_train_model_for_symbol(symbol)
    else:
        model_tuple = global_model_tuple or load_global_model([symbol])

    if not model_tuple:
        logger.warning("No model available for %s", symbol)
        return None
    model, features = model_tuple

    try:
        X_live = pd.DataFrame([[float(latest[feat]) for feat in features]], columns=features)
    except Exception as e:
        logger.exception("Error building live feature frame for %s: %s", symbol, e)
        return None

    # predict and probability
    try:
        proba = float(model.predict_proba(X_live)[0][1])
        pred = int(model.predict(X_live)[0])
    except Exception as e:
        logger.exception("Model prediction failed for %s: %s", symbol, e)
        return None

    # only alert if probability passes threshold
    if proba < CONFIDENCE_THRESHOLD:
        logger.debug("Low confidence for %s: %.3f < %.3f", symbol, proba, CONFIDENCE_THRESHOLD)
        return None

    action = "BUY" if pred == 1 else "SELL"
    price = float(latest["Close"])
    ts = latest["datetime"].to_pydatetime() if hasattr(latest["datetime"], "to_pydatetime") else latest["datetime"]
    msg = f"📊 {action}: {symbol} @ {price:.2f} ({action} prob={proba:.2f}) [{ts}]"
    logger.info("Alert: %s", msg)
    send_telegram_message(msg)
    alerts_sent += 1
    return msg

# -----------------------
# Health endpoint
# -----------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "running" if not stop_event.is_set() else "shutting_down",
        "last_run": last_run_ts.isoformat() if last_run_ts else None,
        "alerts_sent": alerts_sent,
        "loaded_models": loaded_models_count,
        "stocks_count": len(STOCKS),
        "per_symbol_models": USE_PER_SYMBOL_MODEL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    })

@app.route("/")
def home():
    return "🎉 Twelve Data NSE Trading Alert Bot — production mode 🎉"

# -----------------------
# Worker loop
# -----------------------
def run_loop():
    global last_run_ts
    logger.info("Starting main loop. Market open window checker enabled.")
    global_model_tuple = None
    if not USE_PER_SYMBOL_MODEL:
        logger.info("Preparing global model (this may take a while)...")
        global_model_tuple = load_global_model(STOCKS)

    while not stop_event.is_set():
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        last_run_ts = now
        if market_open(now):
            logger.info("Market open — generating alerts.")
            # process in batches to reduce API spikes
            for i in range(0, len(STOCKS), BATCH_SIZE):
                if stop_event.is_set():
                    break
                batch = STOCKS[i:i + BATCH_SIZE]
                # parallel fetch + analyze
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futures = {ex.submit(analyze_and_alert_for_symbol, s, global_model_tuple): s for s in batch}
                    for fut in as_completed(futures):
                        s = futures[fut]
                        try:
                            res = fut.result()
                            if res:
                                logger.debug("Sent alert for %s", s)
                        except Exception as e:
                            logger.exception("Error in future for %s: %s", s, e)
                # gentle pause between batches (helps avoid rate limits)
                time.sleep(1.0)
        else:
            logger.info("Market closed; sleeping until next loop.")
        # sleep (interruptible)
        for _ in range(0, LOOP_SLEEP_SECONDS, 5):
            if stop_event.is_set():
                break
            time.sleep(5)
    logger.info("Main loop exiting due to stop_event.")

# -----------------------
# Graceful shutdown
# -----------------------
def _signal_handler(signum, frame):
    logger.info("Signal %s received: initiating graceful shutdown...", signum)
    stop_event.set()

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# -----------------------
# Entrypoint
# -----------------------
def main():
    # start flask in a thread
    flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()
    # start loop
    try:
        send_telegram_message("🚀 Bot (production) started successfully with Twelve Data API!")
    except Exception:
        logger.debug("Could not send startup message (maybe missing credentials).")
    run_loop()
    logger.info("Shutdown complete. Goodbye.")

if __name__ == "__main__":
    main()
