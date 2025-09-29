# app.py
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd, numpy as np
import requests, os, json, sqlite3, traceback
from datetime import datetime, timedelta
import random

app = Flask(__name__, static_folder='.', static_url_path='/static')

DB_PATH = 'forex_analysis.db'
HISTORICAL = {}

PAIR_MAP = {
    "USDJPY": "USD/JPY",
    "GBPJPY": "GBP/JPY",
    "EURJPY": "EUR/JPY",
    "CHFJPY": "CHF/JPY",
}

# API Keys
TWELVE_API_KEY = "1a5a4b69dae6419c951a4fb62e4ad7b2"
TWELVE_API_URL = "https://api.twelvedata.com"
ALPHA_API_KEY = "G8588U1ISMGM8GZB"
ALPHA_API_URL = "https://www.alphavantage.co/query"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


# ---------------- DB INIT ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT, timeframe TEXT, timestamp TEXT,
        current_price REAL, technical_indicators TEXT,
        ai_analysis TEXT, chart_data TEXT, data_source TEXT
    )''')
    conn.commit(); conn.close()


# ---------------- CSV HISTORICAL ----------------
def load_csv_data():
    search_dirs = [".", "data"]
    for d in search_dirs:
        if not os.path.exists(d): 
            continue
        for f in os.listdir(d):
            if f.endswith("_1D.csv"):
                path = os.path.join(d, f)
                try:
                    df = pd.read_csv(path)
                    df.columns = [c.lower() for c in df.columns]

                    # kalau ada kolom "price" tapi tidak ada "close" → pakai price sebagai close
                    if "close" not in df.columns and "price" in df.columns:
                        df["close"] = df["price"]

                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")

                    pair = os.path.basename(f).split("_")[0].upper()
                    HISTORICAL[pair] = df.sort_values("date")
                    print(f"✅ Loaded {pair} from {path}, {len(df)} rows")
                except Exception as e:
                    print(f"⚠️ Error loading {path}: {e}")
    print("Pairs available in HISTORICAL:", list(HISTORICAL.keys()))


# ---------------- TWELVE DATA ----------------
def get_price_twelvedata(pair):
    try:
        symbol = f"{pair[:3]}/{pair[3:]}"
        url = f"{TWELVE_API_URL}/exchange_rate?symbol={symbol}&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "rate" in data:
            return float(data["rate"])
        return None
    except Exception as e:
        print("TwelveData error:", e)
        return None


# ---------------- FUNDAMENTAL NEWS ----------------
def get_fundamental_news(pair="USDJPY"):
    try:
        ticker = pair[-3:]
        url = f"{ALPHA_API_URL}?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()

        if "feed" in data:
            headlines = [f"{item['title']} ({item.get('source','')})" for item in data["feed"][:2]]
            return " | ".join(headlines)
        else:
            return "No recent fundamental news."
