# app.py (versi dengan scraping Investing.com)
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random
import os, json, sqlite3, traceback

app = Flask(__name__, static_folder='.', static_url_path='/static')

DB_PATH = 'forex_analysis.db'

# Pairs yang akan dipantau dari Investing.com
PAIR_MAP = {
    "USDJPY": "USD/JPY",
    "GBPJPY": "GBP/JPY",
    "EURJPY": "EUR/JPY",
    "CHFJPY": "CHF/JPY",
}

HISTORICAL = {}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT, timeframe TEXT, timestamp TEXT,
            current_price REAL, price_change REAL,
            technical_indicators TEXT, ai_analysis TEXT,
            chart_data TEXT, data_source TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_csv_data():
    files = [f for f in os.listdir('.') if f.endswith('_1D.csv')]
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            pair = os.path.basename(f).split('_')[0].upper()
            HISTORICAL[pair] = df.sort_values('date')
        except Exception as e:
            print("CSV load error:", f, e)

def scrape_investing_prices():
    """Scrape live rates dari Investing.com widget"""
    url = "https://id.investing.com/webmaster-tools/live-currency-cross-rates"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "cr1"})
    data = {}
    if not table:
        return data
    for row in table.find_all("tr"):
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) >= 3:
            pair = cols[0]
            price = cols[1].replace(",", "")
            try:
                price = float(price)
            except:
                continue
            for k,v in PAIR_MAP.items():
                if v == pair:
                    data[k] = price
    return data

def calc_indicators(series):
    close = pd.Series(series)
    cp = close.iloc[-1]
    delta = close.diff().fillna(0)
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = (up / (down.replace(0, np.nan))).fillna(0)
    rsi = (100 - (100/(1+rs))).iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    macd = (close.ewm(span=12).mean() - close.ewm(span=26).mean()).iloc[-1]
    return {
        "current_price": round(cp,4),
        "RSI": round(rsi,2),
        "SMA20": round(sma20,4),
        "SMA50": round(sma50,4),
        "MACD": round(macd,4),
        "Resistance": round(cp*1.003,4),
        "Support": round(cp*0.997,4),
    }

def ai_trade_logic(tech):
    cp = tech["current_price"]
    rsi = tech["RSI"]
    atr = cp * 0.002
    if rsi < 30:
        signal = "BUY"
        entry = cp
        sl = cp - atr
        tp1 = cp + 2*atr
        tp2 = cp + 3*atr
    elif rsi > 70:
        signal = "SELL"
        entry = cp
        sl = cp + atr
        tp1 = cp - 2*atr
        tp2 = cp - 3*atr
    else:
        signal = "HOLD"
        entry = cp
        sl = cp
        tp1 = cp
        tp2 = cp
    return {
        "SIGNAL": signal,
        "ENTRY_PRICE": round(entry,4),
        "STOP_LOSS": round(sl,4),
        "TAKE_PROFIT_1": round(tp1,4),
        "TAKE_PROFIT_2": round(tp2,4),
        "CONFIDENCE_LEVEL": 75 if signal!="HOLD" else 50,
        "TRADING_ADVICE": f"{signal} setup dengan SL/TP dihitung otomatis",
    }

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/get_analysis")
def get_analysis():
    pair = request.args.get("pair","USDJPY").upper()
    timeframe = request.args.get("timeframe","1H")
    use_history = request.args.get("use_history","0")=="1"

    try:
        live_prices = scrape_investing_prices()
        if pair not in live_prices:
            return jsonify({"error":"pair not found"}), 400

        cp = live_prices[pair]
        if use_history and pair in HISTORICAL:
            df = HISTORICAL[pair].tail(100)
            closes = df["close"].tolist() + [cp]
            dates = df["date"].dt.strftime("%Y-%m-%d").tolist() + [datetime.now().strftime("%Y-%m-%d %H:%M")]
        else:
            closes = [cp + random.uniform(-0.1,0.1) for _ in range(50)] + [cp]
            dates = [(datetime.now()-timedelta(minutes=i)).strftime("%H:%M") for i in range(50)] + [datetime.now().strftime("%H:%M")]

        tech = calc_indicators(closes)
        ai = ai_trade_logic(tech)

        result = {
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "current_price": cp,
            "technical_indicators": tech,
            "ai_analysis": ai,
            "chart_data": {"dates":dates,"close":closes},
            "data_source": "Investing.com"
        }
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    init_db()
    load_csv_data()
    app.run(debug=True)
