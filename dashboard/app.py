# app.py
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd, numpy as np
import requests, os, json, sqlite3, traceback
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import random

app = Flask(__name__, static_folder='.', static_url_path='/static')

DB_PATH = 'forex_analysis.db'

PAIR_MAP = {
    "USDJPY": "USD/JPY",
    "GBPJPY": "GBP/JPY",
    "EURJPY": "EUR/JPY",
    "CHFJPY": "CHF/JPY",
}

HISTORICAL = {}

# API DeepSeek
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", " ")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair TEXT, timeframe TEXT, timestamp TEXT,
        current_price REAL, price_change REAL,
        technical_indicators TEXT, ai_analysis TEXT,
        chart_data TEXT, data_source TEXT
    )''')
    conn.commit(); conn.close()

def load_csv_data():
    files = [f for f in os.listdir('.') if f.endswith('_1D.csv')]
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.lower() for c in df.columns]
            if 'date' in df.columns: df['date'] = pd.to_datetime(df['date'])
            pair = os.path.basename(f).split('_')[0].upper()
            HISTORICAL[pair] = df.sort_values('date')
        except Exception as e:
            print("CSV load error:", f, e)

def scrape_investing_prices():
    url = "https://id.investing.com/webmaster-tools/live-currency-cross-rates"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "cr1"})
    data = {}
    if not table: return data
    for row in table.find_all("tr"):
        cols = [c.get_text(strip=True) for c in row.find_all("td")]
        if len(cols) >= 3:
            pair = cols[0]; price = cols[1].replace(",", "")
            try: price = float(price)
            except: continue
            for k,v in PAIR_MAP.items():
                if v == pair: data[k] = price
    return data

def calc_indicators(series):
    close = pd.Series(series)
    cp = close.iloc[-1]
    delta = close.diff().fillna(0)
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = (up/(down.replace(0,np.nan))).fillna(0)
    rsi = (100-(100/(1+rs))).iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    macd = (close.ewm(span=12).mean()-close.ewm(span=26).mean()).iloc[-1]
    return {
        "current_price": round(cp,4),
        "RSI": round(rsi,2),
        "SMA20": round(sma20,4),
        "SMA50": round(sma50,4),
        "MACD": round(macd,4),
        "Resistance": round(cp*1.003,4),
        "Support": round(cp*0.997,4),
    }

def ai_fallback(tech, news_summary=""):
    cp = tech["current_price"]; rsi = tech["RSI"]; atr = cp*0.002
    if rsi<30:
        signal="BUY"; sl=cp-atr; tp1=cp+2*atr; tp2=cp+3*atr
    elif rsi>70:
        signal="SELL"; sl=cp+atr; tp1=cp-2*atr; tp2=cp-3*atr
    else:
        signal="HOLD"; sl=tp1=tp2=cp
    return {
        "SIGNAL": signal,
        "ENTRY_PRICE": round(cp,4),
        "STOP_LOSS": round(sl,4),
        "TAKE_PROFIT_1": round(tp1,4),
        "TAKE_PROFIT_2": round(tp2,4),
        "CONFIDENCE_LEVEL": 75 if signal!="HOLD" else 50,
        "TRADING_ADVICE": f"Fallback RSI-based. News: {news_summary}"
    }

def ai_deepseek_analysis(pair, tech, fundamentals):
    if not DEEPSEEK_API_KEY:
        return ai_fallback(tech, fundamentals)
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type":"application/json"}
        prompt = f"""
Anda adalah analis forex. Berdasarkan indikator teknikal dan fundamental berikut:

Pair: {pair}
Harga sekarang: {tech['current_price']}
RSI: {tech['RSI']}, MACD: {tech['MACD']}
SMA20: {tech['SMA20']}, SMA50: {tech['SMA50']}
Support: {tech['Support']}, Resistance: {tech['Resistance']}
Berita/Fundamental: {fundamentals}

Berikan rekomendasi:
- SIGNAL (BUY/SELL/HOLD)
- ENTRY, STOP LOSS, TAKE PROFIT 1 & 2
- Confidence %
- Ringkasan analisis singkat
"""
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role":"user","content": prompt}],
            "temperature":0.7
        }
        r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        resp = r.json()
        txt = resp["choices"][0]["message"]["content"]
        return {
            "SIGNAL": "AI",
            "ENTRY_PRICE": tech['current_price'],
            "STOP_LOSS": tech['Support'],
            "TAKE_PROFIT_1": tech['Resistance'],
            "TAKE_PROFIT_2": tech['Resistance']*1.002,
            "CONFIDENCE_LEVEL": 80,
            "TRADING_ADVICE": txt[:400]
        }
    except Exception as e:
        print("AI error:", e)
        return ai_fallback(tech, fundamentals)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/get_analysis")
def get_analysis():
    pair = request.args.get("pair","USDJPY").upper()
    timeframe = request.args.get("timeframe","1H")
    use_history = request.args.get("use_history","0")=="1"
    try:
        prices = scrape_investing_prices()
        if pair not in prices: return jsonify({"error":"no live price"}),400
        cp = prices[pair]

        if use_history and pair in HISTORICAL:
            df = HISTORICAL[pair].tail(100)
            closes = df["close"].tolist()+[cp]
            dates = df["date"].dt.strftime("%Y-%m-%d").tolist()+[datetime.now().strftime("%Y-%m-%d %H:%M")]
        else:
            closes = [cp+random.uniform(-0.1,0.1) for _ in range(50)]+[cp]
            dates = [(datetime.now()-timedelta(minutes=i)).strftime("%H:%M") for i in range(50)]+[datetime.now().strftime("%H:%M")]

        tech = calc_indicators(closes)
        fundamentals = "Bank of Japan monitoring yen, mixed risk sentiment."
        ai = ai_deepseek_analysis(pair, tech, fundamentals)

        return jsonify({
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "current_price": cp,
            "technical_indicators": tech,
            "ai_analysis": ai,
            "fundamental_news": fundamentals,
            "chart_data": {"dates":dates,"close":closes},
            "data_source": "Investing.com + DeepSeek"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

if __name__=="__main__":
    init_db(); load_csv_data()
    app.run(debug=True)
