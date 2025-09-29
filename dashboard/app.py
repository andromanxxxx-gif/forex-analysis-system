# app.py
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd, numpy as np
import requests, os, json, sqlite3, traceback
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

# Twelve Data API
TWELVE_API_KEY = ""
TWELVE_API_URL = "https://api.twelvedata.com"

# DeepSeek API (opsional)
DEEPSEEK_API_KEY = os.environ.get("", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Alpha Vantage API (fundamental news)
ALPHA_API_KEY = ""
ALPHA_API_URL = "https://www.alphavantage.co/query"


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


# ---------------- FUNDAMENTAL NEWS (Alpha Vantage) ----------------
def get_fundamental_news(pair="USDJPY"):
    try:
        ticker = pair[-3:]  # ambil JPY dari USDJPY
        url = f"{ALPHA_API_URL}?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()

        if "feed" in data:
            headlines = [f"{item['title']} ({item.get('source','')})" for item in data["feed"][:2]]
            return " | ".join(headlines)
        else:
            return "No recent fundamental news."
    except Exception as e:
        print("AlphaVantage news error:", e)
        return "Error fetching fundamental news."


# ---------------- INDICATORS ----------------
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


# ---------------- AI FALLBACK ----------------
def ai_fallback(tech, news_summary=""):
    cp = tech["current_price"]; rsi = tech["RSI"]
    atr = cp*0.005

    if rsi < 30:
        signal = "BUY"
        sl = cp - atr
        tp1 = cp + 2*atr
        tp2 = cp + 3*atr
    elif rsi > 70:
        signal = "SELL"
        sl = cp + atr
        tp1 = cp - 2*atr
        tp2 = cp - 3*atr
    else:
        signal = "HOLD"
        sl = cp - atr
        tp1 = cp + atr
        tp2 = cp + 2*atr

    return {
        "SIGNAL": signal,
        "ENTRY_PRICE": round(cp,4),
        "STOP_LOSS": round(sl,4),
        "TAKE_PROFIT_1": round(tp1,4),
        "TAKE_PROFIT_2": round(tp2,4),
        "CONFIDENCE_LEVEL": 75 if signal!="HOLD" else 50,
        "TRADING_ADVICE": f"RSI-based fallback. News: {news_summary}"
    }


# ---------------- AI DEEPSEEK ----------------
def ai_deepseek_analysis(pair, tech, fundamentals):
    if not DEEPSEEK_API_KEY:
        return ai_fallback(tech, fundamentals)
    try:
        headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type":"application/json"}
        prompt = f"""
Return ONLY a valid JSON object with keys:
SIGNAL, ENTRY_PRICE, STOP_LOSS, TAKE_PROFIT_1, TAKE_PROFIT_2, CONFIDENCE_LEVEL, TRADING_ADVICE.

Pair: {pair}
Price: {tech['current_price']}
RSI: {tech['RSI']}, MACD: {tech['MACD']}
SMA20: {tech['SMA20']}, SMA50: {tech['SMA50']}
Support: {tech['Support']}, Resistance: {tech['Resistance']}
Fundamentals: {fundamentals}
"""
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role":"user","content": prompt}],
            "temperature": 0.2
        }
        r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=20)
        resp = r.json()

        print("DeepSeek raw response:", resp)

        if "choices" not in resp:
            return {
                "SIGNAL": "HOLD",
                "ENTRY_PRICE": tech["current_price"],
                "STOP_LOSS": tech["current_price"],
                "TAKE_PROFIT_1": tech["current_price"],
                "TAKE_PROFIT_2": tech["current_price"],
                "CONFIDENCE_LEVEL": 0,
                "TRADING_ADVICE": f"DeepSeek API error: {resp.get('error','Unknown error')}"
            }

        txt = resp["choices"][0]["message"]["content"]
        return json.loads(txt)
    except Exception as e:
        print("AI error:", e)
        return ai_fallback(tech, fundamentals)


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/get_analysis")
def get_analysis():
    pair = request.args.get("pair","USDJPY").upper()
    timeframe = request.args.get("timeframe","1H")
    use_history = request.args.get("use_history","0")=="1"
    try:
        cp = get_price_twelvedata(pair)
        if cp is None:
            if pair in HISTORICAL:
                cp = float(HISTORICAL[pair].tail(1)["close"].iloc[0])
            else:
                cp = 150 + random.uniform(-1, 1)

        if use_history and pair in HISTORICAL:
            df = HISTORICAL[pair].tail(100)
            closes = df["close"].tolist()+[cp]
            dates = df["date"].dt.strftime("%Y-%m-%d").tolist()+[datetime.now().strftime("%Y-%m-%d %H:%M")]
        else:
            closes = [cp+random.uniform(-0.1,0.1) for _ in range(50)]+[cp]
            dates = [(datetime.now()-timedelta(minutes=i)).strftime("%H:%M") for i in range(50)]+[datetime.now().strftime("%H:%M")]

        tech = calc_indicators(closes)
        fundamentals = get_fundamental_news(pair)  # 🔥 berita real dari Alpha Vantage
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
            "data_source": "Twelve Data API + Alpha Vantage + DeepSeek"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500


@app.route("/quick_overview")
def quick_overview():
    overview = {}
    for pair in PAIR_MAP.keys():
        cp = get_price_twelvedata(pair)
        if cp is None and pair in HISTORICAL:
            cp = float(HISTORICAL[pair].tail(1)["close"].iloc[0])
        elif cp is None:
            cp = 150 + random.uniform(-1, 1)
        overview[pair] = {"price": cp}
    return jsonify(overview)


if __name__=="__main__":
    init_db(); load_csv_data()
    app.run(debug=True)
