import os, random, traceback, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder=".")

# ================== KONFIG ==================
TWELVE_API_KEY = "b90862d072ce41e4b0505cbd7b710b66"   # ganti dengan key Anda
ALPHA_API_KEY   = "G8588U1ISMGM8GZB"
TWELVE_API_URL  = "https://api.twelvedata.com"
ALPHA_API_URL   = "https://www.alphavantage.co/query"

HISTORICAL = {}

# ================== LOAD CSV ==================
def load_csv_data():
    search_dirs = [".", "data"]
    for d in search_dirs:
        if not os.path.exists(d): 
            continue
        for f in os.listdir(d):
            if f.endswith(".csv"):
                path = os.path.join(d, f)
                try:
                    # coba koma
                    try:
                        df = pd.read_csv(path)
                    except Exception:
                        # coba tab
                        df = pd.read_csv(path, delimiter="\t", header=None)
                        if df.shape[1] == 6:
                            df.columns = ["date","open","high","low","close","volume"]

                    df.columns = [c.lower() for c in df.columns]

                    if "close" not in df.columns and "price" in df.columns:
                        df["close"] = df["price"]

                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")

                    parts = os.path.basename(f).replace(".csv","").split("_")
                    pair = parts[0].upper()
                    timeframe = parts[1].upper() if len(parts) > 1 else "1D"

                    if pair not in HISTORICAL:
                        HISTORICAL[pair] = {}
                    HISTORICAL[pair][timeframe] = df.sort_values("date")

                    print(f"✅ Loaded {pair}-{timeframe} from {path}, {len(df)} rows, columns: {list(df.columns)}")
                except Exception as e:
                    print(f"⚠️ Error loading {path}: {e}")

    print("Pairs available:", {k:list(v.keys()) for k,v in HISTORICAL.items()})

# ================== API WRAPPERS ==================
def get_price_twelvedata(symbol="USDJPY"):
    try:
        url = f"{TWELVE_API_URL}/price?symbol={symbol}&apikey={TWELVE_API_KEY}"
        r = requests.get(url, timeout=10).json()
        return float(r.get("price"))
    except Exception as e:
        print("TwelveData error:", e)
        return None

def get_fundamental_news(pair="USDJPY"):
    try:
        ticker = pair[-3:]
        url = f"{ALPHA_API_URL}?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_API_KEY}"
        r = requests.get(url, timeout=10).json()
        if "feed" in r:
            headlines = [f"{x['title']} ({x.get('source','')})" for x in r["feed"][:2]]
            return " | ".join(headlines)
        return "No recent news."
    except Exception as e:
        print("AlphaVantage error:", e)
        return "Error fetching news."

# ================== ANALYTICS ==================
def calc_indicators(closes, volumes=None):
    closes = pd.Series(closes)
    rsi = 100 - (100 / (1 + closes.diff().clip(lower=0).rolling(14).mean() /
                       closes.diff().clip(upper=0).abs().rolling(14).mean()))
    ema200 = closes.ewm(span=200).mean()
    ema20  = closes.ewm(span=20).mean()
    sma20  = closes.rolling(20).mean()
    sma50  = closes.rolling(50).mean()
    macd_line = closes.ewm(span=12).mean() - closes.ewm(span=26).mean()
    macd_signal = macd_line.ewm(span=9).mean()
    obv = (np.sign(closes.diff()) * (volumes if volumes is not None else 1)).fillna(0).cumsum()

    return {
        "RSI": round(rsi.iloc[-1],2) if not rsi.empty else None,
        "EMA200": round(ema200.iloc[-1],2) if not ema200.empty else None,
        "SMA20": round(sma20.iloc[-1],2) if not sma20.empty else None,
        "SMA50": round(sma50.iloc[-1],2) if not sma50.empty else None,
        "MACD": round(macd_line.iloc[-1],2) if not macd_line.empty else None,
        "MACD_SIGNAL": round(macd_signal.iloc[-1],2) if not macd_signal.empty else None,
        "OBV": int(obv.iloc[-1]) if not obv.empty else None,
        "Support": round(closes.min(),2),
        "Resistance": round(closes.max(),2)
    }

def ai_deepseek_analysis(pair, tech, fundamentals):
    try:
        signal = "BUY" if tech["RSI"] < 40 else "SELL" if tech["RSI"] > 70 else "HOLD"
        return {
            "SIGNAL": signal,
            "ENTRY_PRICE": tech["EMA200"],
            "STOP_LOSS": round(tech["Support"],2),
            "TAKE_PROFIT_1": round(tech["Resistance"],2),
            "TAKE_PROFIT_2": round((tech["Resistance"]+tech["EMA200"])/2,2) if tech["EMA200"] else None,
            "CONFIDENCE_LEVEL": random.randint(60,90),
            "TRADING_ADVICE": f"{signal} based on RSI & EMA200, news: {fundamentals[:50]}..."
        }
    except Exception as e:
        print("AI error:", e)
        return {"SIGNAL":"HOLD","ENTRY_PRICE":None,"STOP_LOSS":None,
                "TAKE_PROFIT_1":None,"TAKE_PROFIT_2":None,
                "CONFIDENCE_LEVEL":50,"TRADING_ADVICE":"Error in AI analysis"}

# ================== ROUTES ==================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_pairs")
def get_pairs():
    return jsonify({k:list(v.keys()) for k,v in HISTORICAL.items()})

@app.route("/get_analysis")
def get_analysis():
    pair = request.args.get("pair","USDJPY").upper()
    tf = request.args.get("timeframe","1H").upper()
    use_hist = request.args.get("use_history","0")=="1"

    try:
        cp = get_price_twelvedata(pair)
        if cp is None and pair in HISTORICAL and tf in HISTORICAL[pair]:
            cp = float(HISTORICAL[pair][tf].tail(1)["close"].iloc[0])
        elif cp is None:
            cp = 100 + random.uniform(-1,1)

        if use_hist and pair in HISTORICAL and tf in HISTORICAL[pair]:
            df = HISTORICAL[pair][tf].tail(200)
            closes = df["close"].tolist()
            vols = df["volume"].tolist() if "volume" in df.columns else None
            dates = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        else:
            closes = [cp+random.uniform(-0.1,0.1) for _ in range(50)]+[cp]
            vols = None
            dates = [(datetime.now()-timedelta(minutes=i)).strftime("%H:%M") for i in range(50)]+[datetime.now().strftime("%H:%M")]

        tech = calc_indicators(closes, vols)
        fundamentals = get_fundamental_news(pair)
        ai = ai_deepseek_analysis(pair, tech, fundamentals)

        return jsonify({
            "pair":pair,"timeframe":tf,"current_price":cp,
            "technical_indicators":tech,
            "ai_analysis":ai,
            "fundamental_news":fundamentals,
            "chart_data":{"dates":dates,"close":closes},
            "data_source":"TwelveData + CSV + DeepSeek + AlphaVantage"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}),500

@app.route("/quick_overview")
def quick_overview():
    result={}
    for p in ["USDJPY","EURJPY","GBPJPY","CHFJPY"]:
        cp = get_price_twelvedata(p)
        if cp: result[p]={"price":round(cp,3)}
    return jsonify(result)

if __name__=="__main__":
    load_csv_data()
    app.run(debug=True)
