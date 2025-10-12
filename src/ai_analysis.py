# ai_analysis.py
import requests
import pandas as pd
import numpy as np
from config import TWELVE_DATA_API_KEY, NEWS_API_KEY, OPENAI_API_KEY
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

# =======================
# 🔹 Ambil data harga forex
# =======================
def get_forex_data(symbol: str, interval="1min", outputsize=100):
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON"
    }

    r = requests.get(base_url, params=params)
    data = r.json()

    if "values" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df = df.astype({"open": float, "high": float, "low": float, "close": float})
    return df


# =======================
# 🔹 Hitung indikator teknikal
# =======================
def calc_indicators(df):
    if df.empty:
        return df

    df["EMA26"] = df["close"].ewm(span=26).mean()
    df["EMA12"] = df["close"].ewm(span=12).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    return df


# =======================
# 🔹 Ambil berita ekonomi
# =======================
def get_news(symbol: str, limit=5):
    keyword = symbol.replace("/", "")
    url = f"https://newsapi.org/v2/everything?q={keyword}&language=en&sortBy=publishedAt&pageSize={limit}&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    articles = r.json().get("articles", [])
    return [
        {"title": a["title"], "url": a["url"], "source": a["source"]["name"]}
        for a in articles
    ]


# =======================
# 🔹 Analisis AI (ChatGPT)
# =======================
def analyze_with_gpt(symbol, df, news_list):
    if df.empty:
        return "⚠️ Data harga tidak tersedia."

    latest = df.iloc[-1]
    trend = "bullish" if latest["EMA12"] > latest["EMA26"] else "bearish"
    sentiment_summary = " | ".join([n["title"] for n in news_list[:3]])

    prompt = f"""
    Kamu adalah analis forex profesional. 
    Analisis data teknikal dan berita ekonomi berikut:

    Pasangan: {symbol}
    Tren teknikal: {trend}
    Harga terakhir: {latest['close']}
    EMA12: {latest['EMA12']:.5f}, EMA26: {latest['EMA26']:.5f}
    MACD: {latest['MACD']:.5f}, Signal: {latest['Signal']:.5f}, RSI: {latest['RSI']:.2f}

    Berita terkait:
    {sentiment_summary}

    Berikan analisis singkat (maks 5 kalimat) tentang arah pasar, potensi entry/exit, dan risiko.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Anda adalah analis forex profesional."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    return response.choices[0].message.content.strip()
