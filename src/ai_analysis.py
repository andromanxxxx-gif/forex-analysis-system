import requests
import pandas as pd
from openai import OpenAI
from config import OPENAI_API_KEY, TWELVE_DATA_API_KEY, NEWS_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_forex_data(symbol, interval="1h", outputsize=100):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    r = requests.get(url)
    data = r.json().get("values", [])
    df = pd.DataFrame(data)
    df = df.rename(columns={"datetime":"date"})
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    df = df.sort_values("date")
    return df

def get_news(symbol):
    q = symbol.replace("/", "")
    url = f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    articles = r.json().get("articles", [])
    summaries = [f"{a['title']} - {a.get('source',{}).get('name','')}" for a in articles]
    return "\n".join(summaries[:5])

def calc_indicators(df):
    df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + df["close"].pct_change().clip(lower=0).rolling(14).mean() / df["close"].pct_change().clip(upper=0).abs().rolling(14).mean()))
    return df

def analyze_with_gpt(symbol, df, news_summary):
    recent = df.tail(1).iloc[0]
    prompt = f"""
    Kamu adalah analis forex profesional. 
    Data terakhir untuk {symbol}:
    Open: {recent['open']}, High: {recent['high']}, Low: {recent['low']}, Close: {recent['close']}
    EMA12: {recent['EMA12']}, EMA26: {recent['EMA26']}, MACD: {recent['MACD']}, Signal: {recent['Signal']}, RSI: {recent['RSI']}
    Berita terbaru:
    {news_summary}

    Berdasarkan data teknikal dan fundamental di atas, berikan analisis singkat (maks 5 kalimat)
    dan rekomendasi akhir: BUY, SELL, atau HOLD.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"system","content":"Kamu adalah analis forex berpengalaman."},
                  {"role":"user","content":prompt}]
    )
    return response.choices[0].message.content.strip()
