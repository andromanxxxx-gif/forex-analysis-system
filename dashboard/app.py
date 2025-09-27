from flask import Flask, render_template, request
import pandas as pd
from src.trading_signals import calculate_indicators, generate_signal
from src.news_analyzer import NewsAnalyzer
import yfinance as yf

app = Flask(__name__)
news_analyzer = NewsAnalyzer()

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X"
}

@app.route("/", methods=["GET", "POST"])
def index():
    pair = request.form.get("pair") or "EURUSD"
    ticker = PAIRS[pair]

    # Ambil data harga
    data = yf.download(ticker, period="3mo", interval="1d")
    if data.empty:
        return "Data tidak tersedia"

    data = calculate_indicators(data)
    signal, tp, sl = generate_signal(data)

    # Contoh berita dummy
    news = [
        {"headline": "EUR/USD naik karena data ekonomi positif", "sentiment": news_analyzer.analyze_sentiment("EUR/USD naik karena data ekonomi positif")},
        {"headline": "USD/JPY stabil menjelang keputusan bank sentral", "sentiment": news_analyzer.analyze_sentiment("USD/JPY stabil menjelang keputusan bank sentral")}
    ]

    return render_template("index.html",
                           pair=pair,
                           pairs=list(PAIRS.keys()),
                           data=data.tail(10).to_dict(orient="records"),
                           signal=signal,
                           tp=round(tp,5),
                           sl=round(sl,5),
                           news=news)

if __name__ == "__main__":
    app.run(debug=True)
