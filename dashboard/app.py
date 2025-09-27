# dashboard/app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from src.trading_signals import calculate_indicators, generate_signal
from src.news_analyzer import NewsAnalyzer
from config.settings import DEEPSEEK_API_KEY

app = Flask(__name__)

# Inisialisasi analisis berita AI
news_ai = NewsAnalyzer(api_key=DEEPSEEK_API_KEY)

# Default pairs
PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X"
}

@app.route("/")
def index():
    return render_template("index.html", pairs=list(PAIRS.keys()))

@app.route("/get_data", methods=["POST"])
def get_data():
    pair_name = request.json.get("pair")
    ticker = PAIRS.get(pair_name, "EURUSD=X")
    
    # Ambil data historis 1 bulan
    df = yf.download(ticker, period="1mo", interval="1h")
    if df.empty:
        return jsonify({"error": "Data not found"}), 404
    
    df.reset_index(inplace=True)
    df = calculate_indicators(df)

    # Generate signals
    signals = generate_signal(df)

    # AI News Analysis
    news_summary = news_ai.analyze(pair_name)

    # Buat chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Candlestick'
    ))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA200'], mode='lines', name='EMA200'))
    
    fig.update_layout(title=f"{pair_name} Chart", xaxis_title="Datetime", yaxis_title="Price")
    
    chart_html = fig.to_html(full_html=False)
    
    return jsonify({
        "chart": chart_html,
        "signals": signals,
        "news_summary": news_summary
    })

if __name__ == "__main__":
    app.run(debug=True)
