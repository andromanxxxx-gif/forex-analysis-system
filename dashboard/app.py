from flask import Flask, render_template, request, jsonify
import pandas as pd
import yfinance as yf
from src.trading_signals import calculate_indicators, generate_signal

app = Flask(__name__)

PAIRS = ["EURUSD=X", "GBPUSD=X", "JPY=X"]  # Yahoo Finance symbols

@app.route("/")
def index():
    return render_template("index.html", pairs=PAIRS)

@app.route("/get_data", methods=["POST"])
def get_data():
    pair = request.json.get("pair")
    df = yf.download(pair, period="6mo", interval="1d")
    if df.empty:
        return jsonify({"error": "Data not found"}), 404
    
    df = calculate_indicators(df)
    signal, tp, sl = generate_signal(df)
    
    data = {
        "dates": df.index.strftime("%Y-%m-%d").tolist(),
        "open": df["Open"].tolist(),
        "high": df["High"].tolist(),
        "low": df["Low"].tolist(),
        "close": df["Close"].tolist(),
        "ema200": df["EMA200"].tolist(),
        "macd": df["MACD"].tolist(),
        "macd_signal": df["MACD_signal"].tolist(),
        "obv": df["OBV"].tolist(),
        "signal": signal,
        "tp": tp,
        "sl": sl
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
