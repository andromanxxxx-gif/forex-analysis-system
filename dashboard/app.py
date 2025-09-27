# dashboard/app.py
from pathlib import Path
import sys
import yfinance as yf
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from flask import Flask, render_template, request
from src.trading_signals import calculate_indicators, generate_signal
from src.ai_analysis import analyze_news

app = Flask(__name__)

# Mapping pair ke ticker yfinance
PAIR_TICKERS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X"
}

def fetch_data(pair):
    ticker = PAIR_TICKERS.get(pair, "EURUSD=X")
    df = yf.download(ticker, period="6mo", interval="1d")
    df.reset_index(inplace=True)
    df = df.rename(columns={"Adj Close":"Close"})
    return df[['Date','Open','High','Low','Close','Volume']]

@app.route("/", methods=["GET", "POST"])
def index():
    pair = request.form.get("pair", "EUR/USD")
    df = fetch_data(pair)
    df = calculate_indicators(df)
    signal_info = generate_signal(df)
    ai_recommendation = analyze_news(pair)
    
    return render_template("index.html",
                           pair=pair,
                           tables=[df.tail(10).to_html(classes='data', index=False)],
                           signal_info=signal_info,
                           ai_recommendation=ai_recommendation)

if __name__ == "__main__":
    app.run(debug=True)
