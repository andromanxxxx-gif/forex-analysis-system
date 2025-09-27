# dashboard/app.py
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from flask import Flask, render_template, request
import pandas as pd
from src.trading_signals import calculate_indicators, generate_signal
from src.ai_analysis import analyze_news

app = Flask(__name__)

# Dummy data untuk testing
def get_dummy_data(pair):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    df = pd.DataFrame({
        "Date": dates,
        "Open": pd.Series(1.1 + 0.01*np.random.randn(len(dates))),
        "Close": pd.Series(1.1 + 0.01*np.random.randn(len(dates))),
        "High": pd.Series(1.1 + 0.01*np.random.randn(len(dates))),
        "Low": pd.Series(1.1 + 0.01*np.random.randn(len(dates))),
        "Volume": pd.Series(1000 + 50*np.random.randn(len(dates))),
    })
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    pair = request.form.get("pair", "EUR/USD")
    df = get_dummy_data(pair)
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
