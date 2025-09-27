# dashboard/app.py
from flask import Flask, render_template, request
import yfinance as yf
from src.trading_signals import calculate_indicators, generate_signal
from src.deepseek_client import analyze_market

app = Flask(__name__)
PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]

def summarize_technical(df):
    last = df.iloc[-1]
    summary = f"""
Pair: {last.name}
Close: {last['Close']:.5f}
EMA200: {last['EMA200']:.5f}
MACD: {last['MACD']:.5f}, Signal: {last['MACD_signal']:.5f}
OBV: {last['OBV']:.2f}
"""
    return summary

@app.route("/", methods=["GET", "POST"])
def index():
    pair = request.form.get("pair", PAIRS[0])
    news_text = request.form.get("news_text", "")
    
    df = yf.download(pair, period="7d", interval="1h")
    df = df.reset_index()
    df = calculate_indicators(df)
    signal = generate_signal(df)
    
    tech_summary = summarize_technical(df)
    ai_recommendation = analyze_market(tech_summary, news_text) if news_text else analyze_market(tech_summary)
    
    chart_data = df.to_dict(orient="records")
    
    return render_template(
        "index.html",
        pairs=PAIRS,
        selected_pair=pair,
        chart_data=chart_data,
        signal=signal,
        ai_recommendation=ai_recommendation
    )

if __name__ == "__main__":
   import webbrowser
import threading
import time

def open_browser():
    time.sleep(2)  # tunggu server siap
    webbrowser.open("http://127.0.0.1:5000")

threading.Thread(target=open_browser).start()

    app.run(debug=True)
