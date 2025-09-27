from flask import Flask, render_template, request
import pandas as pd
from src.trading_signals import calculate_indicators, generate_signal
from src.ai_analysis import analyze_news

app = Flask(__name__)

# Dummy historical data
def get_dummy_data(pair):
    data = pd.DataFrame({
        "Open": [1.0, 1.01, 1.02, 1.03, 1.04],
        "Close": [1.01, 1.02, 1.03, 1.02, 1.05],
        "Volume": [1000, 1200, 1100, 1050, 1300]
    })
    return data

@app.route("/", methods=["GET", "POST"])
def index():
    pair = request.form.get("pair", "EUR/USD")
    data = get_dummy_data(pair)
    data = calculate_indicators(data)
    signal = generate_signal(data)
    
    # Contoh analisis berita AI
    news_text = f"Latest news for {pair}"
    ai_recommendation = analyze_news(news_text)

    tables = [data.to_html(classes='data')]
    return render_template("index.html", pair=pair, signal=signal, ai_recommendation=ai_recommendation, tables=tables)

if __name__ == "__main__":
    app.run(debug=True)
