from flask import Flask, render_template, request
import pandas as pd
from src.trading_signals import calculate_indicators, generate_signal
from src.news_analyzer import analyze_news
from src.ai_analysis import get_ai_recommendation

app = Flask(__name__)

# Dummy data
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"]
DATA = {
    pair: pd.DataFrame({
        "Open": [1.1,1.2,1.15,1.18,1.17],
        "High": [1.2,1.25,1.2,1.22,1.19],
        "Low":  [1.05,1.18,1.1,1.15,1.14],
        "Close":[1.18,1.2,1.15,1.16,1.18],
        "Volume":[1000,1500,1200,1300,1100]
    }) for pair in PAIRS
}

@app.route("/", methods=["GET", "POST"])
def index():
    pair = request.form.get("pair", "EUR/USD")
    df = DATA[pair].copy()
    df = calculate_indicators(df)
    signal_info = generate_signal(df)
    news_summary = analyze_news()
    ai_reco = get_ai_recommendation(news_summary)
    
    return render_template(
        "index.html",
        pairs=PAIRS,
        selected_pair=pair,
        df=df.to_dict(orient="records"),
        signal=signal_info,
        news=news_summary,
        ai_reco=ai_reco
    )

if __name__ == "__main__":
    app.run(debug=True)
