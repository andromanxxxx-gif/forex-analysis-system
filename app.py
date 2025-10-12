from flask import Flask, jsonify, request, render_template
from ai_analysis import get_forex_data, get_news, calc_indicators, analyze_with_gpt
from config import MAJOR_PAIRS
import pandas as pd

app = Flask(__name__, static_folder="static", template_folder="static")

@app.route("/")
def index():
    return render_template("index.html", pairs=MAJOR_PAIRS)

@app.route("/get_data", methods=["GET"])
def get_data():
    pair = request.args.get("pair", "EUR/USD")
    timeframe = request.args.get("timeframe", "1h")
    df = get_forex_data(pair, interval=timeframe)
    df = calc_indicators(df)
    news = get_news(pair)
    analysis = analyze_with_gpt(pair, df, news)
    prices = df.tail(200)[["date","open","high","low","close","EMA12","EMA26","MACD","Signal"]].to_dict(orient="records")
    return jsonify({"pair":pair,"prices":prices,"analysis":analysis})

if __name__ == "__main__":
    app.run(debug=True)
