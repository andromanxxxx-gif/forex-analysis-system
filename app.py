from flask import Flask, render_template, request, jsonify
from ai_analysis import get_forex_data, get_news, calc_indicators, analyze_with_gpt
from config import MAJOR_PAIRS
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "templates")
static_dir = os.path.join(base_dir, "static")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)


@app.route("/")
def index():
    return render_template("index.html", pairs=MAJOR_PAIRS)


@app.route("/analyze", methods=["POST"])
def analyze():
    symbol = request.form.get("pair")
    df = get_forex_data(symbol)
    df = calc_indicators(df)
    news = get_news(symbol)
    analysis = analyze_with_gpt(symbol, df, news)
    return jsonify({
        "pair": symbol,
        "analysis": analysis,
        "news": news,
        "data": df.tail(50).to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)
