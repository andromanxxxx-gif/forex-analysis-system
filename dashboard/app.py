from flask import Flask, render_template, request
import pandas as pd
import yfinance as yf
from src.trading_signals import calculate_indicators, generate_signal
from src.news_analyzer import NewsAnalyzer
import plotly.graph_objs as go
import plotly
import json

app = Flask(__name__)
news_analyzer = NewsAnalyzer()

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X"
}

def create_candlestick_chart(df, pair):
    """Buat grafik candlestick dengan indikator EMA200, MACD, OBV"""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick"
    ))

    # EMA200
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA200'],
        line=dict(color='blue', width=1),
        name='EMA200'
    ))

    # MACD dan Signal line di subchart
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        line=dict(color='green', width=1),
        name='MACD',
        yaxis='y2'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal'],
        line=dict(color='red', width=1),
        name='Signal',
        yaxis='y2'
    ))

    # OBV di subchart lain
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['OBV'],
        name='OBV',
        yaxis='y3',
        marker_color='orange'
    ))

    # Layout multi-axis
    fig.update_layout(
        title=f'{pair} Candlestick Chart with Indicators',
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='MACD', overlaying='y', side='right', showgrid=False, position=0.95),
        yaxis3=dict(title='OBV', overlaying='y', side='left', showgrid=False, position=0.05),
        legend=dict(orientation='h')
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

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

    # Grafik candlestick + indikator
    graphJSON = create_candlestick_chart(data, pair)

    # Contoh berita dummy
    news = [
        {"headline": "EUR/USD naik karena data ekonomi positif",
         "sentiment": news_analyzer.analyze_sentiment("EUR/USD naik karena data ekonomi positif")},
        {"headline": "USD/JPY stabil menjelang keputusan bank sentral",
         "sentiment": news_analyzer.analyze_sentiment("USD/JPY stabil menjelang keputusan bank sentral")}
    ]

    return render_template("index.html",
                           pair=pair,
                           pairs=list(PAIRS.keys()),
                           signal=signal,
                           tp=round(tp,5),
                           sl=round(sl,5),
                           news=news,
                           graphJSON=graphJSON)

if __name__ == "__main__":
    app.run(debug=True)
