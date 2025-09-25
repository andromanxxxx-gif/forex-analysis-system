import os
import json
import subprocess
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# === Konfigurasi ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LAST_FILE = os.path.join(DATA_DIR, "last_signal.json")
PAIRS = ["GBPJPY", "CHFJPY", "EURJPY", "USDJPY"]
TIMEFRAMES = ["4h", "1d"]

# === Fungsi Load Data ===
def load_signals():
    if not os.path.exists(LAST_FILE):
        st.warning("⚠️ Data tidak ditemukan. Menjalankan generate_dummy.py ...")
        subprocess.run(["python", "src/generate_dummy.py"])

    try:
        with open(LAST_FILE, "r", encoding="utf-8") as f:
            signals = json.load(f)
    except Exception as e:
        st.error(f"Gagal load data: {e}")
        return {}

    return signals


# === Main Dashboard ===
st.set_page_config(page_title="Forex ML Dashboard", layout="wide")

st.sidebar.header("⚙️ Settings")
refresh_ms = st.sidebar.number_input("⏱ Auto-refresh (ms)", min_value=1000, value=180000)
st.sidebar.markdown(f"⚡ Dashboard auto-refresh setiap **{refresh_ms//60000} menit**.")

pair = st.sidebar.selectbox("📊 Pilih Pasangan Forex", PAIRS)
timeframe = st.sidebar.selectbox("⏲ Pilih Timeframe", TIMEFRAMES)
n_rows = st.sidebar.slider("📈 Jumlah data terakhir", 50, 500, 200)

st.title("📈 Forex ML Dashboard (MACD + EMA200 + News + Prediction)")

# === Load data ===
signals = load_signals()
if not signals:
    st.stop()

df = pd.DataFrame(signals[pair][timeframe])
df["time"] = pd.to_datetime(df["time"])
df = df.tail(n_rows).copy()

# Hitung EMA200
df["ema200"] = df["price"].ewm(span=200, adjust=False).mean()

# Hitung MACD + Histogram
df["ema12"] = df["price"].ewm(span=12, adjust=False).mean()
df["ema26"] = df["price"].ewm(span=26, adjust=False).mean()
df["macd"] = df["ema12"] - df["ema26"]
df["signal_line"] = df["macd"].ewm(span=9, adjust=False).mean()
df["histogram"] = df["macd"] - df["signal_line"]

# Tambahkan prob_down (1 - prob_up)
if "prob_down" not in df.columns:
    df["prob_down"] = 1 - df["prob_up"]

# === Tabel Data ===
st.subheader(f"📑 Data Sinyal: {pair} ({timeframe})")
st.dataframe(
    df[[
        "time", "signal", "price", "stop_loss", "take_profit",
        "prob_up", "prob_down", "news_compound", "news_headline", "pred_price_next"
    ]],
    use_container_width=True
)

# === Candlestick Chart + Prediksi + EMA200 + SL/TP ===
st.subheader(f"📊 Candlestick Chart + EMA200 + Prediksi Candle + SL/TP ({pair}, {timeframe})")
fig = go.Figure()

# Harga historis
fig.add_trace(go.Scatter(
    x=df["time"], y=df["price"],
    mode="lines", name="Price", line=dict(color="orange")
))

# EMA200
fig.add_trace(go.Scatter(
    x=df["time"], y=df["ema200"],
    mode="lines", name="EMA200", line=dict(color="cyan", width=2)
))

# Prediksi candle berikutnya
last_row = df.iloc[-1]
fig.add_trace(go.Scatter(
    x=[last_row["time"] + pd.Timedelta(hours=4 if timeframe == "4h" else 24)],
    y=[last_row["pred_price_next"]],
    mode="markers+text",
    name="Prediksi Candle Berikutnya",
    marker=dict(color="blue", size=12, symbol="diamond"),
    text=[f"{last_row['pred_price_next']:.2f}"],
    textposition="top center"
))

# Buy/Sell markers
buy_signals = df[df["signal"] == "BUY"]
sell_signals = df[df["signal"] == "SELL"]

fig.add_trace(go.Scatter(
    x=buy_signals["time"], y=buy_signals["price"],
    mode="markers", name="BUY",
    marker=dict(color="green", size=10, symbol="triangle-up")
))

fig.add_trace(go.Scatter(
    x=sell_signals["time"], y=sell_signals["price"],
    mode="markers", name="SELL",
    marker=dict(color="red", size=10, symbol="triangle-down")
))

# SL/TP lines
for _, row in df.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["time"], row["time"] + pd.Timedelta(hours=4 if timeframe == "4h" else 24)],
        y=[row["stop_loss"], row["stop_loss"]],
        mode="lines", line=dict(color="red", dash="dot"),
        name="Stop Loss", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[row["time"], row["time"] + pd.Timedelta(hours=4 if timeframe == "4h" else 24)],
        y=[row["take_profit"], row["take_profit"]],
        mode="lines", line=dict(color="green", dash="dot"),
        name="Take Profit", showlegend=False
    ))

fig.update_layout(
    title=f"{pair} Price Action + EMA200 + Prediksi Candle + SL/TP",
    xaxis_title="Time", yaxis_title="Price",
    template="plotly_dark", height=600
)

st.plotly_chart(fig, use_container_width=True)

# === MACD Chart ===
st.subheader("📊 MACD Indicator")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df["time"], y=df["macd"], mode="lines", name="MACD", line=dict(color="yellow")))
fig_macd.add_trace(go.Scatter(x=df["time"], y=df["signal_line"], mode="lines", name="Signal Line", line=dict(color="white")))
fig_macd.add_trace(go.Bar(x=df["time"], y=df["histogram"], name="Histogram",
                          marker=dict(color=df["histogram"].apply(lambda x: "green" if x >= 0 else "red"))))
fig_macd.update_layout(title="MACD vs Signal Line + Histogram", template="plotly_dark", height=300, barmode="relative")
st.plotly_chart(fig_macd, use_container_width=True)

# === Probability Chart ===
st.subheader("📊 Prediksi Probabilitas Harga Naik vs Turun")
fig_prob = go.Figure()
fig_prob.add_trace(go.Bar(x=df["time"], y=df["prob_up"], name="Prob Up", marker_color="green"))
fig_prob.add_trace(go.Bar(x=df["time"], y=df["prob_down"], name="Prob Down", marker_color="red"))
fig_prob.update_layout(barmode="group", title="Probabilitas Harga Naik vs Turun", xaxis_title="Time", yaxis_title="Probability", template="plotly_dark", height=300)
st.plotly_chart(fig_prob, use_container_width=True)

# === News Section ===
st.subheader("📰 News & Sentiment Timeline")
st.table(df.tail(10)[["time", "news_headline", "news_compound"]])
