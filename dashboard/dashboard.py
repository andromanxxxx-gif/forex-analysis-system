import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Path ke data
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "last_signal.json")

# Judul dashboard
st.set_page_config(page_title="Forex ML Dashboard", layout="wide")
st.title("📊 Forex ML Dashboard (EMA200 + MACD + News)")

# Cek apakah data ada
if not os.path.exists(DATA_FILE):
    st.error("❌ Data tidak ditemukan. Jalankan generate_dummy.py dulu.")
    st.stop()

# Load data JSON
with open(DATA_FILE, "r") as f:
    signals = json.load(f)

# Pilihan pair & timeframe
pairs = list(signals.keys())
pair = st.sidebar.selectbox("Pilih Pair", pairs, index=0)

timeframes = list(signals[pair].keys())
timeframe = st.sidebar.selectbox("Pilih Timeframe", timeframes, index=0)

# Data untuk pair & timeframe terpilih
df = pd.DataFrame(signals[pair][timeframe])

# Pastikan kolom datetime
df["time"] = pd.to_datetime(df["time"])

# === CHART ===
st.subheader(f"📈 Chart {pair} ({timeframe})")

fig = go.Figure()

# Tambah candlestick
fig.add_trace(go.Candlestick(
    x=df["time"],
    open=df["price"] - 0.2,
    high=df["price"] + 0.5,
    low=df["price"] - 0.5,
    close=df["price"],
    name="Candles"
))

# Tambah prediksi (Buy = Hijau, Sell = Merah)
buy_signals = df[df["signal"] == "BUY"]
sell_signals = df[df["signal"] == "SELL"]

fig.add_trace(go.Scatter(
    x=buy_signals["time"], y=buy_signals["price"],
    mode="markers", name="BUY", marker=dict(color="green", size=10, symbol="triangle-up")
))

fig.add_trace(go.Scatter(
    x=sell_signals["time"], y=sell_signals["price"],
    mode="markers", name="SELL", marker=dict(color="red", size=10, symbol="triangle-down")
))

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price",
    height=600,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# === TABEL SINYAL TERBARU ===
st.subheader("📑 Sinyal Terbaru")

n_rows = st.slider("Tampilkan berapa baris terakhir?", 5, 50, 10)

st.dataframe(
    df.tail(n_rows)[["time", "signal", "price", "stop_loss", "take_profit", "prob_up", "news_compound", "news"]],
    use_container_width=True
)
