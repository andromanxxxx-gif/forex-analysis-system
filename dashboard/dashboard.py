import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# === Konfigurasi Streamlit ===
st.set_page_config(page_title="Forex Analysis Dashboard", layout="wide")

# === Load Data Dummy ===
DATA_PATH = os.path.join("data", "last_signal.json")

if not os.path.exists(DATA_PATH):
    st.error("❌ Data tidak ditemukan. Jalankan generate_dummy.py dulu!")
    st.stop()

with open(DATA_PATH, "r") as f:
    signals = json.load(f)

PAIRS = list(signals.keys())

# === Sidebar ===
st.sidebar.title("⚙️ Settings")
pair = st.sidebar.selectbox("Pilih Pair", PAIRS)
n_rows = st.sidebar.slider("Jumlah data ditampilkan", 10, 50, 20)

# === Main Content ===
st.title("📊 Forex ML Dashboard")
st.subheader(f"Pair: {pair}")

# === Convert ke DataFrame ===
df = pd.DataFrame(signals[pair])
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# === Tabel Sinyal Terbaru ===
st.write("📌 **Sinyal Terbaru**")
st.dataframe(df.tail(n_rows)[["time", "signal", "price", "stop_loss", "take_profit"]])

# === Chart Harga + EMA200 ===
st.write("📈 **Harga vs EMA200**")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=df["time"], y=df["price"], mode="lines+markers", name="Price"
))
fig_price.add_trace(go.Scatter(
    x=df["time"], y=df["ema200"], mode="lines", name="EMA200"
))

# Tambahkan marker Buy/Sell
buy_signals = df[df["signal"] == "BUY"]
sell_signals = df[df["signal"] == "SELL"]

fig_price.add_trace(go.Scatter(
    x=buy_signals["time"], y=buy_signals["price"],
    mode="markers", marker=dict(color="green", size=10, symbol="triangle-up"),
    name="BUY"
))
fig_price.add_trace(go.Scatter(
    x=sell_signals["time"], y=sell_signals["price"],
    mode="markers", marker=dict(color="red", size=10, symbol="triangle-down"),
    name="SELL"
))

fig_price.update_layout(title=f"{pair} Price & EMA200", xaxis_title="Time", yaxis_title="Price")
st.plotly_chart(fig_price, use_container_width=True)

# === Chart MACD ===
st.write("📉 **MACD**")

fig_macd = go.Figure()
fig_macd.add_trace(go.Bar(
    x=df["time"], y=df["macd"], name="MACD", marker_color="blue"
))
fig_macd.update_layout(title=f"{pair} MACD", xaxis_title="Time", yaxis_title="MACD Value")
st.plotly_chart(fig_macd, use_container_width=True)

st.success("✅ Dashboard berhasil dimuat!")
