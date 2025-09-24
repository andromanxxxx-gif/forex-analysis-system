import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# === Konfigurasi Streamlit ===
st.set_page_config(page_title="Forex ML Dashboard", layout="wide")

# === Sidebar ===
st.sidebar.title("⚙️ Settings")

# Pilihan interval refresh
refresh_options = {
    "1 menit": 60,
    "15 menit": 900,
    "1 jam": 3600,
    "4 jam": 14400
}
refresh_choice = st.sidebar.selectbox("Auto-refresh interval", list(refresh_options.keys()))
refresh_seconds = refresh_options[refresh_choice]

# === Auto refresh sesuai pilihan user ===
st_autorefresh = st.experimental_rerun  # fallback untuk versi lama
try:
    st_autorefresh = st.experimental_autorefresh
except AttributeError:
    pass

st_autorefresh(interval=refresh_seconds * 1000, limit=None, key="refresh_timer")

# === Load Data Dummy ===
DATA_PATH = os.path.join("data", "last_signal.json")

if not os.path.exists(DATA_PATH):
    st.error("❌ Data tidak ditemukan. Jalankan generate_dummy.py dulu!")
    st.stop()

with open(DATA_PATH, "r") as f:
    signals = json.load(f)

PAIRS = list(signals["pairs"].keys())

pair = st.sidebar.selectbox("Pilih Pair", PAIRS)
n_rows = st.sidebar.slider("Jumlah data ditampilkan", 10, 50, 20)

# === Main Content ===
st.title("📊 Forex ML Dashboard")
st.subheader(f"Pair: {pair}")

# === Convert ke DataFrame ===
pair_data = signals["pairs"][pair]
df = pd.DataFrame(pair_data)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

# === Tabel Sinyal Terbaru ===
st.write("📌 **Sinyal Terbaru**")
st.dataframe(
    df.tail(n_rows)[
        ["time", "signal", "price", "stop_loss", "take_profit", "prob_up", "news_compound"]
    ]
)

# === Gauge: Probabilitas & Sentimen ===
st.write("🧭 **Probabilitas & Sentimen**")
last_row = df.iloc[-1]  # ambil sinyal terbaru
col1, col2 = st.columns(2)

with col1:
    fig_prob = go.Figure(go.Indicator(
        mode="gauge+number",
        value=last_row["prob_up"] * 100,
        title={"text": "Probabilitas Naik (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "blue"},
            "steps": [
                {"range": [0, 40], "color": "red"},
                {"range": [40, 60], "color": "yellow"},
                {"range": [60, 100], "color": "green"},
            ],
        }
    ))
    st.plotly_chart(fig_prob, use_container_width=True)

with col2:
    fig_sent = go.Figure(go.Indicator(
        mode="gauge+number",
        value=last_row["news_compound"],
        title={"text": "Sentimen Berita (-1 = Negatif, +1 = Positif)"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": "purple"},
            "steps": [
                {"range": [-1, -0.2], "color": "red"},
                {"range": [-0.2, 0.2], "color": "yellow"},
                {"range": [0.2, 1], "color": "green"},
            ],
        }
    ))
    st.plotly_chart(fig_sent, use_container_width=True)

# === Candlestick Chart Harga + EMA200 + Buy/Sell markers + TP/SL ===
st.write("📈 **Candlestick Chart + EMA200 + TP/SL**")

fig_candle = go.Figure()

# Candlestick (pakai price sebagai proxy OHLC dummy)
fig_candle.add_trace(go.Candlestick(
    x=df["time"],
    open=df["price"] - 0.2,
    high=df["price"] + 0.4,
    low=df["price"] - 0.4,
    close=df["price"],
    name="Price"
))

# EMA200
fig_candle.add_trace(go.Scatter(
    x=df["time"], y=df["ema200"],
    mode="lines", name="EMA200", line=dict(color="orange")
))

# Buy markers
buy_df = df[df["signal"] == "BUY"]
fig_candle.add_trace(go.Scatter(
    x=buy_df["time"], y=buy_df["price"],
    mode="markers", marker=dict(color="green", size=12, symbol="triangle-up"),
    name="BUY"
))

# Sell markers
sell_df = df[df["signal"] == "SELL"]
fig_candle.add_trace(go.Scatter(
    x=sell_df["time"], y=sell_df["price"],
    mode="markers", marker=dict(color="red", size=12, symbol="triangle-down"),
    name="SELL"
))

# TP/SL lines
for idx, row in df.iterrows():
    fig_candle.add_trace(go.Scatter(
        x=[row["time"], row["time"]],
        y=[row["take_profit"], row["take_profit"]],
        mode="lines",
        line=dict(color="green", dash="dot"),
        name="TP" if idx == 0 else None,
        showlegend=(idx == 0)
    ))
    fig_candle.add_trace(go.Scatter(
        x=[row["time"], row["time"]],
        y=[row["stop_loss"], row["stop_loss"]],
        mode="lines",
        line=dict(color="red", dash="dot"),
        name="SL" if idx == 0 else None,
        showlegend=(idx == 0)
    ))

fig_candle.update_layout(
    title=f"{pair} Price Action with EMA200 & TP/SL",
    xaxis_title="Time",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig_candle, use_container_width=True)

# === Chart MACD ===
st.write("📉 **MACD**")

fig_macd = go.Figure()
fig_macd.add_trace(go.Bar(
    x=df["time"], y=df["macd"], name="MACD", marker_color="blue"
))
fig_macd.update_layout(title=f"{pair} MACD", xaxis_title="Time", yaxis_title="MACD Value")
st.plotly_chart(fig_macd, use_container_width=True)

st.success(f"✅ Dashboard berhasil dimuat! Auto-refresh setiap {refresh_choice}.")
