import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# --- Fix path agar bisa import src/ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predictor import Predictor

# ==============================
# Konfigurasi Dashboard
# ==============================
st.set_page_config(
    page_title="Forex ML Dashboard (H4)",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = os.path.join("data", "last_signal.json")

# ==============================
# Sidebar Settings
# ==============================
st.sidebar.header("⚙️ Settings")
refresh_interval = st.sidebar.selectbox("Auto-refresh interval", ["1 menit", "5 menit", "10 menit"])
pair_selected = st.sidebar.selectbox("Pilih Pair", ["GBPJPY", "USDJPY", "EURJPY", "CHFJPY"])
n_rows = st.sidebar.slider("Jumlah bar historis ditampilkan", min_value=50, max_value=500, value=100)

if st.sidebar.button("Refresh Data"):
    st.experimental_rerun()

# ==============================
# Load Data
# ==============================
if not os.path.exists(DATA_FILE):
    st.error("❌ Data tidak ditemukan. Jalankan generate_dummy.py dulu!")
    st.stop()

with open(DATA_FILE, "r") as f:
    signals = json.load(f)

if pair_selected not in signals:
    st.error(f"❌ Pair {pair_selected} tidak ada di data dummy.")
    st.stop()

df = pd.DataFrame(signals[pair_selected])

# pastikan ada kolom time
if "time" not in df.columns:
    st.error("❌ Data tidak memiliki kolom 'time'.")
    st.stop()

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

# filter hanya ambil n_rows terakhir
df_show = df.tail(n_rows)

# ==============================
# Prediksi Candle Berikutnya
# ==============================
predictor = Predictor(horizon_hours=4)
pred = predictor.predict_next(df_show)

st.subheader("🔮 Prediksi Candle Berikutnya (H4)")
st.table(pd.DataFrame([pred]))

# ==============================
# Chart Candlestick + EMA200 + Prediksi
# ==============================
fig_candle = go.Figure()

fig_candle.add_trace(go.Candlestick(
    x=df_show["time"],
    open=df_show["open"],
    high=df_show["high"],
    low=df_show["low"],
    close=df_show["price"],
    name="Price"
))

# EMA200
if "price" in df_show.columns:
    df_show["ema200"] = df_show["price"].ewm(span=200).mean()
    fig_candle.add_trace(go.Scatter(
        x=df_show["time"],
        y=df_show["ema200"],
        mode="lines",
        line=dict(color="orange", width=1),
        name="EMA200"
    ))

# Marker prediksi
fig_candle.add_trace(go.Scatter(
    x=[pred["Next Time"]],
    y=[pred["Predicted Price"]],
    mode="markers+text",
    marker=dict(color="blue", size=16, symbol="star"),
    text=[f"{pred['Predicted Signal']}"],
    textfont=dict(size=12, color="white"),
    textposition="top center",
    name="Prediction"
))

# TP
fig_candle.add_trace(go.Scatter(
    x=[pred["Next Time"]],
    y=[pred["Take Profit"]],
    mode="markers+text",
    marker=dict(color="green", size=12, symbol="triangle-up"),
    text=["TP"],
    textposition="bottom center",
    name="Pred TP"
))

# SL
fig_candle.add_trace(go.Scatter(
    x=[pred["Next Time"]],
    y=[pred["Stop Loss"]],
    mode="markers+text",
    marker=dict(color="red", size=12, symbol="triangle-down"),
    text=["SL"],
    textposition="top center",
    name="Pred SL"
))

fig_candle.update_layout(
    title=f"{pair_selected} Price Action with EMA200 & Prediction (Timeframe H4)",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
)

# extend axis supaya prediksi kelihatan
fig_candle.update_xaxes(range=[df_show["time"].iloc[-50], pred["Next Time"] + pd.Timedelta(hours=4)])

st.plotly_chart(fig_candle, use_container_width=True)

# ==============================
# Tabel Data Historis
# ==============================
st.subheader("📊 Data Historis")
cols = ["time", "signal", "price", "stop_loss", "take_profit", "prob_up", "news_compound"]
available_cols = [c for c in cols if c in df_show.columns]
st.dataframe(df_show[available_cols].tail(n_rows), use_container_width=True)
