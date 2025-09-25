import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime

# ======================
# Fix Import Path untuk src/
# ======================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.predictor import Predictor  # sekarang aman

# ======================
# Konfigurasi Halaman
# ======================
st.set_page_config(page_title="Forex ML Dashboard", layout="wide")
st.title("📊 Forex ML Dashboard (MACD + EMA200 + News)")

# ======================
# Load Dummy Signals
# ======================
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "last_signal.json")

signals = {}
if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r") as f:
        signals = json.load(f)
else:
    st.warning("⚠️ Data dummy belum tersedia. Jalankan generate_dummy.py dulu.")

# ======================
# Init Predictor
# ======================
predictor = Predictor(model_path=os.path.join(PROJECT_ROOT, "models", "predictor.pkl"))

# ======================
# Sidebar
# ======================
pairs = list(signals.keys()) if signals else ["GBPJPY", "CHFJPY", "EURJPY", "USDJPY"]
selected_pair = st.sidebar.selectbox("Pilih Pasangan Forex", pairs)

timeframes = ["10m", "30m", "1h", "4h", "1d"]
selected_tf = st.sidebar.selectbox("Pilih Timeframe", timeframes, index=3)  # default 4h

n_rows = st.sidebar.slider("Jumlah data terakhir", 10, 200, 50)

# ======================
# Tampilkan Data
# ======================
if selected_pair in signals:
    df = pd.DataFrame(signals[selected_pair])
    if not df.empty:
        df = df.copy()  # hindari SettingWithCopyWarning

        # Pastikan kolom waktu ada
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        else:
            df["time"] = pd.date_range(end=datetime.now(), periods=len(df), freq="4h")

        # Tambahkan prediksi model
        try:
            df["predicted"], df["prob_up"] = predictor.predict(df)
        except Exception as e:
            st.warning(f"⚠️ Model not found, fallback ke prediksi sederhana. ({e})")
            df["predicted"] = df["signal"]
            df["prob_up"] = 0.5

        # ======================
        # Chart Harga
        # ======================
        fig_price = go.Figure()
        fig_price.add_trace(
            go.Candlestick(
                x=df["time"],
                open=df["price"],
                high=df["price"] * 1.002,
                low=df["price"] * 0.998,
                close=df["price"],
                name="Candles"
            )
        )
        fig_price.update_layout(title=f"{selected_pair} - Timeframe {selected_tf}", xaxis_rangeslider_visible=False)

        st.subheader("📈 Chart Harga")
        st.plotly_chart(fig_price, config={"responsive": True}, width="stretch")

        # ======================
        # Chart Probabilitas
        # ======================
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(x=df["time"], y=df["prob_up"], mode="lines+markers", name="Prob Naik"))
        fig_prob.update_layout(title="Prediksi Probabilitas Naik")

        st.subheader("📊 Prediksi Probabilitas")
        st.plotly_chart(fig_prob, config={"responsive": True}, width="stretch")

        # ======================
        # Tabel Sinyal
        # ======================
        st.subheader("📋 Data Sinyal Terakhir")
        expected_cols = ["time", "signal", "predicted", "price", "stop_loss", "take_profit", "prob_up", "news_compound"]
        available_cols = [col for col in expected_cols if col in df.columns]
        st.dataframe(df.tail(n_rows)[available_cols])
    else:
        st.warning("⚠️ Data kosong untuk pasangan ini.")
else:
    st.error("❌ Pasangan tidak ditemukan dalam data.")
