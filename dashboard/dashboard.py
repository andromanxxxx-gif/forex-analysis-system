import os
import sys
import json
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from datetime import datetime

# ======================
# Pastikan src bisa diimport
# ======================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.predictor import Predictor  # ✅ Import aman

# ======================
# Konfigurasi Halaman
# ======================
st.set_page_config(page_title="Forex ML Dashboard", layout="wide")
st.title("📊 Forex ML Dashboard (MACD + EMA200 + News)")

# ======================
# Auto Refresh Setting
# ======================
REFRESH_INTERVAL = 4 * 60 * 60 * 1000  # 4 jam dalam milidetik
refresh_rate = st.sidebar.number_input("⏱️ Auto-refresh (ms)", min_value=60000, value=REFRESH_INTERVAL, step=60000)

st.sidebar.markdown(f"⚡ Dashboard akan auto-refresh setiap **{refresh_rate/1000/60:.0f} menit**.")

# Inject JS untuk refresh otomatis
st.markdown(
    f"""
    <meta http-equiv="refresh" content="{int(refresh_rate/1000)}">
    """,
    unsafe_allow_html=True
)

# ======================
# Load Data Sinyal
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
        df = df.copy()  # ✅ hindari SettingWithCopyWarning

        # Pastikan kolom waktu ada
