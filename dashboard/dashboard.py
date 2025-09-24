import streamlit as st
import pandas as pd
import json
import os
import plotly.graph_objects as go

DATA_DIR = "data"
LAST_SIGNAL_FILE = os.path.join(DATA_DIR, "last_signal.json")

# --- Load dummy signals ---
def load_signals():
    if os.path.exists(LAST_SIGNAL_FILE):
        with open(LAST_SIGNAL_FILE, "r") as f:
            return json.load(f)
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Forex Dashboard", layout="wide")
st.title("📊 Forex Analysis Dashboard")

signals = load_signals()

if signals:
    st.subheader("Latest Signals")
    st.json(signals)

    # Chart untuk setiap pair
    for pair in signals.keys():
        st.subheader(f"Chart {pair}")
        df = pd.DataFrame(signals[pair])
        df["time"] = pd.to_datetime(df["time"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["time"], y=df["price"], mode="lines", name="Price"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema200"], mode="lines", name="EMA200"))
        fig.add_trace(go.Bar(x=df["time"], y=df["macd"], name="MACD"))

        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No signals found. Run generate_dummy.py first.")
