import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from pathlib import Path
import sys

# Tambahkan src ke path
sys.path.append(str(Path(__file__).parent.parent))
from src.trading_signals import calculate_indicators, generate_signal

st.set_page_config(page_title="Forex Analysis Dashboard", layout="wide")
st.title("📈 Forex Analysis System")

# Dropdown pair
pair_map = {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X"}
selected_pair = st.selectbox("Select Forex Pair", list(pair_map.keys()))
ticker = pair_map[selected_pair]

# Ambil data 6 bulan terakhir
data = yf.download(ticker, period="6mo", interval="1d").reset_index()
data = calculate_indicators(data)
signals = generate_signal(data)

# Chart candlestick
fig = go.Figure(data=[go.Candlestick(
    x=data['Date'],
    open=data['Open'], high=data['High'],
    low=data['Low'], close=data['Close'],
    name='Price'
)])

# EMA200
fig.add_trace(go.Scatter(
    x=data['Date'], y=data['EMA200'],
    line=dict(color='orange', width=2),
    name='EMA 200'
))

# MACD
fig.add_trace(go.Scatter(
    x=data['Date'], y=data['MACD'],
    line=dict(color='blue', width=1),
    name='MACD'
))
fig.add_trace(go.Scatter(
    x=data['Date'], y=data['Signal_MACD'],
    line=dict(color='red', width=1),
    name='MACD Signal'
))

# OBV
fig.add_trace(go.Scatter(
    x=data['Date'], y=data['OBV'],
    line=dict(color='green', width=1),
    name='On Balance Volume'
))

fig.update_layout(title=f"{selected_pair} Analysis", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)

# Latest signal
latest = signals.iloc[-1]
st.subheader("Latest Trading Signal")
signal_color = "green" if latest['Signal'] == "Buy" else "red" if latest['Signal'] == "Sell" else "gray"
st.markdown(f"<span style='color:{signal_color}; font-size:24px'><b>{latest['Signal']}</b></span>", unsafe_allow_html=True)
st.write(f"Take Profit: {latest['Take_Profit']}")
st.write(f"Stop Loss: {latest['Stop_Loss']}")
