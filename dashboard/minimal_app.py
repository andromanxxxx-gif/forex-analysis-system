# dashboard/minimal_app.py - File sederhana tanpa external imports
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Minimal Forex Dashboard", layout="wide")
st.title("Minimal Forex Dashboard")
st.write("This is a basic version without complex imports")

# Simple functionality
st.subheader("Sample Data")
data = pd.DataFrame({
    'Pair': ['GBP/JPY', 'USD/JPY', 'EUR/JPY', 'CHF/JPY'],
    'Price': [187.25, 150.80, 160.45, 170.20],
    'Trend': ['Up', 'Down', 'Up', 'Neutral']
})
st.dataframe(data)

st.subheader("Simple Chart")
chart_data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Price': np.random.randn(100).cumsum() + 150
})
st.line_chart(chart_data.set_index('Date'))

st.success("Minimal dashboard is working!")
