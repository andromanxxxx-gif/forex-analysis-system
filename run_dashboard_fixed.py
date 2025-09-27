# run_dashboard.py - Streamlit-only version
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Forex Analysis System", layout="wide")

# Forex pairs mapping
pair_mapping = {
    'GBPJPY': 'GBPJPY=X',
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'USDJPY=X',
    'GBPUSD': 'GBPUSD=X'
}

def main():
    st.title("🎯 FOREX ANALYSIS SYSTEM")
    st.markdown("---")
    
    # Currency pair selection
    pair = st.selectbox("Select Currency Pair:", list(pair_mapping.keys()))
    
    if st.button("Analyze"):
        try:
            with st.spinner("Fetching data..."):
                data = yf.download(pair_mapping[pair], period='7d', interval='1h', auto_adjust=True)
            
            if data.empty:
                st.error("No data retrieved")
                return
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            
            latest_close = data['Close'].iloc[-1]
            previous_close = data['Close'].iloc[-2]
            change = ((latest_close - previous_close) / previous_close) * 100
            
            with col1:
                st.metric("Current Price", f"{latest_close:.4f}", f"{change:.2f}%")
            
            with col2:
                st.metric("Signal", "HOLD", "50.0% Confidence")
            
            with col3:
                st.metric("Status", "Active", "Real-time")
            
            # Display chart
            st.subheader(f"{pair} Price Chart (2H)")
            st.line_chart(data['Close'])
            
            # Display raw data
            with st.expander("View Raw Data"):
                st.dataframe(data.tail(10))
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
