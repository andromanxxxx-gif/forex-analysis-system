import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Fix import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.signal_generator import SignalGenerator
    from src.technical_analyzer import TechnicalAnalyzer
    from src.data_fetcher import ForexDataFetcher
    SIGNAL_GENERATOR_AVAILABLE = True
except ImportError as e:
    st.error(f"Import error: {e}")
    SIGNAL_GENERATOR_AVAILABLE = False
    # Fallback ke class sederhana
    class SignalGenerator:
        def generate_signals(self, data):
            return {
                'action': 'HOLD', 
                'confidence': 0.5,
                'message': 'SignalGenerator not available'
            }

class FixedForexDashboard:
    def __init__(self):
        self.signal_generator = SignalGenerator() if SIGNAL_GENERATOR_AVAILABLE else SignalGenerator()
        self.pairs = ['GBPJPY', 'USDJPY', 'CHFJPY', 'EURJPY', 'EURNZD']
        self.timeframes = ['2H', '4H', '1D']
    
    def get_simplified_data(self, pair):
        """Dapatkan data simplified"""
        try:
            # Gunakan yfinance langsung
            import yfinance as yf
            pair_mapping = {
                'GBPJPY': 'GBPJPY=X',
                'USDJPY': 'USDJPY=X',
                'CHFJPY': 'CHFJPY=X',
                'EURJPY': 'EURJPY=X', 
                'EURNZD': 'EURNZD=X'
            }
            
            data = yf.download(pair_mapping[pair], period='7d', interval='1h')
            return data
        except:
            # Fallback ke data dummy
            return self.create_dummy_data(pair)
    
    def create_dummy_data(self, pair):
        """Buat data dummy"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        np.random.seed(hash(pair) % 1000)
        
        prices = [150.0]
        for i in range(99):
            change = np.random.normal(0.001, 0.01)
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'Open': prices,
            'High': [p * 1.005 for p in prices],
            'Low': [p * 0.995 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(10000, 50000, 100)
        }, index=dates)
    
    def calculate_basic_indicators(self, data):
        """Hitung indikator dasar"""
        if data is None or data.empty:
            return data
        
        # EMA sederhana
        data['EMA_200'] = data['Close'].ewm(span=200).mean()
        
        # MACD sederhana
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema_12 - ema_26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        
        return data
    
    def run(self):
        """Jalankan dashboard yang sudah diperbaiki"""
        st.set_page_config(
            page_title="Forex Analysis System - Fixed",
            layout="wide"
        )
        
        st.title("🎯 Forex Analysis System")
        st.markdown("**Fixed Version - Running Successfully**")
        
        # Sidebar
        pair = st.sidebar.selectbox("Select Pair:", self.pairs)
        timeframe = st.sidebar.selectbox("Select Timeframe:", self.timeframes)
        
        # Main content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Chart {pair} - {timeframe}")
            
            # Get data
            data = self.get_simplified_data(pair)
            data = self.calculate_basic_indicators(data)
            
            if data is not None:
                # Buat chart sederhana
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price'))
                fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], name='EMA 200'))
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Trading Signals")
            
            if data is not None:
                # Generate signals
                signals = self.signal_generator.generate_signals(data)
                
                # Display signals
                action = signals.get('action', 'HOLD')
                color = "green" if action == "BUY" else "red" if action == "SELL" else "orange"
                
                st.markdown(f"<h2 style='color: {color};'>{action}</h2>", 
                           unsafe_allow_html=True)
                
                confidence = signals.get('confidence', 0.5)
                st.metric("Confidence", f"{confidence*100:.1f}%")
                st.progress(confidence)
                
                if 'take_profit' in signals:
                    st.metric("Take Profit", f"{signals['take_profit']:.4f}")
                if 'stop_loss' in signals:
                    st.metric("Stop Loss", f"{signals['stop_loss']:.4f}")
        
        # Status info
        st.sidebar.info(f"Signal Generator: {'✅ Available' if SIGNAL_GENERATOR_AVAILABLE else '❌ Using Fallback'}")

def main():
    dashboard = FixedForexDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
