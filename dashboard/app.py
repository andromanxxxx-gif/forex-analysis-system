# dashboard/app.py - GANTI SEMUA KODE DENGAN INI

import sys
import os
from pathlib import Path

print("=" * 60)
print("STARTING FOREX DASHBOARD - PATH CONFIGURATION")
print("=" * 60)

# ===== FIX PATH - METHOD 1: Absolute Path =====
PROJECT_ROOT = Path(r"C:\Users\HP\forex-analysis-system")
SRC_PATH = PROJECT_ROOT / "src"
CONFIG_PATH = PROJECT_ROOT / "config"

# Add to Python path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(CONFIG_PATH))

print(f"Project Root: {PROJECT_ROOT}")
print(f"Python Path: {sys.path}")

# ===== TRY IMPORTS WITH PROPER ERROR HANDLING =====
try:
    from src.data_collection import DataCollector
    print("✅ DataCollector imported successfully")
except ImportError as e:
    print(f"❌ DataCollector import failed: {e}")
    # Create dummy class as fallback
    class DataCollector:
        def fetch_all_data(self, pairs=None):
            return {'GBPJPY=X': None, 'USDJPY=X': None}

try:
    from src.technical_analysis import TechnicalAnalyzer
    print("✅ TechnicalAnalyzer imported successfully")
except ImportError as e:
    print(f"❌ TechnicalAnalyzer import failed: {e}")
    class TechnicalAnalyzer:
        def analyze(self, data):
            return data if data is not None else None

try:
    from src.news_analyzer import NewsAnalyzer
    print("✅ NewsAnalyzer imported successfully")
except ImportError as e:
    print(f"❌ NewsAnalyzer import failed: {e}")
    class NewsAnalyzer:
        def analyze_sentiment(self, news_items):
            return 0.0, []

try:
    from src.signal_generator import SignalGenerator
    print("✅ SignalGenerator imported successfully")
except ImportError as e:
    print(f"❌ SignalGenerator import failed: {e}")
    class SignalGenerator:
        def generate_signals(self, data, news_sentiment=0):
            return {}

try:
    from src.ml_predictor import MLForexPredictor
    print("✅ MLForexPredictor imported successfully")
except ImportError as e:
    print(f"❌ MLForexPredictor import failed: {e}")
    class MLForexPredictor:
        def load_models(self):
            return True
        def generate_predictions(self, data_dict):
            return {}

try:
    from config import settings
    print("✅ Settings imported successfully")
except ImportError as e:
    print(f"❌ Settings import failed: {e}")
    # Fallback settings
    class settings:
        FOREX_PAIRS = ['GBPJPY=X', 'USDJPY=X', 'EURJPY=X', 'CHFJPY=X']
        PAIR_NAMES = ['GBP/JPY', 'USD/JPY', 'EUR/JPY', 'CHF/JPY']

print("=" * 60)
print("IMPORT PHASE COMPLETED")
print("=" * 60)

# ===== STREAMLIT DASHBOARD CODE =====
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize components with error handling
try:
    data_collector = DataCollector()
    technical_analyzer = TechnicalAnalyzer()
    news_analyzer = NewsAnalyzer()
    signal_generator = SignalGenerator()
    ml_predictor = MLForexPredictor()
    
    # Try to load ML models
    try:
        ml_predictor.load_models()
        print("✅ ML models loaded successfully")
    except Exception as e:
        print(f"⚠️ ML models loading failed: {e}")
        
except Exception as e:
    st.error(f"Error initializing components: {e}")

# Streamlit App
st.set_page_config(
    page_title="Forex Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("Forex Analysis Dashboard")
st.markdown("Real-time forex analysis with technical indicators and ML predictions")

# Sidebar
st.sidebar.header("Configuration")
selected_pair_name = st.sidebar.selectbox(
    "Select Forex Pair:",
    options=settings.PAIR_NAMES
)

# Simple display - we'll enhance this once imports work
st.header(f"Analysis for {selected_pair_name}")

# Placeholder for chart
st.subheader("Price Chart")
st.info("Chart functionality will be available once all modules are properly imported")

# Placeholder for signals
st.subheader("Trading Signals")
st.warning("Signal analysis will be available once all modules are properly imported")

# Debug information
with st.expander("Debug Information"):
    st.write("Python Path:", sys.path)
    st.write("Current Directory:", os.getcwd())
    st.write("Project Root:", PROJECT_ROOT)
    st.write("Files in src directory:", os.listdir(SRC_PATH) if os.path.exists(SRC_PATH) else "Directory not found")

st.success("Dashboard is running! Module imports are being handled with fallbacks.")

print("=" * 60)
print("DASHBOARD STARTED SUCCESSFULLY")
print("=" * 60)
