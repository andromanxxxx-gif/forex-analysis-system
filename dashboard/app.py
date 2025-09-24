import sys
import os
from pathlib import Path
import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import json
import os
# Dapatkan path ke root project (direktori yang mengandung folder 'dashboard' dan 'src')
project_root = Path(__file__).parent.parent
# Tambahkan path project_root ke sys.path
sys.path.insert(0, str(project_root))

# Sekarang coba import modul dari src
# Tambahkan path ke src folder
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

# Import modules setelah path setup
try:
    from src.data_collection import DataCollector
    from src.technical_analysis import TechnicalAnalyzer
    from src.news_analyzer import NewsAnalyzer
    from src.signal_generator import SignalGenerator
    from src.ml_predictor import MLForexPredictor
    from config import settings
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
# Import custom modules
from src.data_collection import DataCollector
from src.technical_analysis import TechnicalAnalyzer
from src.news_analyzer import NewsAnalyzer
from src.signal_generator import SignalGenerator
from src.ml_predictor import MLForexPredictor
from config import settings
# dashboard/app.py (bagian yang berhubungan dengan auth)
import sys
from pathlib import Path
import os

# Tambahkan path ke src
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.google_drive_auth import drive_auth
    HAS_DRIVE_ACCESS = True
except ImportError:
    HAS_DRIVE_ACCESS = False

# Di dalam fungsi yang perlu akses Google Drive:
def load_data_from_drive():
    if not HAS_DRIVE_ACCESS:
        st.warning("Google Drive access not available")
        return None
    
    try:
        drive_service = drive_auth.get_service()
        # ... kode untuk akses data
    except Exception as e:
        st.error(f"Error accessing Google Drive: {e}")
        return None

# Set page config
st.set_page_config(
    page_title="Forex Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    data_collector = DataCollector()
    technical_analyzer = TechnicalAnalyzer()
    news_analyzer = NewsAnalyzer()
    signal_generator = SignalGenerator()
    ml_predictor = MLForexPredictor()
    return data_collector, technical_analyzer, news_analyzer, signal_generator, ml_predictor

data_collector, technical_analyzer, news_analyzer, signal_generator, ml_predictor = init_components()

# Load ML models
@st.cache_resource
def load_models():
    pairs = ['GBPJPY=X', 'CHFJPY=X', 'USDJPY=X', 'EURJPY=X']
    for pair in pairs:
        model_path = f"models/saved_models/{pair.replace('=', '').lower()}_model.h5"
        scaler_path = f"models/saved_models/{pair.replace('=', '').lower()}_scaler.joblib"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            ml_predictor.load_model(pair, model_path, scaler_path)

load_models()

# Title
st.title("Forex Analysis Dashboard")
st.markdown("Analisis teknikal dan fundamental untuk pasangan forex utama")

# Sidebar
st.sidebar.header("Konfigurasi")
selected_pair = st.sidebar.selectbox(
    "Pilih Pasangan Forex:",
    options=settings.PAIR_NAMES,
    index=0
)

# Mapping nama tampilan ke simbol
pair_mapping = {
    'GBP/JPY': 'GBPJPY=X',
    'CHF/JPY': 'CHFJPY=X',
    'USD/JPY': 'USDJPY=X',
    'EUR/JPY': 'EURJPY=X'
}

selected_symbol = pair_mapping[selected_pair]

# Checkbox untuk menampilkan prediksi ML
show_ml_predictions = st.sidebar.checkbox("Tampilkan Prediksi ML", value=True)

# Load data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    all_data = data_collector.fetch_all_data([selected_symbol])
    analyzed_data = {}
    
    for pair, data in all_data.items():
        analyzed_data[pair] = technical_analyzer.analyze(data)
    
    # Load news
    news_file = 'data/news/latest_news.json'
    if os.path.exists(news_file):
        with open(news_file, 'r') as f:
            news_data = json.load(f)
        avg_sentiment = news_data['avg_sentiment']
        analyzed_news = news_data['news_items']
    else:
        # Fallback to live news
        news_items = data_collector.scrape_news()
        avg_sentiment, analyzed_news = news_analyzer.analyze_sentiment(news_items)
    
    return analyzed_data, avg_sentiment, analyzed_news

analyzed_data, avg_sentiment, analyzed_news = load_data()

# Generate predictions
predictions = {}
if show_ml_predictions:
    for pair, data in analyzed_data.items():
        predictions[pair] = ml_predictor.predict(data, pair)

# Generate signals
signals = {}
for pair, data in analyzed_data.items():
    signals[pair] = signal_generator.generate_signals(
        data, avg_sentiment, 
        {'values': predictions.get(pair), 'next_close': predictions.get(pair)[0] if predictions.get(pair) is not None else None}
    )

# Tampilkan chart
st.header(f"Chart {selected_pair}")

if selected_symbol in analyzed_data:
    data = analyzed_data[selected_symbol]
    
    # Buat chart dengan subplot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price', 'MACD'),
        row_width=[0.7, 0.3]
    )
    
    # Chart harga
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # EMA200
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA200'],
            line=dict(color='orange', width=2),
            name='EMA200'
        ),
        row=1, col=1
    )
    
    # Tampilkan prediksi ML jika ada
    if show_ml_predictions and selected_symbol in predictions and predictions[selected_symbol]:
        prediction = predictions[selected_symbol]
        if prediction is not None and len(prediction) > 0:
            # Buat tanggal untuk prediksi masa depan
            last_date = data.index[-1]
            prediction_dates = [last_date + timedelta(hours=4*i) for i in range(1, len(prediction)+1)]
            
            fig.add_trace(
                go.Scatter(
                    x=prediction_dates,
                    y=prediction,
                    mode='lines+markers',
                    line=dict(color='purple', width=2, dash='dot'),
                    name='ML Prediction'
                ),
                row=1, col=1
            )
    
    # MACD
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD'],
            line=dict(color='blue', width=2),
            name='MACD'
        ),
        row=2, col=1
    )
    
    # MACD Signal Line
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MACD_Signal'],
            line=dict(color='red', width=2),
            name='Signal Line'
        ),
        row=2, col=1
    )
    
    # MACD Histogram
    colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['MACD_Histogram'],
            name='Histogram',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"Tidak ada data untuk {selected_pair}")

# Tampilkan sinyal trading
st.header("Sinyal Trading")

if selected_symbol in signals and signals[selected_symbol]:
    signal = signals[selected_symbol]
    
    # Tampilkan sinyal
    if signal['signal'] == 'BUY':
        st.success(f"**Sinyal: BUY** {selected_pair}")
        st.metric("Confidence", f"{signal['confidence']*100:.2f}%")
    elif signal['signal'] == 'SELL':
        st.error(f"**Sinyal: SELL** {selected_pair}")
        st.metric("Confidence", f"{signal['confidence']*100:.2f}%")
    else:
        st.warning(f"**Sinyal: NEUTRAL** {selected_pair}")
        st.metric("Confidence", f"{signal['confidence']*100:.2f}%")
    
    # Tampilkan level trading
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entry Price", f"{signal['entry']:.5f}")
    with col2:
        st.metric("Stop Loss", f"{signal['stop_loss']:.5f}")
    with col3:
        st.metric("Take Profit", f"{signal['take_profit']:.5f}")
    
    # Tampilkan faktor yang mempengaruhi
    st.subheader("Faktor yang Mempengaruhi Sinyal")
    
    factors_col1, factors_col2, factors_col3 = st.columns(3)
    
    with factors_col1:
        st.metric("Sentimen Berita", f"{signal['news_sentiment']:.4f}")
    
    with factors_col2:
        if signal['ml_prediction'] and signal['ml_prediction'].get('next_close'):
            ml_change = ((signal['ml_prediction']['next_close'] - signal['entry']) / signal['entry']) * 100
            st.metric("Prediksi ML", f"{ml_change:.2f}%")
        else:
            st.metric("Prediksi ML", "N/A")
    
    with factors_col3:
        st.metric("Analisis Teknikal", "Strong" if signal['confidence'] > 0.6 else "Moderate")
else:
    st.warning("Tidak ada sinyal trading yang dihasilkan")

# Tampilkan analisis sentimen berita
st.header("Analisis Sentimen Berita")
st.metric("Sentimen Berita Rata-rata", f"{avg_sentiment:.4f}")

if avg_sentiment > 0.1:
    st.success("Sentimen berita overall POSITIVE")
elif avg_sentiment < -0.1:
    st.error("Sentimen berita overall NEGATIVE")
else:
    st.warning("Sentimen berita overall NEUTRAL")

# Tampilkan berita
st.subheader("Berita Terkini")
for news in analyzed_news[:5]:
    sentiment_color = {
        'Positive': 'green',
        'Negative': 'red',
        'Neutral': 'gray'
    }.get(news['sentiment'], 'gray')
    
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; border-left: 4px solid {sentiment_color}; margin-bottom: 10px;">
            <strong>{news['title']}</strong><br>
            <small>Sumber: {news['source']} | Waktu: {news['time']} | Sentimen: <span style="color: {sentiment_color}">{news['sentiment']}</span></small>
        </div>
        """,
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** Sinyal trading ini hanya untuk tujuan edukasi dan bukan saran finansial. Trading forex memiliki risiko kerugian yang tinggi.")
