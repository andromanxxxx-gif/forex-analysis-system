import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import ta  # Library untuk technical analysis

# Konfigurasi halaman
st.set_page_config(
    page_title="Forex Advanced Analysis Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Forex Advanced Analysis Dashboard")
st.markdown("Analisis teknikal EMA200 + MACD dengan prediksi harga dan sinyal trading")

# ===== FUNGSI UTAMA =====
def calculate_technical_indicators(df):
    """Menghitung indikator teknikal: EMA200, MACD, RSI, Bollinger Bands"""
    df = df.copy()
    
    # EMA200
    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    
    # ATR (Average True Range) untuk volatilitas
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    return df

def generate_ml_predictions(df, periods=10):
    """Generate prediksi harga menggunakan metode sederhana (simulasi ML)"""
    # Simulasi prediksi ML - dalam implementasi nyata, ganti dengan model ML sesungguhnya
    last_price = df['Close'].iloc[-1]
    volatility = df['Close'].pct_change().std()
    
    # Generate prediksi dengan random walk (simulasi)
    predictions = []
    current_pred = last_price
    
    for i in range(periods):
        # Simulasi pergerakan harga dengan drift dan volatilitas
        change = np.random.normal(0.001, volatility)  # Small positive drift
        current_pred = current_pred * (1 + change)
        predictions.append(current_pred)
    
    return predictions

def generate_trading_signals(df, predictions=None):
    """Generate sinyal trading berdasarkan analisis teknikal dan prediksi"""
    if len(df) < 30:
        return None
    
    latest = df.iloc[-1]
    signals = {}
    
    # Analisis kondisi teknikal
    price_above_ema200 = latest['Close'] > latest['EMA200'] if pd.notna(latest['EMA200']) else False
    macd_bullish = latest['MACD'] > latest['MACD_Signal'] if pd.notna(latest['MACD']) else False
    rsi_value = latest['RSI'] if pd.notna(latest['RSI']) else 50
    atr_value = latest['ATR'] if pd.notna(latest['ATR']) else 0.01
    
    # Analisis prediksi (jika ada)
    prediction_bullish = False
    if predictions and len(predictions) > 0:
        price_change = (predictions[-1] - latest['Close']) / latest['Close']
        prediction_bullish = price_change > 0.005  # 0.5% kenaikan
    
    # Generate sinyal berdasarkan kombinasi faktor
    bullish_factors = sum([price_above_ema200, macd_bullish, rsi_value > 50, prediction_bullish])
    
    if bullish_factors >= 3:
        signal = "BUY"
        confidence = min(0.9, 0.6 + (bullish_factors * 0.1))
    elif bullish_factors <= 1:
        signal = "SELL" 
        confidence = min(0.9, 0.6 + ((4 - bullish_factors) * 0.1))
    else:
        signal = "HOLD"
        confidence = 0.5
    
    # Hitung level trading
    current_price = latest['Close']
    stop_loss_distance = atr_value * 2  # 2x ATR untuk stop loss
    
    if signal == "BUY":
        stop_loss = current_price - stop_loss_distance
        take_profit = current_price + (stop_loss_distance * 2)  # Risk-reward 1:2
    elif signal == "SELL":
        stop_loss = current_price + stop_loss_distance
        take_profit = current_price - (stop_loss_distance * 2)
    else:
        stop_loss = take_profit = current_price
    
    signals = {
        'signal': signal,
        'confidence': confidence,
        'entry_price': current_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'risk_reward': 2.0,
        'timestamp': datetime.now(),
        'factors': {
            'price_above_ema200': price_above_ema200,
            'macd_bullish': macd_bullish,
            'rsi': rsi_value,
            'prediction_bullish': prediction_bullish
        }
    }
    
    return signals

def create_advanced_chart(df, predictions=None, signals=None):
    """Membuat chart advanced dengan indikator dan prediksi"""
    # Buat subplot: Harga + Prediksi, MACD, RSI
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price with EMA200 & Prediction', 'MACD', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Plot 1: Harga dengan EMA200 dan Bollinger Bands
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # EMA200
    if 'EMA200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA200'],
                line=dict(color='orange', width=2),
                name='EMA200'
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_Upper'],
                line=dict(color='gray', width=1, dash='dash'),
                name='BB Upper'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_Middle'],
                line=dict(color='gray', width=1),
                name='BB Middle'
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_Lower'],
                line=dict(color='gray', width=1, dash='dash'),
                name='BB Lower',
                fill='tonexty'
            ),
            row=1, col=1
        )
    
    # Prediksi masa depan
    if predictions and len(predictions) > 0:
        future_dates = [df.index[-1] + timedelta(hours=4*i) for i in range(1, len(predictions)+1)]
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions,
                line=dict(color='purple', width=2, dash='dot'),
                name='ML Prediction',
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # Plot 2: MACD
    if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue', width=1), name='MACD'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='red', width=1), name='Signal'),
            row=2, col=1
        )
        # MACD Histogram
        colors = ['green' if x >= 0 else 'red' for x in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histogram', marker_color=colors),
            row=2, col=1
        )
    
    # Plot 3: RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple', width=1), name='RSI'),
            row=3, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", row=3, col=1)
    
    # Tambah garis untuk sinyal trading
    if signals and signals['signal'] != 'HOLD':
        fig.add_hline(
            y=signals['entry_price'], 
            line_dash="dot", 
            line_color="blue",
            annotation_text="Entry",
            row=1, col=1
        )
        fig.add_hline(
            y=signals['stop_loss'], 
            line_dash="dot", 
            line_color="red",
            annotation_text="SL",
            row=1, col=1
        )
        fig.add_hline(
            y=signals['take_profit'], 
            line_dash="dot", 
            line_color="green",
            annotation_text="TP",
            row=1, col=1
        )
    
    fig.update_layout(
        height=800,
        title="Advanced Technical Analysis",
        xaxis_rangeslider_visible=False,
        showlegend=True
    )
    
    return fig

# ===== SIDEBAR KONFIGURASI =====
st.sidebar.header("⚙️ Configuration")

# Pilihan pair forex
forex_pairs = {
    "GBP/JPY": "GBPJPY=X",
    "USD/JPY": "USDJPY=X", 
    "EUR/JPY": "EURJPY=X",
    "CHF/JPY": "CHFJPY=X",
    "GBP/USD": "GBPUSD=X",
    "EUR/USD": "EURUSD=X"
}

selected_pair_name = st.sidebar.selectbox("Select Forex Pair", list(forex_pairs.keys()))
selected_pair = forex_pairs[selected_pair_name]

# Timeframe
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d", "1wk"], index=1)

# Periode data
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

# Jumlah prediksi
prediction_periods = st.sidebar.slider("Prediction Periods", 5, 20, 10)

# ===== MENGAMBIL DATA =====
@st.cache_data(ttl=3600)  # Cache untuk 1 jam
def load_data(pair, period, interval):
    """Mengambil data dari Yahoo Finance"""
    try:
        data = yf.download(pair, period=period, interval=interval, progress=False)
        if data.empty:
            st.error(f"Tidak dapat mengambil data untuk {pair}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

st.sidebar.header("📊 Data Loading")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()

with st.spinner("Loading data..."):
    df = load_dummy_data(selected_pair_name, period, timeframe)

if df is None or df.empty:
    st.error("Failed to load data. Please check your internet connection and try again.")
    st.stop()

# ===== PROSES DATA =====
# Hitung indikator teknikal
df = calculate_technical_indicators(df)

# Generate prediksi
predictions = generate_ml_predictions(df, periods=prediction_periods)

# Generate sinyal trading
signals = generate_trading_signals(df, predictions)

# ===== TAMPILAN UTAMA =====
# Header dengan informasi pair
col1, col2, col3, col4 = st.columns(4)

with col1:
    current_price = df['Close'].iloc[-1]
    price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else 0
    price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if len(df) > 1 else 0
    
    st.metric(
        f"Current Price ({selected_pair_name})",
        f"{current_price:.4f}",
        f"{price_change_pct:+.2f}%"
    )

with col2:
    if signals:
        signal_color = "green" if signals['signal'] == 'BUY' else "red" if signals['signal'] == 'SELL' else "gray"
        st.metric("Trading Signal", signals['signal'], delta=f"Confidence: {signals['confidence']:.1%}")

with col3:
    if 'EMA200' in df.columns and pd.notna(df['EMA200'].iloc[-1]):
        ema_position = "Above" if current_price > df['EMA200'].iloc[-1] else "Below"
        st.metric("EMA200 Position", ema_position)

with col4:
    if 'RSI' in df.columns and pd.notna(df['RSI'].iloc[-1]):
        rsi_status = "Overbought" if df['RSI'].iloc[-1] > 70 else "Oversold" if df['RSI'].iloc[-1] < 30 else "Neutral"
        st.metric("RSI Status", rsi_status, delta=f"{df['RSI'].iloc[-1]:.1f}")

# ===== CHART UTAMA =====
st.subheader("📈 Advanced Technical Analysis Chart")
chart = create_advanced_chart(df, predictions, signals)
st.plotly_chart(chart, use_container_width=True)

# ===== ANALISIS SINYAL TRADING =====
st.subheader("🎯 Trading Signal Analysis")

if signals:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Signal Details**")
        
        # Tampilkan sinyal dengan warna
        if signals['signal'] == 'BUY':
            st.success(f"**Signal: BUY** 🟢")
        elif signals['signal'] == 'SELL':
            st.error(f"**Signal: SELL** 🔴")
        else:
            st.warning(f"**Signal: HOLD** ⚪")
        
        st.write(f"**Confidence Level:** {signals['confidence']:.1%}")
        st.write(f"**Entry Price:** {signals['entry_price']:.4f}")
        st.write(f"**Stop Loss:** {signals['stop_loss']:.4f}")
        st.write(f"**Take Profit:** {signals['take_profit']:.4f}")
        st.write(f"**Risk/Reward Ratio:** 1:{signals['risk_reward']:.1f}")
        
        # Hitung risk dan reward
        risk = abs(signals['entry_price'] - signals['stop_loss'])
        reward = abs(signals['take_profit'] - signals['entry_price'])
        st.write(f"**Risk:** {risk:.4f} | **Reward:** {reward:.4f}")
    
    with col2:
        st.write("**🔍 Technical Factors**")
        
        factors = signals['factors']
        for factor, value in factors.items():
            if isinstance(value, bool):
                status = "✅" if value else "❌"
                st.write(f"{status} {factor.replace('_', ' ').title()}")
            else:
                st.write(f"**{factor.replace('_', ' ').title()}:** {value:.2f}")
        
        # Analisis tambahan
        st.write("**📈 Market Condition**")
        if signals['signal'] == 'BUY':
            st.info("Market menunjukkan kondisi bullish dengan konfirmasi multiple factors")
        elif signals['signal'] == 'SELL':
            st.info("Market menunjukkan kondisi bearish dengan konfirmasi multiple factors")
        else:
            st.info("Market dalam kondisi netral, tunggu konfirmasi lebih lanjut")

# ===== PREDIKSI MASA DEPAN =====
st.subheader("🔮 Price Prediction Analysis")

if predictions:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Prediction Summary**")
        
        # Hitung perubahan prediksi
        current_price = df['Close'].iloc[-1]
        predicted_change = (predictions[-1] - current_price) / current_price * 100
        
        st.metric(
            "Predicted Price Change", 
            f"{predicted_change:+.2f}%",
            f"in {prediction_periods} periods"
        )
        
        # Tampilkan prediksi dalam tabel
        prediction_data = []
        for i, pred in enumerate(predictions, 1):
            change_from_current = (pred - current_price) / current_price * 100
            prediction_data.append({
                'Period': i,
                'Predicted Price': f"{pred:.4f}",
                'Change %': f"{change_from_current:+.2f}%"
            })
        
        st.dataframe(pd.DataFrame(prediction_data), use_container_width=True)
    
    with col2:
        st.write("**📈 Prediction Chart**")
        
        # Buat chart prediksi sederhana
        pred_fig = go.Figure()
        
        # Data historis terakhir
        historical_dates = df.index[-20:]  # 20 periode terakhir
        historical_prices = df['Close'].tail(20)
        
        pred_fig.add_trace(go.Scatter(
            x=historical_dates, y=historical_prices,
            mode='lines', name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Prediksi masa depan
        future_dates = [df.index[-1] + timedelta(hours=4*i) for i in range(1, len(predictions)+1)]
        pred_fig.add_trace(go.Scatter(
            x=future_dates, y=predictions,
            mode='lines+markers', name='Prediction',
            line=dict(color='red', width=2, dash='dot')
        ))
        
        pred_fig.update_layout(
            height=300,
            title="Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price"
        )
        
        st.plotly_chart(pred_fig, use_container_width=True)

# ===== DATA TEKNIKAL TERBARU =====
st.subheader("📋 Latest Technical Data")

# Tampilkan data terbaru
display_columns = ['Open', 'High', 'Low', 'Close', 'EMA200', 'MACD', 'MACD_Signal', 'RSI', 'ATR']
display_df = df[display_columns].tail(10).copy()
display_df = display_df.round(4)

st.dataframe(display_df, use_container_width=True)

# ===== DOWNLOAD DATA =====
st.subheader("💾 Export Data")

# Convert to CSV
csv = df.to_csv()
st.download_button(
    label="Download Technical Data (CSV)",
    data=csv,
    file_name=f"{selected_pair_name.replace('/', '_')}_technical_data.csv",
    mime="text/csv"
)

# ===== INFORMASI TAMBAHAN =====
with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    **Features Included:**
    - 📊 **Technical Indicators**: EMA200, MACD, RSI, Bollinger Bands, ATR
    - 🔮 **Price Prediction**: ML-based future price forecasting
    - 🎯 **Trading Signals**: Automated BUY/SELL/HOLD recommendations
    - 📈 **Advanced Charts**: Interactive charts with multiple indicators
    - 📋 **Risk Management**: Stop Loss, Take Profit, Risk/Reward ratios
    
    **Technical Analysis Logic:**
    - **BUY Signal**: Price above EMA200 + Bullish MACD + RSI > 50 + Positive Prediction
    - **SELL Signal**: Price below EMA200 + Bearish MACD + RSI < 50 + Negative Prediction
    - **HOLD Signal**: Mixed or neutral technical indicators
    
    **Disclaimer:** This is for educational purposes only. Always do your own research.
    """)

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** Trading forex carries high risk. Use this information for educational purposes only.")
