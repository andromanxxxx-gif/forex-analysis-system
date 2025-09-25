import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_forex_dummy_data(pair_name="GBP/JPY", periods=500, timeframe="4h", initial_price=185.00):
    """
    Generate dummy forex data yang sesuai dengan kebutuhan analisis teknikal
    
    Parameters:
    - pair_name: Nama pair forex
    - periods: Jumlah periode data
    - timeframe: Timeframe data (1h, 4h, 1d, etc.)
    - initial_price: Harga awal
    """
    
    # Setup dasar
    end_date = datetime.now()
    
    # Tentukan interval waktu berdasarkan timeframe
    if timeframe == "1h":
        time_delta = timedelta(hours=1)
    elif timeframe == "4h":
        time_delta = timedelta(hours=4)
    elif timeframe == "1d":
        time_delta = timedelta(days=1)
    elif timeframe == "1wk":
        time_delta = timedelta(weeks=1)
    else:
        time_delta = timedelta(hours=4)
    
    start_date = end_date - (periods * time_delta)
    
    # Inisialisasi lists untuk data
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    # Parameter untuk simulasi pergerakan harga yang realistis
    current_price = initial_price
    volatility = 0.002  # Volatilitas harian (0.2%)
    drift = 0.0001      # Drift kecil positif
    
    # Trend dan siklus untuk membuat pola yang lebih realistis
    trend_period = random.randint(50, 150)
    cycle_period = random.randint(10, 30)
    
    # Generate data
    for i in range(periods):
        current_date = start_date + (i * time_delta)
        dates.append(current_date)
        
        # Simulasi pergerakan harga dengan trend dan siklus
        trend_component = np.sin(2 * np.pi * i / trend_period) * 0.005
        cycle_component = np.sin(2 * np.pi * i / cycle_period) * 0.002
        
        # Random walk dengan drift, trend, dan siklus
        price_change = (random.gauss(0, 1) * volatility + 
                       drift + trend_component + cycle_component)
        
        new_price = current_price * (1 + price_change)
        
        # Simulasi candle stick (Open, High, Low, Close)
        open_price = current_price
        close_price = new_price
        
        # Tentukan high dan low yang realistis
        price_range = abs(close_price - open_price) * random.uniform(1.0, 3.0)
        high_price = max(open_price, close_price) + price_range * random.uniform(0.1, 0.4)
        low_price = min(open_price, close_price) - price_range * random.uniform(0.1, 0.4)
        
        # Pastikan high >= max(open, close) dan low <= min(open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Volume acak dengan sedikit korelasi dengan volatilitas
        volume = random.randint(1000, 10000) * (1 + abs(price_change) * 100)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)
        
        current_price = close_price
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates)
    
    # Set index name
    df.index.name = 'Datetime'
    
    return df

def calculate_technical_indicators_dummy(df):
    """
    Menghitung indikator teknikal untuk data dummy
    (Sama dengan fungsi di dashboard utama)
    """
    df = df.copy()
    
    # EMA200
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # MACD (12, 26, 9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # ATR (14)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = np.maximum(np.maximum(high_low, high_close), low_close)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def save_dummy_data_to_csv(df, filename=None):
    """
    Menyimpan data dummy ke file CSV
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forex_dummy_data_{timestamp}.csv"
    
    df.to_csv(filename, index=True)
    print(f"Data dummy disimpan sebagai: {filename}")
    return filename

# Contoh penggunaan
if __name__ == "__main__":
    import streamlit as st
    
    st.title("📊 Forex Dummy Data Generator")
    
    st.sidebar.header("Configuration")
    
    # Parameter input
    pair_name = st.sidebar.selectbox(
        "Forex Pair",
        ["GBP/JPY", "USD/JPY", "EUR/JPY", "CHF/JPY", "GBP/USD", "EUR/USD"]
    )
    
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ["1h", "4h", "1d", "1wk"]
    )
    
    periods = st.sidebar.slider("Number of Periods", 100, 1000, 500)
    
    initial_price = st.sidebar.number_input(
        "Initial Price", 
        min_value=0.1, 
        max_value=1000.0, 
        value=185.00 if "JPY" in pair_name else 1.2000
    )
    
    if st.sidebar.button("Generate Dummy Data"):
        with st.spinner("Generating dummy data..."):
            # Generate data
            df = generate_forex_dummy_data(
                pair_name=pair_name,
                periods=periods,
                timeframe=timeframe,
                initial_price=initial_price
            )
            
            # Calculate technical indicators
            df = calculate_technical_indicators_dummy(df)
            
            # Display results
            st.success(f"✅ Successfully generated {len(df)} periods of {pair_name} data")
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"{df['Close'].iloc[-1]:.4f}")
            with col2:
                st.metric("Data Points", len(df))
            with col3:
                st.metric("Date Range", f"{df.index[0].date()} to {df.index[-1].date()}")
            
            # Show recent data
            st.subheader("📋 Recent Data (Last 10 periods)")
            st.dataframe(df.tail(10).round(4))
            
            # Show technical indicators summary
            st.subheader("📊 Technical Indicators Summary")
            tech_cols = ['EMA200', 'MACD', 'MACD_Signal', 'RSI', 'ATR']
            tech_summary = df[tech_cols].iloc[-1:].T.round(4)
            tech_summary.columns = ['Latest Values']
            st.dataframe(tech_summary)
            
            # Download option
            csv = df.to_csv()
            st.download_button(
                label="📥 Download Dummy Data (CSV)",
                data=csv,
                file_name=f"{pair_name.replace('/', '_')}_dummy_data.csv",
                mime="text/csv"
            )
            
            # Visualize basic chart
            st.subheader("📈 Price Chart")
            st.line_chart(df['Close'])
            
            # Show statistics
            st.subheader("📊 Data Statistics")
            st.dataframe(df.describe().round(4))

# Fungsi untuk digunakan dalam dashboard utama
def load_dummy_data(pair_name="GBP/JPY", period="3mo", interval="4h"):
    """
    Fungsi yang kompatibel dengan dashboard utama untuk load dummy data
    """
    # Map period to number of periods
    period_map = {
        "1mo": 30 * 6,   # 30 days * 6 periods per day (4h)
        "3mo": 90 * 6,
        "6mo": 180 * 6,
        "1y": 365 * 6,
        "2y": 730 * 6
    }
    
    periods = period_map.get(period, 500)
    
    # Map interval to timeframe
    timeframe_map = {
        "1h": "1h",
        "4h": "4h", 
        "1d": "1d",
        "1wk": "1wk"
    }
    
    timeframe = timeframe_map.get(interval, "4h")
    
    # Set initial price based on pair
    initial_prices = {
        "GBP/JPY": 185.00,
        "USD/JPY": 150.00,
        "EUR/JPY": 160.00,
        "CHF/JPY": 170.00,
        "GBP/USD": 1.2500,
        "EUR/USD": 1.0800
    }
    
    initial_price = initial_prices.get(pair_name, 100.00)
    
    # Generate data
    df = generate_forex_dummy_data(
        pair_name=pair_name,
        periods=periods,
        timeframe=timeframe,
        initial_price=initial_price
    )
    
    # Calculate technical indicators
    df = calculate_technical_indicators_dummy(df)
    
    return dfimport os
import json
import random
from datetime import datetime, timedelta

# Folder penyimpanan data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

LAST_SIGNAL_FILE = os.path.join(DATA_DIR, "last_signal.json")

# Pasangan forex
PAIRS = ["GBPJPY", "USDJPY", "EURJPY", "CHFJPY"]

# Timeframe yang kita dukung
TIMEFRAMES = {
    "4H": 4,
    "1D": 24,
}

N_BARS = 100  # jumlah candle per timeframe

def generate_dummy_signals():
    signals = {}

    for pair in PAIRS:
        signals[pair] = {}

        for tf, hours in TIMEFRAMES.items():
            data = []
            time = datetime.utcnow() - timedelta(hours=N_BARS * hours)

            price = random.uniform(100, 200)

            for _ in range(N_BARS):
                signal = random.choice(["BUY", "SELL"])
                prob_up = round(random.uniform(0.4, 0.8), 2)

                price += random.uniform(-1, 1)

                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "signal": signal,
                    "price": round(price, 3),
                    "stop_loss": round(price - random.uniform(0.5, 1.5), 3),
                    "take_profit": round(price + random.uniform(0.5, 1.5), 3),
                    "prob_up": prob_up,
                    "news_compound": round(random.uniform(-1, 1), 2),
                    "news": random.choice([
                        "BOE mempertahankan suku bunga.",
                        "Fed memberi sinyal dovish.",
                        "Yen menguat karena risk-off sentiment.",
                        "Inflasi Eropa lebih rendah dari perkiraan."
                    ])
                }

                data.append(row)
                time += timedelta(hours=hours)

            signals[pair][tf] = data

    return signals

if __name__ == "__main__":
    signals = generate_dummy_signals()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_path = os.path.join(DATA_DIR, f"signals_dummy_{ts}.json")

    # Simpan file snapshot
    with open(file_path, "w") as f:
        json.dump(signals, f, indent=2)

    # Simpan juga last_signal.json
    with open(LAST_SIGNAL_FILE, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"✅ Dummy signals saved to:\n  {file_path}\n  {LAST_SIGNAL_FILE}")
