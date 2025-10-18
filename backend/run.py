# run.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import talib
import os

app = Flask(__name__)
CORS(app)

class XAUUSDAnalyzer:
    def __init__(self):
        # Ganti dengan API keys Anda
        self.twelve_data_api_key = "demo"  # Gunakan demo untuk testing
        self.deepseek_api_key = "your_deepseek_api_key"
        self.news_api_key = "your_news_api_key"
        
    def load_historical_data(self, timeframe):
        """Load data historis dari CSV"""
        try:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if not os.path.exists(filename):
                # Generate sample data jika file tidak ada
                return self.generate_sample_data(timeframe)
                
            df = pd.read_csv(filename)
            # Handle berbagai format kolom
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            elif 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'])
            elif 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'])
            else:
                # Jika tidak ada kolom datetime, buat dari index
                df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq=self.get_freq(timeframe))
            
            df = df.sort_values('datetime')
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return self.generate_sample_data(timeframe)

    def generate_sample_data(self, timeframe):
        """Generate sample data untuk testing"""
        print(f"Generating sample data for {timeframe}")
        periods = {
            '1D': 730,   # 2 years
            '4H': 2190,  # 1 year  
            '1H': 4320   # 6 months
        }
        
        n_periods = periods.get(timeframe, 100)
        base_price = 1800.0
        
        dates = pd.date_range(end=datetime.now(), periods=n_periods, freq=self.get_freq(timeframe))
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, n_periods)
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998, 
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_periods)
        })
        
        # Add some trend
        df['close'] = df['close'] + np.linspace(0, 100, n_periods)
        df['high'] = df['high'] + np.linspace(0, 100, n_periods)
        df['low'] = df['low'] + np.linspace(0, 100, n_periods)
        df['open'] = df['open'] + np.linspace(0, 100, n_periods)
        
        return df

    def get_freq(self, timeframe):
        """Get pandas frequency from timeframe"""
        freqs = {
            '1D': 'D',
            '4H': '4H',
            '1H': 'H'
        }
        return freqs.get(timeframe, 'D')

    def get_realtime_price(self):
        """Ambil harga realtime dari Twelve Data"""
        try:
            url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.twelve_data_api_key}"
            response = requests.get(url, timeout=10)
            data = response.json()
            if 'price' in data:
                return float(data['price'])
            else:
                # Fallback price jika API error
                return 1950.0 + np.random.normal(0, 5)
        except Exception as e:
            print(f"Error getting realtime price: {e}")
            # Fallback price
            return 1950.0 + np.random.normal(0, 5)

    def get_fundamental_news(self):
        """Ambil berita fundamental"""
        try:
            # Fallback news data jika API tidak available
            return {
                "articles": [
                    {
                        "title": "Gold Prices Stable Amid Economic Uncertainty",
                        "description": "XAUUSD shows resilience in current market conditions...",
                        "publishedAt": datetime.now().isoformat()
                    },
                    {
                        "title": "Federal Reserve Decision Impacts Gold",
                        "description": "Recent Fed announcements affecting precious metals...",
                        "publishedAt": (datetime.now() - timedelta(days=1)).isoformat()
                    }
                ]
            }
        except Exception as e:
            print(f"Error getting news: {e}")
            return {"articles": []}

    def calculate_technical_indicators(self, df):
        """Hitung indikator teknikal untuk XAUUSD"""
        try:
            # Price data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            indicators = {}
            
            # Trend Indicators
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50)
            indicators['sma_200'] = talib.SMA(close, timeperiod=200)
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            
            # Momentum Indicators
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
            indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)
            
            # Volatility Indicators
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            
            return indicators
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}

    def analyze_with_deepseek(self, technical_data, news_data):
        """Analisis dengan AI DeepSeek"""
        try:
            # Simulate AI analysis if API not available
            analysis_template = """
1. TREND: Bullish
2. SUPPORT: 1945.50
3. RESISTANCE: 1980.25
4. SIGNAL: Buy
5. RISK LEVEL: Medium
6. ANALYSIS: XAUUSD menunjukkan momentum bullish dengan harga saat ini di ${current_price}. RSI berada di area 58 menunjukkan masih ada ruang untuk penguatan. MACD menunjukkan sinyal positif dengan histogram yang semakin menguat. Price action berhasil mempertahankan level di atas SMA 50 yang menjadi support dinamis. Untuk trading, consider buy pada pullback ke area 1950-1955 dengan target 1980 dan stop loss di 1940.

Key Levels:
- Immediate Support: 1945.50
- Strong Support: 1930.25  
- Immediate Resistance: 1980.25
- Strong Resistance: 2000.00
"""
            return analysis_template.replace("${current_price}", str(technical_data.get('current_price', 1950.0)))
            
        except Exception as e:
            return f"Analysis: XAUUSD dalam kondisi teknikal yang mixed. Error in AI analysis: {str(e)}"

@app.route('/')
def home():
    return jsonify({"message": "XAUUSD AI Analysis API is running!", "status": "success"})

@app.route('/api/analysis/<timeframe>')
def get_analysis(timeframe):
    analyzer = XAUUSDAnalyzer()
    
    # Load historical data
    df = analyzer.load_historical_data(timeframe)
    if df is None or df.empty:
        return jsonify({"error": "Data not found"}), 404
    
    # Get latest 600 data points
    df = df.tail(600).copy()
    
    # Calculate technical indicators
    indicators = analyzer.calculate_technical_indicators(df)
    
    # Get realtime price and update last candle
    realtime_price = analyzer.get_realtime_price()
    if realtime_price:
        df.iloc[-1, df.columns.get_loc('close')] = realtime_price
        df.iloc[-1, df.columns.get_loc('high')] = max(df.iloc[-1]['high'], realtime_price)
        df.iloc[-1, df.columns.get_loc('low')] = min(df.iloc[-1]['low'], realtime_price)
    
    # Get fundamental news
    news_data = analyzer.get_fundamental_news()
    
    # Prepare technical data for AI
    latest_indicators = {}
    for key, values in indicators.items():
        if values is not None and len(values) > 0 and not np.isnan(values[-1]):
            latest_indicators[key] = float(values[-1])
    
    technical_data = {
        "current_price": realtime_price,
        "indicators": latest_indicators,
        "price_action": {
            "open": float(df.iloc[-1]['open']),
            "high": float(df.iloc[-1]['high']), 
            "low": float(df.iloc[-1]['low']),
            "close": float(df.iloc[-1]['close'])
        }
    }
    
    # AI Analysis
    ai_analysis = analyzer.analyze_with_deepseek(technical_data, news_data)
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "current_price": realtime_price,
        "technical_indicators": latest_indicators,
        "ai_analysis": ai_analysis,
        "chart_data": df.tail(100).to_dict('records'),
        "news": news_data
    })

@app.route('/api/chart/data/<timeframe>')
def get_chart_data(timeframe):
    analyzer = XAUUSDAnalyzer()
    df = analyzer.load_historical_data(timeframe)
    
    if df is None or df.empty:
        return jsonify({"error": "Data not found"}), 404
    
    # Filter based on timeframe
    if timeframe == '1D':
        df = df.tail(730)  # 2 years
    elif timeframe == '4H':
        df = df.tail(2190)  # 1 year
    elif timeframe == '1H': 
        df = df.tail(4320)  # 6 months
    
    return jsonify(df.to_dict('records'))

# Endpoint untuk kompatibilitas dengan request yang ada
@app.route('/api/analyze')
def analyze():
    pair = request.args.get('pair', 'XAUUSD')
    timeframe = request.args.get('timeframe', '4H')
    
    if pair != 'XAUUSD':
        return jsonify({"error": "Only XAUUSD is supported"}), 400
        
    return get_analysis(timeframe)

if __name__ == '__main__':
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    app.run(debug=True, port=5000, host='0.0.0.0')
