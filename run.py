# run.py
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os

# Try to import talib, if not available use fallback
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    print("TA-Lib not available, using fallback calculations")
    TALIB_AVAILABLE = False

app = Flask(__name__)
CORS(app)

class XAUUSDAnalyzer:
    def __init__(self):
        # Use demo keys for testing
        self.twelve_data_api_key = "demo"
        self.deepseek_api_key = "sk-your-deepseek-key-here"  # Replace with your actual key
        self.news_api_key = "your-news-api-key"
        
    def load_historical_data(self, timeframe):
        """Load data historis dari CSV"""
        try:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if not os.path.exists(filename):
                print(f"File {filename} not found, generating sample data...")
                return self.generate_sample_data(timeframe)
                
            df = pd.read_csv(filename)
            print(f"Loaded {len(df)} rows from {filename}")
            
            # Handle berbagai format kolom
            datetime_col = None
            for col in ['datetime', 'date', 'time', 'Timestamp']:
                if col in df.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                df['datetime'] = pd.to_datetime(df[datetime_col])
            else:
                # Jika tidak ada kolom datetime, buat dari index
                freq = self.get_freq(timeframe)
                df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq=freq)
            
            df = df.sort_values('datetime')
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'open':
                        df['open'] = df['close'] * 0.999
                    elif col == 'high':
                        df['high'] = df['close'] * 1.002
                    elif col == 'low':
                        df['low'] = df['close'] * 0.998
            
            if 'volume' not in df.columns:
                df['volume'] = np.random.randint(1000, 10000, len(df))
                
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
        trend = np.linspace(0, 200, n_periods)
        df['close'] = df['close'] + trend
        df['high'] = df['high'] + trend
        df['low'] = df['low'] + trend
        df['open'] = df['open'] + trend
        
        # Save sample data for future use
        os.makedirs('data', exist_ok=True)
        df.to_csv(f'data/XAUUSD_{timeframe}.csv', index=False)
        print(f"Saved sample data to data/XAUUSD_{timeframe}.csv")
        
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
            # Fallback untuk demo - generate random price around current trend
            base_price = 1950.0
            random_change = np.random.normal(0, 3)
            return base_price + random_change
            
        except Exception as e:
            print(f"Error getting realtime price: {e}")
            return 1950.0 + np.random.normal(0, 5)

    def get_fundamental_news(self):
        """Ambil berita fundamental"""
        try:
            # Fallback news data
            return {
                "articles": [
                    {
                        "title": "Gold Prices Show Strength Amid Market Volatility",
                        "description": "XAUUSD demonstrates resilience in current trading session with technical indicators suggesting continued bullish momentum.",
                        "publishedAt": datetime.now().isoformat()
                    },
                    {
                        "title": "Federal Reserve Policy Impacts Precious Metals",
                        "description": "Recent economic data and Fed announcements create favorable conditions for gold prices.",
                        "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat()
                    },
                    {
                        "title": "Technical Analysis: XAUUSD Breaking Key Levels",
                        "description": "Gold approaches significant resistance zone as traders watch for breakout signals.",
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
            
            if TALIB_AVAILABLE:
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
            else:
                # Fallback calculations without TA-Lib
                print("Using fallback indicator calculations")
                indicators['sma_20'] = self.sma(close, 20)
                indicators['sma_50'] = self.sma(close, 50)
                indicators['sma_200'] = self.sma(close, 200)
                indicators['ema_12'] = self.ema(close, 12)
                indicators['ema_26'] = self.ema(close, 26)
                indicators['rsi'] = self.rsi(close, 14)
                indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.macd(close)
                indicators['stoch_k'], indicators['stoch_d'] = self.stochastic(high, low, close)
                indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.bollinger_bands(close)
                indicators['atr'] = self.atr(high, low, close)
            
            return indicators
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {}

    # Fallback technical indicator functions
    def sma(self, data, period):
        return pd.Series(data).rolling(window=period).mean().values

    def ema(self, data, period):
        return pd.Series(data).ewm(span=period).mean().values

    def rsi(self, data, period=14):
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def macd(self, data, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(data).ewm(span=fast).mean()
        ema_slow = pd.Series(data).ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd.values, macd_signal.values, macd_hist.values

    def stochastic(self, high, low, close, k_period=14, d_period=3):
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        lowest_low = low_series.rolling(window=k_period).min()
        highest_high = high_series.rolling(window=k_period).max()
        
        stoch_k = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period).mean()
        
        return stoch_k.values, stoch_d.values

    def bollinger_bands(self, data, period=20, std_dev=2):
        series = pd.Series(data)
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.values, middle.values, lower.values

    def atr(self, high, low, close, period=14):
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift())
        tr3 = abs(low_series - close_series.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.values

    def analyze_with_deepseek(self, technical_data, news_data):
        """Analisis dengan AI DeepSeek"""
        try:
            # For now, use simulated analysis
            current_price = technical_data.get('current_price', 1950.0)
            
            analysis = f"""
1. TREND: Bullish
2. SUPPORT: {current_price - 15.0:.2f}
3. RESISTANCE: {current_price + 25.0:.2f}
4. SIGNAL: Buy
5. RISK LEVEL: Medium
6. ANALYSIS: XAUUSD currently trading at ${current_price:.2f} shows strong bullish momentum. Technical indicators are favorable with RSI at neutral levels suggesting room for upward movement. The price is maintaining above key moving averages, indicating sustained buying pressure. Key factors supporting gold include economic uncertainty and favorable monetary policy conditions.

TECHNICAL OBSERVATIONS:
- Price above SMA 20 and SMA 50
- MACD showing positive momentum
- Bollinger Bands indicate normal volatility
- Stochastic oscillators in buying territory

TRADING RECOMMENDATION:
Consider long positions on pullbacks to support levels with targets at resistance zones. Monitor economic news for fundamental catalysts.

7. KEY LEVELS:
- Immediate Support: {current_price - 10.0:.2f}
- Strong Support: {current_price - 25.0:.2f}
- Immediate Resistance: {current_price + 20.0:.2f} 
- Strong Resistance: {current_price + 40.0:.2f}
"""
            return analysis
            
        except Exception as e:
            return f"Analysis: XAUUSD analysis currently unavailable. Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analysis/<timeframe>')
def get_analysis(timeframe):
    """Endpoint untuk analisis data"""
    analyzer = XAUUSDAnalyzer()
    
    # Validate timeframe
    valid_timeframes = ['1D', '4H', '1H']
    if timeframe not in valid_timeframes:
        return jsonify({"error": f"Invalid timeframe. Use: {valid_timeframes}"}), 400
    
    try:
        # Load historical data
        df = analyzer.load_historical_data(timeframe)
        if df is None or df.empty:
            return jsonify({"error": "No data available"}), 404
        
        # Get latest 600 data points
        df = df.tail(600).copy()
        
        # Calculate technical indicators
        indicators = analyzer.calculate_technical_indicators(df)
        
        # Get realtime price and update last candle
        realtime_price = analyzer.get_realtime_price()
        if realtime_price and len(df) > 0:
            df.iloc[-1, df.columns.get_loc('close')] = realtime_price
            current_high = df.iloc[-1]['high']
            current_low = df.iloc[-1]['low']
            df.iloc[-1, df.columns.get_loc('high')] = max(current_high, realtime_price)
            df.iloc[-1, df.columns.get_loc('low')] = min(current_low, realtime_price)
        
        # Get fundamental news
        news_data = analyzer.get_fundamental_news()
        
        # Prepare technical data for AI
        latest_indicators = {}
        for key, values in indicators.items():
            if values is not None and len(values) > 0:
                last_val = values[-1]
                if last_val is not None and not np.isnan(last_val):
                    latest_indicators[key] = float(last_val)
        
        technical_data = {
            "current_price": realtime_price,
            "indicators": latest_indicators,
            "price_action": {
                "open": float(df.iloc[-1]['open']) if len(df) > 0 else 0,
                "high": float(df.iloc[-1]['high']) if len(df) > 0 else 0,
                "low": float(df.iloc[-1]['low']) if len(df) > 0 else 0,
                "close": float(df.iloc[-1]['close']) if len(df) > 0 else 0
            }
        }
        
        # AI Analysis
        ai_analysis = analyzer.analyze_with_deepseek(technical_data, news_data)
        
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "current_price": realtime_price,
            "technical_indicators": latest_indicators,
            "ai_analysis": ai_analysis,
            "chart_data": df.tail(100).to_dict('records'),
            "news": news_data
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analysis endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/chart/data/<timeframe>')
def get_chart_data(timeframe):
    """Endpoint untuk data chart"""
    analyzer = XAUUSDAnalyzer()
    
    # Validate timeframe
    valid_timeframes = ['1D', '4H', '1H']
    if timeframe not in valid_timeframes:
        return jsonify({"error": f"Invalid timeframe. Use: {valid_timeframes}"}), 400
    
    try:
        df = analyzer.load_historical_data(timeframe)
        if df is None or df.empty:
            return jsonify({"error": "No data available"}), 404
        
        # Filter based on timeframe
        if timeframe == '1D':
            df = df.tail(730)  # 2 years
        elif timeframe == '4H':
            df = df.tail(2190)  # 1 year
        elif timeframe == '1H': 
            df = df.tail(4320)  # 6 months
        
        return jsonify(df.to_dict('records'))
        
    except Exception as e:
        print(f"Error in chart data endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Endpoint untuk kompatibilitas dengan request yang ada
@app.route('/api/analyze')
def analyze():
    """Legacy endpoint untuk kompatibilitas"""
    pair = request.args.get('pair', 'XAUUSD')
    timeframe = request.args.get('timeframe', '4H')
    
    if pair != 'XAUUSD':
        return jsonify({"error": "Only XAUUSD is supported"}), 400
        
    return get_analysis(timeframe)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("Starting XAUUSD AI Analysis Server...")
    print("Available endpoints:")
    print("  GET / - Main application")
    print("  GET /api/analysis/<timeframe> - Get analysis for timeframe (1D, 4H, 1H)")
    print("  GET /api/chart/data/<timeframe> - Get chart data")
    print("  GET /health - Health check")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
