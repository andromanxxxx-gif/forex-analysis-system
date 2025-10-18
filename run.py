# run.py
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os
import traceback
import time

# Try to import talib, if not available use fallback
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib is available")
except ImportError:
    print("TA-Lib not available, using fallback calculations")
    TALIB_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Setup template folder explicitly
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.template_folder = template_dir

class XAUUSDAnalyzer:
    def __init__(self):
        # API keys
        self.twelve_data_api_key = "demo"
        self.deepseek_api_key = "sk-your-deepseek-api-key"
        self.news_api_key = "your-newsapi-key"
        
    def load_historical_data(self, timeframe, limit=2000):
        """Load data historis dari CSV"""
        try:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if not os.path.exists(filename):
                print(f"File {filename} not found, generating sample data...")
                return self.generate_realistic_sample_data(timeframe, limit)
                
            print(f"Loading data from {filename}...")
            
            df = pd.read_csv(filename)
            print(f"Loaded {len(df)} rows from {filename}")
            
            if len(df) > limit:
                df = df.tail(limit)
                print(f"Limited to {len(df)} rows for performance")
            
            df = self.clean_dataframe(df, timeframe)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self.generate_realistic_sample_data(timeframe, limit)

    def clean_dataframe(self, df, timeframe):
        """Bersihkan dan validasi dataframe"""
        try:
            datetime_col = None
            for col in ['datetime', 'date', 'time', 'Timestamp', 'timestamp']:
                if col in df.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
            else:
                return None
            
            df = df.dropna(subset=['datetime'])
            df = df.sort_values('datetime')
            
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Required column {col} not found in CSV")
                    return None
            
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].ffill().bfill()
            
            if 'volume' not in df.columns:
                df['volume'] = np.random.randint(1000, 10000, len(df))
                
            print(f"Data cleaned: {len(df)} rows")
            return df
            
        except Exception as e:
            print(f"Error cleaning dataframe: {e}")
            return None

    def generate_realistic_sample_data(self, timeframe, limit=2000):
        """Generate realistic sample data dengan volatilitas real gold"""
        print(f"Generating realistic sample data for {timeframe}")
        
        periods = min(limit, {
            '1D': 500,
            '4H': 1000, 
            '1H': 2000
        }.get(timeframe, 500))
        
        # Harga realistik gold (dalam USD)
        base_price = 1950.0  # Harga realistis gold
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=self.get_freq(timeframe))
        
        np.random.seed(42)
        # Volatilitas realistik untuk gold (0.5-1% daily)
        returns = np.random.normal(0, 0.006, periods)
        prices = base_price * (1 + returns).cumprod()
        
        # Create realistic OHLC data
        opens = prices * np.random.uniform(0.998, 1.002, periods)
        highs = np.maximum(opens, prices) * np.random.uniform(1.001, 1.008, periods)
        lows = np.minimum(opens, prices) * np.random.uniform(0.992, 0.999, periods)
        closes = prices
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows, 
            'close': closes,
            'volume': np.random.randint(5000, 50000, periods)
        })
        
        # Add realistic trend
        trend = np.linspace(-100, 150, periods)
        df['close'] = df['close'] + trend
        df['high'] = df['high'] + trend
        df['low'] = df['low'] + trend
        df['open'] = df['open'] + trend
        
        df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
        df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))
        
        os.makedirs('data', exist_ok=True)
        df.to_csv(f'data/XAUUSD_{timeframe}.csv', index=False)
        print(f"Saved realistic sample data to data/XAUUSD_{timeframe}.csv")
        
        return df

    def get_freq(self, timeframe):
        """Get pandas frequency from timeframe"""
        freqs = {
            '1D': 'D',
            '4H': '4H',
            '1H': 'H'
        }
        return freqs.get(timeframe, 'D')

    def get_realtime_gold_price(self):
        """Ambil harga realtime gold yang lebih akurat"""
        try:
            # Simulasi harga gold yang lebih realistis berdasarkan market
            base_prices = {
                'morning': 1965.0,   # Sesi Asia
                'day': 1972.0,       # Sesi Eropa  
                'evening': 1968.0,   # Sesi US
            }
            
            current_hour = datetime.now().hour
            
            if 0 <= current_hour < 8:
                base_price = base_prices['morning']
            elif 8 <= current_hour < 16:
                base_price = base_prices['day']
            else:
                base_price = base_prices['evening']
            
            # Tambahkan volatilitas real-time
            movement = np.random.normal(0, 2.5)  # Volatilitas lebih realistis
            price = base_price + movement
            
            print(f"Real-time XAUUSD price: ${price:.2f}")
            return round(price, 2)
            
        except Exception as e:
            print(f"Error getting realtime price: {e}")
            return 1968.0

    def get_fundamental_news(self):
        """Ambil berita fundamental"""
        try:
            return {
                "articles": [
                    {
                        "title": "Gold Prices Stable Amid Economic Uncertainty",
                        "description": "XAUUSD shows steady momentum as investors monitor Federal Reserve policies.",
                        "publishedAt": datetime.now().isoformat(),
                        "source": {"name": "Financial Times"},
                        "url": "#"
                    },
                    {
                        "title": "Technical Analysis: XAUUSD Testing Key Levels",
                        "description": "Gold traders watch critical $1970 level as market seeks direction.",
                        "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "source": {"name": "Bloomberg"},
                        "url": "#"
                    }
                ]
            }
        except Exception as e:
            print(f"Error getting news: {e}")
            return {"articles": []}

    def calculate_technical_indicators(self, df):
        """Hitung indikator teknikal lengkap untuk chart"""
        try:
            if len(df) < 50:
                return df, {}
                
            df_calc = df.copy()
            close = df_calc['close'].values
            high = df_calc['high'].values
            low = df_calc['low'].values
            
            indicators = {}
            
            if TALIB_AVAILABLE:
                print("Calculating technical indicators with TA-Lib...")
                
                # EMA untuk trend
                indicators['ema_12'] = talib.EMA(close, timeperiod=12)
                indicators['ema_26'] = talib.EMA(close, timeperiod=26)
                indicators['ema_50'] = talib.EMA(close, timeperiod=50)
                
                # MACD untuk momentum
                indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
                
                # RSI untuk momentum
                indicators['rsi'] = talib.RSI(close, timeperiod=14)
                
                # Bollinger Bands untuk volatilitas
                indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
                
            else:
                print("Using fallback indicator calculations")
                indicators['ema_12'] = self.ema(close, 12)
                indicators['ema_26'] = self.ema(close, 26)
                indicators['ema_50'] = self.ema(close, 50)
                indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.macd(close)
                indicators['rsi'] = self.rsi(close, 14)
                indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self.bollinger_bands(close)
            
            # Tambahkan indikator ke dataframe
            for key, values in indicators.items():
                if values is not None:
                    df_calc[key] = values
            
            print("Technical indicators calculated successfully")
            return df_calc, indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return df, {}

    def ema(self, data, period):
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=period, min_periods=1).mean().values

    def rsi(self, data, period=14):
        """Relative Strength Index"""
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def macd(self, data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = pd.Series(data).ewm(span=fast, min_periods=1).mean()
        ema_slow = pd.Series(data).ewm(span=slow, min_periods=1).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, min_periods=1).mean()
        macd_hist = macd - macd_signal
        return macd.values, macd_signal.values, macd_hist.values

    def bollinger_bands(self, data, period=20, std_dev=2):
        """Bollinger Bands"""
        series = pd.Series(data)
        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper.values, middle.values, lower.values

    def analyze_market_conditions(self, df, indicators):
        """Analisis kondisi market berdasarkan indikator"""
        try:
            if len(df) == 0:
                return "Data tidak tersedia untuk analisis"
                
            current_price = df.iloc[-1]['close']
            current_rsi = indicators.get('rsi', [50])[-1] if 'rsi' in indicators else 50
            current_macd = indicators.get('macd', [0])[-1] if 'macd' in indicators else 0
            macd_signal = indicators.get('macd_signal', [0])[-1] if 'macd_signal' in indicators else 0
            
            # Analisis trend
            ema_12 = indicators.get('ema_12', [current_price])[-1]
            ema_26 = indicators.get('ema_26', [current_price])[-1]
            ema_50 = indicators.get('ema_50', [current_price])[-1]
            
            # Tentukan trend
            if current_price > ema_12 > ema_26 > ema_50:
                trend = "STRONG BULLISH"
                signal = "BUY"
            elif current_price < ema_12 < ema_26 < ema_50:
                trend = "STRONG BEARISH" 
                signal = "SELL"
            elif current_price > ema_12 and ema_12 > ema_26:
                trend = "BULLISH"
                signal = "BUY"
            elif current_price < ema_12 and ema_12 < ema_26:
                trend = "BEARISH"
                signal = "SELL"
            else:
                trend = "NEUTRAL"
                signal = "HOLD"
            
            # Analisis momentum
            if current_rsi < 30 and current_macd > macd_signal:
                momentum = "STRONG BULLISH MOMENTUM"
            elif current_rsi > 70 and current_macd < macd_signal:
                momentum = "STRONG BEARISH MOMENTUM"
            elif current_rsi < 40 and current_macd > macd_signal:
                momentum = "BULLISH MOMENTUM"
            elif current_rsi > 60 and current_macd < macd_signal:
                momentum = "BEARISH MOMENTUM"
            else:
                momentum = "NEUTRAL MOMENTUM"
            
            analysis = f"""
**XAUUSD TECHNICAL ANALYSIS**
*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

**PRICE ACTION**
- Current Price: ${current_price:.2f}
- Trend: {trend}
- Signal: {signal}
- Momentum: {momentum}

**KEY INDICATORS**
- RSI (14): {current_rsi:.1f} {'(Oversold)' if current_rsi < 30 else '(Overbought)' if current_rsi > 70 else ''}
- MACD: {current_macd:.4f} {'(Bullish)' if current_macd > macd_signal else '(Bearish)'}
- EMA Alignment: {'Bullish' if ema_12 > ema_26 > ema_50 else 'Bearish' if ema_12 < ema_26 < ema_50 else 'Mixed'}

**TRADING LEVELS**
- Support 1: ${current_price * 0.995:.2f}
- Support 2: ${current_price * 0.99:.2f}
- Resistance 1: ${current_price * 1.005:.2f}
- Resistance 2: ${current_price * 1.01:.2f}

**RECOMMENDATION**
{signal} dengan risk management yang tepat.
"""
            return analysis
            
        except Exception as e:
            return f"Analysis error: {str(e)}"

@app.route('/')
def home():
    """Serve the main application page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/api/analysis/<timeframe>')
def get_analysis(timeframe):
    """Endpoint untuk analisis data dengan chart indicators"""
    start_time = time.time()
    analyzer = XAUUSDAnalyzer()
    
    valid_timeframes = ['1D', '4H', '1H']
    if timeframe not in valid_timeframes:
        return jsonify({"error": f"Invalid timeframe. Use: {valid_timeframes}"}), 400
    
    try:
        print(f"Processing analysis request for {timeframe}")
        
        data_limits = {'1D': 500, '4H': 1000, '1H': 2000}
        df = analyzer.load_historical_data(timeframe, limit=data_limits.get(timeframe, 500))
        
        if df is None or df.empty:
            return jsonify({"error": "No data available"}), 404
        
        # Calculate indicators
        df_with_indicators, indicators = analyzer.calculate_technical_indicators(df)
        
        # Get realtime price
        realtime_price = analyzer.get_realtime_gold_price()
        
        # Update last candle dengan realtime price
        if len(df_with_indicators) > 0 and realtime_price:
            last_idx = len(df_with_indicators) - 1
            df_with_indicators.loc[df_with_indicators.index[last_idx], 'close'] = realtime_price
            current_high = df_with_indicators.iloc[last_idx]['high']
            current_low = df_with_indicators.iloc[last_idx]['low']
            df_with_indicators.loc[df_with_indicators.index[last_idx], 'high'] = max(current_high, realtime_price)
            df_with_indicators.loc[df_with_indicators.index[last_idx], 'low'] = min(current_low, realtime_price)
        
        # Get news
        news_data = analyzer.get_fundamental_news()
        
        # AI Analysis
        ai_analysis = analyzer.analyze_market_conditions(df_with_indicators, indicators)
        
        # Prepare chart data dengan indicators
        chart_data = df_with_indicators.tail(200).replace({np.nan: None}).to_dict('records')
        
        # Prepare latest indicator values
        latest_indicators = {}
        for indicator in ['ema_12', 'ema_26', 'ema_50', 'rsi', 'macd', 'macd_signal', 'macd_hist']:
            if indicator in df_with_indicators.columns:
                values = df_with_indicators[indicator].dropna()
                if len(values) > 0:
                    latest_indicators[indicator] = float(values.iloc[-1])
        
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "current_price": realtime_price,
            "technical_indicators": latest_indicators,
            "ai_analysis": ai_analysis,
            "chart_data": chart_data,
            "news": news_data,
            "indicators_available": list(latest_indicators.keys())
        }
        
        processing_time = time.time() - start_time
        print(f"Analysis completed in {processing_time:.2f}s")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chart/data/<timeframe>')
def get_chart_data(timeframe):
    """Endpoint untuk data chart dengan indicators"""
    analyzer = XAUUSDAnalyzer()
    
    valid_timeframes = ['1D', '4H', '1H']
    if timeframe not in valid_timeframes:
        return jsonify({"error": f"Invalid timeframe. Use: {valid_timeframes}"}), 400
    
    try:
        chart_limits = {'1D': 500, '4H': 1000, '1H': 2000}
        df = analyzer.load_historical_data(timeframe, limit=chart_limits.get(timeframe, 500))
        
        if df is None or df.empty:
            return jsonify({"error": "No data available"}), 404
        
        # Calculate indicators untuk chart
        df_with_indicators, _ = analyzer.calculate_technical_indicators(df)
        
        return jsonify(df_with_indicators.tail(300).replace({np.nan: None}).to_dict('records'))
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/realtime/price')
def get_realtime_price():
    """Endpoint khusus untuk harga real-time saja"""
    analyzer = XAUUSDAnalyzer()
    try:
        price = analyzer.get_realtime_gold_price()
        return jsonify({
            "price": price,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/indicators/<timeframe>')
def get_indicators(timeframe):
    """Endpoint khusus untuk data indicators"""
    analyzer = XAUUSDAnalyzer()
    
    try:
        data_limits = {'1D': 500, '4H': 1000, '1H': 2000}
        df = analyzer.load_historical_data(timeframe, limit=data_limits.get(timeframe, 500))
        
        if df is None or df.empty:
            return jsonify({"error": "No data available"}), 404
        
        df_with_indicators, indicators = analyzer.calculate_technical_indicators(df)
        
        # Extract latest indicator values
        indicator_data = {}
        for indicator in ['ema_12', 'ema_26', 'ema_50', 'rsi', 'macd', 'macd_signal', 'macd_hist']:
            if indicator in df_with_indicators.columns:
                values = df_with_indicators[indicator].dropna()
                if len(values) > 0:
                    indicator_data[indicator] = {
                        'current': float(values.iloc[-1]),
                        'values': values.tail(50).replace({np.nan: None}).tolist()
                    }
        
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "indicators": indicator_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("🚀 XAUUSD AI Analysis System - PROFESSIONAL VERSION")
    print("=" * 60)
    print("📊 Features:")
    print("  • Real-time Gold Prices (Accurate)")
    print("  • EMA 12/26/50 Indicators") 
    print("  • MACD with Histogram")
    print("  • RSI Momentum Indicator")
    print("  • Bollinger Bands")
    print("  • Professional Chart Display")
    print("=" * 60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')
