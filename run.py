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

print(f"Template directory: {template_dir}")
print(f"Template folder exists: {os.path.exists(template_dir)}")

class XAUUSDAnalyzer:
    def __init__(self):
        # Use demo keys for testing
        self.twelve_data_api_key = "demo"
        self.deepseek_api_key = "sk-your-deepseek-key-here"
        self.news_api_key = "your-news-api-key"
        
    def load_historical_data(self, timeframe, limit=None):
        """Load data historis dari CSV dengan optimasi"""
        try:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if not os.path.exists(filename):
                print(f"File {filename} not found, generating sample data...")
                return self.generate_sample_data(timeframe, limit)
                
            print(f"Loading data from {filename}...")
            
            # Baca hanya kolom yang diperlukan untuk menghemat memory
            usecols = ['datetime', 'open', 'high', 'low', 'close']
            df = pd.read_csv(filename, usecols=lambda col: col in usecols)
            
            print(f"Loaded {len(df)} rows from {filename}")
            
            # Limit data jika diperlukan
            if limit and len(df) > limit:
                df = df.tail(limit)
                print(f"Limited to {len(df)} rows for performance")
            
            # Clean the data
            df = self.clean_dataframe(df, timeframe)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return self.generate_sample_data(timeframe, limit)

    def clean_dataframe(self, df, timeframe):
        """Bersihkan dan validasi dataframe dengan optimasi"""
        try:
            # Handle berbagai format kolom
            datetime_col = None
            for col in ['datetime', 'date', 'time', 'Timestamp', 'timestamp']:
                if col in df.columns:
                    datetime_col = col
                    break
            
            if datetime_col:
                df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
            else:
                # Jika tidak ada kolom datetime, buat dari index
                freq = self.get_freq(timeframe)
                df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq=freq)
            
            # Remove rows with invalid datetime
            df = df.dropna(subset=['datetime'])
            df = df.sort_values('datetime')
            
            # Ensure required columns exist dengan nilai yang valid
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Column {col} not found, generating...")
                    if col == 'open':
                        df['open'] = df['close'] * 0.999 if 'close' in df.columns else 1800.0
                    elif col == 'high':
                        df['high'] = (df['close'] * 1.002) if 'close' in df.columns else 1800.0
                    elif col == 'low':
                        df['low'] = (df['close'] * 0.998) if 'close' in df.columns else 1800.0
                    elif col == 'close':
                        df['close'] = 1800.0  # Default value
            
            # Convert to numeric and handle invalid values
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN values dengan method yang efisien
                df[col] = df[col].ffill().bfill().fillna(1800.0)
            
            print(f"Data cleaned: {len(df)} rows")
            if len(df) > 0:
                print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"Error cleaning dataframe: {e}")
            return self.generate_sample_data(timeframe, 100)

    def generate_sample_data(self, timeframe, limit=1000):
        """Generate sample data untuk testing dengan optimasi"""
        print(f"Generating sample data for {timeframe} (limit: {limit})")
        
        periods = min(limit, {
            '1D': 500,   # ~2 years
            '4H': 1000,  # ~1 year  
            '1H': 2000   # ~6 months
        }.get(timeframe, 500))
        
        base_price = 1800.0
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=self.get_freq(timeframe))
        
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, periods)
        prices = base_price * (1 + returns).cumprod()
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': prices * 0.999,
            'high': prices * 1.002,
            'low': prices * 0.998, 
            'close': prices
        })
        
        # Add some trend
        trend = np.linspace(0, 200, periods)
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
        """Ambil harga realtime - optimasi untuk demo"""
        try:
            # Untuk demo, gunakan harga acak yang realistis
            base_price = 1950.0
            random_change = np.random.normal(0, 2)  # Smaller random change
            price = base_price + random_change
            print(f"Generated realtime price: ${price:.2f}")
            return price
            
        except Exception as e:
            print(f"Error getting realtime price: {e}")
            return 1950.0

    def get_fundamental_news(self):
        """Ambil berita fundamental - optimasi"""
        try:
            # Sample news data yang ringan
            return {
                "articles": [
                    {
                        "title": "Gold Market Analysis - Real-time Updates",
                        "description": "Latest gold price movements and technical analysis.",
                        "publishedAt": datetime.now().isoformat(),
                        "source": {"name": "Market Data"}
                    }
                ]
            }
        except Exception as e:
            print(f"Error getting news: {e}")
            return {"articles": []}

    def calculate_technical_indicators(self, df):
        """Hitung indikator teknikal dengan optimasi performa"""
        try:
            # Gunakan hanya data terakhir untuk perhitungan yang lebih cepat
            if len(df) > 500:
                df_calc = df.tail(500).copy()
            else:
                df_calc = df.copy()
                
            # Price data
            high = df_calc['high'].values
            low = df_calc['low'].values
            close = df_calc['close'].values
            
            indicators = {}
            
            if TALIB_AVAILABLE:
                print("Using TA-Lib for indicator calculations (optimized)")
                
                # Hanya hitung indikator utama untuk performa
                if len(close) >= 20:
                    indicators['sma_20'] = talib.SMA(close, timeperiod=20)
                if len(close) >= 50:
                    indicators['sma_50'] = talib.SMA(close, timeperiod=50)
                if len(close) >= 14:
                    indicators['rsi'] = talib.RSI(close, timeperiod=14)
                if len(close) >= 26:
                    indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
                
                # Extend arrays to match original dataframe length
                for key in indicators:
                    if indicators[key] is not None:
                        original_len = len(df)
                        calc_len = len(indicators[key])
                        if calc_len < original_len:
                            # Pad with NaN values at the beginning
                            padding = np.full(original_len - calc_len, np.nan)
                            indicators[key] = np.concatenate([padding, indicators[key]])
                        
            else:
                print("Using fallback indicator calculations (optimized)")
                # Fallback calculations yang lebih ringan
                if len(close) >= 20:
                    indicators['sma_20'] = self.sma(close, 20)
                if len(close) >= 50:
                    indicators['sma_50'] = self.sma(close, 50)
                if len(close) >= 14:
                    indicators['rsi'] = self.rsi(close, 14)
                if len(close) >= 26:
                    indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.macd(close)
            
            print(f"Calculated {len(indicators)} technical indicators")
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            traceback.print_exc()
            return {}

    # Fallback technical indicator functions yang dioptimasi
    def sma(self, data, period):
        return pd.Series(data).rolling(window=period, min_periods=1).mean().values

    def ema(self, data, period):
        return pd.Series(data).ewm(span=period, min_periods=1).mean().values

    def rsi(self, data, period=14):
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values

    def macd(self, data, fast=12, slow=26, signal=9):
        ema_fast = pd.Series(data).ewm(span=fast, min_periods=1).mean()
        ema_slow = pd.Series(data).ewm(span=slow, min_periods=1).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, min_periods=1).mean()
        macd_hist = macd - macd_signal
        return macd.values, macd_signal.values, macd_hist.values

    def analyze_with_deepseek(self, technical_data, news_data):
        """Analisis dengan AI - versi ringan untuk performa"""
        try:
            current_price = technical_data.get('current_price', 1950.0)
            indicators = technical_data.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            # Analisis sederhana berdasarkan indikator
            if rsi > 70 and macd < macd_signal:
                trend = "Bearish"
                signal = "Sell"
                risk = "High"
            elif rsi < 30 and macd > macd_signal:
                trend = "Bullish"
                signal = "Buy" 
                risk = "Medium"
            else:
                trend = "Sideways"
                signal = "Hold"
                risk = "Low"
                
            analysis = f"""
1. TREND: {trend}
2. SUPPORT: {current_price - 15.0:.2f}
3. RESISTANCE: {current_price + 20.0:.2f}
4. SIGNAL: {signal}
5. RISK LEVEL: {risk}
6. ANALYSIS: XAUUSD saat ini diperdagangkan di ${current_price:.2f} dengan kondisi {trend.lower()}. RSI di level {rsi:.1f} menunjukkan kondisi {'jenuh beli' if rsi > 70 else 'jenuh jual' if rsi < 30 else 'netral'}.

7. KEY LEVELS:
- Support: {current_price - 10.0:.2f}
- Resistance: {current_price + 15.0:.2f}
"""
            print("Generated optimized AI analysis")
            return analysis
            
        except Exception as e:
            error_msg = f"Analysis: Simple analysis - {trend} trend detected"
            print(error_msg)
            return error_msg

@app.route('/')
def home():
    """Serve the main application page"""
    try:
        # Check if template exists
        template_path = os.path.join(app.template_folder, 'index.html')
        if os.path.exists(template_path):
            print(f"Template found at: {template_path}")
            return render_template('index.html')
        else:
            print(f"Template not found at: {template_path}")
            return create_fallback_html()
    except Exception as e:
        print(f"Error rendering template: {e}")
        return create_fallback_html()

def create_fallback_html():
    """Create fallback HTML when template is not found"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>XAUUSD AI Analysis System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #0f1b2b; color: white; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #1e2b3a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .endpoints { background: #1e2b3a; padding: 20px; border-radius: 10px; }
            a { color: #00d4aa; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .status { color: #00d4aa; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 XAUUSD AI Analysis System</h1>
                <p class="status">Backend is running! However, the dashboard template was not found.</p>
            </div>
            
            <div class="endpoints">
                <h2>📊 Available API Endpoints:</h2>
                <ul>
                    <li><a href="/api/analysis/1D" target="_blank">/api/analysis/1D</a> - Analysis for 1D timeframe</li>
                    <li><a href="/api/analysis/4H" target="_blank">/api/analysis/4H</a> - Analysis for 4H timeframe</li>
                    <li><a href="/api/analysis/1H" target="_blank">/api/analysis/1H</a> - Analysis for 1H timeframe</li>
                    <li><a href="/health" target="_blank">/health</a> - Health check</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/api/analysis/<timeframe>')
def get_analysis(timeframe):
    """Endpoint untuk analisis data dengan optimasi performa"""
    start_time = time.time()
    analyzer = XAUUSDAnalyzer()
    
    # Validate timeframe
    valid_timeframes = ['1D', '4H', '1H']
    if timeframe not in valid_timeframes:
        return jsonify({"error": f"Invalid timeframe. Use: {valid_timeframes}"}), 400
    
    try:
        print(f"🚀 Processing analysis request for timeframe: {timeframe}")
        
        # Limit data berdasarkan timeframe untuk performa
        data_limits = {
            '1D': 500,   # ~2 years
            '4H': 1000,  # ~1 year
            '1H': 2000   # ~6 months
        }
        
        limit = data_limits.get(timeframe, 500)
        
        # Load historical data dengan limit
        df = analyzer.load_historical_data(timeframe, limit=limit)
        if df is None or df.empty:
            return jsonify({"error": f"No historical data found for {timeframe}"}), 404
        
        print(f"📊 Loaded {len(df)} rows of data")
        
        # Calculate technical indicators
        print("🔧 Calculating technical indicators...")
        indicators = analyzer.calculate_technical_indicators(df)
        print(f"✅ Calculated {len(indicators)} indicators")
        
        # Get realtime price
        print("💰 Getting realtime price...")
        realtime_price = analyzer.get_realtime_price()
        print(f"📈 Realtime price: {realtime_price}")
        
        # Get fundamental news
        print("📰 Getting news data...")
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
        print("🤖 Generating AI analysis...")
        ai_analysis = analyzer.analyze_with_deepseek(technical_data, news_data)
        
        # Siapkan data chart yang lebih kecil untuk performa
        chart_data_limit = min(200, len(df))
        chart_data = df.tail(chart_data_limit).to_dict('records')
        
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "current_price": realtime_price,
            "technical_indicators": latest_indicators,
            "ai_analysis": ai_analysis,
            "chart_data": chart_data,
            "news": news_data
        }
        
        processing_time = time.time() - start_time
        print(f"✅ Analysis completed successfully in {processing_time:.2f} seconds")
        
        return jsonify(response_data)
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"❌ Error in analysis endpoint after {processing_time:.2f}s: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/chart/data/<timeframe>')
def get_chart_data(timeframe):
    """Endpoint untuk data chart dengan optimasi performa"""
    analyzer = XAUUSDAnalyzer()
    
    # Validate timeframe
    valid_timeframes = ['1D', '4H', '1H']
    if timeframe not in valid_timeframes:
        return jsonify({"error": f"Invalid timeframe. Use: {valid_timeframes}"}), 400
    
    try:
        # Limit data chart untuk performa
        chart_limits = {
            '1D': 500,   # 500 candles max
            '4H': 1000,  # 1000 candles max  
            '1H': 2000   # 2000 candles max
        }
        
        limit = chart_limits.get(timeframe, 500)
        df = analyzer.load_historical_data(timeframe, limit=limit)
        
        if df is None or df.empty:
            return jsonify({"error": "No data available"}), 404
        
        print(f"📈 Returning {len(df)} chart data points for {timeframe}")
        return jsonify(df.to_dict('records'))
        
    except Exception as e:
        print(f"Error in chart data endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "analysis": "/api/analysis/<timeframe>",
            "chart_data": "/api/chart/data/<timeframe>",
            "health": "/health"
        }
    })

if __name__ == '__main__':
    # Create directories if not exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("🚀 XAUUSD AI Analysis System - OPTIMIZED VERSION")
    print("=" * 60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Template folder: {template_dir}")
    print("=" * 60)
    print("📊 Available Endpoints:")
    print("  GET / - Main dashboard")
    print("  GET /api/analysis/<1D|4H|1H> - Technical analysis")
    print("  GET /api/chart/data/<1D|4H|1H> - Chart data")
    print("  GET /health - System health")
    print("=" * 60)
    print("⚡ Performance Optimizations:")
    print("  • Limited data loading (500-2000 records max)")
    print("  • Optimized technical indicator calculations")
    print("  • Faster response times")
    print("=" * 60)
    
    try:
        app.run(debug=True, port=5000, host='0.0.0.0')
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
