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
        # Ganti dengan API keys Anda yang sesungguhnya
        self.twelve_data_api_key = "demo"  # Dapatkan dari https://twelvedata.com/
        self.deepseek_api_key = "sk-YOUR-DEEPSEEK-API-KEY"  # Dapatkan dari https://platform.deepseek.com/
        self.news_api_key = "YOUR-NEWSAPI-KEY"  # Dapatkan dari https://newsapi.org/
        
    def load_historical_data(self, timeframe):
        """Load data historis dari CSV yang sesungguhnya"""
        try:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if not os.path.exists(filename):
                print(f"File {filename} not found")
                return None
                
            df = pd.read_csv(filename)
            print(f"Loaded {len(df)} rows from {filename}")
            
            # Clean the data
            df = self.clean_dataframe(df, timeframe)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None

    def clean_dataframe(self, df, timeframe):
        """Bersihkan dan validasi dataframe"""
        # Handle berbagai format kolom
        datetime_col = None
        for col in ['datetime', 'date', 'time', 'Timestamp', 'timestamp']:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col:
            df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
        else:
            return None
        
        # Remove rows with invalid datetime
        df = df.dropna(subset=['datetime'])
        df = df.sort_values('datetime')
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print(f"Required column {col} not found in CSV")
                return None
        
        # Convert to numeric and handle invalid values
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN values with forward/backward fill
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        if 'volume' not in df.columns:
            df['volume'] = 0  # Default value jika tidak ada volume
            
        print(f"Data cleaned: {len(df)} rows, columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
        
        return df

    def get_realtime_price(self):
        """Ambil harga realtime dari Twelve Data API"""
        try:
            # Untuk demo, kita gunakan harga acak dulu
            # Ganti dengan API call sesungguhnya nanti
            base_price = 1950.0
            random_change = np.random.normal(0, 5)
            price = base_price + random_change
            print(f"Generated realtime price: ${price:.2f}")
            return price
            
        except Exception as e:
            print(f"Error getting realtime price: {e}")
            return 1950.0

    def get_fundamental_news(self):
        """Ambil berita fundamental"""
        try:
            # Untuk demo, gunakan sample news dulu
            news = {
                "articles": [
                    {
                        "title": "Gold Prices Stable Amid Economic Uncertainty",
                        "description": "XAUUSD shows resilience in current market conditions with technical indicators pointing to continued bullish momentum.",
                        "publishedAt": datetime.now().isoformat(),
                        "source": {"name": "Market Analysis"}
                    },
                    {
                        "title": "Federal Reserve Decision Impacts Precious Metals",
                        "description": "Recent Fed announcements affecting gold prices and creating favorable conditions for XAUUSD.",
                        "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "source": {"name": "Financial News"}
                    },
                    {
                        "title": "Technical Analysis: XAUUSD Breaking Key Levels",
                        "description": "Gold approaches significant resistance zone as traders watch for breakout signals in the current session.",
                        "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                        "source": {"name": "Trading Insights"}
                    }
                ]
            }
            print("Generated sample news data")
            return news
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
                print("Using TA-Lib for indicator calculations")
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
            
            print(f"Calculated {len(indicators)} technical indicators")
            return indicators
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            traceback.print_exc()
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
        """Analisis dengan AI"""
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
6. ANALYSIS: XAUUSD saat ini diperdagangkan di ${current_price:.2f} dengan kondisi {trend.lower()}. RSI berada di level {rsi:.1f} menunjukkan kondisi {'jenuh beli' if rsi > 70 else 'jenuh jual' if rsi < 30 else 'netral'}. MACD {'positif' if macd > macd_signal else 'negatif'} menunjukkan momentum {'naik' if macd > macd_signal else 'turun'}.

REKOMENDASI TRADING:
- {signal} dengan target resistance di ${current_price + 20.0:.2f}
- Stop loss di ${current_price - 15.0:.2f}
- Monitor level kunci untuk konfirmasi

7. KEY LEVELS:
- Support 1: ${current_price - 10.0:.2f}
- Support 2: ${current_price - 25.0:.2f}  
- Resistance 1: ${current_price + 15.0:.2f}
- Resistance 2: ${current_price + 35.0:.2f}
"""
            print("Generated AI analysis")
            return analysis
            
        except Exception as e:
            error_msg = f"Analysis: XAUUSD analysis currently unavailable. Error: {str(e)}"
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
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>XAUUSD AI Analysis System</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #0f1b2b; color: white; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: #1e2b3a; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .endpoints {{ background: #1e2b3a; padding: 20px; border-radius: 10px; }}
            a {{ color: #00d4aa; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            .status {{ color: #00d4aa; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 XAUUSD AI Analysis System</h1>
                <p class="status">Backend is running! However, the dashboard template was not found.</p>
                <p>Please ensure 'index.html' is in the 'templates' folder.</p>
            </div>
            
            <div class="endpoints">
                <h2>📊 Available API Endpoints:</h2>
                <ul>
                    <li><a href="/api/analysis/1D" target="_blank">/api/analysis/1D</a> - Analysis for 1D timeframe</li>
                    <li><a href="/api/analysis/4H" target="_blank">/api/analysis/4H</a> - Analysis for 4H timeframe</li>
                    <li><a href="/api/analysis/1H" target="_blank">/api/analysis/1H</a> - Analysis for 1H timeframe</li>
                    <li><a href="/api/chart/data/1D" target="_blank">/api/chart/data/1D</a> - Chart data for 1D</li>
                    <li><a href="/health" target="_blank">/health</a> - Health check</li>
                </ul>
            </div>
            
            <div class="endpoints">
                <h2>🔧 Next Steps:</h2>
                <ol>
                    <li>Create a 'templates' folder in the same directory as run.py</li>
                    <li>Place 'index.html' inside the 'templates' folder</li>
                    <li>Restart the server</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/api/analysis/<timeframe>')
def get_analysis(timeframe):
    """Endpoint untuk analisis data"""
    analyzer = XAUUSDAnalyzer()
    
    # Validate timeframe
    valid_timeframes = ['1D', '4H', '1H']
    if timeframe not in valid_timeframes:
        return jsonify({"error": f"Invalid timeframe. Use: {valid_timeframes}"}), 400
    
    try:
        print(f"Processing analysis request for timeframe: {timeframe}")
        
        # Load historical data
        df = analyzer.load_historical_data(timeframe)
        if df is None or df.empty:
            return jsonify({"error": f"No historical data found for {timeframe}. Please ensure CSV files exist in data folder."}), 404
        
        print(f"Loaded {len(df)} rows of data")
        
        # Get latest 600 data points
        df = df.tail(600).copy()
        
        # Calculate technical indicators
        print("Calculating technical indicators...")
        indicators = analyzer.calculate_technical_indicators(df)
        print(f"Calculated {len(indicators)} indicators")
        
        # Get realtime price and update last candle
        print("Getting realtime price...")
        realtime_price = analyzer.get_realtime_price()
        print(f"Realtime price: {realtime_price}")
        
        if realtime_price and len(df) > 0:
            # Update the last candle with realtime price
            last_idx = len(df) - 1
            df.iloc[last_idx, df.columns.get_loc('close')] = realtime_price
            current_high = df.iloc[last_idx]['high']
            current_low = df.iloc[last_idx]['low']
            df.iloc[last_idx, df.columns.get_loc('high')] = max(current_high, realtime_price)
            df.iloc[last_idx, df.columns.get_loc('low')] = min(current_low, realtime_price)
        
        # Get fundamental news
        print("Getting news data...")
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
        print("Generating AI analysis...")
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
        
        print("Analysis completed successfully")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analysis endpoint: {e}")
        traceback.print_exc()
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
        
        print(f"Returning {len(df)} chart data points for {timeframe}")
        return jsonify(df.to_dict('records'))
        
    except Exception as e:
        print(f"Error in chart data endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "template_folder": app.template_folder,
        "template_exists": os.path.exists(os.path.join(app.template_folder, 'index.html')) if app.template_folder else False
    })

if __name__ == '__main__':
    # Create directories if not exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("🚀 XAUUSD AI Analysis System")
    print("=" * 60)
    print(f"Current directory: {os.getcwd()}")
    print(f"Template folder: {template_dir}")
    print(f"Template exists: {os.path.exists(template_dir)}")
    print("=" * 60)
    print("📊 Available Endpoints:")
    print("  GET / - Main dashboard")
    print("  GET /api/analysis/<1D|4H|1H> - Technical analysis")
    print("  GET /api/chart/data/<1D|4H|1H> - Chart data")
    print("  GET /health - System health")
    print("=" * 60)
    
    # Check if template exists
    template_path = os.path.join(template_dir, 'index.html')
    if not os.path.exists(template_path):
        print("❌ WARNING: index.html not found in templates folder!")
        print("💡 Please create 'templates/index.html' for the dashboard")
    else:
        print("✅ Template found: templates/index.html")
    
    try:
        app.run(debug=True, port=5000, host='0.0.0.0')
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
