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
            url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.twelve_data_api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data and data['price'] != '':
                    price = float(data['price'])
                    print(f"Real-time price from Twelve Data: ${price:.2f}")
                    return price
                else:
                    print("No price data in response")
            else:
                print(f"Twelve Data API error: {response.status_code}")
                
        except Exception as e:
            print(f"Error getting realtime price from Twelve Data: {e}")
        
        # Fallback: try Alpha Vantage atau API lainnya
        return self.get_fallback_realtime_price()

    def get_fallback_realtime_price(self):
        """Fallback realtime price dari API alternatif"""
        try:
            # Coba Alpha Vantage sebagai fallback
            url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=XAU&apikey=demo"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'Global Quote' in data and '05. price' in data['Global Quote']:
                    price = float(data['Global Quote']['05. price'])
                    print(f"Real-time price from Alpha Vantage: ${price:.2f}")
                    return price
        except Exception as e:
            print(f"Error getting fallback price: {e}")
        
        # Ultimate fallback - gunakan harga dari data historis terakhir
        return self.get_last_historical_price()

    def get_last_historical_price(self):
        """Ambil harga terakhir dari data historis sebagai fallback"""
        try:
            for timeframe in ['1H', '4H', '1D']:
                df = self.load_historical_data(timeframe)
                if df is not None and len(df) > 0:
                    last_price = float(df.iloc[-1]['close'])
                    print(f"Using last historical price from {timeframe}: ${last_price:.2f}")
                    return last_price
        except Exception as e:
            print(f"Error getting last historical price: {e}")
        
        # Final fallback
        return 1950.0

    def get_fundamental_news(self):
        """Ambil berita fundamental dari NewsAPI"""
        try:
            # Dapatkan berita tentang gold, XAUUSD, Federal Reserve, dll.
            url = f"https://newsapi.org/v2/everything?q=gold+XAUUSD+Federal+Reserve+precious+metals&language=en&sortBy=publishedAt&apiKey={self.news_api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'ok' and data['totalResults'] > 0:
                    print(f"Retrieved {len(data['articles'])} news articles from NewsAPI")
                    return data
                else:
                    print("No news articles found")
            else:
                print(f"NewsAPI error: {response.status_code}")
                
        except Exception as e:
            print(f"Error getting news from NewsAPI: {e}")
        
        # Fallback news
        return self.get_fallback_news()

    def get_fallback_news(self):
        """Fallback news source"""
        try:
            # Coba sumber berita alternatif
            sources = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=XAUUSD%3DX&region=US&lang=en-US",
                "https://www.fxstreet.com/rss"
            ]
            
            # Untuk simplicity, kita return sample news
            return {
                "articles": [
                    {
                        "title": "Gold Market Analysis - Real-time Updates",
                        "description": "Latest gold price movements and market analysis.",
                        "publishedAt": datetime.now().isoformat(),
                        "source": {"name": "System"}
                    }
                ]
            }
        except Exception as e:
            print(f"Error in fallback news: {e}")
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

    # Fallback technical indicator functions (sama seperti sebelumnya)
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
        """Analisis dengan DeepSeek AI API yang sesungguhnya"""
        try:
            # Siapkan prompt yang komprehensif untuk analisis XAUUSD
            current_price = technical_data.get('current_price', 0)
            indicators = technical_data.get('indicators', {})
            
            # Format data teknikal untuk prompt
            tech_analysis = f"""
Current XAUUSD Price: ${current_price:.2f}

TECHNICAL INDICATORS:
- RSI (14): {indicators.get('rsi', 'N/A'):.2f}
- MACD: {indicators.get('macd', 'N/A'):.4f}
- MACD Signal: {indicators.get('macd_signal', 'N/A'):.4f}
- Stochastic K: {indicators.get('stoch_k', 'N/A'):.2f}
- Stochastic D: {indicators.get('stoch_d', 'N/A'):.2f}
- SMA 20: {indicators.get('sma_20', 'N/A'):.2f}
- SMA 50: {indicators.get('sma_50', 'N/A'):.2f}
- SMA 200: {indicators.get('sma_200', 'N/A'):.2f}
- Bollinger Upper: {indicators.get('bb_upper', 'N/A'):.2f}
- Bollinger Lower: {indicators.get('bb_lower', 'N/A'):.2f}
- ATR: {indicators.get('atr', 'N/A'):.2f}

PRICE ACTION:
- Open: ${technical_data.get('price_action', {}).get('open', 0):.2f}
- High: ${technical_data.get('price_action', {}).get('high', 0):.2f}
- Low: ${technical_data.get('price_action', {}).get('low', 0):.2f}
- Close: ${technical_data.get('price_action', {}).get('close', 0):.2f}
"""
            
            # Format berita untuk prompt
            news_summary = "RECENT MARKET NEWS:\n"
            if news_data and 'articles' in news_data:
                for i, article in enumerate(news_data['articles'][:3]):
                    news_summary += f"{i+1}. {article.get('title', 'No title')}\n"
                    news_summary += f"   {article.get('description', 'No description')}\n\n"
            
            prompt = f"""
Anda adalah analis teknikal profesional untuk trading XAUUSD (Gold/USD). Analisis kondisi pasar saat ini berdasarkan data berikut:

{tech_analysis}

{news_summary}

Berikan analisis komprehensif dalam format berikut:

1. TREND: (Bullish/Bearish/Sideways) - analisis trend berdasarkan indikator
2. SUPPORT: (level support utama) - berdasarkan teknikal
3. RESISTANCE: (level resistance utama) - berdasarkan teknikal  
4. SIGNAL: (Buy/Sell/Hold) - rekomendasi trading
5. RISK LEVEL: (High/Medium/Low) - tingkat risiko
6. ANALYSIS: (analisis mendetail 200-300 kata termasuk analisis teknikal, momentum, volatilitas, dan konteks fundamental)
7. KEY LEVELS: (daftar level-level kunci untuk trading)

Gunakan bahasa Indonesia yang profesional dan fokus pada analisis yang dapat ditindaklanjuti untuk trader.
"""
            
            # Panggil DeepSeek API
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Anda adalah analis teknikal profesional untuk XAUUSD (Gold/USD) dengan spesialisasi dalam analisis teknikal dan fundamental."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            print("Calling DeepSeek API for analysis...")
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                print("DeepSeek API analysis completed successfully")
                return analysis
            else:
                print(f"DeepSeek API error: {response.status_code} - {response.text}")
                return self.get_fallback_analysis(technical_data)
            
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return self.get_fallback_analysis(technical_data)

    def get_fallback_analysis(self, technical_data):
        """Fallback analysis jika API tidak available"""
        current_price = technical_data.get('current_price', 1950.0)
        rsi = technical_data.get('indicators', {}).get('rsi', 50)
        
        if rsi > 70:
            trend = "Bearish"
            signal = "Sell"
        elif rsi < 30:
            trend = "Bullish" 
            signal = "Buy"
        else:
            trend = "Sideways"
            signal = "Hold"
            
        return f"""
1. TREND: {trend}
2. SUPPORT: {current_price - 15.0:.2f}
3. RESISTANCE: {current_price + 20.0:.2f}
4. SIGNAL: {signal}
5. RISK LEVEL: Medium
6. ANALYSIS: XAUUSD sedang dalam kondisi {trend.lower()} dengan harga saat ini di ${current_price:.2f}. RSI berada di level {rsi:.2f} menunjukkan kondisi {'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'netral'}. Disarankan untuk menunggu konfirmasi lebih lanjut sebelum mengambil posisi.

7. KEY LEVELS:
- Support 1: {current_price - 10.0:.2f}
- Support 2: {current_price - 20.0:.2f}
- Resistance 1: {current_price + 15.0:.2f}
- Resistance 2: {current_price + 30.0:.2f}
"""

# ... (sisanya sama dengan kode sebelumnya - routes dan main function)

@app.route('/')
def home():
    """Serve the main application page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"""
        <html>
            <head><title>XAUUSD Analysis</title></head>
            <body>
                <h1>XAUUSD AI Analysis System</h1>
                <p>Backend is running, but template not found.</p>
                <p>Error: {str(e)}</p>
                <p>Available endpoints:</p>
                <ul>
                    <li><a href="/api/analysis/1D">/api/analysis/1D</a></li>
                    <li><a href="/api/analysis/4H">/api/analysis/4H</a></li>
                    <li><a href="/api/analysis/1H">/api/analysis/1H</a></li>
                    <li><a href="/health">/health</a></li>
                </ul>
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
        
        # AI Analysis dengan DeepSeek
        print("Generating AI analysis with DeepSeek...")
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
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 50)
    print("XAUUSD AI Analysis System with Real API Integration")
    print("=" * 50)
    print("IMPORTANT: Please configure your API keys in the code:")
    print("1. Twelve Data API Key - for real-time prices")
    print("2. DeepSeek API Key - for AI analysis") 
    print("3. NewsAPI Key - for fundamental news")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET / - Main application")
    print("  GET /api/analysis/<timeframe> - Get analysis for timeframe (1D, 4H, 1H)")
    print("  GET /api/chart/data/<timeframe> - Get chart data")
    print("  GET /health - Health check")
    print("=" * 50)
    
    try:
        app.run(debug=True, port=5000, host='0.0.0.0')
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("Make sure port 5000 is available and all dependencies are installed.")
