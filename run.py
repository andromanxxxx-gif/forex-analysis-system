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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import talib
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib is available")
except ImportError:
    print("TA-Lib not available, using fallback calculations")
    TALIB_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Setup template folder
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.template_folder = template_dir

class XAUUSDAnalyzer:
    def __init__(self):
        self.data_cache = {}
        self.twelve_data_api_key = os.getenv('TWELVE_DATA_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.last_api_call = 0  # Rate limiting
        
        print(f"API Keys loaded: TwelveData: {'Yes' if self.twelve_data_api_key else 'No'}, "
              f"DeepSeek: {'Yes' if self.deepseek_api_key else 'No'}, "
              f"NewsAPI: {'Yes' if self.news_api_key else 'No'}")
        
    def load_historical_data(self, timeframe, limit=500):
        """Load data historis dari CSV file yang sudah ada"""
        try:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if os.path.exists(filename):
                print(f"Loading historical data from {filename}")
                df = pd.read_csv(filename)
                
                # Pastikan kolom datetime ada dan format benar
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                elif 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'])
                    df = df.rename(columns={'date': 'datetime'})
                elif 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])
                    df = df.rename(columns={'time': 'datetime'})
                else:
                    # Jika tidak ada kolom datetime, buat berdasarkan index
                    df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')
                
                # Pastikan kolom OHLC ada
                required_cols = ['open', 'high', 'low', 'close']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"Warning: Column {col} not found in CSV")
                        # Buat data dummy jika kolom tidak ada
                        df[col] = np.random.uniform(1800, 2000, len(df))
                
                # Konversi ke numeric
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                if 'volume' not in df.columns:
                    df['volume'] = np.random.randint(1000, 10000, len(df))
                
                df = df.sort_values('datetime')
                print(f"Successfully loaded {len(df)} records from {filename}")
                return df.tail(limit)
            else:
                print(f"File {filename} not found, using generated data")
                return self.generate_sample_data(timeframe, limit)
                
        except Exception as e:
            print(f"Error loading historical data: {e}")
            traceback.print_exc()
            return self.generate_sample_data(timeframe, limit)

    def generate_sample_data(self, timeframe, limit=500):
        """Generate sample data tanpa save ke file"""
        print(f"Generating sample data for {timeframe} (in memory only)")
        
        periods = limit
        base_price = 1968.0
        
        # Create dates
        if timeframe == '1H':
            freq = 'H'
        elif timeframe == '4H':
            freq = '4H'
        else:  # 1D
            freq = 'D'
            
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        
        # Generate realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0, 0.005, periods)
        prices = base_price * (1 + returns).cumprod()
        
        # Create OHLC data
        data = []
        for i in range(periods):
            open_price = prices[i] * np.random.uniform(0.998, 1.002)
            close_price = prices[i] * np.random.uniform(0.998, 1.002)
            high_price = max(open_price, close_price) * np.random.uniform(1.001, 1.008)
            low_price = min(open_price, close_price) * np.random.uniform(0.992, 0.999)
            
            data.append({
                'datetime': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(5000, 50000)
            })
        
        df = pd.DataFrame(data)
        
        # Simpan di cache memory
        self.data_cache[timeframe] = df
        print(f"Generated {len(df)} records for {timeframe} (cached in memory)")
        return df

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            if len(df) < 20:
                print("Not enough data for indicators")
                return df
                
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            # Calculate EMAs
            df['ema_12'] = self.ema(close, 12)
            df['ema_26'] = self.ema(close, 26)
            df['ema_50'] = self.ema(close, 50)
            
            # Calculate MACD
            macd, signal, hist = self.macd(close)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Calculate RSI
            df['rsi'] = self.rsi(close, 14)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.bollinger_bands(close)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Calculate Stochastic
            stoch_k, stoch_d = self.stochastic(high, low, close)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            print("All technical indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            traceback.print_exc()
            return df

    def ema(self, data, period):
        """Exponential Moving Average"""
        series = pd.Series(data)
        return series.ewm(span=period, adjust=False).mean()

    def macd(self, data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = pd.Series(data).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(data).ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def rsi(self, data, period=14):
        """RSI Indicator"""
        series = pd.Series(data)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def bollinger_bands(self, data, period=20, std_dev=2):
        """Bollinger Bands"""
        series = pd.Series(data)
        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def stochastic(self, high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        lowest_low = low_series.rolling(window=k_period, min_periods=1).min()
        highest_high = high_series.rolling(window=k_period, min_periods=1).max()
        
        stoch_k = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
        
        return stoch_k, stoch_d

    def get_realtime_price_twelvedata(self):
        """Get real-time gold price from Twelve Data API"""
        try:
            if not self.twelve_data_api_key:
                print("Twelve Data API key not set")
                return self.get_simulated_price()
            
            url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.twelve_data_api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data and data['price'] != '':
                    price = float(data['price'])
                    print(f"Real-time XAUUSD price from Twelve Data: ${price:.2f}")
                    return price
                else:
                    print(f"Twelve Data API error: {data.get('message', 'No price data')}")
                    return self.get_simulated_price()
            else:
                print(f"Twelve Data API HTTP error: {response.status_code}")
                return self.get_simulated_price()
                
        except Exception as e:
            print(f"Error getting price from Twelve Data: {e}")
            return self.get_simulated_price()

    def get_simulated_price(self):
        """Fallback simulated price"""
        base_price = 1968.0
        movement = np.random.normal(0, 1.5)
        price = base_price + movement
        print(f"Simulated XAUUSD price: ${price:.2f}")
        return round(price, 2)

    def get_realtime_price(self):
        """Main function to get real-time price"""
        return self.get_realtime_price_twelvedata()

    def get_fundamental_news(self):
        """Get fundamental news from NewsAPI"""
        try:
            if not self.news_api_key:
                print("NewsAPI key not set, using sample news")
                return self.get_sample_news()
            
            # Get news from last 7 days about gold and economy
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            url = f"https://newsapi.org/v2/everything?q=gold+XAUUSD+Federal+Reserve+inflation&from={from_date}&sortBy=publishedAt&language=en&apiKey={self.news_api_key}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'ok' and data['totalResults'] > 0:
                    articles = data['articles'][:3]  # Get top 3 articles
                    print(f"Retrieved {len(articles)} news articles from NewsAPI")
                    return {"articles": articles}
                else:
                    print("No articles found from NewsAPI")
                    return self.get_sample_news()
            else:
                print(f"NewsAPI HTTP error: {response.status_code}")
                return self.get_sample_news()
                
        except Exception as e:
            print(f"Error getting news from NewsAPI: {e}")
            return self.get_sample_news()

    def get_sample_news(self):
        """Sample news data as fallback"""
        return {
            "articles": [
                {
                    "title": "Gold Prices Hold Steady Amid Economic Uncertainty",
                    "description": "XAUUSD maintains strong support levels as investors seek safe-haven assets.",
                    "publishedAt": datetime.now().isoformat(),
                    "source": {"name": "Financial Times"},
                    "url": "#"
                },
                {
                    "title": "Federal Reserve Policy Impacts Precious Metals",
                    "description": "Recent Fed announcements create favorable conditions for gold prices.",
                    "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "source": {"name": "Bloomberg"},
                    "url": "#"
                }
            ]
        }

    def analyze_with_deepseek(self, technical_data, news_data):
        """Get AI analysis from DeepSeek API with improved error handling"""
        try:
            # Rate limiting - minimum 10 seconds between API calls
            current_time = time.time()
            if current_time - self.last_api_call < 10:
                print("Skipping DeepSeek API call (rate limiting)")
                return self.comprehensive_fallback_analysis(technical_data, news_data)
            
            if not self.deepseek_api_key:
                print("DeepSeek API key not set, using comprehensive analysis")
                return self.comprehensive_fallback_analysis(technical_data, news_data)
            
            # Prepare context for AI
            current_price = technical_data.get('current_price', 0)
            indicators = technical_data.get('indicators', {})
            
            # Extract news headlines
            news_headlines = []
            if news_data and 'articles' in news_data:
                for article in news_data['articles'][:3]:  # Top 3 articles
                    news_headlines.append(f"- {article['title']} ({article['source']['name']})")
            
            news_context = "\n".join(news_headlines) if news_headlines else "No significant news"
            
            prompt = f"""
Sebagai analis pasar keuangan profesional, berikan analisis komprehensif untuk XAUUSD (Gold/USD) berdasarkan data berikut:

DATA TEKNIKAL:
- Harga Saat Ini: ${current_price:.2f}
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- EMA 12: {indicators.get('ema_12', 'N/A')}
- EMA 26: {indicators.get('ema_26', 'N/A')}
- EMA 50: {indicators.get('ema_50', 'N/A')}
- Stochastic K: {indicators.get('stoch_k', 'N/A')}
- Stochastic D: {indicators.get('stoch_d', 'N/A')}
- Bollinger Band Upper: {indicators.get('bb_upper', 'N/A')}
- Bollinger Band Lower: {indicators.get('bb_lower', 'N/A')}

BERITA TERKINI:
{news_context}

Tolong berikan analisis yang mencakup:
1. Kondisi trend saat ini (bullish/bearish/neutral)
2. Sinyal trading dengan level entry, stop loss, dan take profit
3. Analisis momentum dan kekuatan trend
4. Faktor fundamental yang perlu diperhatikan
5. Rekomendasi risk management

Format respons dalam bahasa Indonesia yang profesional.
"""
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.deepseek_api_key}'
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            self.last_api_call = current_time
            response = requests.post(
                'https://api.deepseek.com/chat/completions',
                headers=headers,
                json=data,
                timeout=45  # Increased timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                print("DeepSeek AI analysis generated successfully")
                return analysis
            else:
                print(f"DeepSeek API error: {response.status_code} - {response.text}")
                return self.comprehensive_fallback_analysis(technical_data, news_data)
                
        except requests.exceptions.Timeout:
            print("DeepSeek API timeout, using fallback analysis")
            return self.comprehensive_fallback_analysis(technical_data, news_data)
        except Exception as e:
            print(f"Error getting DeepSeek analysis: {e}")
            return self.comprehensive_fallback_analysis(technical_data, news_data)

    def comprehensive_fallback_analysis(self, technical_data, news_data):
        """Comprehensive fallback analysis when AI fails"""
        current_price = technical_data.get('current_price', 0)
        indicators = technical_data.get('indicators', {})
        
        # Extract values with defaults
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        ema_12 = indicators.get('ema_12', current_price)
        ema_26 = indicators.get('ema_26', current_price)
        ema_50 = indicators.get('ema_50', current_price)
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        bb_upper = indicators.get('bb_upper', current_price * 1.02)
        bb_lower = indicators.get('bb_lower', current_price * 0.98)
        
        # Calculate signals
        bullish_signals = 0
        bearish_signals = 0
        
        # RSI Analysis
        if rsi < 30:
            bullish_signals += 2
            rsi_signal = "OVERSOLD - STRONG BUY"
        elif rsi < 40:
            bullish_signals += 1
            rsi_signal = "NEARLY OVERSOLD - BUY"
        elif rsi > 70:
            bearish_signals += 2
            rsi_signal = "OVERBOUGHT - STRONG SELL"
        elif rsi > 60:
            bearish_signals += 1
            rsi_signal = "NEARLY OVERBOUGHT - SELL"
        else:
            rsi_signal = "NEUTRAL"
        
        # MACD Analysis
        if macd > macd_signal:
            bullish_signals += 1
            macd_signal_text = "BULLISH CROSSOVER"
        else:
            bearish_signals += 1
            macd_signal_text = "BEARISH CROSSOVER"
        
        # EMA Analysis
        if current_price > ema_12 > ema_26 > ema_50:
            bullish_signals += 2
            ema_signal = "STRONG BULLISH TREND"
        elif current_price < ema_12 < ema_26 < ema_50:
            bearish_signals += 2
            ema_signal = "STRONG BEARISH TREND"
        elif current_price > ema_12 and ema_12 > ema_26:
            bullish_signals += 1
            ema_signal = "BULLISH TREND"
        elif current_price < ema_12 and ema_12 < ema_26:
            bearish_signals += 1
            ema_signal = "BEARISH TREND"
        else:
            ema_signal = "MIXED TREND"
        
        # Stochastic Analysis
        if stoch_k < 20 and stoch_d < 20:
            bullish_signals += 1
            stoch_signal = "OVERSOLD - BUY"
        elif stoch_k > 80 and stoch_d > 80:
            bearish_signals += 1
            stoch_signal = "OVERBOUGHT - SELL"
        else:
            stoch_signal = "NEUTRAL"
        
        # Bollinger Bands Analysis
        if current_price < bb_lower:
            bullish_signals += 1
            bb_signal = "PRICE BELOW LOWER BAND - POTENTIAL BUY"
        elif current_price > bb_upper:
            bearish_signals += 1
            bb_signal = "PRICE ABOVE UPPER BAND - POTENTIAL SELL"
        else:
            bb_signal = "PRICE WITHIN BANDS - NEUTRAL"
        
        # Determine overall trend and signal
        if bullish_signals - bearish_signals >= 3:
            trend = "STRONG BULLISH"
            signal = "STRONG BUY"
            risk = "LOW"
        elif bullish_signals - bearish_signals >= 1:
            trend = "BULLISH"
            signal = "BUY"
            risk = "MEDIUM"
        elif bearish_signals - bullish_signals >= 3:
            trend = "STRONG BEARISH"
            signal = "STRONG SELL"
            risk = "HIGH"
        elif bearish_signals - bullish_signals >= 1:
            trend = "BEARISH"
            signal = "SELL"
            risk = "MEDIUM"
        else:
            trend = "NEUTRAL"
            signal = "HOLD"
            risk = "LOW"
        
        # Calculate trading levels
        if signal in ["STRONG BUY", "BUY"]:
            stop_loss = current_price * 0.99  # 1% below current price
            take_profit_1 = current_price * 1.01  # 1% above
            take_profit_2 = current_price * 1.02  # 2% above
        elif signal in ["STRONG SELL", "SELL"]:
            stop_loss = current_price * 1.01  # 1% above current price
            take_profit_1 = current_price * 0.99  # 1% below
            take_profit_2 = current_price * 0.98  # 2% below
        else:
            stop_loss = current_price * 0.995
            take_profit_1 = current_price * 1.005
            take_profit_2 = current_price * 1.01
        
        analysis = f"""
** ANALISIS XAUUSD KOMPREHENSIF **
*Dibuat: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

** RINGKASAN EKSEKUTIF **
- Harga Saat Ini: ${current_price:.2f}
- Trend Pasar: {trend}
- Sinyal Trading: {signal}
- Level Risiko: {risk}

** ANALISIS TEKNIKAL **

** Analisis Trend: **
- Alignment EMA: {ema_signal}
- Harga vs EMA 12: {'Diatas' if current_price > ema_12 else 'Dibawah'} (${ema_12:.2f})
- Harga vs EMA 26: {'Diatas' if current_price > ema_26 else 'Dibawah'} (${ema_26:.2f})

** Indikator Momentum: **
- RSI (14): {rsi:.1f} - {rsi_signal}
- MACD: {macd:.4f} - {macd_signal_text}
- Stochastic: K={stoch_k:.1f}, D={stoch_d:.1f} - {stoch_signal}
- Bollinger Bands: {bb_signal}

** Kekuatan Sinyal: **
- Sinyal Bullish: {bullish_signals}
- Sinyal Bearish: {bearish_signals}

** REKOMENDASI TRADING **

** Strategi Utama: **
{signal} XAUUSD dengan pendekatan risiko {risk.lower()}.

** Level Kunci: **
- Support Kuat: ${bb_lower:.2f}
- Resistance Kuat: ${bb_upper:.2f}

** Manajemen Risiko: **
- Stop Loss: ${stop_loss:.2f}
- Take Profit 1: ${take_profit_1:.2f}
- Take Profit 2: ${take_profit_2:.2f}

** CATATAN: **
Analisis ini menggunakan fallback system. Pastikan koneksi internet stabil untuk analisis AI DeepSeek.
"""
        return analysis

    def analyze_market_conditions(self, df, indicators, news_data):
        """Comprehensive market analysis using AI"""
        try:
            if len(df) == 0:
                return "No data available for analysis"
                
            current_price = df.iloc[-1]['close']
            
            # Prepare technical data for AI analysis
            technical_data = {
                'current_price': current_price,
                'indicators': indicators
            }
            
            # Get AI analysis
            analysis = self.analyze_with_deepseek(technical_data, news_data)
            return analysis
            
        except Exception as e:
            return f"Market analysis completed. Error in processing: {str(e)}"

# Create analyzer instance
analyzer = XAUUSDAnalyzer()

@app.route('/')
def home():
    """Serve main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/api/analysis/<timeframe>')
def get_analysis(timeframe):
    """Main analysis endpoint"""
    try:
        print(f"Processing analysis for {timeframe}")
        
        # Validate timeframe
        if timeframe not in ['1H', '4H', '1D']:
            return jsonify({"error": "Invalid timeframe"}), 400
        
        # Load and prepare data
        df = analyzer.load_historical_data(timeframe, 200)
        df_with_indicators = analyzer.calculate_indicators(df)
        
        # Get current price from Twelve Data
        current_price = analyzer.get_realtime_price()
        
        # Update last price
        if len(df_with_indicators) > 0:
            df_with_indicators.iloc[-1, df_with_indicators.columns.get_loc('close')] = current_price
        
        # Get news from NewsAPI
        news_data = analyzer.get_fundamental_news()
        
        # Prepare indicators for analysis
        latest_indicators = {}
        if len(df_with_indicators) > 0:
            last_row = df_with_indicators.iloc[-1]
            for indicator in ['ema_12', 'ema_26', 'ema_50', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                             'stoch_k', 'stoch_d', 'bb_upper', 'bb_lower']:
                if indicator in df_with_indicators.columns and not pd.isna(last_row[indicator]):
                    latest_indicators[indicator] = float(last_row[indicator])
        
        # Generate comprehensive AI analysis
        analysis = analyzer.analyze_market_conditions(df_with_indicators, latest_indicators, news_data)
        
        # Prepare chart data (limit to 100 points for better performance)
        chart_data = []
        for _, row in df_with_indicators.tail(100).iterrows():
            chart_data.append({
                'datetime': row['datetime'].isoformat() if hasattr(row['datetime'], 'isoformat') else str(row['datetime']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'ema_12': float(row['ema_12']) if 'ema_12' in df_with_indicators.columns and not pd.isna(row['ema_12']) else None,
                'ema_26': float(row['ema_26']) if 'ema_26' in df_with_indicators.columns and not pd.isna(row['ema_26']) else None,
                'ema_50': float(row['ema_50']) if 'ema_50' in df_with_indicators.columns and not pd.isna(row['ema_50']) else None,
                'macd': float(row['macd']) if 'macd' in df_with_indicators.columns and not pd.isna(row['macd']) else None,
                'macd_signal': float(row['macd_signal']) if 'macd_signal' in df_with_indicators.columns and not pd.isna(row['macd_signal']) else None,
                'macd_hist': float(row['macd_hist']) if 'macd_hist' in df_with_indicators.columns and not pd.isna(row['macd_hist']) else None,
                'rsi': float(row['rsi']) if 'rsi' in df_with_indicators.columns and not pd.isna(row['rsi']) else None,
                'bb_upper': float(row['bb_upper']) if 'bb_upper' in df_with_indicators.columns and not pd.isna(row['bb_upper']) else None,
                'bb_lower': float(row['bb_lower']) if 'bb_lower' in df_with_indicators.columns and not pd.isna(row['bb_lower']) else None
            })
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "current_price": current_price,
            "technical_indicators": latest_indicators,
            "ai_analysis": analysis,
            "chart_data": chart_data,
            "data_points": len(chart_data),
            "news": news_data,
            "api_sources": {
                "twelve_data": bool(analyzer.twelve_data_api_key),
                "deepseek": bool(analyzer.deepseek_api_key),
                "newsapi": bool(analyzer.news_api_key)
            }
        }
        
        print(f"Analysis completed for {timeframe}. Sent {len(chart_data)} data points.")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/analyze')
def legacy_analyze():
    """Legacy endpoint for compatibility - redirects to XAUUSD analysis"""
    pair = request.args.get('pair', 'XAUUSD')
    timeframe = request.args.get('timeframe', '4H')
    
    if pair.upper() != 'XAUUSD':
        return jsonify({
            "error": f"Pair {pair} not supported. Only XAUUSD is supported.",
            "supported_pairs": ["XAUUSD"]
        }), 400
    
    # Redirect to the main analysis endpoint
    return get_analysis(timeframe)

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/realtime/price')
def realtime_price():
    """Realtime price endpoint"""
    try:
        price = analyzer.get_realtime_price()
        source = "Twelve Data API" if analyzer.twelve_data_api_key else "Simulated"
        return jsonify({
            "price": price,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug')
def debug():
    """Debug endpoint to check data files and API status"""
    try:
        files = {}
        for timeframe in ['1H', '4H', '1D']:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                files[timeframe] = {
                    "exists": True,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "first_date": df.iloc[0]['datetime'] if 'datetime' in df.columns else "N/A",
                    "last_date": df.iloc[-1]['datetime'] if 'datetime' in df.columns else "N/A"
                }
            else:
                files[timeframe] = {"exists": False}
        
        # Test API connections
        api_status = {
            "twelve_data": bool(analyzer.twelve_data_api_key),
            "deepseek": bool(analyzer.deepseek_api_key),
            "newsapi": bool(analyzer.news_api_key)
        }
        
        return jsonify({
            "status": "debug",
            "data_files": files,
            "api_status": api_status,
            "current_dir": os.getcwd(),
            "data_dir": os.path.join(os.getcwd(), 'data')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Install required packages if not already installed
    try:
        import dotenv
    except ImportError:
        print("Installing python-dotenv...")
        os.system("pip install python-dotenv")
        from dotenv import load_dotenv
        load_dotenv()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("🚀 XAUUSD Professional Trading Analysis")
    print("=" * 60)
    print("📊 Available Endpoints:")
    print("  • GET / → Dashboard")
    print("  • GET /api/analysis/1H → 1Hour Analysis") 
    print("  • GET /api/analysis/4H → 4Hour Analysis")
    print("  • GET /api/analysis/1D → Daily Analysis")
    print("  • GET /api/analyze?pair=XAUUSD&timeframe=4H → Legacy Support")
    print("  • GET /api/realtime/price → Current Price")
    print("  • GET /api/health → Health Check")
    print("  • GET /api/debug → Debug Info")
    print("=" * 60)
    print("🔧 Integrated APIs:")
    print("  • Twelve Data → Real-time Prices")
    print("  • DeepSeek AI → Market Analysis") 
    print("  • NewsAPI → Fundamental News")
    print("=" * 60)
    
    print("Starting server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
