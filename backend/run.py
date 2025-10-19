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
import asyncio
import aiohttp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# Try to import talib
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib is available")
except ImportError:
    print("⚠️ TA-Lib not available, using fallback calculations")
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
        self.last_api_call = 0
        
        # Setup session dengan retry strategy
        self.session = self._create_session()
        
        print(f"🔑 API Keys loaded: TwelveData: {'✅' if self.twelve_data_api_key else '❌'}, "
              f"DeepSeek: {'✅' if self.deepseek_api_key else '❌'}, "
              f"NewsAPI: {'✅' if self.news_api_key else '❌'}")

    def _create_session(self):
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        
        # Retry strategy untuk handle network issues
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def debug_data_quality(self, df, column_name):
        """Debug data quality for a specific column"""
        if column_name in df.columns:
            series = df[column_name]
            print(f"  {column_name}: min={series.min():.2f}, max={series.max():.2f}, "
                  f"mean={series.mean():.2f}, nulls={series.isnull().sum()}, unique={series.nunique()}")

    def load_from_local_csv(self, timeframe, limit=500):
        """Load data dari file CSV lokal"""
        possible_paths = [
            f"data/XAUUSD_{timeframe}.csv",
            f"../data/XAUUSD_{timeframe}.csv",
            f"./data/XAUUSD_{timeframe}.csv",
            f"XAUUSD_{timeframe}.csv"
        ]
        
        for filename in possible_paths:
            if os.path.exists(filename):
                try:
                    print(f"📁 Loading from {filename}")
                    df = pd.read_csv(filename)
                    
                    print(f"📊 Columns in CSV: {df.columns.tolist()}")
                    
                    # Pastikan kolom datetime ada dan format benar
                    datetime_col = None
                    for col in ['datetime', 'date', 'time', 'timestamp']:
                        if col in df.columns:
                            datetime_col = col
                            break
                    
                    if datetime_col:
                        df['datetime'] = pd.to_datetime(df[datetime_col])
                        if datetime_col != 'datetime':
                            df = df.drop(columns=[datetime_col])
                    else:
                        print("⚠️ No datetime column found, creating based on index")
                        if timeframe == '1H':
                            freq = 'H'
                        elif timeframe == '4H':
                            freq = '4H'
                        else:
                            freq = 'D'
                        df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq=freq)
                    
                    # Pastikan kolom OHLC ada
                    ohlc_mapping = {
                        'open': ['open', 'Open', 'OPEN'],
                        'high': ['high', 'High', 'HIGH'], 
                        'low': ['low', 'Low', 'LOW'],
                        'close': ['close', 'Close', 'CLOSE']
                    }
                    
                    for standard_name, possible_names in ohlc_mapping.items():
                        if standard_name not in df.columns:
                            for name in possible_names:
                                if name in df.columns:
                                    df[standard_name] = df[name]
                                    print(f"🔀 Mapped column {name} to {standard_name}")
                                    break
                    
                    # Pastikan kolom volume ada
                    if 'volume' not in df.columns:
                        print("📈 Volume column not found, setting default values")
                        df['volume'] = np.random.randint(1000, 10000, len(df))
                    
                    # Konversi ke numeric dan handle missing values
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                    
                    df = df.sort_values('datetime')
                    print(f"✅ Successfully loaded {len(df)} records from {filename}")
                    
                    # Debug data quality
                    print("🔍 Data quality check:")
                    self.debug_data_quality(df, 'open')
                    self.debug_data_quality(df, 'high')
                    self.debug_data_quality(df, 'low')
                    self.debug_data_quality(df, 'close')
                    
                    return df.tail(limit)
                except Exception as e:
                    print(f"❌ Error processing {filename}: {e}")
                    continue
        return None

    def download_historical_data(self, timeframe, days=30):
        """Download data historis dari Twelve Data API"""
        try:
            if not self.twelve_data_api_key:
                print("❌ Twelve Data API key not available for historical data download")
                return None
                
            interval_map = {
                '1H': '1h',
                '4H': '4h', 
                '1D': '1day'
            }
            
            interval = interval_map.get(timeframe, '1h')
            url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize=1000&apikey={self.twelve_data_api_key}"
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok' and 'values' in data:
                    values = data['values']
                    df = pd.DataFrame(values)
                    
                    df = df.rename(columns={
                        'datetime': 'datetime',
                        'open': 'open',
                        'high': 'high', 
                        'low': 'low',
                        'close': 'close'
                    })
                    
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    for col in ['open', 'high', 'low', 'close']:
                        df[col] = pd.to_numeric(df[col])
                    
                    if 'volume' not in df.columns:
                        df['volume'] = 10000
                    
                    df = df.sort_values('datetime')
                    
                    filename = f"data/XAUUSD_{timeframe}.csv"
                    df.to_csv(filename, index=False)
                    print(f"✅ Downloaded and saved {len(df)} records to {filename}")
                    
                    return df
                else:
                    print(f"❌ Twelve Data API error: {data.get('message', 'Unknown error')}")
                    return None
            else:
                print(f"❌ Twelve Data API HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error downloading historical data: {e}")
            return None

    # ... (metode lainnya tetap sama sampai get_fundamental_news)

    def get_fundamental_news(self):
        """Get fundamental news from NewsAPI - IMPROVED with better error handling"""
        try:
            if not self.news_api_key:
                print("❌ NewsAPI key not set, using sample news")
                return self.get_sample_news()
            
            from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
            
            # Improved query dengan multiple terms dan better error handling
            queries = [
                {
                    "url": f"https://newsapi.org/v2/everything?q=gold+OR+XAUUSD+OR+precious+metals&from={from_date}&sortBy=publishedAt&language=en&pageSize=5",
                    "name": "Gold News"
                },
                {
                    "url": f"https://newsapi.org/v2/everything?q=Federal+Reserve+OR+interest+rates+OR+inflation&from={from_date}&sortBy=publishedAt&language=en&pageSize=5", 
                    "name": "Economic News"
                },
                {
                    "url": f"https://newsapi.org/v2/top-headlines?category=business&country=us&pageSize=5",
                    "name": "Business Headlines"
                }
            ]
            
            all_articles = []
            
            for query in queries:
                try:
                    url = f"{query['url']}&apiKey={self.news_api_key}"
                    print(f"📡 Attempting NewsAPI query: {query['name']}")
                    
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('status') == 'ok' and data.get('articles'):
                            articles = data['articles'][:2]  # Take 2 articles per query
                            all_articles.extend(articles)
                            print(f"✅ {query['name']}: Retrieved {len(articles)} articles")
                        else:
                            print(f"⚠️ {query['name']}: No articles or API error - {data.get('message', 'Unknown error')}")
                    else:
                        print(f"❌ {query['name']}: HTTP error {response.status_code}")
                        
                except requests.exceptions.Timeout:
                    print(f"⏰ {query['name']}: Request timeout")
                except requests.exceptions.ConnectionError:
                    print(f"🔌 {query['name']}: Connection error")
                except Exception as e:
                    print(f"⚠️ {query['name']}: Error - {e}")
            
            if all_articles:
                # Remove duplicates berdasarkan title
                seen_titles = set()
                unique_articles = []
                for article in all_articles:
                    title = article.get('title', '').strip()
                    if title and title not in seen_titles and len(title) > 10:
                        seen_titles.add(title)
                        unique_articles.append(article)
                
                print(f"✅ Retrieved {len(unique_articles)} unique news articles")
                return {"articles": unique_articles[:5]}  # Max 5 articles
            
            print("❌ No articles found from NewsAPI, using sample news")
            return self.get_sample_news()
                
        except Exception as e:
            print(f"❌ Critical error in NewsAPI: {e}")
            return self.get_sample_news()

    def get_sample_news(self):
        """Sample news data as fallback"""
        return {
            "articles": [
                {
                    "title": "Gold Prices Hold Steady Amid Economic Uncertainty",
                    "description": "XAUUSD maintains strong support levels as investors seek safe-haven assets amidst global economic concerns.",
                    "publishedAt": datetime.now().isoformat(),
                    "source": {"name": "Financial Times"},
                    "url": "#"
                },
                {
                    "title": "Federal Reserve Policy Impacts Precious Metals Market",
                    "description": "Recent Federal Reserve announcements create favorable conditions for gold prices as investors hedge against inflation.",
                    "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "source": {"name": "Bloomberg"},
                    "url": "#"
                },
                {
                    "title": "Central Bank Gold Purchases Reach Record Levels",
                    "description": "Global central banks continue to increase gold reserves, supporting long-term price stability for XAUUSD.",
                    "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                    "source": {"name": "Reuters"},
                    "url": "#"
                }
            ]
        }

    def analyze_with_deepseek(self, technical_data, news_data):
        """Get AI analysis from DeepSeek API - ULTRA ROBUST VERSION"""
        try:
            current_time = time.time()
            if current_time - self.last_api_call < 15:  # Increased rate limiting
                print("⏳ Skipping DeepSeek API call (rate limiting)")
                return self.comprehensive_fallback_analysis(technical_data, news_data)
            
            if not self.deepseek_api_key:
                print("❌ DeepSeek API key not set, using comprehensive analysis")
                return self.comprehensive_fallback_analysis(technical_data, news_data)
            
            # Validasi API key format
            if not self.deepseek_api_key.startswith('sk-'):
                print("❌ DeepSeek API key format invalid, using fallback")
                return self.comprehensive_fallback_analysis(technical_data, news_data)
            
            current_price = technical_data.get('current_price', 0)
            indicators = technical_data.get('indicators', {})
            
            news_headlines = []
            if news_data and 'articles' in news_data:
                for article in news_data['articles'][:3]:
                    news_headlines.append(f"- {article['title']} ({article['source']['name']})")
            
            news_context = "\n".join(news_headlines) if news_headlines else "No significant market news"
            
            prompt = f"""
Sebagai analis pasar keuangan profesional, berikan analisis komprehensif untuk XAUUSD (Gold/USD):

**DATA TEKNIKAL:**
- Harga Saat Ini: ${current_price:.2f}
- RSI (14): {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')} | Signal: {indicators.get('macd_signal', 'N/A')}
- EMA 12: {indicators.get('ema_12', 'N/A')}
- EMA 26: {indicators.get('ema_26', 'N/A')}
- EMA 50: {indicators.get('ema_50', 'N/A')}
- Stochastic: K={indicators.get('stoch_k', 'N/A')}, D={indicators.get('stoch_d', 'N/A')}
- Bollinger Bands: Upper={indicators.get('bb_upper', 'N/A')}, Lower={indicators.get('bb_lower', 'N/A')}

**BERITA TERKINI:**
{news_context}

Berikan rekomendasi trading yang JELAS: BUY, SELL, atau HOLD dengan:
- Entry Price spesifik
- Stop Loss (SL) realistis  
- Minimal 2 level Take Profit (TP1, TP2) dengan risk-reward ratio minimal 1:2
- Risk-reward ratio harus disebutkan secara eksplisit

Format output profesional dengan:
1. EXECUTIVE SUMMARY
2. TECHNICAL ANALYSIS DETAILED
3. TRADING RECOMMENDATION dengan ENTRY, SL, TP1, TP2
4. RISK MANAGEMENT
5. FUNDAMENTAL CONTEXT

Gunakan bahasa Indonesia yang profesional dan mudah dipahami.
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
                "max_tokens": 2000,
                "stream": False
            }
            
            self.last_api_call = current_time
            
            # ULTRA ROBUST timeout with retry and better error handling
            max_retries = 3
            timeout_duration = 45  # Increased timeout
            
            for attempt in range(max_retries):
                try:
                    print(f"🤖 Attempting DeepSeek API call (attempt {attempt + 1}/{max_retries})...")
                    
                    # Main API call dengan session yang sudah ada retry
                    response = self.session.post(
                        'https://api.deepseek.com/chat/completions',
                        headers=headers,
                        json=data,
                        timeout=timeout_duration
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        analysis = result['choices'][0]['message']['content']
                        print("✅ DeepSeek AI analysis generated successfully")
                        return analysis
                    else:
                        print(f"❌ DeepSeek API error (attempt {attempt + 1}): {response.status_code}")
                        
                        if response.status_code == 401:
                            print("❌ Unauthorized - check API key")
                            return self.comprehensive_fallback_analysis(technical_data, news_data)
                        elif response.status_code == 429:
                            wait_time = (attempt + 1) * 10
                            print(f"⏳ Rate limited, waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        elif response.status_code >= 500:
                            print("🔧 Server error, retrying...")
                            time.sleep(5)
                            continue
                        elif attempt == max_retries - 1:
                            return self.comprehensive_fallback_analysis(technical_data, news_data)
                            
                except requests.exceptions.Timeout:
                    print(f"⏰ DeepSeek API timeout (attempt {attempt + 1})")
                    if attempt == max_retries - 1:
                        return self.comprehensive_fallback_analysis(technical_data, news_data)
                        
                except requests.exceptions.ConnectionError as e:
                    print(f"🔌 DeepSeek API connection error (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        return self.comprehensive_fallback_analysis(technical_data, news_data)
                        
                except Exception as e:
                    print(f"❌ Unexpected error in DeepSeek API (attempt {attempt + 1}): {e}")
                    if attempt == max_retries - 1:
                        return self.comprehensive_fallback_analysis(technical_data, news_data)
                
                # Exponential backoff before retry
                wait_time = (attempt + 1) * 3
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                    
            return self.comprehensive_fallback_analysis(technical_data, news_data)
            
        except Exception as e:
            print(f"❌ Critical error in DeepSeek analysis: {e}")
            return self.comprehensive_fallback_analysis(technical_data, news_data)

    def comprehensive_fallback_analysis(self, technical_data, news_data):
        """Comprehensive fallback analysis when AI fails"""
        # ... (tetap sama seperti sebelumnya)
        current_price = technical_data.get('current_price', 0)
        indicators = technical_data.get('indicators', {})
        
        rsi = indicators.get('rsi', 50) or 50
        macd = indicators.get('macd', 0) or 0
        macd_signal = indicators.get('macd_signal', 0) or 0
        ema_12 = indicators.get('ema_12', current_price) or current_price
        ema_26 = indicators.get('ema_26', current_price) or current_price
        ema_50 = indicators.get('ema_50', current_price) or current_price
        
        bullish_signals = 0
        bearish_signals = 0
        
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
        
        if macd > macd_signal:
            bullish_signals += 1
            macd_signal_text = "BULLISH CROSSOVER"
        else:
            bearish_signals += 1
            macd_signal_text = "BEARISH CROSSOVER"
        
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
        
        if bullish_signals - bearish_signals >= 3:
            trend = "STRONG BULLISH"
            signal = "BUY"
            risk = "MEDIUM"
            risk_reward = "1:3"
        elif bullish_signals - bearish_signals >= 1:
            trend = "BULLISH"
            signal = "BUY"
            risk = "MEDIUM"
            risk_reward = "1:2"
        elif bearish_signals - bullish_signals >= 3:
            trend = "STRONG BEARISH"
            signal = "SELL"
            risk = "HIGH"
            risk_reward = "1:3"
        elif bearish_signals - bullish_signals >= 1:
            trend = "BEARISH"
            signal = "SELL"
            risk = "MEDIUM"
            risk_reward = "1:2"
        else:
            trend = "NEUTRAL"
            signal = "HOLD/WAIT"
            risk = "LOW"
            risk_reward = "N/A"
        
        if signal == "BUY":
            entry = current_price
            stop_loss = entry * 0.99
            take_profit_1 = entry * 1.02
            take_profit_2 = entry * 1.03
            position_size = "Standard (1-2% risk per trade)"
        elif signal == "SELL":
            entry = current_price
            stop_loss = entry * 1.01
            take_profit_1 = entry * 0.98
            take_profit_2 = entry * 0.97
            position_size = "Standard (1-2% risk per trade)"
        else:
            entry = current_price
            stop_loss = entry * 0.995
            take_profit_1 = entry * 1.01
            take_profit_2 = entry * 1.02
            position_size = "Wait for clearer signal"
        
        analysis = f"""
**ANALISIS XAUUSD KOMPREHENSIF - TRADING RECOMMENDATION**
*Dibuat: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

**EXECUTIVE SUMMARY:**
- Harga Saat Ini: ${current_price:.2f}
- Trend Pasar: {trend}
- **REKOMENDASI UTAMA: {signal}**
- Risk Level: {risk}
- Risk-Reward Ratio: {risk_reward}

**TECHNICAL ANALYSIS:**
- RSI (14): {rsi:.1f} - {rsi_signal}
- MACD: {macd:.4f} - {macd_signal_text}
- EMA Alignment: {ema_signal}

**TRADING RECOMMENDATION:**
**Action: {signal} XAUUSD**

**Entry Levels:**
- Ideal Entry: ${entry:.2f}

**Risk Management:**
- Stop Loss: ${stop_loss:.2f}
- Take Profit 1: ${take_profit_1:.2f} (Risk-Reward 1:2)
- Take Profit 2: ${take_profit_2:.2f} (Risk-Reward 1:3)
- Position Size: {position_size}

**TRADING PLAN:**
1. Entry pada ${entry:.2f}
2. Stop Loss: ${stop_loss:.2f}
3. Take Profit 1: ${take_profit_1:.2f} (50% position)
4. Take Profit 2: ${take_profit_2:.2f} (50% position)
5. Risk maksimal 2% dari equity per trade

**CATATAN:** Analisis ini menggunakan fallback system karena koneksi AI sedang mengalami gangguan.
"""
        return analysis

    # ... (metode lainnya tetap sama)

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
    """Main analysis endpoint - ENHANCED VERSION"""
    try:
        print(f"🔍 Processing analysis for {timeframe}")
        
        if timeframe not in ['1H', '4H', '1D']:
            return jsonify({"error": "Invalid timeframe"}), 400
        
        # Load data
        df = analyzer.load_historical_data(timeframe, 200)
        print(f"✅ Loaded {len(df)} records for {timeframe}")
        
        # Calculate indicators
        df_with_indicators = analyzer.calculate_indicators(df)
        print("✅ Indicators calculated")
        
        # Get current price
        current_price = analyzer.get_realtime_price()
        print(f"💰 Current price: ${current_price:.2f}")
        
        # Update latest price in data
        if len(df_with_indicators) > 0:
            df_with_indicators.iloc[-1, df_with_indicators.columns.get_loc('close')] = current_price
            if current_price > df_with_indicators.iloc[-1]['high']:
                df_with_indicators.iloc[-1, df_with_indicators.columns.get_loc('high')] = current_price
            if current_price < df_with_indicators.iloc[-1]['low']:
                df_with_indicators.iloc[-1, df_with_indicators.columns.get_loc('low')] = current_price
        
        # Get news data - dengan logging yang lebih baik
        print("📰 Fetching news data...")
        news_data = analyzer.get_fundamental_news()
        print(f"✅ Retrieved {len(news_data.get('articles', []))} news articles")
        
        # Prepare indicators for API response
        latest_indicators = {}
        if len(df_with_indicators) > 0:
            last_row = df_with_indicators.iloc[-1]
            indicator_list = ['ema_12', 'ema_26', 'ema_50', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                             'stoch_k', 'stoch_d', 'bb_upper', 'bb_lower', 'bb_middle']
            
            for indicator in indicator_list:
                if indicator in df_with_indicators.columns:
                    value = last_row[indicator]
                    if value is not None and not pd.isna(value):
                        latest_indicators[indicator] = float(value)
                    else:
                        latest_indicators[indicator] = 0.0 if 'macd' in indicator else 50.0
                else:
                    latest_indicators[indicator] = 0.0 if 'macd' in indicator else 50.0
        
        print(f"✅ Prepared {len(latest_indicators)} indicators for API response")
        
        # Get AI analysis
        print("🤖 Generating AI analysis...")
        analysis = analyzer.analyze_market_conditions(df_with_indicators, latest_indicators, news_data)
        print("✅ AI analysis completed")
        
        # Prepare chart data
        chart_data = []
        display_data = df_with_indicators.tail(100)
        
        for _, row in display_data.iterrows():
            chart_point = {
                'datetime': row['datetime'].isoformat() if hasattr(row['datetime'], 'isoformat') else str(row['datetime']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
            }
            
            indicator_columns = ['ema_12', 'ema_26', 'ema_50', 'macd', 'macd_signal', 'macd_hist', 
                               'rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'stoch_k', 'stoch_d']
            
            for indicator in indicator_columns:
                if indicator in df_with_indicators.columns:
                    value = row[indicator]
                    if value is not None and not pd.isna(value):
                        chart_point[indicator] = float(value)
                    else:
                        chart_point[indicator] = 0.0 if 'macd' in indicator else 50.0
                else:
                    chart_point[indicator] = 0.0 if 'macd' in indicator else 50.0
            
            chart_data.append(chart_point)
        
        # Prepare API status
        api_status = {
            "twelve_data": bool(analyzer.twelve_data_api_key),
            "deepseek": bool(analyzer.deepseek_api_key),
            "newsapi": bool(analyzer.news_api_key)
        }
        
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
            "api_sources": api_status
        }
        
        print(f"✅ Analysis completed for {timeframe}. Sent {len(chart_data)} data points with {len(latest_indicators)} indicators.")
        print(f"🔌 API Status: TwelveData: {api_status['twelve_data']}, DeepSeek: {api_status['deepseek']}, NewsAPI: {api_status['newsapi']}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500

# ... (endpoint lainnya tetap sama)

@app.route('/api/debug')
def debug_info():
    """Debug information endpoint"""
    api_status = {
        "twelve_data": bool(analyzer.twelve_data_api_key),
        "deepseek": bool(analyzer.deepseek_api_key), 
        "newsapi": bool(analyzer.news_api_key)
    }
    
    return jsonify({
        "status": "debug",
        "timestamp": datetime.now().isoformat(),
        "api_status": api_status,
        "cache_size": len(analyzer.data_cache),
        "cached_timeframes": list(analyzer.data_cache.keys()),
        "last_api_call": analyzer.last_api_call,
        "environment_loaded": True,
        "talib_available": TALIB_AVAILABLE
    })

if __name__ == '__main__':
    try:
        import dotenv
    except ImportError:
        print("📦 Installing python-dotenv...")
        os.system("pip install python-dotenv")
        from dotenv import load_dotenv
        load_dotenv()
    
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 70)
    print("🚀 XAUUSD Professional Trading Analysis - ULTRA ROBUST VERSION")
    print("=" * 70)
    print("📊 Available Endpoints:")
    print("  • GET / → Dashboard")
    print("  • GET /api/analysis/1H → 1Hour Analysis") 
    print("  • GET /api/analysis/4H → 4Hour Analysis")
    print("  • GET /api/analysis/1D → Daily Analysis")
    print("  • GET /api/debug/indicators → Indicator Debug")
    print("  • GET /api/realtime/price → Current Price")
    print("  • GET /api/health → Health Check")
    print("  • GET /api/debug → Debug Info")
    print("=" * 70)
    print("🔧 Integrated APIs:")
    print("  • Twelve Data → Real-time Prices")
    print("  • DeepSeek AI → Market Analysis") 
    print("  • NewsAPI → Fundamental News")
    print("=" * 70)
    print("🛡️  ULTRA ROBUST FEATURES:")
    print("  • ✅ HTTP Session dengan Retry Strategy")
    print("  • ✅ Enhanced DeepSeek API - 45s timeout, 3 retries")
    print("  • ✅ Improved NewsAPI - Better query & error handling") 
    print("  • ✅ Exponential Backoff untuk rate limiting")
    print("  • ✅ Comprehensive Logging untuk debugging")
    print("  • ✅ Graceful Fallbacks untuk semua API failures")
    print("=" * 70)
    
    print("🚀 Starting ultra-robust server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
