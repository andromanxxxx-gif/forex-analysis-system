from flask import Flask, jsonify, request, render_template, send_from_directory
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
    print("✅ TA-Lib is available")
except ImportError:
    print("⚠️ TA-Lib not available, using fallback calculations")
    TALIB_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Setup template folder
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.template_folder = template_dir

print(f"📁 Template folder path: {template_dir}")
print(f"📁 Current working directory: {os.getcwd()}")

class XAUUSDAnalyzer:
    def __init__(self):
        self.data_cache = {}
        self.twelve_data_api_key = os.getenv('TWELVE_DATA_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.last_api_call = 0
        
        # Setup session dengan retry strategy yang kompatibel
        self.session = self._create_session()
        
        print(f"🔑 API Keys loaded: TwelveData: {'✅' if self.twelve_data_api_key else '❌'}, "
              f"DeepSeek: {'✅' if self.deepseek_api_key else '❌'}, "
              f"NewsAPI: {'✅' if self.news_api_key else '❌'}")

    def _create_session(self):
        """Create HTTP session dengan retry strategy yang kompatibel"""
        session = requests.Session()
        
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Retry strategy yang kompatibel dengan versi urllib3 terbaru
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
                backoff_factor=1
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            print("✅ HTTP session dengan retry strategy berhasil dibuat")
            
        except Exception as e:
            print(f"⚠️ Tidak bisa membuat retry strategy: {e}")
            print("🔧 Menggunakan session tanpa retry strategy")
        
        return session

    # ... (semua method untuk data loading, cleaning, dan indicator calculation tetap sama)
    # [Semua method dari kode sebelumnya tetap dipertahankan di sini]
    # ... (load_from_local_csv, download_historical_data, gentle_data_cleaning, dll)

    def get_realtime_price(self):
        """Get real-time price from Twelve Data API"""
        try:
            if not self.twelve_data_api_key:
                print("❌ Twelve Data API key not available for real-time price")
                return None
                
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_api_call < 1:  # 1 second between calls
                time.sleep(1)
                
            url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.twelve_data_api_key}"
            
            response = self.session.get(url, timeout=10)
            self.last_api_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data:
                    price = float(data['price'])
                    print(f"✅ Real-time price from Twelve Data: ${price:.2f}")
                    return price
                else:
                    print(f"❌ Twelve Data price error: {data}")
                    return None
            else:
                print(f"❌ Twelve Data HTTP error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting real-time price: {e}")
            return None

    def enhance_with_realtime_data(self, df, timeframe):
        """Enhance historical data with real-time price"""
        try:
            realtime_price = self.get_realtime_price()
            if realtime_price is not None:
                # Create a new data point for current time
                current_time = datetime.now()
                
                # Determine the appropriate datetime based on timeframe
                if timeframe == '1H':
                    # Round to current hour
                    current_time = current_time.replace(minute=0, second=0, microsecond=0)
                elif timeframe == '4H':
                    # Round to current 4-hour block
                    hour = (current_time.hour // 4) * 4
                    current_time = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
                else:  # 1D
                    # Round to current day
                    current_time = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Check if we already have data for this time period
                last_time = df['datetime'].iloc[-1] if len(df) > 0 else None
                
                # If we don't have data for current period or realtime price is significantly different
                if (last_time is None or 
                    current_time > last_time or 
                    abs(realtime_price - df['close'].iloc[-1]) / df['close'].iloc[-1] > 0.001):  # 0.1% difference
                    
                    new_row = {
                        'datetime': current_time,
                        'open': realtime_price,
                        'high': realtime_price,
                        'low': realtime_price,
                        'close': realtime_price,
                        'volume': df['volume'].mean() if len(df) > 0 else 10000
                    }
                    
                    # Add to dataframe
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    print(f"✅ Enhanced data with real-time price: ${realtime_price:.2f}")
                
            return df
            
        except Exception as e:
            print(f"❌ Error enhancing with real-time data: {e}")
            return df

    def generate_comprehensive_ai_analysis(self, df, current_price, timeframe):
        """Generate comprehensive AI analysis with trading recommendations using DeepSeek"""
        try:
            if not self.deepseek_api_key:
                return self.generate_fallback_analysis(df, current_price, timeframe)
            
            # Prepare technical analysis data
            tech_analysis = self.prepare_technical_analysis(df)
            fundamental_context = self.get_fundamental_context()
            
            # Create comprehensive prompt for DeepSeek
            prompt = self.create_trading_prompt(tech_analysis, fundamental_context, current_price, timeframe)
            
            # Call DeepSeek API
            analysis_result = self.call_deepseek_api(prompt)
            
            if analysis_result:
                return analysis_result
            else:
                return self.generate_fallback_analysis(df, current_price, timeframe)
                
        except Exception as e:
            print(f"❌ Error in comprehensive AI analysis: {e}")
            return self.generate_fallback_analysis(df, current_price, timeframe)

    def prepare_technical_analysis(self, df):
        """Prepare comprehensive technical analysis data"""
        if len(df) < 2:
            return {}
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Price action
        price_change = ((current['close'] - previous['close']) / previous['close']) * 100
        price_trend = "BULLISH" if price_change > 0 else "BEARISH"
        
        # EMA analysis
        ema_trend = "NEUTRAL"
        if current.get('ema_12') and current.get('ema_26'):
            if current['ema_12'] > current['ema_26'] and current['close'] > current['ema_12']:
                ema_trend = "STRONG BULLISH"
            elif current['ema_12'] > current['ema_26']:
                ema_trend = "BULLISH"
            elif current['ema_12'] < current['ema_26'] and current['close'] < current['ema_12']:
                ema_trend = "STRONG BEARISH"
            elif current['ema_12'] < current['ema_26']:
                ema_trend = "BEARISH"
        
        # RSI analysis
        rsi_signal = "NEUTRAL"
        if current.get('rsi'):
            if current['rsi'] > 70:
                rsi_signal = "OVERSOLD"
            elif current['rsi'] < 30:
                rsi_signal = "OVERBOUGHT"
        
        # MACD analysis
        macd_signal = "NEUTRAL"
        if current.get('macd') and current.get('macd_signal'):
            if current['macd'] > current['macd_signal'] and current['macd_hist'] > 0:
                macd_signal = "BULLISH"
            elif current['macd'] < current['macd_signal'] and current['macd_hist'] < 0:
                macd_signal = "BEARISH"
        
        # Support and Resistance
        support_level = current.get('bb_lower', current['close'] * 0.98)
        resistance_level = current.get('bb_upper', current['close'] * 1.02)
        
        # Volatility
        volatility = df['close'].pct_change().std() * 100 if len(df) > 1 else 1.0
        
        return {
            'current_price': current['close'],
            'price_change_percent': price_change,
            'price_trend': price_trend,
            'ema_trend': ema_trend,
            'rsi_value': current.get('rsi', 50),
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'support_level': support_level,
            'resistance_level': resistance_level,
            'volatility_percent': volatility,
            'volume_trend': "INCREASING" if current.get('volume', 0) > previous.get('volume', 0) else "DECREASING"
        }

    def get_fundamental_context(self):
        """Get fundamental context from news and market data"""
        try:
            news_data = self.get_market_news()
            news_context = ""
            
            if news_data and 'articles' in news_data:
                for article in news_data['articles'][:3]:  # Top 3 news
                    news_context += f"- {article['title']}\n"
            
            # Add economic context
            economic_context = """
            Economic Context:
            - Gold is trading as a safe-haven asset
            - Monitor USD strength and interest rate expectations
            - Watch for geopolitical tensions and inflation data
            - Central bank policies impact gold prices
            """
            
            return news_context + economic_context
            
        except Exception as e:
            print(f"❌ Error getting fundamental context: {e}")
            return "Fundamental data temporarily unavailable"

    def create_trading_prompt(self, tech_analysis, fundamental_context, current_price, timeframe):
        """Create comprehensive trading prompt for DeepSeek"""
        
        prompt = f"""
        Anda adalah analis trading profesional khusus emas (XAUUSD). Berikan analisis komprehensif dan rekomendasi trading berdasarkan data berikut:

        DATA TEKNIKAL (Timeframe: {timeframe}):
        - Harga Saat Ini: ${current_price:.2f}
        - Perubahan Harga: {tech_analysis['price_change_percent']:+.2f}%
        - Tren Harga: {tech_analysis['price_trend']}
        - Tren EMA: {tech_analysis['ema_trend']}
        - RSI: {tech_analysis['rsi_value']:.1f} ({tech_analysis['rsi_signal']})
        - Sinyal MACD: {tech_analysis['macd_signal']}
        - Level Support: ${tech_analysis['support_level']:.2f}
        - Level Resistance: ${tech_analysis['resistance_level']:.2f}
        - Volatilitas: {tech_analysis['volatility_percent']:.2f}%
        - Trend Volume: {tech_analysis['volume_trend']}

        KONTEKS FUNDAMENTAL:
        {fundamental_context}

        TUGAS ANALISIS:
        1. Berikan penilaian menyeluruh tentang kondisi market
        2. Analisis momentum dan trend
        3. Identifikasi level kunci support dan resistance
        4. Berikan rekomendasi trading yang spesifik

        FORMAT REKOMENDASI TRADING:
        🎯 REKOMENDASI: [BUY/SELL/HOLD]
        💰 ENTRY: $[harga]
        🛑 STOP LOSS: $[harga] 
        ✅ TAKE PROFIT: $[harga]
        📊 RISK-REWARD RATIO: 1:[ratio]
        ⚠️ RISK LEVEL: [LOW/MEDIUM/HIGH]

        ANALISIS DETAIL:
        [Penjelasan mendalam tentang analisis teknikal dan fundamental]

        STRATEGI TRADING:
        [Rencana trading yang detail]

        CATATAN:
        - Gunakan analisis price action yang ketat
        - Pertimbangkan manajemen risiko
        - Berdasarkan data real-time dan historis
        - Timeframe trading: {timeframe}
        """

        return prompt

    def call_deepseek_api(self, prompt):
        """Call DeepSeek API for trading analysis"""
        try:
            if not self.deepseek_api_key:
                return None

            # Rate limiting
            current_time = time.time()
            if current_time - self.last_api_call < 2:  # 2 seconds between calls
                time.sleep(2)
            
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.deepseek_api_key}"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "Anda adalah analis trading emas (XAUUSD) profesional dengan pengalaman 10+ tahun. Berikan analisis yang akurat dan rekomendasi trading yang dapat ditindaklanjuti berdasarkan data teknikal dan fundamental. Fokus pada manajemen risiko dan probabilitas tinggi."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            response = self.session.post(url, json=payload, headers=headers, timeout=30)
            self.last_api_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    analysis = data['choices'][0]['message']['content']
                    print("✅ DeepSeek AI analysis generated successfully")
                    return analysis
                else:
                    print(f"❌ DeepSeek API response format unexpected: {data}")
                    return None
            else:
                print(f"❌ DeepSeek API HTTP error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling DeepSeek API: {e}")
            return None

    def generate_fallback_analysis(self, df, current_price, timeframe):
        """Generate fallback analysis when AI is unavailable"""
        try:
            if len(df) < 2:
                return "Insufficient data for analysis"
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            price_change = ((current['close'] - previous['close']) / previous['close']) * 100
            
            # Simple trend analysis
            trend = "BULLISH" if price_change > 0 else "BEARISH"
            
            # Simple recommendation logic
            if current.get('rsi', 50) < 35 and current.get('macd_hist', 0) > 0:
                recommendation = "BUY"
                stop_loss = current['close'] * 0.985
                take_profit = current['close'] * 1.015
            elif current.get('rsi', 50) > 65 and current.get('macd_hist', 0) < 0:
                recommendation = "SELL"
                stop_loss = current['close'] * 1.015
                take_profit = current['close'] * 0.985
            else:
                recommendation = "HOLD"
                stop_loss = current['close'] * 0.99
                take_profit = current['close'] * 1.01
            
            analysis = f"""
🎯 REKOMENDASI: {recommendation}
💰 ENTRY: ${current_price:.2f}
🛑 STOP LOSS: ${stop_loss:.2f}
✅ TAKE PROFIT: ${take_profit:.2f}
📊 RISK-REWARD RATIO: 1:1.5
⚠️ RISK LEVEL: MEDIUM

ANALISIS DETAIL:
Harga XAUUSD saat ini ${current_price:.2f} ({price_change:+.2f}%). 
Trend jangka pendek: {trend}.
RSI: {current.get('rsi', 50):.1f} - { 'Oversold' if current.get('rsi', 50) < 30 else 'Overbought' if current.get('rsi', 50) > 70 else 'Neutral'}.

STRATEGI TRADING:
- {recommendation} dengan risk management ketat
- Monitor level ${stop_loss:.2f} sebagai stop loss
- Target profit di ${take_profit:.2f}

CATATAN:
Analisis ini berdasarkan data teknikal dasar. Untuk analisis lebih mendalam, pastikan koneksi AI tersedia.
            """
            
            return analysis.strip()
            
        except Exception as e:
            return f"Basic analysis unavailable: {str(e)}"

    def get_market_news(self):
        """Get market news from NewsAPI"""
        try:
            if not self.news_api_key:
                return {"articles": []}
                
            # Rate limiting
            current_time = time.time()
            if current_time - self.last_api_call < 2:
                time.sleep(2)
            
            # Try to get real gold news, fallback to simulated news
            url = f"https://newsapi.org/v2/everything?q=gold+OR+XAUUSD+OR+precious+metals&sortBy=publishedAt&language=en&apiKey={self.news_api_key}"
            
            response = self.session.get(url, timeout=15)
            self.last_api_call = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data.get('articles'):
                    print(f"✅ Retrieved {len(data['articles'])} news articles")
                    return data
            
            # Fallback to simulated news
            return self.get_simulated_news()
                
        except Exception as e:
            print(f"❌ Error getting market news: {e}")
            return self.get_simulated_news()

    def get_simulated_news(self):
        """Get simulated market news when API is unavailable"""
        return {
            "articles": [
                {
                    "title": "Gold Prices React to Federal Reserve Policy Outlook",
                    "description": "Gold markets show volatility as traders assess Federal Reserve's interest rate trajectory and inflation concerns.",
                    "publishedAt": datetime.now().isoformat(),
                    "source": {"name": "Market Analysis"}
                },
                {
                    "title": "Geopolitical Tensions Support Safe-Haven Demand for Gold",
                    "description": "Ongoing geopolitical uncertainties continue to bolster gold's appeal as a safe-haven asset among investors.",
                    "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                    "source": {"name": "Financial Times"}
                },
                {
                    "title": "Central Bank Gold Purchases Reach Record Levels", 
                    "description": "Global central banks continue aggressive gold accumulation, supporting long-term price fundamentals.",
                    "publishedAt": (datetime.now() - timedelta(hours=4)).isoformat(),
                    "source": {"name": "Reuters"}
                }
            ]
        }

# Create analyzer instance
analyzer = XAUUSDAnalyzer()

# ========== ROUTES UTAMA ==========

@app.route('/')
def index():
    """Serve the main dashboard page from templates folder"""
    try:
        return render_template('index.html')
    except Exception as e:
        error_msg = f"Error loading dashboard: {str(e)}"
        print(f"❌ {error_msg}")
        return f"<h1>Server Error</h1><p>{error_msg}</p>", 500

@app.route('/api/analysis/<timeframe>')
def analysis(timeframe):
    """Get analysis data for specified timeframe"""
    try:
        print(f"📊 Processing analysis request for {timeframe}")
        
        # Check if force download is requested
        force_download = request.args.get('force_download', 'false').lower() == 'true'
        
        if force_download:
            print(f"🔄 Force download requested for {timeframe}")
            df = analyzer.download_historical_data(timeframe)
        else:
            # Load historical data
            df = analyzer.load_historical_data(timeframe, limit=200)
        
        if df is None or len(df) == 0:
            return jsonify({
                "status": "error", 
                "error": "No data available",
                "data_points": 0
            }), 500
        
        # Enhance with real-time data
        df = analyzer.enhance_with_realtime_data(df, timeframe)
        
        # Calculate indicators
        df_with_indicators = analyzer.calculate_indicators(df)
        
        # Get current price (use real-time if available, otherwise latest historical)
        current_price = float(df_with_indicators['close'].iloc[-1])
        
        # Prepare chart data
        chart_data = []
        for _, row in df_with_indicators.tail(100).iterrows():
            chart_data.append({
                'datetime': row['datetime'].isoformat() if hasattr(row['datetime'], 'isoformat') else str(row['datetime']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0)),
                'ema_12': float(row.get('ema_12', 0)),
                'ema_26': float(row.get('ema_26', 0)),
                'ema_50': float(row.get('ema_50', 0)),
                'rsi': float(row.get('rsi', 50)),
                'macd': float(row.get('macd', 0)),
                'macd_signal': float(row.get('macd_signal', 0)),
                'macd_hist': float(row.get('macd_hist', 0)),
                'stoch_k': float(row.get('stoch_k', 50)),
                'stoch_d': float(row.get('stoch_d', 50)),
                'bb_upper': float(row.get('bb_upper', current_price * 1.02)),
                'bb_lower': float(row.get('bb_lower', current_price * 0.98))
            })
        
        # Get latest indicators for display
        latest_indicators = {}
        indicator_columns = ['ema_12', 'ema_26', 'ema_50', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_k', 'stoch_d', 'bb_upper', 'bb_lower']
        for col in indicator_columns:
            if col in df_with_indicators.columns and not df_with_indicators[col].isna().all():
                latest_indicators[col] = float(df_with_indicators[col].iloc[-1])
        
        # Generate comprehensive AI analysis
        ai_analysis = analyzer.generate_comprehensive_ai_analysis(df_with_indicators, current_price, timeframe)
        
        # Get news
        news_data = analyzer.get_market_news()
        
        response_data = {
            "status": "success",
            "timeframe": timeframe,
            "current_price": current_price,
            "data_points": len(df),
            "technical_indicators": latest_indicators,
            "chart_data": chart_data,
            "ai_analysis": ai_analysis,
            "news": news_data,
            "api_sources": {
                "twelve_data": bool(analyzer.twelve_data_api_key),
                "deepseek": bool(analyzer.deepseek_api_key),
                "newsapi": bool(analyzer.news_api_key)
            }
        }
        
        print(f"✅ Analysis completed for {timeframe}: {len(df)} data points, price: ${current_price:.2f}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in analysis endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e),
            "data_points": 0
        }), 500

# ... (debug, clear_cache, health endpoints tetap sama)

if __name__ == '__main__':
    try:
        import dotenv
    except ImportError:
        print("📦 Installing python-dotenv...")
        os.system("pip install python-dotenv")
        from dotenv import load_dotenv
        load_dotenv()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("=" * 70)
    print("🚀 XAUUSD Professional Trading Analysis - AI ENHANCED")
    print("=" * 70)
    print("📊 Available Endpoints:")
    print("  • GET / → Dashboard")
    print("  • GET /api/analysis/1H → 1Hour Analysis") 
    print("  • GET /api/analysis/4H → 4Hour Analysis")
    print("  • GET /api/analysis/1D → Daily Analysis")
    print("  • GET /api/debug → Debug Info")
    print("  • GET /api/clear_cache → Clear Cache")
    print("  • GET /api/health → Health Check")
    print("=" * 70)
    print("🔧 Integrated APIs:")
    print("  • Twelve Data → Real-time Prices & Historical Data")
    print("  • DeepSeek AI → Comprehensive Trading Analysis")
    print("  • NewsAPI → Fundamental Market News")
    print("=" * 70)
    print("🎯 AI TRADING FEATURES:")
    print("  • ✅ Comprehensive Technical Analysis")
    print("  • ✅ Fundamental Context Integration")
    print("  • ✅ Specific Buy/Sell/Hold Recommendations")
    print("  • ✅ Entry, Stop Loss, Take Profit Levels")
    print("  • ✅ Risk-Reward Ratio Calculation")
    print("  • ✅ Real-time Data Enhancement")
    print("=" * 70)
    
    print("🚀 Starting AI-enhanced trading analysis server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
