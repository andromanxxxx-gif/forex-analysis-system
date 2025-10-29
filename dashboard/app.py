# app.py - Enhanced Forex Trading System with Real Data Sources
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import requests
import os
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import talib
import random

# ==================== KONFIGURASI LOGGING ====================
def setup_logging():
    """Setup logging yang compatible dengan Windows"""
    logger = logging.getLogger()
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler('forex_trading.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger

logger = setup_logging()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'forex-secure-key-2024')

# ==================== KONFIGURASI SISTEM YANG DIPERBAIKI ====================
@dataclass
class SystemConfig:
    # API Configuration - KEMBALI KE STRUKTUR AWAL
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "demo")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "demo") 
    TWELVE_DATA_KEY: str = os.environ.get("TWELVE_DATA_KEY", "demo")
    
    # ENHANCED Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.01
    MAX_DAILY_LOSS: float = 0.02
    MAX_DRAWDOWN: float = 0.08
    MAX_POSITIONS: int = 3
    STOP_LOSS_PCT: float = 0.008
    TAKE_PROFIT_PCT: float = 0.02
    
    # Enhanced Risk Management Parameters
    CORRELATION_THRESHOLD: float = 0.75
    VOLATILITY_THRESHOLD: float = 0.02
    DAILY_TRADE_LIMIT: int = 15
    MAX_POSITION_SIZE_PCT: float = 0.04
    
    # Trading Conditions Filter
    MIN_ADX: float = 20.0
    MAX_VOLATILITY_PCT: float = 3.0
    MIN_CONFIDENCE: int = 65
    
    # Backtesting-specific parameters
    BACKTEST_DAILY_TRADE_LIMIT: int = 100
    BACKTEST_MIN_CONFIDENCE: int = 60
    BACKTEST_RISK_SCORE_THRESHOLD: int = 8
    
    # Supported Instruments
    FOREX_PAIRS: List[str] = field(default_factory=lambda: [
        "USDJPY", "GBPJPY", "EURJPY", "CHFJPY", "CADJPY",
        "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
    ])
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M30", "1H", "4H", "1D", "1W"])
    
    # Backtesting
    DEFAULT_BACKTEST_DAYS: int = 90
    MIN_DATA_POINTS: int = 100
    
    # Trading Hours (UTC)
    HIGH_IMPACT_HOURS: List[Tuple[int, int]] = field(default_factory=lambda: [(8, 10), (13, 15)])

config = SystemConfig()

# ==================== DATA MANAGER DENGAN CSV HISTORICAL DATA ====================
class DataManager:
    def __init__(self):
        self.historical_data = {}
        self.historical_data_path = "historical_data"
        logger.info("Data Manager with CSV historical data initialized")
    
    def get_price_data(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Mendapatkan data harga dari CSV historical data"""
        try:
            cache_key = f"{pair}_{timeframe}_{days}"
            
            if cache_key in self.historical_data:
                return self.historical_data[cache_key]
            
            # Coba baca dari file CSV
            csv_file = f"{self.historical_data_path}/{pair}_{timeframe}.csv"
            if os.path.exists(csv_file):
                data = self._load_from_csv(csv_file, days)
                if not data.empty:
                    self.historical_data[cache_key] = data
                    logger.info(f"Loaded {len(data)} records for {pair}-{timeframe} from CSV")
                    return data
            
            # Fallback ke TwelveData API
            logger.info(f"Trying to fetch data from TwelveData for {pair}-{timeframe}")
            data = self._get_twelvedata_data(pair, timeframe, days)
            if not data.empty:
                self.historical_data[cache_key] = data
                return data
            
            # Final fallback ke data sample
            logger.warning(f"Using sample data for {pair}-{timeframe}")
            data = self._generate_sample_data(pair, timeframe, days)
            self.historical_data[cache_key] = data
            return data
            
        except Exception as e:
            logger.error(f"Error getting price data for {pair}: {e}")
            return self._generate_sample_data(pair, timeframe, days)
    
    def _load_from_csv(self, csv_file: str, days: int) -> pd.DataFrame:
        """Load data dari file CSV"""
        try:
            df = pd.read_csv(csv_file)
            
            # Pastikan kolom yang diperlukan ada
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing column {col} in CSV file {csv_file}")
                    return pd.DataFrame()
            
            # Konversi timestamp
            df['date'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Filter untuk jumlah hari yang diminta
            if len(df) > days:
                df = df.tail(days)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            return pd.DataFrame()
    
    def _get_twelvedata_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Get real-time data dari TwelveData API"""
        try:
            if config.TWELVE_DATA_KEY == "demo":
                logger.warning("Using demo TwelveData key - limited functionality")
                return pd.DataFrame()
            
            # Map timeframe ke interval TwelveData
            interval_map = {
                'M30': '30min',
                '1H': '1h',
                '4H': '4h',
                '1D': '1day',
                '1W': '1week'
            }
            
            interval = interval_map.get(timeframe, '1h')
            symbol = f"{pair[:3]}/{pair[3:]}"
            
            url = f"https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'outputsize': min(days * 5, 5000),  # Max 5000 data points
                'apikey': config.TWELVE_DATA_KEY,
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data:
                    df = pd.DataFrame(data['values'])
                    
                    # Convert dan rename columns
                    df['date'] = pd.to_datetime(df['datetime'])
                    df = df.rename(columns={
                        'open': 'open',
                        'high': 'high', 
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume'
                    })
                    
                    # Convert to numeric
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna().sort_values('date').reset_index(drop=True)
                    logger.info(f"Successfully fetched {len(df)} records from TwelveData for {pair}")
                    return df
                else:
                    logger.warning(f"No data in TwelveData response for {pair}")
                    return pd.DataFrame()
            else:
                logger.warning(f"TwelveData API error: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"TwelveData API error for {pair}: {e}")
            return pd.DataFrame()
    
    def _generate_sample_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate sample data hanya untuk fallback"""
        logger.info(f"Generating sample data for {pair}-{timeframe}")
        
        base_prices = {
            'USDJPY': 150.0, 'EURUSD': 1.0800, 'GBPUSD': 1.2600,
            'USDCHF': 0.8800, 'AUDUSD': 0.6500, 'USDCAD': 1.3600,
            'NZDUSD': 0.5900, 'EURJPY': 162.0, 'GBPJPY': 189.0,
            'CHFJPY': 170.0, 'CADJPY': 110.0
        }
        
        base_price = base_prices.get(pair, 150.0)
        points = min(days * 24, 1000)  # Max 1000 points
        
        dates = [datetime.now() - timedelta(hours=i) for i in range(points)][::-1]
        
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Realistic price movement
            change = np.random.normal(0, 0.001)
            current_price = current_price * (1 + change)
            
            # Generate OHLC
            open_price = current_price
            high_price = open_price * (1 + abs(np.random.normal(0, 0.0005)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.0005)))
            close_price = open_price * (1 + np.random.normal(0, 0.0005))
            
            prices.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(1000, 10000)
            })
        
        df = pd.DataFrame(prices)
        logger.info(f"Generated {len(df)} sample data points for {pair}")
        return df

    def get_current_price(self, pair: str) -> float:
        """Get current price dari TwelveData API"""
        try:
            if config.TWELVE_DATA_KEY == "demo":
                # Return price dari historical data sebagai fallback
                data = self.get_price_data(pair, '1H', 1)
                if not data.empty:
                    return float(data['close'].iloc[-1])
                return 150.0
            
            symbol = f"{pair[:3]}/{pair[3:]}"
            url = f"https://api.twelvedata.com/price"
            params = {
                'symbol': symbol,
                'apikey': config.TWELVE_DATA_KEY
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data.get('price', 150.0))
            else:
                logger.warning(f"TwelveData price API error: {response.status_code}")
                data = self.get_price_data(pair, '1H', 1)
                return float(data['close'].iloc[-1]) if not data.empty else 150.0
                
        except Exception as e:
            logger.error(f"Error getting current price for {pair}: {e}")
            data = self.get_price_data(pair, '1H', 1)
            return float(data['close'].iloc[-1]) if not data.empty else 150.0

# ==================== DEEPSEEK AI ANALYZER ====================
class DeepSeekAnalyzer:
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        logger.info("DeepSeek AI Analyzer initialized")
    
    def analyze_market_sentiment(self, pair: str, technical_data: Dict, news_data: List = None) -> Dict:
        """Analisis sentimen pasar menggunakan DeepSeek AI"""
        try:
            if self.api_key == "demo":
                return self._demo_sentiment_analysis(pair, technical_data)
            
            # Prepare technical analysis summary
            tech_summary = self._prepare_technical_summary(technical_data)
            
            # Prepare prompt
            prompt = f"""
            Analisis sentimen pasar untuk pair forex {pair} berdasarkan data teknikal berikut:
            
            {tech_summary}
            
            Berikan analisis dalam format JSON dengan struktur:
            {{
                "sentiment": "BULLISH|BEARISH|NEUTRAL",
                "confidence": 0-100,
                "reasoning": "analisis singkat",
                "key_levels": {{
                    "support": [level1, level2],
                    "resistance": [level1, level2]
                }},
                "recommendation": "BUY|SELL|HOLD",
                "risk_level": "LOW|MEDIUM|HIGH"
            }}
            """
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "Anda adalah analis forex profesional. Berikan analisis yang objektif dan berdasarkan data."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                analysis_text = result['choices'][0]['message']['content']
                
                # Parse JSON dari response
                try:
                    analysis = json.loads(analysis_text)
                    logger.info(f"DeepSeek analysis completed for {pair}: {analysis.get('sentiment', 'UNKNOWN')}")
                    return analysis
                except json.JSONDecodeError:
                    return self._parse_text_analysis(analysis_text)
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return self._demo_sentiment_analysis(pair, technical_data)
                
        except Exception as e:
            logger.error(f"Error in DeepSeek analysis: {e}")
            return self._demo_sentiment_analysis(pair, technical_data)
    
    def _prepare_technical_summary(self, technical_data: Dict) -> str:
        """Siapkan summary data teknikal untuk AI"""
        trend = technical_data.get('trend', {})
        momentum = technical_data.get('momentum', {})
        levels = technical_data.get('levels', {})
        
        summary = f"""
        TREND ANALYSIS:
        - Direction: {trend.get('trend_direction', 'UNKNOWN')}
        - Strength: {trend.get('trend_strength', 'UNKNOWN')}
        - SMA 20: {trend.get('sma_20', 0):.4f}
        - SMA 50: {trend.get('sma_50', 0):.4f}
        - ADX: {trend.get('adx', 0):.1f}
        
        MOMENTUM INDICATORS:
        - RSI: {momentum.get('rsi', 0):.1f}
        - MACD Histogram: {momentum.get('macd_histogram', 0):.6f}
        - Price Change: {momentum.get('price_change_pct', 0):.2f}%
        
        KEY LEVELS:
        - Current Price: {levels.get('current_price', 0):.4f}
        - Support: {levels.get('support', 0):.4f}
        - Resistance: {levels.get('resistance', 0):.4f}
        - Pivot: {levels.get('pivot_point', 0):.4f}
        """
        
        return summary
    
    def _parse_text_analysis(self, analysis_text: str) -> Dict:
        """Parse analysis text jika JSON tidak valid"""
        sentiment = "NEUTRAL"
        confidence = 50
        
        if "BULLISH" in analysis_text.upper():
            sentiment = "BULLISH"
            confidence = 70
        elif "BEARISH" in analysis_text.upper():
            sentiment = "BEARISH" 
            confidence = 70
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "reasoning": "AI analysis completed",
            "key_levels": {
                "support": [],
                "resistance": []
            },
            "recommendation": "HOLD",
            "risk_level": "MEDIUM"
        }
    
    def _demo_sentiment_analysis(self, pair: str, technical_data: Dict) -> Dict:
        """Demo sentiment analysis ketika API key demo"""
        rsi = technical_data.get('momentum', {}).get('rsi', 50)
        trend = technical_data.get('trend', {}).get('trend_direction', 'NEUTRAL')
        
        if rsi < 30 and trend == 'BULLISH':
            sentiment = "BULLISH"
            confidence = 75
            recommendation = "BUY"
        elif rsi > 70 and trend == 'BEARISH':
            sentiment = "BEARISH"
            confidence = 75
            recommendation = "SELL"
        else:
            sentiment = "NEUTRAL"
            confidence = 50
            recommendation = "HOLD"
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "reasoning": f"Demo analysis based on RSI {rsi:.1f} and {trend} trend",
            "key_levels": {
                "support": [
                    technical_data.get('levels', {}).get('support', 0) * 0.995,
                    technical_data.get('levels', {}).get('support', 0) * 0.99
                ],
                "resistance": [
                    technical_data.get('levels', {}).get('resistance', 0) * 1.005,
                    technical_data.get('levels', {}).get('resistance', 0) * 1.01
                ]
            },
            "recommendation": recommendation,
            "risk_level": "MEDIUM"
        }

# ==================== NEWS API INTEGRATION ====================
class NewsAnalyzer:
    def __init__(self):
        self.api_key = config.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"
        logger.info("News Analyzer initialized")
    
    def get_forex_news(self, pair: str = None, days: int = 7) -> List[Dict]:
        """Dapatkan berita forex terkini"""
        try:
            if self.api_key == "demo":
                return self._get_demo_news(pair)
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            query_terms = ["forex", "currency", "FX"]
            if pair:
                base_currency = pair[:3]
                quote_currency = pair[3:]
                query_terms.extend([base_currency, quote_currency, f"{base_currency}/{quote_currency}"])
            
            query = " OR ".join(query_terms)
            
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.api_key,
                'pageSize': 20
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                processed_articles = []
                for article in articles:
                    processed_articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'sentiment': self._analyze_article_sentiment(article)
                    })
                
                logger.info(f"Retrieved {len(processed_articles)} news articles")
                return processed_articles
            else:
                logger.error(f"News API error: {response.status_code}")
                return self._get_demo_news(pair)
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return self._get_demo_news(pair)
    
    def _analyze_article_sentiment(self, article: Dict) -> str:
        """Analisis sentimen artikel berita sederhana"""
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        
        text = f"{title} {description}"
        
        positive_words = ['bullish', 'up', 'rise', 'gain', 'strong', 'positive', 'buy', 'growth']
        negative_words = ['bearish', 'down', 'fall', 'drop', 'weak', 'negative', 'sell', 'recession']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "POSITIVE"
        elif negative_count > positive_count:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    
    def _get_demo_news(self, pair: str) -> List[Dict]:
        """Demo news data"""
        demo_articles = [
            {
                'title': f'Forex Market Analysis: {pair} Shows Strong Momentum',
                'description': f'Technical indicators suggest potential breakout for {pair} in coming sessions.',
                'url': '#',
                'published_at': datetime.now().isoformat(),
                'source': 'Demo News',
                'sentiment': 'POSITIVE'
            },
            {
                'title': 'Central Bank Policy Decision Impacts Currency Markets',
                'description': 'Recent policy announcements affecting major currency pairs volatility.',
                'url': '#', 
                'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': 'Demo News',
                'sentiment': 'NEUTRAL'
            }
        ]
        return demo_articles

# ==================== TECHNICAL ANALYSIS ENGINE ====================
class TechnicalAnalysisEngine:
    def __init__(self):
        self.indicators = {}
        logger.info("Technical Analysis Engine initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Menghitung semua indikator teknikal"""
        try:
            if df.empty or len(df) < 20:
                return self._fallback_indicators(df)
                
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return self._fallback_indicators(df)
            
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            opens = df['open'].values
            
            # Handle NaN values
            closes = np.nan_to_num(closes, nan=closes[-1] if len(closes) > 0 else 150.0)
            highs = np.nan_to_num(highs, nan=closes[-1] if len(closes) > 0 else 150.0)
            lows = np.nan_to_num(lows, nan=closes[-1] if len(closes) > 0 else 150.0)
            opens = np.nan_to_num(opens, nan=closes[-1] if len(closes) > 0 else 150.0)
            
            # Trend Indicators
            sma_20 = talib.SMA(closes, timeperiod=20)
            sma_50 = talib.SMA(closes, timeperiod=50)
            ema_12 = talib.EMA(closes, timeperiod=12)
            ema_26 = talib.EMA(closes, timeperiod=26)
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            
            # Momentum Indicators
            rsi = talib.RSI(closes, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(closes)
            stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
            williams_r = talib.WILLR(highs, lows, closes, timeperiod=14)
            
            # Volatility Indicators
            bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(closes)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            
            # Support & Resistance
            lookback_period = min(50, len(highs))
            recent_high = np.max(highs[-lookback_period:]) if len(highs) >= lookback_period else np.max(highs)
            recent_low = np.min(lows[-lookback_period:]) if len(lows) >= lookback_period else np.min(lows)
            
            def safe_float(value, default=0):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
            
            current_price = safe_float(closes[-1], 150.0)
            price_change_pct = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0
            
            return {
                'trend': {
                    'sma_20': safe_float(sma_20[-1], current_price),
                    'sma_50': safe_float(sma_50[-1], current_price),
                    'ema_12': safe_float(ema_12[-1], current_price),
                    'ema_26': safe_float(ema_26[-1], current_price),
                    'adx': safe_float(adx[-1], 25),
                    'trend_direction': 'BULLISH' if safe_float(sma_20[-1], current_price) > safe_float(sma_50[-1], current_price) else 'BEARISH',
                    'trend_strength': 'STRONG' if safe_float(adx[-1], 25) > 40 else 'WEAK' if safe_float(adx[-1], 25) < 20 else 'MODERATE'
                },
                'momentum': {
                    'rsi': safe_float(rsi[-1], 50),
                    'macd': safe_float(macd[-1], 0),
                    'macd_signal': safe_float(macd_signal[-1], 0),
                    'macd_histogram': safe_float(macd_hist[-1], 0),
                    'stoch_k': safe_float(stoch_k[-1], 50),
                    'stoch_d': safe_float(stoch_d[-1], 50),
                    'williams_r': safe_float(williams_r[-1], -50),
                    'price_change_pct': safe_float(price_change_pct, 0)
                },
                'volatility': {
                    'bollinger_upper': safe_float(bollinger_upper[-1], current_price * 1.02),
                    'bollinger_middle': safe_float(bollinger_middle[-1], current_price),
                    'bollinger_lower': safe_float(bollinger_lower[-1], current_price * 0.98),
                    'atr': safe_float(atr[-1], current_price * 0.005),
                    'volatility_pct': safe_float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100, 1.0) if len(closes) >= 20 else 1.0
                },
                'levels': {
                    'support': safe_float(recent_low, current_price * 0.99),
                    'resistance': safe_float(recent_high, current_price * 1.01),
                    'current_price': current_price,
                    'pivot_point': safe_float((highs[-1] + lows[-1] + closes[-1]) / 3, current_price) if len(highs) > 0 else current_price
                }
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._fallback_indicators(df)

    def _fallback_indicators(self, df: pd.DataFrame) -> Dict:
        """Fallback indicators jika perhitungan gagal"""
        try:
            if len(df) > 0 and 'close' in df.columns:
                current_price = float(df['close'].iloc[-1])
            else:
                current_price = 150.0
        except:
            current_price = 150.0
            
        return {
            'trend': {
                'sma_20': current_price * 0.998,
                'sma_50': current_price * 0.995,
                'ema_12': current_price * 0.999,
                'ema_26': current_price * 0.997,
                'adx': 25,
                'trend_direction': 'BULLISH',
                'trend_strength': 'MODERATE'
            },
            'momentum': {
                'rsi': 50,
                'macd': 0.001,
                'macd_signal': 0.0005,
                'macd_histogram': 0.0005,
                'stoch_k': 50,
                'stoch_d': 50,
                'williams_r': -50,
                'price_change_pct': 0
            },
            'volatility': {
                'bollinger_upper': current_price * 1.02,
                'bollinger_middle': current_price,
                'bollinger_lower': current_price * 0.98,
                'atr': current_price * 0.005,
                'volatility_pct': 1.0
            },
            'levels': {
                'support': current_price * 0.99,
                'resistance': current_price * 1.01,
                'current_price': current_price,
                'pivot_point': current_price
            }
        }

# ==================== RISK MANAGEMENT SYSTEM ====================
class AdvancedRiskManager:
    def __init__(self, backtest_mode: bool = False):
        self.max_daily_loss_pct = config.MAX_DAILY_LOSS
        self.max_drawdown_pct = config.MAX_DRAWDOWN
        self.daily_trade_limit = config.BACKTEST_DAILY_TRADE_LIMIT if backtest_mode else config.DAILY_TRADE_LIMIT
        self.backtest_mode = backtest_mode
        
        self.today_trades = 0
        self.daily_pnl = 0.0
        self.peak_balance = 10000.0
        self.current_drawdown = 0.0
        self.last_reset_date = datetime.now().date()
        
        logger.info(f"Risk Manager initialized - Backtest Mode: {backtest_mode}")
    
    def validate_trade(self, pair: str, signal: str, confidence: int, 
                      account_balance: float, current_price: float) -> Dict:
        """Validasi trade dengan risk management"""
        self.reset_daily_limits()
        
        validation_result = {
            'approved': True,
            'risk_score': 0,
            'rejection_reasons': [],
            'warnings': []
        }
        
        # Check daily trade limit
        if self.today_trades >= self.daily_trade_limit:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(f"Daily trade limit reached ({self.daily_trade_limit})")
        
        # Check daily loss limit
        daily_loss_limit = account_balance * self.max_daily_loss_pct
        if self.daily_pnl <= -daily_loss_limit:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(f"Daily loss limit reached (${-self.daily_pnl:.2f})")
        
        # Check confidence level
        min_confidence = config.BACKTEST_MIN_CONFIDENCE if self.backtest_mode else config.MIN_CONFIDENCE
        if confidence < min_confidence:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(f"Low confidence ({confidence}% < {min_confidence}%)")
        
        logger.info(f"Risk validation for {pair}-{signal}: {'APPROVED' if validation_result['approved'] else 'REJECTED'}")
        return validation_result
    
    def reset_daily_limits(self):
        """Reset daily limits"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.today_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today

    def update_trade_result(self, pnl: float):
        """Update hasil trade"""
        self.daily_pnl += pnl
        self.today_trades += 1
        
        if pnl < 0:
            self.current_drawdown = abs(self.daily_pnl) / self.peak_balance if self.peak_balance > 0 else 0
        else:
            if self.daily_pnl > self.peak_balance:
                self.peak_balance = self.daily_pnl

# ==================== BACKTESTING ENGINE ====================
class AdvancedBacktestingEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.risk_manager = AdvancedRiskManager(backtest_mode=True)
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.trade_history = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        
        logger.info("Backtesting engine reset")
    
    def run_backtest(self, signals: List[Dict], price_data: pd.DataFrame, pair: str) -> Dict:
        """Jalankan backtest"""
        self.reset()
        
        logger.info(f"Running backtest for {pair} with {len(signals)} signals")
        
        if not signals or price_data.empty:
            return self._empty_result(pair)
        
        for signal in signals:
            self._execute_trade(signal, price_data)
        
        return self._generate_report(pair)
    
    def _execute_trade(self, signal: Dict, price_data: pd.DataFrame):
        """Eksekusi trade dalam backtest"""
        try:
            action = signal['action']
            confidence = signal.get('confidence', 50)
            entry_price = signal['price']
            
            # Risk validation
            risk_check = self.risk_manager.validate_trade(
                signal.get('pair', 'UNKNOWN'),
                action,
                confidence,
                self.balance,
                entry_price
            )
            
            if not risk_check['approved']:
                return
            
            # Simulate trade outcome
            if np.random.random() < (confidence / 100.0):
                # Winning trade
                profit = entry_price * 0.01 * (1 if action == 'BUY' else -1) * 100
                outcome = 'WIN'
                self.winning_trades += 1
            else:
                # Losing trade  
                profit = -entry_price * 0.005 * (1 if action == 'BUY' else -1) * 100
                outcome = 'LOSS'
                self.losing_trades += 1
            
            self.balance += profit
            
            trade_record = {
                'date': signal['date'].strftime('%Y-%m-%d') if hasattr(signal['date'], 'strftime') else str(signal['date']),
                'pair': signal.get('pair', 'UNKNOWN'),
                'action': action,
                'entry_price': round(entry_price, 4),
                'profit': round(profit, 2),
                'outcome': outcome,
                'confidence': confidence,
                'balance_after': round(self.balance, 2)
            }
            
            self.trade_history.append(trade_record)
            self.risk_manager.update_trade_result(profit)
            
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            
            current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
        except Exception as e:
            logger.error(f"Error executing trade in backtest: {e}")
    
    def _generate_report(self, pair: str) -> Dict:
        """Generate laporan backtest"""
        total_trades = len(self.trade_history)
        
        if total_trades == 0:
            return self._empty_result(pair)
        
        win_rate = (self.winning_trades / total_trades * 100)
        total_profit = sum(trade['profit'] for trade in self.trade_history)
        total_return_pct = ((self.balance - self.initial_balance) / self.initial_balance * 100)
        
        return {
            'status': 'success',
            'summary': {
                'total_trades': total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': round(win_rate, 2),
                'total_profit': round(total_profit, 2),
                'final_balance': round(self.balance, 2),
                'return_percentage': round(total_return_pct, 2),
                'max_drawdown': round(self.max_drawdown * 100, 2)
            },
            'trade_history': self.trade_history[-20:],  # Last 20 trades
            'metadata': {
                'pair': pair,
                'initial_balance': self.initial_balance,
                'testing_date': datetime.now().isoformat()
            }
        }
    
    def _empty_result(self, pair: str) -> Dict:
        return {
            'status': 'no_trades',
            'summary': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'final_balance': self.initial_balance,
                'return_percentage': 0,
                'max_drawdown': 0
            },
            'trade_history': [],
            'metadata': {
                'pair': pair,
                'message': 'No trades executed during backtest period'
            }
        }

# ==================== TRADING SIGNAL GENERATOR ====================
def generate_trading_signals(price_data: pd.DataFrame, pair: str, timeframe: str) -> List[Dict]:
    """Generate sinyal trading berdasarkan analisis teknikal"""
    signals = []
    
    try:
        if len(price_data) < 20:
            return signals
        
        tech_engine = TechnicalAnalysisEngine()
        step_size = max(1, len(price_data) // 20)  # Sample points untuk efisiensi
        
        for i in range(20, len(price_data), step_size):
            try:
                window_data = price_data.iloc[:i+1]
                tech_analysis = tech_engine.calculate_all_indicators(window_data)
                
                rsi = tech_analysis['momentum']['rsi']
                macd_hist = tech_analysis['momentum']['macd_histogram']
                trend = tech_analysis['trend']['trend_direction']
                
                signal = None
                confidence = 50
                
                # Buy conditions
                if rsi < 35 and macd_hist > 0 and trend == 'BULLISH':
                    signal = 'BUY'
                    confidence = 70
                # Sell conditions
                elif rsi > 65 and macd_hist < 0 and trend == 'BEARISH':
                    signal = 'SELL'
                    confidence = 70
                
                if signal and confidence >= config.BACKTEST_MIN_CONFIDENCE:
                    current_date = window_data.iloc[-1]['date']
                    signals.append({
                        'date': current_date,
                        'pair': pair,
                        'action': signal,
                        'confidence': confidence,
                        'price': tech_analysis['levels']['current_price'],
                        'rsi': rsi,
                        'macd_hist': macd_hist
                    })
                    
            except Exception as e:
                continue
        
        logger.info(f"Generated {len(signals)} trading signals for {pair}-{timeframe}")
        return signals
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        return []

# ==================== INISIALISASI SISTEM ====================
logger.info("Initializing Enhanced Forex Analysis System...")

# Initialize components
data_manager = DataManager()
tech_engine = TechnicalAnalysisEngine()
deepseek_analyzer = DeepSeekAnalyzer()
news_analyzer = NewsAnalyzer()
risk_manager = AdvancedRiskManager()
backtesting_engine = AdvancedBacktestingEngine()

logger.info("All system components initialized successfully")

# ==================== UTILITY FUNCTIONS ====================
def convert_numpy_types(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', 
                         pairs=config.FOREX_PAIRS,
                         timeframes=config.TIMEFRAMES,
                         initial_balance=config.INITIAL_BALANCE)

@app.route('/api/analyze')
def api_analyze():
    """Endpoint untuk analisis market komprehensif"""
    try:
        pair = request.args.get('pair', 'USDJPY').upper()
        timeframe = request.args.get('timeframe', '4H').upper()
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        # Get price data
        price_data = data_manager.get_price_data(pair, timeframe, days=60)
        if price_data.empty:
            return jsonify({'error': 'No price data available'}), 400
        
        # Technical analysis
        technical_analysis = tech_engine.calculate_all_indicators(price_data)
        
        # AI analysis
        ai_analysis = deepseek_analyzer.analyze_market_sentiment(pair, technical_analysis)
        
        # News analysis
        news_articles = news_analyzer.get_forex_news(pair)
        
        # Current price
        current_price = data_manager.get_current_price(pair)
        
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'technical_analysis': technical_analysis,
            'ai_analysis': ai_analysis,
            'news_analysis': {
                'total_articles': len(news_articles),
                'articles': news_articles[:5]  # First 5 articles
            },
            'price_data': {
                'current': current_price,
                'support': technical_analysis['levels']['support'],
                'resistance': technical_analysis['levels']['resistance'],
                'change_pct': technical_analysis['momentum']['price_change_pct']
            }
        }
        
        # Convert numpy types
        response = convert_numpy_types(response)
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Endpoint untuk backtesting"""
    try:
        data = request.get_json()
        pair = data.get('pair', 'USDJPY')
        timeframe = data.get('timeframe', '4H')
        days = int(data.get('days', 30))
        initial_balance = float(data.get('initial_balance', config.INITIAL_BALANCE))
        
        logger.info(f"Backtest request: {pair}-{timeframe} for {days} days")
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        # Get price data
        price_data = data_manager.get_price_data(pair, timeframe, days)
        if price_data.empty:
            return jsonify({'error': 'No price data available for backtesting'}), 400
        
        # Generate signals
        signals = generate_trading_signals(price_data, pair, timeframe)
        
        # Run backtest
        backtesting_engine.initial_balance = initial_balance
        result = backtesting_engine.run_backtest(signals, price_data, pair)
        
        # Add metadata
        result['backtest_parameters'] = {
            'pair': pair,
            'timeframe': timeframe,
            'days': days,
            'initial_balance': initial_balance,
            'signals_generated': len(signals)
        }
        
        result = convert_numpy_types(result)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500

@app.route('/api/system_status')
def api_system_status():
    """Status sistem"""
    return jsonify({
        'system': 'RUNNING',
        'supported_pairs': config.FOREX_PAIRS,
        'data_sources': {
            'historical_data': 'CSV Files',
            'realtime_data': 'TwelveData API',
            'ai_analysis': 'DeepSeek AI',
            'news': 'NewsAPI'
        },
        'server_time': datetime.now().isoformat(),
        'version': '2.0 - Real Data Sources'
    })

@app.route('/api/current_price/<pair>')
def api_current_price(pair):
    """Get current price untuk pair tertentu"""
    try:
        price = data_manager.get_current_price(pair)
        return jsonify({
            'pair': pair,
            'price': price,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Current price error for {pair}: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("Starting Enhanced Forex Analysis System v2.0...")
    logger.info(f"Supported pairs: {config.FOREX_PAIRS}")
    logger.info("Data Sources: CSV Historical + TwelveData Real-time + DeepSeek AI + NewsAPI")
    
    # Create necessary directories
    os.makedirs('historical_data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Forex Analysis System is ready and running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
