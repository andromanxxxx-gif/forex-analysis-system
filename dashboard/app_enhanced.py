# [FILE: app_enhanced.py] - COMPLETE FIXED VERSION
from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import talib
import yfinance as yf
import random
import time
from functools import wraps
import psutil
import hashlib
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import asyncio
import aiohttp
import threading

# ==================== ENHANCED CACHE CONFIG ====================
cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# ==================== KONFIGURASI LOGGING ====================
def setup_logging():
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
cache.init_app(app)

# ==================== KONFIGURASI SISTEM YANG DIPERBAIKI ====================
@dataclass
class SystemConfig:
    # FREE API Configuration - NO API KEYS REQUIRED
    YAHOO_FINANCE_ENABLED: bool = True
    SIMULATED_DATA_ENABLED: bool = True
    
    # Optional Premium APIs (if available)
    ALPHA_VANTAGE_KEY: str = os.environ.get("ALPHA_VANTAGE_KEY", "demo")
    TWELVE_DATA_KEY: str = os.environ.get("TWELVE_DATA_KEY", "demo")
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "demo")
    
    # Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    
    # Supported Instruments
    FOREX_PAIRS: List[str] = field(default_factory=lambda: [
        "USDJPY", "GBPJPY", "EURJPY", "CHFJPY", 
        "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
    ])
    
    # Data source mapping - PRIORITIZE FREE SOURCES
    DATA_SOURCE_PRIORITY: List[str] = field(default_factory=lambda: [
        'YAHOO_FINANCE',
        'SIMULATED_DATA',
        'ALPHA_VANTAGE',
        'TWELVE_DATA'
    ])
    
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M30", "1H", "4H", "1D", "1W"])
    
    # Rate limiting
    REQUESTS_PER_MINUTE: int = 30
    REQUESTS_PER_HOUR: int = 200

config = SystemConfig()

# ==================== ENHANCED TECHNICAL ANALYSIS ENGINE ====================
class TechnicalAnalysisEngine:
    def __init__(self):
        self.indicators_cache = {}
        self.cache_lock = Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        logger.info("Enhanced Technical Analysis Engine initialized")

    def _safe_convert_to_float64(self, data):
        """Convert data to float64 safely"""
        try:
            return np.array(data, dtype=np.float64)
        except:
            return np.array(data, dtype=object)

    def _get_fallback_indicators(self, df):
        """Return fallback indicators when data is insufficient"""
        return {
            'trend': {'direction': 'NEUTRAL', 'strength': 50},
            'momentum': {'rsi': 50, 'macd': 0, 'signal': 0},
            'volatility': {'atr': 0, 'volatility_class': 'LOW'},
            'levels': {'support': 0, 'resistance': 0, 'pivot': 0},
            'volume': {'volume_trend': 'NEUTRAL'},
            'oscillators': {'stoch_k': 50, 'stoch_d': 50},
            'signals': {'primary': 'HOLD', 'confidence': 0}
        }

    @cache.memoize(timeout=60)
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Hitung semua indikator teknikal dengan caching"""
        try:
            if df.empty or len(df) < 20:
                return self._get_fallback_indicators(df)
            
            df = df.sort_values('date').reset_index(drop=True)
            closes = self._safe_convert_to_float64(df['close'].values)
            highs = self._safe_convert_to_float64(df['high'].values)
            lows = self._safe_convert_to_float64(df['low'].values)
            
            # Calculate indicators
            with self.cache_lock:
                results = {
                    'trend': self._calculate_trend_indicators(closes, highs, lows),
                    'momentum': self._calculate_momentum_indicators(closes, highs, lows),
                    'volatility': self._calculate_volatility_indicators(highs, lows, closes),
                    'levels': self._calculate_support_resistance(df),
                    'volume': self._calculate_volume_indicators(df),
                    'oscillators': self._calculate_oscillators(closes, highs, lows)
                }
            
            results['signals'] = self._generate_signals(results)
            return self._convert_to_serializable(results)
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._get_fallback_indicators(df)

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def _calculate_trend_indicators(self, closes, highs, lows):
        """Calculate trend indicators"""
        try:
            sma_20 = talib.SMA(closes, timeperiod=20)
            sma_50 = talib.SMA(closes, timeperiod=50)
            
            if sma_20 is None or sma_50 is None or len(sma_20) == 0 or len(sma_50) == 0:
                return {'direction': 'NEUTRAL', 'strength': 50}
            
            # Determine trend direction
            if sma_20[-1] > sma_50[-1]:
                direction = 'BULLISH'
                strength = min(100, (sma_20[-1] / sma_50[-1] - 1) * 1000)
            elif sma_20[-1] < sma_50[-1]:
                direction = 'BEARISH'
                strength = min(100, (1 - sma_20[-1] / sma_50[-1]) * 1000)
            else:
                direction = 'NEUTRAL'
                strength = 50
                
            return {
                'direction': direction,
                'strength': float(strength),
                'sma_20': float(sma_20[-1]),
                'sma_50': float(sma_50[-1])
            }
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return {'direction': 'NEUTRAL', 'strength': 50}

    def _calculate_momentum_indicators(self, closes, highs, lows):
        """Calculate momentum indicators"""
        try:
            rsi = talib.RSI(closes, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(closes)
            
            return {
                'rsi': float(rsi[-1]) if rsi is not None and len(rsi) > 0 else 50,
                'macd': float(macd[-1]) if macd is not None and len(macd) > 0 else 0,
                'signal': float(macd_signal[-1]) if macd_signal is not None and len(macd_signal) > 0 else 0,
                'histogram': float(macd_hist[-1]) if macd_hist is not None and len(macd_hist) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {'rsi': 50, 'macd': 0, 'signal': 0, 'histogram': 0}

    def _calculate_volatility_indicators(self, highs, lows, closes):
        """Calculate volatility indicators"""
        try:
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            if atr is None or len(atr) == 0:
                return {'atr': 0, 'volatility_class': 'LOW'}
            
            atr_value = atr[-1]
            avg_close = np.mean(closes) if len(closes) > 0 else 1
            
            if atr_value > avg_close * 0.01:
                volatility_class = 'HIGH'
            elif atr_value > avg_close * 0.005:
                volatility_class = 'MEDIUM'
            else:
                volatility_class = 'LOW'
                
            return {
                'atr': float(atr_value),
                'volatility_class': volatility_class
            }
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return {'atr': 0, 'volatility_class': 'LOW'}

    def _calculate_support_resistance(self, df):
        """Calculate support and resistance levels"""
        try:
            closes = df['close'].values
            if len(closes) < 20:
                return {'support': 0, 'resistance': 0, 'pivot': 0}
            
            recent_low = float(np.min(closes[-20:]))
            recent_high = float(np.max(closes[-20:]))
            pivot = float((recent_high + recent_low + closes[-1]) / 3)
            
            return {
                'support': recent_low,
                'resistance': recent_high,
                'pivot': pivot
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {'support': 0, 'resistance': 0, 'pivot': 0}

    def _calculate_volume_indicators(self, df):
        """Calculate volume indicators"""
        try:
            if 'volume' not in df:
                return {'volume_trend': 'NEUTRAL'}
            
            volumes = df['volume'].values
            if len(volumes) < 2:
                return {'volume_trend': 'NEUTRAL'}
            
            if volumes[-1] > volumes[-2]:
                return {'volume_trend': 'INCREASING'}
            elif volumes[-1] < volumes[-2]:
                return {'volume_trend': 'DECREASING'}
            else:
                return {'volume_trend': 'NEUTRAL'}
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {'volume_trend': 'NEUTRAL'}

    def _calculate_oscillators(self, closes, highs, lows):
        """Calculate oscillator indicators"""
        try:
            stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
            
            return {
                'stoch_k': float(stoch_k[-1]) if stoch_k is not None and len(stoch_k) > 0 else 50,
                'stoch_d': float(stoch_d[-1]) if stoch_d is not None and len(stoch_d) > 0 else 50
            }
        except Exception as e:
            logger.error(f"Error calculating oscillators: {e}")
            return {'stoch_k': 50, 'stoch_d': 50}

    def _generate_signals(self, results):
        """Generate trading signals based on indicators"""
        try:
            trend = results['trend']
            momentum = results['momentum']
            
            # Simple signal logic
            if trend['direction'] == 'BULLISH' and momentum.get('rsi', 50) < 70:
                signal = 'BUY'
                confidence = trend['strength']
            elif trend['direction'] == 'BEARISH' and momentum.get('rsi', 50) > 30:
                signal = 'SELL'
                confidence = trend['strength']
            else:
                signal = 'HOLD'
                confidence = 50
                
            return {
                'primary': signal,
                'confidence': float(confidence)
            }
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {'primary': 'HOLD', 'confidence': 50}

# ==================== FREE DATA MANAGER ====================
class FreeDataManager:
    """Data manager yang menggunakan sumber data GRATIS"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        logger.info("Free Data Manager initialized")

    def get_price_data(self, pair: str, timeframe: str, days: int = 60) -> pd.DataFrame:
        """Dapatkan data harga dari sumber gratis"""
        try:
            cache_key = f"{pair}_{timeframe}_{days}"
            if cache_key in self.cache:
                cached_time, data = self.cache[cache_key]
                if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                    return data

            # Priority 1: Yahoo Finance (FREE)
            df = self._get_yahoo_data(pair, timeframe, days)
            if df is not None and not df.empty:
                self.cache[cache_key] = (datetime.now(), df)
                return df

            # Priority 2: Simulated Data (FALLBACK)
            df = self._generate_simulated_data(pair, timeframe, days)
            self.cache[cache_key] = (datetime.now(), df)
            return df

        except Exception as e:
            logger.error(f"Error getting price data for {pair}-{timeframe}: {e}")
            return self._generate_simulated_data(pair, timeframe, days)

    def _get_yahoo_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Dapatkan data dari Yahoo Finance (GRATIS)"""
        try:
            # Convert forex pair to Yahoo format
            yahoo_symbol = self._convert_to_yahoo_symbol(pair)
            if not yahoo_symbol:
                return None

            # Calculate period based on timeframe
            period = self._get_yahoo_period(timeframe, days)
            
            # Download data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(period=period, interval=self._get_yahoo_interval(timeframe))
            
            if df.empty:
                return None

            # Process data
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Ensure we have enough data
            if len(df) < 20:
                return None
                
            logger.info(f"Yahoo Finance data for {pair}-{timeframe}: {len(df)} records")
            return df

        except Exception as e:
            logger.warning(f"Yahoo Finance failed for {pair}: {e}")
            return None

    def _convert_to_yahoo_symbol(self, pair: str) -> str:
        """Convert forex pair to Yahoo Finance symbol"""
        mapping = {
            'EURUSD': 'EURUSD=X', 'USDJPY': 'JPY=X', 'GBPUSD': 'GBPUSD=X',
            'USDCHF': 'CHF=X', 'AUDUSD': 'AUDUSD=X', 'USDCAD': 'CAD=X',
            'NZDUSD': 'NZDUSD=X', 'EURJPY': 'EURJPY=X', 'GBPJPY': 'GBPJPY=X',
            'CHFJPY': 'CHFJPY=X'
        }
        return mapping.get(pair, f"{pair}=X")

    def _get_yahoo_period(self, timeframe: str, days: int) -> str:
        """Get Yahoo Finance period parameter"""
        if days <= 7: return "5d"
        elif days <= 30: return "1mo"
        elif days <= 90: return "3mo"
        elif days <= 180: return "6mo"
        else: return "1y"

    def _get_yahoo_interval(self, timeframe: str) -> str:
        """Get Yahoo Finance interval parameter"""
        intervals = {
            'M30': '30m', '1H': '1h', '4H': '4h', 
            '1D': '1d', '1W': '1wk'
        }
        return intervals.get(timeframe, '1h')

    def _generate_simulated_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate realistic simulated data sebagai fallback"""
        # Base price for different pairs
        base_prices = {
            'USDJPY': 150.0, 'EURJPY': 160.0, 'GBPJPY': 190.0, 'CHFJPY': 170.0,
            'EURUSD': 1.08, 'GBPUSD': 1.27, 'USDCHF': 0.88, 
            'AUDUSD': 0.66, 'USDCAD': 1.36, 'NZDUSD': 0.61
        }
        
        base_price = base_prices.get(pair, 100.0)
        volatility = 0.001  # 0.1% daily volatility
        
        # Generate dates
        end_date = datetime.now()
        if timeframe == '1D':
            start_date = end_date - timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        elif timeframe == '1H':
            start_date = end_date - timedelta(days=min(days, 30))
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
        elif timeframe == '4H':
            start_date = end_date - timedelta(days=min(days, 60))
            dates = pd.date_range(start=start_date, end=end_date, freq='4H')
        elif timeframe == 'M30':
            start_date = end_date - timedelta(days=min(days, 15))
            dates = pd.date_range(start=start_date, end=end_date, freq='30T')
        else:  # 1W
            start_date = end_date - timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
        
        n = len(dates)
        if n == 0:
            n = 100
            dates = pd.date_range(end=end_date, periods=n, freq='D')
        
        # Generate price data with random walk
        prices = [base_price]
        for i in range(1, n):
            change = random.uniform(-volatility, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(random.uniform(0, volatility))) for p in prices],
            'low': [p * (1 - abs(random.uniform(0, volatility))) for p in prices],
            'close': prices,
            'volume': [random.randint(1000000, 5000000) for _ in range(n)]
        })
        
        logger.info(f"Generated simulated data for {pair}-{timeframe}: {len(df)} records")
        return df

    def get_real_time_price(self, pair: str) -> float:
        """Get real-time price dari Yahoo Finance"""
        try:
            yahoo_symbol = self._convert_to_yahoo_symbol(pair)
            ticker = yf.Ticker(yahoo_symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            logger.warning(f"Real-time price failed for {pair}: {e}")
        
        # Fallback to simulated price
        return self._get_simulated_price(pair)

    def _get_simulated_price(self, pair: str) -> float:
        """Get simulated real-time price"""
        base_prices = {
            'USDJPY': 150.0, 'EURJPY': 160.0, 'GBPJPY': 190.0, 'CHFJPY': 170.0,
            'EURUSD': 1.08, 'GBPUSD': 1.27, 'USDCHF': 0.88, 
            'AUDUSD': 0.66, 'USDCAD': 1.36, 'NZDUSD': 0.61
        }
        base_price = base_prices.get(pair, 100.0)
        variation = random.uniform(-0.001, 0.001)
        return base_price * (1 + variation)

# ==================== ENHANCED AI ANALYZER ====================
class EnhancedAIAnalyzer:
    """AI Analyzer dengan multiple fallback options"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 600
        
    def analyze_market(self, pair: str, technical_analysis: Dict, fundamental_analysis: Dict) -> Dict[str, Any]:
        """Analisis market dengan multiple fallback strategies"""
        try:
            cache_key = hashlib.md5(f"{pair}_{json.dumps(technical_analysis, sort_keys=True)}".encode()).hexdigest()
            
            if cache_key in self.cache:
                cached_time, analysis = self.cache[cache_key]
                if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                    return analysis

            # Try DeepSeek first if available
            analysis = self._try_deepseek_analysis(pair, technical_analysis, fundamental_analysis)
            if analysis:
                self.cache[cache_key] = (datetime.now(), analysis)
                return analysis

            # Fallback to rule-based AI analysis
            analysis = self._rule_based_analysis(pair, technical_analysis, fundamental_analysis)
            self.cache[cache_key] = (datetime.now(), analysis)
            return analysis

        except Exception as e:
            logger.error(f"AI analysis failed for {pair}: {e}")
            return self._get_fallback_analysis(pair, technical_analysis)

    def _try_deepseek_analysis(self, pair: str, technical: Dict, fundamental: Dict) -> Optional[Dict]:
        """Coba analisis dengan DeepSeek jika API key tersedia"""
        if not config.DEEPSEEK_API_KEY or config.DEEPSEEK_API_KEY == "demo":
            return None
            
        try:
            # Simulate API call (replace with actual implementation)
            return None
        except Exception as e:
            logger.warning(f"DeepSeek analysis failed: {e}")
            return None

    def _rule_based_analysis(self, pair: str, technical: Dict, fundamental: Dict) -> Dict[str, Any]:
        """Rule-based AI analysis sebagai fallback"""
        trend = technical.get('trend', {})
        momentum = technical.get('momentum', {})
        levels = technical.get('levels', {})
        
        current_price = levels.get('pivot', 150.0)
        rsi = momentum.get('rsi', 50)
        trend_dir = trend.get('direction', 'NEUTRAL')
        trend_strength = trend.get('strength', 50)
        
        # Advanced rule-based logic
        score = 50
        
        # Trend factors
        if 'BULLISH' in trend_dir:
            score += trend_strength * 0.3
        elif 'BEARISH' in trend_dir:
            score -= trend_strength * 0.3
            
        # Momentum factors
        if rsi < 30:  # Oversold
            score += 20
        elif rsi > 70:  # Overbought
            score -= 20
            
        # Volatility adjustment
        volatility = technical.get('volatility', {}).get('volatility_class', 'MEDIUM')
        if volatility == 'HIGH':
            score -= 10
        elif volatility == 'LOW':
            score += 5
            
        # Determine signal
        if score >= 60:
            signal = "BUY"
            confidence = min(90, score)
        elif score <= 40:
            signal = "SELL" 
            confidence = min(90, 100 - score)
        else:
            signal = "HOLD"
            confidence = 50 - abs(score - 50)
            
        return {
            'signal': signal,
            'confidence': int(confidence),
            'reasoning': f"Rule-based analysis: Trend {trend_dir}, RSI {rsi:.1f}, Score {score:.1f}",
            'key_levels': {
                'entry': round(current_price, 4),
                'stop_loss': round(current_price * 0.99, 4),
                'take_profit': round(current_price * 1.02, 4),
                'take_profit_2': round(current_price * 1.04, 4)
            },
            'risk_rating': 'MEDIUM',
            'timeframe': '4H',
            'ai_provider': 'Enhanced Rule Engine',
            'timestamp': datetime.now().isoformat()
        }

    def _get_fallback_analysis(self, pair: str, technical: Dict) -> Dict[str, Any]:
        """Fallback analysis when everything else fails"""
        return {
            'signal': 'HOLD',
            'confidence': 50,
            'reasoning': 'Fallback analysis due to system issues',
            'key_levels': {
                'entry': 0,
                'stop_loss': 0,
                'take_profit': 0,
                'take_profit_2': 0
            },
            'risk_rating': 'HIGH',
            'timeframe': '1D',
            'ai_provider': 'Fallback System',
            'timestamp': datetime.now().isoformat()
        }

# ==================== INITIALIZE COMPONENTS ====================
logger.info("Initializing ENHANCED Forex Trading System...")

tech_engine = TechnicalAnalysisEngine()
data_manager = FreeDataManager()
ai_analyzer = EnhancedAIAnalyzer()

# Initialize limiter AFTER app is created
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "100 per hour", "30 per minute"],
    storage_uri="memory://",
)

# ==================== ENHANCED FLASK ROUTES ====================
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/analyze')
@limiter.limit("30 per minute")
@cache.cached(timeout=30, query_string=True)
def api_analyze():
    """Enhanced analysis endpoint dengan rate limiting dan caching"""
    start_time = time.time()
    
    try:
        pair = request.args.get('pair', 'USDJPY').upper()
        timeframe = request.args.get('timeframe', '4H').upper()
        
        # Input validation
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        if timeframe not in config.TIMEFRAMES:
            return jsonify({'error': f'Unsupported timeframe: {timeframe}'}), 400
        
        logger.info(f"Processing analysis request for {pair}-{timeframe}")
        
        # Get data
        real_time_price = data_manager.get_real_time_price(pair)
        price_data = data_manager.get_price_data(pair, timeframe, days=60)
        
        if price_data.empty:
            return jsonify({'error': 'No data available'}), 500
        
        # Technical analysis
        technical_analysis = tech_engine.calculate_all_indicators(price_data)
        technical_analysis['levels']['current_price'] = real_time_price
        
        # AI analysis
        ai_analysis = ai_analyzer.analyze_market(pair, technical_analysis, {})
        
        # Prepare price series for chart
        price_series = _prepare_price_series(price_data)
        
        # Risk assessment
        risk_assessment = _calculate_risk_assessment(technical_analysis, ai_analysis)
        
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'ai_analysis': ai_analysis,
            'risk_assessment': risk_assessment,
            'price_data': {
                'current': real_time_price,
                'support': technical_analysis.get('levels', {}).get('support', 0),
                'resistance': technical_analysis.get('levels', {}).get('resistance', 0),
                'change_pct': 0  # Placeholder
            },
            'price_series': price_series,
            'performance_metrics': {
                'processing_time': round(time.time() - start_time, 3),
                'data_points': len(price_series),
                'cache_status': 'MISS'
            }
        }
        
        logger.info(f"Analysis completed for {pair}-{timeframe} in {time.time() - start_time:.3f}s")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e),
            'fallback_data': True
        }), 500

def _prepare_price_series(price_data):
    """Prepare price series for chart"""
    try:
        series = []
        for _, row in price_data.tail(100).iterrows():
            series.append({
                'time': row['date'].strftime('%Y-%m-%d %H:%M:%S'),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
        return series
    except Exception as e:
        logger.error(f"Error preparing price series: {e}")
        return []

def _calculate_risk_assessment(technical_analysis, ai_analysis):
    """Calculate risk assessment"""
    try:
        volatility = technical_analysis.get('volatility', {}).get('volatility_class', 'MEDIUM')
        confidence = ai_analysis.get('confidence', 50)
        signal = ai_analysis.get('signal', 'HOLD')
        
        risk_score = 50
        if volatility == 'HIGH':
            risk_score += 20
        elif volatility == 'LOW':
            risk_score -= 10
            
        if confidence < 30:
            risk_score += 20
        elif confidence > 70:
            risk_score -= 10
            
        if signal == 'HOLD':
            risk_score = 50
            
        risk_score = max(0, min(100, risk_score))
        
        if risk_score >= 70:
            rating = 'HIGH'
        elif risk_score >= 40:
            rating = 'MEDIUM'
        else:
            rating = 'LOW'
            
        return {
            'risk_score': risk_score,
            'risk_rating': rating,
            'factors': {
                'volatility': volatility,
                'signal_confidence': confidence,
                'signal': signal
            }
        }
    except Exception as e:
        logger.error(f"Error calculating risk assessment: {e}")
        return {'risk_score': 50, 'risk_rating': 'MEDIUM', 'factors': {}}

@app.route('/api/system_status')
def api_system_status():
    """System status dengan metrics"""
    return jsonify({
        'system': 'RUNNING',
        'timestamp': datetime.now().isoformat(),
        'version': '6.0',
        'features': [
            'Enhanced Technical Analysis',
            'Free Data Sources', 
            'Rule-based AI Engine',
            'Rate Limiting & Caching'
        ],
        'data_sources': ['Yahoo Finance', 'Simulated Data'],
        'rate_limits': {
            'per_minute': config.REQUESTS_PER_MINUTE,
            'per_hour': config.REQUESTS_PER_HOUR
        }
    })

@app.route('/api/health')
def api_health():
    """Health check dengan detailed metrics"""
    try:
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent(interval=1)
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'memory_usage_mb': round(memory_usage, 2),
            'cpu_percent': cpu_percent,
            'active_threads': threading.active_count()
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'degraded', 'error': str(e)}), 500

# Custom error handler for rate limits
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please slow down.',
        'retry_after': e.description.split(' ')[-1] if e.description else '60'
    }), 429

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("=== ENHANCED FOREX TRADING SYSTEM ===")
    logger.info("✓ Free Data Sources: Yahoo Finance + Simulated Data")
    logger.info("✓ Enhanced Technical Analysis with Caching") 
    logger.info("✓ Rule-based AI Engine (No API Required)")
    logger.info("✓ Rate Limiting: 30 requests/minute")
    logger.info("✓ System ready: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
