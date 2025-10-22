# [FILE: app.py] - FOREX TRADING ANALYSIS SYSTEM v4.0
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
import yfinance as yf
import random
import time

# ==================== KONFIGURASI LOGGING YANG DIPERBAIKI ====================
def setup_logging():
    """Setup logging yang compatible dengan Windows"""
    logger = logging.getLogger()
    
    # Hapus handler lama jika ada
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter tanpa emoji untuk Windows compatibility
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Stream handler untuk console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # File handler untuk log file
    file_handler = logging.FileHandler('forex_trading.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Add handlers ke root logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger

# Setup logging
logger = setup_logging()

app = Flask(__name__)
CORS(app)  # Enable CORS untuk semua route
app.secret_key = os.environ.get('SECRET_KEY', 'forex-secure-key-2024')

# ==================== KONFIGURASI SISTEM YANG DIPERBAIKI ====================
@dataclass
class SystemConfig:
    # API Configuration
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "demo")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "demo") 
    TWELVE_DATA_KEY: str = os.environ.get("TWELVE_DATA_KEY", "demo")
    
    # Enhanced Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02  # 2% risk per trade
    MAX_DAILY_LOSS: float = 0.03  # 3% max daily loss
    MAX_DRAWDOWN: float = 0.10    # 10% max drawdown
    MAX_POSITIONS: int = 3
    STOP_LOSS_PCT: float = 0.01
    TAKE_PROFIT_PCT: float = 0.02
    
    # Risk Management Parameters
    CORRELATION_THRESHOLD: float = 0.7
    VOLATILITY_THRESHOLD: float = 0.02
    DAILY_TRADE_LIMIT: int = 50  # Increased for backtesting
    MAX_POSITION_SIZE_PCT: float = 0.05  # 5% max per position
    
    # Backtesting-specific parameters
    BACKTEST_DAILY_TRADE_LIMIT: int = 100  # Higher limit for backtesting
    BACKTEST_MIN_CONFIDENCE: int = 40  # Lower confidence threshold for backtesting
    BACKTEST_RISK_SCORE_THRESHOLD: int = 8  # Higher risk tolerance for backtesting
    
    # Supported Instruments - SEMUA 10 PASANGAN FOREX
    FOREX_PAIRS: List[str] = field(default_factory=lambda: [
        "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", 
        "USDCAD", "NZDUSD", "EURJPY", "GBPJPY", "CHFJPY"
    ])
    
    # Crypto pairs tambahan
    CRYPTO_PAIRS: List[str] = field(default_factory=lambda: [
        "BTCUSD", "ETHUSD", "XRPUSD", "LTCUSD", "BCHUSD"
    ])
    
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M5", "M15", "M30", "1H", "4H", "1D", "1W"])
    
    # EURUSD Specific Configuration
    EURUSD_SPECIFIC = {
        'volatility_adjustment': 0.8,
        'preferred_timeframes': ['1H', '4H', '1D'],
        'trading_hours_priority': [(8, 11), (13, 16)],  # London-NY overlap
        'news_sensitivity': 'HIGH',
        'correlation_pairs': ['GBPUSD', 'USDCHF', 'GBPJPY', 'USDJPY'],
        'typical_range_pct': 0.007,
        'support_resistance_levels': {
            'major_support': [1.0650, 1.0750, 1.0850],
            'major_resistance': [1.0950, 1.1050, 1.1150],
            'psychological_levels': [1.0500, 1.0750, 1.1000, 1.1250]
        }
    }
    
    # JPY Pairs Specific Configuration
    JPY_PAIRS_SPECIFIC = {
        'volatility_adjustment': 1.2,
        'preferred_timeframes': ['1H', '4H'],
        'trading_hours_priority': [(0, 3), (8, 11)],  # Asian + London sessions
        'news_sensitivity': 'HIGH',
        'typical_range_pct': 0.012,
        'intervention_levels': {
            'USDJPY': [145.00, 150.00],
            'EURJPY': [155.00, 160.00],
            'GBPJPY': [185.00, 190.00]
        }
    }
    
    # Backtesting
    DEFAULT_BACKTEST_DAYS: int = 90
    MIN_DATA_POINTS: int = 100
    
    # Trading Hours (UTC)
    HIGH_IMPACT_HOURS: List[Tuple[int, int]] = field(default_factory=lambda: [(8, 10), (13, 15)])

config = SystemConfig()

# ==================== ENGINE ANALISIS TEKNIKAL YANG DIPERBAIKI ====================
class TechnicalAnalysisEngine:
    def __init__(self):
        self.indicators = {}
        self.special_patterns = {}
        logger.info("Technical Analysis Engine initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Menghitung semua indikator teknikal dari DataFrame OHLC dengan error handling"""
        try:
            # PERBAIKAN: Pastikan DataFrame tidak kosong dan memiliki kolom required
            if df.empty:
                return self._fallback_indicators(df)
                
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return self._fallback_indicators(df)
            
            if len(df) < 20:
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
            
            # PERBAIKAN: Gunakan try-except untuk setiap indikator
            try:
                # Trend Indicators
                sma_20 = talib.SMA(closes, timeperiod=20)
                sma_50 = talib.SMA(closes, timeperiod=50)
                ema_12 = talib.EMA(closes, timeperiod=12)
                ema_26 = talib.EMA(closes, timeperiod=26)
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
            except Exception as e:
                logger.warning(f"Error calculating trend indicators: {e}")
                sma_20 = closes
                sma_50 = closes
                ema_12 = closes
                ema_26 = closes
                adx = np.full_like(closes, 25)
            
            try:
                # Momentum Indicators
                rsi = talib.RSI(closes, timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(closes)
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
                williams_r = talib.WILLR(highs, lows, closes, timeperiod=14)
            except Exception as e:
                logger.warning(f"Error calculating momentum indicators: {e}")
                rsi = np.full_like(closes, 50)
                macd, macd_signal, macd_hist = np.zeros_like(closes), np.zeros_like(closes), np.zeros_like(closes)
                stoch_k, stoch_d = np.full_like(closes, 50), np.full_like(closes, 50)
                williams_r = np.full_like(closes, -50)
            
            try:
                # Volatility Indicators
                bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(closes)
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
            except Exception as e:
                logger.warning(f"Error calculating volatility indicators: {e}")
                bollinger_upper = closes * 1.02
                bollinger_lower = closes * 0.98
                bollinger_middle = closes
                atr = np.full_like(closes, closes[-1] * 0.005)
            
            # Support & Resistance menggunakan recent highs/lows
            lookback_period = min(50, len(highs))
            recent_high = np.max(highs[-lookback_period:]) if len(highs) >= lookback_period else np.max(highs)
            recent_low = np.min(lows[-lookback_period:]) if len(lows) >= lookback_period else np.min(lows)
            
            # Handle NaN values dengan aman
            def safe_float(value, default=0):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
            
            current_price = safe_float(closes[-1], 150.0)
            
            # Calculate additional metrics
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
                    'volatility_pct': safe_float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100, 1.0) if len(closes) >= 20 else 1.0,
                    'bollinger_bandwidth': safe_float((bollinger_upper[-1] - bollinger_lower[-1]) / bollinger_middle[-1] * 100, 2.0)
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
        """Fallback indicators jika TA-Lib gagal"""
        try:
            if len(df) > 0 and 'close' in df.columns:
                closes = df['close'].values
                current_price = float(closes[-1]) if len(closes) > 0 else 150.0
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
                'volatility_pct': 1.0,
                'bollinger_bandwidth': 4.0
            },
            'levels': {
                'support': current_price * 0.99,
                'resistance': current_price * 1.01,
                'current_price': current_price,
                'pivot_point': current_price
            }
        }

    def calculate_pair_specific_analysis(self, pair: str, df: pd.DataFrame) -> Dict:
        """Analisis teknikal khusus untuk pair tertentu"""
        try:
            if df.empty or len(df) < 50:
                return {}
                
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Pair-specific patterns
            if pair == 'EURUSD':
                analysis = self._calculate_eurusd_specific_analysis(df)
            elif pair in ['USDJPY', 'EURJPY', 'GBPJPY', 'CHFJPY']:
                analysis = self._calculate_jpy_pairs_specific_analysis(pair, df)
            elif pair in ['GBPUSD', 'USDCHF']:
                analysis = self._calculate_major_pairs_specific_analysis(pair, df)
            elif pair in ['AUDUSD', 'USDCAD', 'NZDUSD']:
                analysis = self._calculate_commodity_pairs_specific_analysis(pair, df)
            else:
                analysis = self._calculate_general_analysis(df)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Pair specific analysis error for {pair}: {e}")
            return {}
    
    def _calculate_eurusd_specific_analysis(self, df: pd.DataFrame) -> Dict:
        """Analisis khusus EURUSD"""
        return {
            'london_newyork_overlap': self._analyze_session_overlap(df),
            'european_us_news_impact': self._analyze_news_impact_sensitivity(df),
            'correlation_strength': self._calculate_eurusd_correlation_strength(df),
            'pivot_levels': self._calculate_eurusd_pivot_levels(df),
            'market_hours_recommendation': 'Trade during London-NY overlap (8:00-11:00 EST)',
            'news_sensitivity': 'HIGH'
        }
    
    def _calculate_jpy_pairs_specific_analysis(self, pair: str, df: pd.DataFrame) -> Dict:
        """Analisis khusus JPY pairs"""
        return {
            'asian_session_analysis': self._analyze_asian_session(df),
            'bank_of_japan_impact': 'HIGH',
            'intervention_risk': self._assess_intervention_risk(pair, df),
            'carry_trade_analysis': self._analyze_carry_trade_conditions(df['close'].values),
            'market_hours_recommendation': 'Trade during Asian session or London-Asian overlap',
            'volatility_note': 'High volatility during Tokyo session'
        }
    
    def _calculate_major_pairs_specific_analysis(self, pair: str, df: pd.DataFrame) -> Dict:
        """Analisis khusus major pairs"""
        return {
            'london_session_analysis': self._analyze_london_session(df),
            'correlation_analysis': self._calculate_major_pairs_correlation(pair),
            'volatility_profile': self._analyze_volatility_profile(df),
            'market_hours_recommendation': 'Trade during London session for best liquidity'
        }
    
    def _calculate_commodity_pairs_specific_analysis(self, pair: str, df: pd.DataFrame) -> Dict:
        """Analisis khusus commodity pairs"""
        commodity_map = {
            'AUDUSD': 'Gold and mining',
            'USDCAD': 'Oil prices', 
            'NZDUSD': 'Dairy and agriculture'
        }
        
        return {
            'commodity_correlation': commodity_map.get(pair, 'General commodities'),
            'asian_pacific_session': self._analyze_asian_pacific_session(df),
            'risk_sentiment_analysis': self._analyze_risk_sentiment(df),
            'market_hours_recommendation': 'Trade during Asian-Pacific session overlap'
        }
    
    def _analyze_session_overlap(self, df: pd.DataFrame) -> Dict:
        """Analisis performa selama overlap session"""
        try:
            volatility_by_hour = {}
            if 'date' in df.columns:
                df['hour'] = pd.to_datetime(df['date']).dt.hour
                for hour in range(24):
                    hour_data = df[df['hour'] == hour]
                    if not hour_data.empty:
                        volatility = hour_data['close'].pct_change().std() * 100
                        volatility_by_hour[hour] = round(volatility, 4)
            
            return {
                'high_volatility_hours': [k for k, v in volatility_by_hour.items() if v > 0.05],
                'volatility_profile': volatility_by_hour,
                'recommended_trading_hours': [8, 9, 10, 14, 15]  # London open & NY overlap
            }
        except Exception as e:
            logger.warning(f"Session overlap analysis error: {e}")
            return {}
    
    def _analyze_news_impact_sensitivity(self, df: pd.DataFrame) -> Dict:
        """Analisis sensitivitas terhadap news"""
        return {
            'high_impact_events': ['NFP', 'ECB Rate Decision', 'FOMC', 'CPI'],
            'sensitivity_score': 85,
            'typical_reaction_pct': 0.008,
            'recovery_time_hours': 4
        }
    
    def _analyze_asian_session(self, df: pd.DataFrame) -> Dict:
        """Analisis performa Asian session"""
        return {
            'active_hours': [0, 1, 2, 3, 4, 5, 6],
            'volatility_characteristics': 'Breakout oriented',
            'recommended_pairs': ['USDJPY', 'EURJPY', 'AUDUSD']
        }
    
    def _assess_intervention_risk(self, pair: str, df: pd.DataFrame) -> Dict:
        """Assess Bank of Japan intervention risk"""
        intervention_levels = {
            'USDJPY': {'high_risk': 150.00, 'medium_risk': 148.00},
            'EURJPY': {'high_risk': 160.00, 'medium_risk': 158.00},
            'GBPJPY': {'high_risk': 190.00, 'medium_risk': 188.00}
        }
        
        current_price = df['close'].iloc[-1] if not df.empty else 0
        levels = intervention_levels.get(pair, {})
        
        risk = 'LOW'
        if current_price >= levels.get('high_risk', 200):
            risk = 'HIGH'
        elif current_price >= levels.get('medium_risk', 180):
            risk = 'MEDIUM'
            
        return {
            'intervention_risk': risk,
            'current_price': current_price,
            'risk_levels': levels
        }
    
    def _analyze_carry_trade_conditions(self, closes: np.array) -> Dict:
        """Analisis kondisi carry trade"""
        try:
            if len(closes) < 20:
                return {}
                
            price_trend = "BULLISH" if closes[-1] > closes[-20] else "BEARISH"
            volatility = np.std(np.diff(closes[-20:]) / closes[-21:-1])
            
            return {
                'carry_attractiveness': 'MEDIUM',
                'trend_alignment': price_trend,
                'volatility_environment': 'HIGH' if volatility > 0.0008 else 'MEDIUM',
                'risk_appetite_indicator': 'NEUTRAL'
            }
        except Exception as e:
            logger.warning(f"Carry trade analysis error: {e}")
            return {}
    
    def _calculate_eurusd_correlation_strength(self, df: pd.DataFrame) -> Dict:
        """Hitung strength correlation untuk EURUSD"""
        return {
            'usd_index': -0.85,
            'gbpusd': 0.75,
            'usdchf': -0.80,
            'gold': 0.45,
            'us10y': 0.60
        }
    
    def _calculate_major_pairs_correlation(self, pair: str) -> Dict:
        """Hitung correlation untuk major pairs"""
        correlations = {
            'GBPUSD': {'eurusd': 0.75, 'usdchf': -0.60, 'usdjpy': -0.55},
            'USDCHF': {'eurusd': -0.80, 'gbpusd': -0.60, 'usdjpy': 0.85}
        }
        return correlations.get(pair, {})
    
    def _analyze_volatility_profile(self, df: pd.DataFrame) -> Dict:
        """Analisis profil volatilitas"""
        try:
            if len(df) < 20:
                return {}
                
            closes = df['close'].values
            volatility = np.std(np.diff(closes) / closes[:-1]) * 100
            
            return {
                'current_volatility': round(volatility, 4),
                'volatility_class': 'HIGH' if volatility > 0.8 else 'MEDIUM' if volatility > 0.4 else 'LOW',
                'average_true_range': round(np.mean([high - low for high, low in zip(df['high'], df['low'])]), 4)
            }
        except Exception as e:
            logger.warning(f"Volatility profile analysis error: {e}")
            return {}
    
    def _analyze_london_session(self, df: pd.DataFrame) -> Dict:
        """Analisis performa London session"""
        return {
            'active_hours': [8, 9, 10, 11, 12],
            'volatility_characteristics': 'Trend continuation',
            'recommended_approach': 'Breakout trading'
        }
    
    def _analyze_asian_pacific_session(self, df: pd.DataFrame) -> Dict:
        """Analisis performa Asian-Pacific session"""
        return {
            'active_hours': [22, 23, 0, 1, 2, 3, 4, 5],
            'volatility_characteristics': 'Range bound with occasional breakouts',
            'recommended_approach': 'Range trading with tight stops'
        }
    
    def _analyze_risk_sentiment(self, df: pd.DataFrame) -> Dict:
        """Analisis risk sentiment untuk commodity pairs"""
        try:
            if len(df) < 10:
                return {}
                
            price_trend = "BULLISH" if df['close'].iloc[-1] > df['close'].iloc[-10] else "BEARISH"
            
            return {
                'risk_appetite': 'HIGH' if price_trend == "BULLISH" else 'LOW',
                'market_sentiment': 'RISK_ON' if price_trend == "BULLISH" else 'RISK_OFF',
                'correlation_with_equities': 'POSITIVE'
            }
        except Exception as e:
            logger.warning(f"Risk sentiment analysis error: {e}")
            return {}
    
    def _calculate_eurusd_pivot_levels(self, df: pd.DataFrame) -> Dict:
        """Hitung pivot levels khusus untuk EURUSD"""
        try:
            if len(df) < 5:
                return {}
                
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-1]
            
            # Standard pivot points
            pivot = (high + low + close) / 3
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': round(pivot, 4),
                'resistance_1': round(r1, 4),
                'resistance_2': round(r2, 4),
                'resistance_3': round(r3, 4),
                'support_1': round(s1, 4),
                'support_2': round(s2, 4),
                'support_3': round(s3, 4)
            }
        except Exception as e:
            logger.warning(f"Pivot levels calculation error: {e}")
            return {}
    
    def _calculate_general_analysis(self, df: pd.DataFrame) -> Dict:
        """Analisis umum untuk pairs tanpa spesifikasi khusus"""
        return {
            'general_recommendation': 'Monitor multiple timeframes for confirmation',
            'risk_management': 'Use standard position sizing',
            'trading_approach': 'Technical analysis driven'
        }

# ==================== TWELVEDATA REAL-TIME INTEGRATION ====================
class TwelveDataClient:
    def __init__(self):
        self.api_key = config.TWELVE_DATA_KEY
        self.base_url = "https://api.twelvedata.com"
        self.price_cache = {}
        self.cache_timeout = 60
        self.demo_mode = not self.api_key or self.api_key == "demo"
        
        if self.demo_mode:
            logger.info("TwelveData running in DEMO mode with simulated real-time prices")
        else:
            logger.info("TwelveData running in LIVE mode with real API data")
    
    def get_real_time_price(self, pair: str) -> float:
        """Ambil current price real-time dari TwelveData atau simulasi"""
        cache_key = f"{pair}_{datetime.now().strftime('%Y%m%d%H%M')}"
        
        # Check cache dulu
        if pair in self.price_cache:
            cached_time, price = self.price_cache[pair]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                return price
        
        # Jika demo mode, gunakan harga simulasi yang lebih realistis
        if self.demo_mode:
            return self._get_simulated_real_time_price(pair)
        
        try:
            # Format pair untuk TwelveData (USDJPY -> USD/JPY)
            formatted_pair = f"{pair[:3]}/{pair[3:]}"
            url = f"{self.base_url}/price?symbol={formatted_pair}&apikey={self.api_key}"
            
            logger.info(f"Fetching real-time price for {pair} from TwelveData...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data and data['price'] is not None:
                    price = float(data['price'])
                    
                    # Cache the price
                    self.price_cache[pair] = (datetime.now(), price)
                    logger.info(f"Real-time price for {pair}: {price}")
                    return price
                else:
                    logger.error(f"Invalid response from TwelveData: {data}")
                    return self._get_simulated_real_time_price(pair)
            else:
                logger.error(f"TwelveData API error: {response.status_code} - {response.text}")
                return self._get_simulated_real_time_price(pair)
                
        except Exception as e:
            logger.error(f"Error getting real-time price for {pair}: {e}")
            return self._get_simulated_real_time_price(pair)
    
    def _get_simulated_real_time_price(self, pair: str) -> float:
        """Harga real-time simulasi untuk demo mode"""
        try:
            # Base prices untuk SEMUA 10 pasangan forex
            base_prices = {
                'EURUSD': 1.0835, 'USDJPY': 147.25, 'GBPUSD': 1.2640, 'USDCHF': 0.8840,
                'AUDUSD': 0.6545, 'USDCAD': 1.3510, 'NZDUSD': 0.6095, 'EURJPY': 159.80,
                'GBPJPY': 186.20, 'CHFJPY': 166.75
            }
            
            base_price = base_prices.get(pair, 150.0)
            
            # Tambahkan variasi acak kecil (±0.1%) untuk simulasi pergerakan market
            variation = random.uniform(-0.001, 0.001)
            simulated_price = round(base_price * (1 + variation), 4)
            
            # Cache the price
            self.price_cache[pair] = (datetime.now(), simulated_price)
            
            logger.info(f"Simulated real-time price for {pair}: {simulated_price:.4f}")
            return simulated_price
            
        except Exception as e:
            logger.error(f"Error in simulated price for {pair}: {e}")
            return 150.0

    def get_multiple_prices(self, pairs: List[str]) -> Dict[str, float]:
        """Dapatkan harga multiple pairs sekaligus"""
        prices = {}
        for pair in pairs:
            prices[pair] = self.get_real_time_price(pair)
        return prices

# ==================== ADVANCED RISK MANAGEMENT SYSTEM ====================
class AdvancedRiskManager:
    def __init__(self, backtest_mode: bool = False):
        self.max_daily_loss_pct = config.MAX_DAILY_LOSS
        self.max_drawdown_pct = config.MAX_DRAWDOWN
        self.max_position_size_pct = config.MAX_POSITION_SIZE_PCT
        self.daily_trade_limit = config.BACKTEST_DAILY_TRADE_LIMIT if backtest_mode else config.DAILY_TRADE_LIMIT
        self.correlation_threshold = config.CORRELATION_THRESHOLD
        self.backtest_mode = backtest_mode
        
        # Trading session tracking
        self.today_trades = 0
        self.daily_pnl = 0.0
        self.peak_balance = 10000.0
        self.current_drawdown = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Correlation matrix untuk SEMUA 10 pasangan forex
        self.correlation_matrix = {
            'EURUSD': {'USDJPY': -0.7, 'GBPUSD': 0.8, 'USDCHF': -0.7, 'AUDUSD': 0.6, 'USDCAD': -0.5, 'NZDUSD': 0.5, 'EURJPY': 0.9, 'GBPJPY': 0.6, 'CHFJPY': -0.5},
            'USDJPY': {'EURUSD': -0.7, 'GBPUSD': -0.6, 'USDCHF': 0.9, 'AUDUSD': -0.5, 'USDCAD': 0.4, 'NZDUSD': -0.5, 'EURJPY': 0.8, 'GBPJPY': 0.7, 'CHFJPY': 0.6},
            'GBPUSD': {'EURUSD': 0.8, 'USDJPY': -0.6, 'USDCHF': -0.6, 'AUDUSD': 0.5, 'USDCAD': -0.4, 'NZDUSD': 0.4, 'EURJPY': 0.7, 'GBPJPY': 0.9, 'CHFJPY': -0.5},
            'USDCHF': {'EURUSD': -0.7, 'USDJPY': 0.9, 'GBPUSD': -0.6, 'AUDUSD': -0.4, 'USDCAD': 0.6, 'NZDUSD': -0.3, 'EURJPY': -0.6, 'GBPJPY': -0.5, 'CHFJPY': 0.8},
            'AUDUSD': {'EURUSD': 0.6, 'USDJPY': -0.5, 'GBPUSD': 0.5, 'USDCHF': -0.4, 'USDCAD': -0.4, 'NZDUSD': 0.8, 'EURJPY': 0.5, 'GBPJPY': 0.4, 'CHFJPY': -0.3},
            'USDCAD': {'EURUSD': -0.5, 'USDJPY': 0.4, 'GBPUSD': -0.4, 'USDCHF': 0.6, 'AUDUSD': -0.4, 'NZDUSD': -0.3, 'EURJPY': -0.4, 'GBPJPY': -0.3, 'CHFJPY': 0.5},
            'NZDUSD': {'EURUSD': 0.5, 'USDJPY': -0.5, 'GBPUSD': 0.4, 'USDCHF': -0.3, 'AUDUSD': 0.8, 'USDCAD': -0.3, 'EURJPY': 0.4, 'GBPJPY': 0.3, 'CHFJPY': -0.2},
            'EURJPY': {'EURUSD': 0.9, 'USDJPY': 0.8, 'GBPUSD': 0.7, 'USDCHF': -0.6, 'AUDUSD': 0.5, 'USDCAD': -0.4, 'NZDUSD': 0.4, 'GBPJPY': 0.8, 'CHFJPY': 0.6},
            'GBPJPY': {'EURUSD': 0.6, 'USDJPY': 0.7, 'GBPUSD': 0.9, 'USDCHF': -0.5, 'AUDUSD': 0.4, 'USDCAD': -0.3, 'NZDUSD': 0.3, 'EURJPY': 0.8, 'CHFJPY': 0.5},
            'CHFJPY': {'EURUSD': -0.5, 'USDJPY': 0.6, 'GBPUSD': -0.5, 'USDCHF': 0.8, 'AUDUSD': -0.3, 'USDCAD': 0.5, 'NZDUSD': -0.2, 'EURJPY': 0.6, 'GBPJPY': 0.5}
        }
        
        logger.info(f"Advanced Risk Manager initialized - Backtest Mode: {backtest_mode}")
    
    def reset_daily_limits(self):
        """Reset daily limits jika hari baru"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.today_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today
            logger.info("Daily risk limits reset")
    
    def validate_trade(self, pair: str, signal: str, confidence: int, 
                      proposed_lot_size: float, account_balance: float, 
                      current_price: float, open_positions: List[Dict]) -> Dict:
        """
        Validasi trade dengan multiple risk factors untuk SEMUA pasangan
        """
        self.reset_daily_limits()
        
        validation_result = {
            'approved': True,
            'adjusted_lot_size': proposed_lot_size,
            'risk_score': 0,
            'rejection_reasons': [],
            'warnings': [],
            'max_allowed_lot_size': proposed_lot_size,
            'risk_factors': {}
        }
        
        risk_factors = {}
        
        # 1. Check daily trade limit
        if self.today_trades >= self.daily_trade_limit:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(
                f"Daily trade limit reached ({self.daily_trade_limit})"
            )
            risk_factors['daily_limit'] = 'HIGH'
        
        # 2. Check daily loss limit
        daily_loss_limit = account_balance * self.max_daily_loss_pct
        if self.daily_pnl <= -daily_loss_limit:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(
                f"Daily loss limit reached (${-self.daily_pnl:.2f})"
            )
            risk_factors['daily_loss'] = 'HIGH'
        
        # 3. Check drawdown limit
        if self.current_drawdown >= self.max_drawdown_pct:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(
                f"Max drawdown reached ({self.current_drawdown:.1%})"
            )
            risk_factors['drawdown'] = 'HIGH'
        
        # 4. Position size validation
        max_position_value = account_balance * self.max_position_size_pct
        proposed_position_value = proposed_lot_size * 100000 * current_price
        
        if proposed_position_value > max_position_value:
            adjusted_lot_size = max_position_value / (100000 * current_price)
            validation_result['adjusted_lot_size'] = max(0.01, adjusted_lot_size)
            validation_result['warnings'].append(
                f"Position size reduced from {proposed_lot_size:.2f} to {validation_result['adjusted_lot_size']:.2f} lots"
            )
            validation_result['risk_score'] += 2
            risk_factors['position_size'] = 'MEDIUM'
        
        # 5. Correlation risk assessment
        correlation_risk = self._check_correlation_risk(pair, signal, open_positions)
        if correlation_risk['high_risk']:
            validation_result['risk_score'] += 3
            validation_result['warnings'].append(
                f"High correlation with {correlation_risk['correlated_pairs']}"
            )
            risk_factors['correlation'] = 'HIGH'
        
        # 6. Market volatility check
        volatility_risk = self._check_volatility_risk(pair, current_price)
        if volatility_risk['high_volatility']:
            validation_result['risk_score'] += 2
            validation_result['warnings'].append(
                f"High volatility detected: {volatility_risk['volatility_pct']:.1%}"
            )
            risk_factors['volatility'] = 'HIGH'
        
        # 7. Confidence-based risk adjustment
        min_confidence = config.BACKTEST_MIN_CONFIDENCE if self.backtest_mode else 60
        if confidence < min_confidence:
            validation_result['risk_score'] += 1
            validation_result['warnings'].append("Low confidence signal")
            risk_factors['confidence'] = 'MEDIUM'
        
        # 8. Time-based risk (avoid trading during high impact news)
        time_risk = self._check_time_risk()
        if time_risk['high_risk_period']:
            validation_result['risk_score'] += 2
            validation_result['warnings'].append(f"Trading during {time_risk['period_name']}")
            risk_factors['timing'] = 'MEDIUM'
        
        # 9. Liquidity check (avoid illiquid periods)
        liquidity_risk = self._check_liquidity_risk()
        if liquidity_risk['low_liquidity']:
            validation_result['risk_score'] += 1
            validation_result['warnings'].append("Low liquidity period")
            risk_factors['liquidity'] = 'LOW'
        
        # 10. Pair-specific risk adjustments
        pair_risk = self._check_pair_specific_risk(pair)
        if pair_risk['additional_risk']:
            validation_result['risk_score'] += pair_risk['risk_score_adjustment']
            validation_result['warnings'].append(pair_risk['warning'])
            risk_factors['pair_specific'] = pair_risk['risk_level']
        
        # Final approval decision based on risk score
        validation_result['risk_factors'] = risk_factors
        
        risk_threshold = config.BACKTEST_RISK_SCORE_THRESHOLD if self.backtest_mode else 6
        if validation_result['risk_score'] >= risk_threshold:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(
                f"Overall risk score too high: {validation_result['risk_score']}/10"
            )
        
        logger.info(f"Risk validation for {pair}-{signal}: {'APPROVED' if validation_result['approved'] else 'REJECTED'} - Score: {validation_result['risk_score']}")
        
        return validation_result
    
    def _check_pair_specific_risk(self, pair: str) -> Dict:
        """Check pair-specific risk factors"""
        # JPY pairs typically have higher volatility and intervention risk
        jpy_pairs = ['USDJPY', 'EURJPY', 'GBPJPY', 'CHFJPY']
        
        if pair in jpy_pairs:
            return {
                'additional_risk': True,
                'risk_score_adjustment': 1,
                'warning': f"{pair} has higher volatility and intervention risk",
                'risk_level': 'MEDIUM'
            }
        elif pair == 'GBPUSD':
            return {
                'additional_risk': True,
                'risk_score_adjustment': 1,
                'warning': "GBPUSD known for sudden volatility spikes",
                'risk_level': 'MEDIUM'
            }
        else:
            return {
                'additional_risk': False,
                'risk_score_adjustment': 0,
                'warning': '',
                'risk_level': 'LOW'
            }
    
    def _check_correlation_risk(self, pair: str, signal: str, open_positions: List[Dict]) -> Dict:
        """Check correlation risk dengan open positions"""
        high_risk = False
        correlated_pairs = []
        
        for position in open_positions:
            open_pair = position['pair']
            open_signal = position['signal']
            
            if open_pair in self.correlation_matrix and pair in self.correlation_matrix[open_pair]:
                correlation = self.correlation_matrix[open_pair][pair]
                
                # Jika highly correlated dan signal sama, itu meningkatkan risk
                if abs(correlation) > self.correlation_threshold and open_signal == signal:
                    high_risk = True
                    correlated_pairs.append(f"{open_pair} (corr: {correlation:.2f})")
        
        return {
            'high_risk': high_risk,
            'correlated_pairs': correlated_pairs
        }
    
    def _check_volatility_risk(self, pair: str, current_price: float) -> Dict:
        """Check volatility risk berdasarkan historical data"""
        try:
            # Get recent price data untuk volatility calculation
            price_data = data_manager.get_price_data(pair, '1H', days=5)
            if len(price_data) > 10:
                closes = price_data['close'].values
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns) * np.sqrt(24)
                
                return {
                    'high_volatility': volatility > config.VOLATILITY_THRESHOLD,
                    'volatility_pct': volatility
                }
        except Exception as e:
            logger.warning(f"Volatility calculation error for {pair}: {e}")
        
        return {'high_volatility': False, 'volatility_pct': 0.01}
    
    def _check_time_risk(self) -> Dict:
        """Check jika sedang dalam periode high impact news"""
        now = datetime.utcnow()
        current_hour = now.hour
        
        for start_hour, end_hour in config.HIGH_IMPACT_HOURS:
            if start_hour <= current_hour < end_hour:
                return {
                    'high_risk_period': True,
                    'period_name': f'High Impact Hours ({start_hour:02d}:00-{end_hour:02d}:00 UTC)'
                }
        
        return {'high_risk_period': False, 'period_name': 'Normal Hours'}
    
    def _check_liquidity_risk(self) -> Dict:
        """Check liquidity risk berdasarkan waktu trading"""
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Low liquidity periods (Asian session overlap, weekends)
        low_liquidity_periods = [
            (21, 24),  # Late NY / Early Asia
            (0, 5),    # Asian session
            (23, 24),  # Weekend start
            (0, 1)     # Weekend
        ]
        
        for start_hour, end_hour in low_liquidity_periods:
            if start_hour <= current_hour < end_hour:
                return {'low_liquidity': True}
        
        # Weekend check
        if now.weekday() >= 5:
            return {'low_liquidity': True}
        
        return {'low_liquidity': False}
    
    def update_trade_result(self, pnl: float, trade_success: bool):
        """Update risk manager dengan hasil trade"""
        self.daily_pnl += pnl
        self.today_trades += 1
        
        # Update drawdown calculation
        if pnl < 0:
            self.current_drawdown = abs(self.daily_pnl) / self.peak_balance if self.peak_balance > 0 else 0
        else:
            if self.daily_pnl > self.peak_balance:
                self.peak_balance = self.daily_pnl
        
        logger.info(f"Trade result: PnL ${pnl:.2f}, Daily PnL: ${self.daily_pnl:.2f}, Trades today: {self.today_trades}")
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        daily_loss_limit = self.peak_balance * self.max_daily_loss_pct
        
        return {
            'daily_metrics': {
                'trades_today': self.today_trades,
                'daily_pnl': round(self.daily_pnl, 2),
                'daily_trade_limit': self.daily_trade_limit,
                'remaining_trades': max(0, self.daily_trade_limit - self.today_trades),
                'daily_loss_limit': round(daily_loss_limit, 2),
                'remaining_daily_loss': round(daily_loss_limit + self.daily_pnl, 2)
            },
            'risk_limits': {
                'max_daily_loss_pct': self.max_daily_loss_pct,
                'max_drawdown_pct': self.max_drawdown_pct,
                'max_position_size_pct': self.max_position_size_pct,
                'current_drawdown_pct': round(self.current_drawdown * 100, 2)
            },
            'warnings': self._generate_risk_warnings(),
            'current_risk_level': self._calculate_overall_risk_level()
        }
    
    def _generate_risk_warnings(self) -> List[str]:
        """Generate risk warnings berdasarkan current state"""
        warnings = []
        
        if self.today_trades >= self.daily_trade_limit * 0.8:
            warnings.append(f"Approaching daily trade limit ({self.today_trades}/{self.daily_trade_limit})")
        
        daily_loss_limit = self.peak_balance * self.max_daily_loss_pct
        if self.daily_pnl < 0 and abs(self.daily_pnl) > (daily_loss_limit * 0.7):
            warnings.append(f"Approaching daily loss limit (${self.daily_pnl:.2f}/${daily_loss_limit:.2f})")
        
        if self.current_drawdown > self.max_drawdown_pct * 0.8:
            warnings.append(f"Approaching maximum drawdown ({self.current_drawdown:.1%}/{self.max_drawdown_pct:.1%})")
        
        # Market hours warning
        time_risk = self._check_time_risk()
        if time_risk['high_risk_period']:
            warnings.append(f"Trading during {time_risk['period_name']}")
        
        return warnings
    
    def _calculate_overall_risk_level(self) -> str:
        """Calculate overall risk level"""
        risk_score = 0
        
        # Daily trades usage
        trade_usage = self.today_trades / self.daily_trade_limit
        if trade_usage > 0.8:
            risk_score += 3
        elif trade_usage > 0.6:
            risk_score += 2
        elif trade_usage > 0.4:
            risk_score += 1
        
        # Daily PnL risk
        daily_loss_limit = self.peak_balance * self.max_daily_loss_pct
        loss_usage = abs(self.daily_pnl) / daily_loss_limit if self.daily_pnl < 0 else 0
        if loss_usage > 0.8:
            risk_score += 3
        elif loss_usage > 0.6:
            risk_score += 2
        elif loss_usage > 0.4:
            risk_score += 1
        
        # Drawdown risk
        drawdown_usage = self.current_drawdown / self.max_drawdown_pct
        if drawdown_usage > 0.8:
            risk_score += 3
        elif drawdown_usage > 0.6:
            risk_score += 2
        elif drawdown_usage > 0.4:
            risk_score += 1
        
        # Time risk
        time_risk = self._check_time_risk()
        if time_risk['high_risk_period']:
            risk_score += 2
        
        if risk_score >= 7:
            return "HIGH"
        elif risk_score >= 4:
            return "MEDIUM"
        else:
            return "LOW"

# ... (Lanjutkan dengan bagian-bagian lainnya yang sama)

# ==================== INITIALIZE SYSTEM ====================
logger.info("Initializing Forex Analysis System with ALL 10 Forex Pairs...")

# Inisialisasi komponen sistem dengan urutan yang benar
tech_engine = TechnicalAnalysisEngine()
# fundamental_engine = FundamentalAnalysisEngine()
# deepseek_analyzer = DeepSeekAnalyzer()
data_manager = DataManager()
twelve_data_client = TwelveDataClient()

# Tampilkan status yang lebih informatif
logger.info(f"ALL 10 Forex Pairs: ENABLED")
logger.info(f"Supported pairs: {config.FOREX_PAIRS}")
logger.info(f"Crypto pairs: {config.CRYPTO_PAIRS}")
logger.info(f"Total instruments: {len(config.FOREX_PAIRS) + len(config.CRYPTO_PAIRS)}")
logger.info(f"Historical data: {len(data_manager.historical_data)} pairs loaded")
logger.info(f"TwelveData Real-time: {'LIVE MODE' if not twelve_data_client.demo_mode else 'DEMO MODE'}")
logger.info(f"Advanced Risk Management: ENABLED")
logger.info(f"Enhanced Backtesting: ENABLED")

logger.info("All system components initialized successfully")

# ==================== GLOBAL VARIABLES ====================
risk_manager = AdvancedRiskManager()
# advanced_backtester = AdvancedBacktestingEngine()

# ==================== FLASK ROUTES YANG DIPERBAIKI ====================
@app.route('/')
def index():
    return render_template('index.html', 
                         pairs=config.FOREX_PAIRS,
                         crypto_pairs=config.CRYPTO_PAIRS,
                         timeframes=config.TIMEFRAMES,
                         initial_balance=config.INITIAL_BALANCE)

@app.route('/api/analyze')
def api_analyze():
    """Endpoint untuk analisis market real-time untuk SEMUA pasangan"""
    try:
        pair = request.args.get('pair', 'EURUSD').upper()
        timeframe = request.args.get('timeframe', '4H').upper()
        
        # Validasi pair - support semua 10 forex pairs
        if pair not in config.FOREX_PAIRS and pair not in config.CRYPTO_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}. Supported: {config.FOREX_PAIRS + config.CRYPTO_PAIRS}'}), 400
        
        # 1) Ambil harga realtime (TwelveData)
        real_time_price = twelve_data_client.get_real_time_price(pair)
        
        # 2) Ambil data harga historis
        price_data = data_manager.get_price_data(pair, timeframe, days=60)
        if price_data.empty:
            logger.warning(f"No price data for {pair}-{timeframe}, generating sample data")
            data_manager._generate_sample_data(pair, timeframe)
            price_data = data_manager.get_price_data(pair, timeframe, days=60)
        
        # 3) Analisis teknikal
        technical_analysis = tech_engine.calculate_all_indicators(price_data)
        
        # Override current price dengan realtime untuk konsistensi
        technical_analysis['levels']['current_price'] = real_time_price
        
        # 4) Analisis khusus pair
        pair_specific_analysis = tech_engine.calculate_pair_specific_analysis(pair, price_data)
        
        # 5) Siapkan price_series
        price_series = []
        try:
            hist_df = data_manager.get_price_data(pair, timeframe, days=200)
            if not hist_df.empty:
                hist_df = hist_df.sort_values('date')
                
                for _, row in hist_df.iterrows():
                    date_value = row['date']
                    
                    if hasattr(date_value, 'isoformat'):
                        date_str = date_value.isoformat()
                    else:
                        date_str = str(date_value)
                    
                    price_series.append({
                        'date': date_str,
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': int(row['volume']) if 'volume' in row and not pd.isna(row['volume']) else 0
                    })
                    
                logger.info(f"Prepared {len(price_series)} price series records for {pair}-{timeframe}")
                
        except Exception as e:
            logger.error(f"Error preparing price series for {pair}-{timeframe}: {e}")
            price_series = []
        
        # 6) Susun response
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'pair_specific_analysis': pair_specific_analysis,
            'price_data': {
                'current': real_time_price,
                'support': technical_analysis.get('levels', {}).get('support'),
                'resistance': technical_analysis.get('levels', {}).get('resistance'),
                'change_pct': technical_analysis.get('momentum', {}).get('price_change_pct', 0)
            },
            'price_series': price_series,
            'analysis_summary': f"{pair} currently trading at {real_time_price:.4f}",
            'data_source': 'TwelveData Live' if not twelve_data_client.demo_mode else 'TwelveData Demo',
            'pair_type': 'FOREX' if pair in config.FOREX_PAIRS else 'CRYPTO'
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/market_overview')
def api_market_overview():
    """Overview market untuk SEMUA 10 pasangan forex"""
    overview = {}
    
    for pair in config.FOREX_PAIRS:  # SEMUA 10 pasangan
        try:
            # Dapatkan current price REAL-TIME dari TwelveData
            real_time_price = twelve_data_client.get_real_time_price(pair)
            
            price_data = data_manager.get_price_data(pair, '1H', days=3)
            
            if not price_data.empty:
                tech = tech_engine.calculate_all_indicators(price_data)
                
                # Calculate price change
                change_pct = 0
                if len(price_data) > 1 and 'close' in price_data.columns:
                    try:
                        prev_price = float(price_data['close'].iloc[-2])
                        change_pct = ((real_time_price - prev_price) / prev_price) * 100
                    except:
                        change_pct = 0
                
                # Determine trading recommendation
                rsi = tech['momentum']['rsi']
                trend = tech['trend']['trend_direction']
                
                if rsi < 35 and trend == 'BULLISH':
                    recommendation = 'BUY'
                    confidence = 'HIGH'
                elif rsi > 65 and trend == 'BEARISH':
                    recommendation = 'SELL'
                    confidence = 'HIGH'
                elif abs(change_pct) > 1.0:
                    recommendation = 'AVOID'
                    confidence = 'MEDIUM'
                else:
                    recommendation = 'HOLD'
                    confidence = 'LOW'
                
                overview[pair] = {
                    'price': real_time_price,
                    'change': round(float(change_pct), 2),
                    'rsi': float(tech['momentum']['rsi']),
                    'trend': tech['trend']['trend_direction'],
                    'trend_strength': tech['trend']['trend_strength'],
                    'volatility': round(tech['volatility']['volatility_pct'], 2),
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'support': tech['levels']['support'],
                    'resistance': tech['levels']['resistance'],
                    'data_source': 'TwelveData Live' if not twelve_data_client.demo_mode else 'TwelveData Demo'
                }
            else:
                overview[pair] = {
                    'price': real_time_price,
                    'change': 0,
                    'rsi': 50,
                    'trend': 'UNKNOWN',
                    'trend_strength': 'UNKNOWN',
                    'volatility': 0,
                    'recommendation': 'HOLD',
                    'confidence': 'LOW',
                    'error': 'No historical data available'
                }
        except Exception as e:
            logger.error(f"Error getting overview for {pair}: {e}")
            overview[pair] = {
                'price': 0,
                'change': 0,
                'rsi': 50,
                'trend': 'UNKNOWN',
                'trend_strength': 'UNKNOWN',
                'volatility': 0,
                'recommendation': 'HOLD',
                'confidence': 'LOW',
                'error': str(e)
            }
    
    return jsonify(overview)

@app.route('/api/system_status')
def api_system_status():
    """Status sistem dan ketersediaan semua pasangan"""
    return jsonify({
        'system': 'RUNNING',
        'forex_pairs': config.FOREX_PAIRS,
        'crypto_pairs': config.CRYPTO_PAIRS,
        'total_instruments': len(config.FOREX_PAIRS) + len(config.CRYPTO_PAIRS),
        'timeframes': config.TIMEFRAMES,
        'historical_data': f"{len(data_manager.historical_data)} pairs loaded",
        'twelve_data': 'LIVE MODE' if not twelve_data_client.demo_mode else 'DEMO MODE',
        'risk_management': 'ADVANCED',
        'server_time': datetime.now().isoformat(),
        'version': '4.0',
        'features': [
            '10 Major Forex Pairs',
            '5 Crypto Pairs', 
            'Advanced Technical Analysis',
            'Real-time Price Data',
            'Pair-specific Analysis',
            'Risk Management',
            'Multiple Timeframes'
        ]
    })

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("Starting Forex Analysis System Flask Server...")
    logger.info(f"Available pairs: {config.FOREX_PAIRS}")
    logger.info(f"Available crypto: {config.CRYPTO_PAIRS}")
    app.run(debug=True, host='0.0.0.0', port=5000)
