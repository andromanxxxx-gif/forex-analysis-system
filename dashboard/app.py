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
import yfinance as yf
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
    # API Configuration
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "demo")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "demo") 
    TWELVE_DATA_KEY: str = os.environ.get("TWELVE_DATA_KEY", "demo")
    
    # Enhanced Trading Parameters - LEBIH REALISTIS DAN KONSERVATIF
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.01  # Reduced from 0.02
    MAX_DAILY_LOSS: float = 0.015 # Reduced from 0.02
    MAX_DRAWDOWN: float = 0.03    # Reduced from 0.05
    MAX_POSITIONS: int = 2        # Reduced from 3
    STOP_LOSS_PCT: float = 0.008  # Reduced from 0.01
    TAKE_PROFIT_PCT: float = 0.02 # Reduced from 0.03
    
    # Risk Management Parameters - LEBIH KETAT
    CORRELATION_THRESHOLD: float = 0.7
    VOLATILITY_THRESHOLD: float = 0.025  # Reduced from 0.035
    DAILY_TRADE_LIMIT: int = 15          # Reduced from 25
    MAX_POSITION_SIZE_PCT: float = 0.03  # Reduced from 0.04
    
    # Signal Quality Parameters - LEBIH KETAT
    MIN_CONFIDENCE: int = 72             # Increased from 65
    MIN_RSI_BUY: int = 35                # Stricter RSI for BUY
    MAX_RSI_SELL: int = 65               # Stricter RSI for SELL
    MIN_ADX: int = 20                    # Minimum trend strength
    REQUIRED_CONDITIONS: int = 2         # Minimum conditions to be met
    
    # Backtesting-specific parameters - LEBIH REALISTIS
    BACKTEST_DAILY_TRADE_LIMIT: int = 20
    BACKTEST_MIN_CONFIDENCE: int = 72    # Increased from 65
    BACKTEST_RISK_SCORE_THRESHOLD: int = 5 # Reduced from 6
    
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

# ==================== ENGINE ANALISIS TEKNIKAL YANG DIPERBAIKI ====================
class TechnicalAnalysisEngine:
    def __init__(self):
        self.indicators = {}
        logger.info("Technical Analysis Engine initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Menghitung semua indikator teknikal termasuk EMA 20, 50, 200"""
        try:
            if df.empty:
                return self._fallback_indicators(df)
                
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return self._fallback_indicators(df)
            
            if len(df) < 50:
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
            
            # PERBAIKAN: Tambahkan EMA 20, 50, 200
            try:
                # Trend Indicators
                sma_20 = talib.SMA(closes, timeperiod=20)
                sma_50 = talib.SMA(closes, timeperiod=50)
                ema_12 = talib.EMA(closes, timeperiod=12)
                ema_26 = talib.EMA(closes, timeperiod=26)
                ema_20 = talib.EMA(closes, timeperiod=20)
                ema_50 = talib.EMA(closes, timeperiod=50)
                ema_200 = talib.EMA(closes, timeperiod=200)
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
            except Exception as e:
                logger.warning(f"Error calculating trend indicators: {e}")
                sma_20 = closes
                sma_50 = closes
                ema_12 = closes
                ema_26 = closes
                ema_20 = closes
                ema_50 = closes
                ema_200 = closes
                adx = np.full_like(closes, 25)
            
            try:
                # Momentum Indicators
                rsi = talib.RSI(closes, timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(closes)
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
                williams_r = talib.WILLR(highs, lows, closes, timeperiod=14)
                cci = talib.CCI(highs, lows, closes, timeperiod=14)
            except Exception as e:
                logger.warning(f"Error calculating momentum indicators: {e}")
                rsi = np.full_like(closes, 50)
                macd, macd_signal, macd_hist = np.zeros_like(closes), np.zeros_like(closes), np.zeros_like(closes)
                stoch_k, stoch_d = np.full_like(closes, 50), np.full_like(closes, 50)
                williams_r = np.full_like(closes, -50)
                cci = np.full_like(closes, 0)
            
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
            
            # Calculate additional metrics
            price_change_pct = ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0
            
            # PERBAIKAN: Tambahkan EMA alignment analysis
            ema_alignment = self._check_ema_alignment(
                safe_float(ema_20[-1], current_price),
                safe_float(ema_50[-1], current_price), 
                safe_float(ema_200[-1], current_price),
                safe_float(ema_20[-2] if len(ema_20) > 1 else ema_20[-1], current_price),
                safe_float(ema_50[-2] if len(ema_50) > 1 else ema_50[-1], current_price),
                safe_float(ema_200[-2] if len(ema_200) > 1 else ema_200[-1], current_price)
            )
            
            # PERBAIKAN: Tambahkan momentum divergence detection
            momentum_divergence = self._check_momentum_divergence(closes, rsi, macd_hist)
            
            return {
                'trend': {
                    'sma_20': safe_float(sma_20[-1], current_price),
                    'sma_50': safe_float(sma_50[-1], current_price),
                    'ema_12': safe_float(ema_12[-1], current_price),
                    'ema_26': safe_float(ema_26[-1], current_price),
                    'ema_20': safe_float(ema_20[-1], current_price),
                    'ema_50': safe_float(ema_50[-1], current_price),
                    'ema_200': safe_float(ema_200[-1], current_price),
                    'adx': safe_float(adx[-1], 25),
                    'trend_direction': 'BULLISH' if safe_float(ema_20[-1], current_price) > safe_float(ema_50[-1], current_price) else 'BEARISH',
                    'trend_strength': 'STRONG' if safe_float(adx[-1], 25) > 40 else 'WEAK' if safe_float(adx[-1], 25) < 20 else 'MODERATE',
                    'ema_alignment': ema_alignment
                },
                'momentum': {
                    'rsi': safe_float(rsi[-1], 50),
                    'macd': safe_float(macd[-1], 0),
                    'macd_signal': safe_float(macd_signal[-1], 0),
                    'macd_histogram': safe_float(macd_hist[-1], 0),
                    'stoch_k': safe_float(stoch_k[-1], 50),
                    'stoch_d': safe_float(stoch_d[-1], 50),
                    'williams_r': safe_float(williams_r[-1], -50),
                    'cci': safe_float(cci[-1], 0),
                    'price_change_pct': safe_float(price_change_pct, 0),
                    'momentum_divergence': momentum_divergence
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
                },
                # PERBAIKAN: Tambahkan data EMA lengkap untuk chart
                'ema_data': {
                    'ema_20': [safe_float(x, current_price) for x in ema_20[-50:]],  # Last 50 values
                    'ema_50': [safe_float(x, current_price) for x in ema_50[-50:]],
                    'ema_200': [safe_float(x, current_price) for x in ema_200[-50:]]
                }
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._fallback_indicators(df)

    def _check_ema_alignment(self, ema20_curr, ema50_curr, ema200_curr, ema20_prev, ema50_prev, ema200_prev):
        """Cek alignment EMA untuk konfirmasi trend"""
        try:
            if (ema20_curr > ema50_curr > ema200_curr and
                ema20_curr > ema20_prev and ema50_curr > ema50_prev and ema200_curr > ema200_prev):
                return "STRONG_BULLISH"
            elif (ema20_curr < ema50_curr < ema200_curr and
                  ema20_curr < ema20_prev and ema50_curr < ema50_prev and ema200_curr < ema200_prev):
                return "STRONG_BEARISH"
            elif ema20_curr > ema50_curr and ema50_curr > ema200_curr:
                return "BULLISH"
            elif ema20_curr < ema50_curr and ema50_curr < ema200_curr:
                return "BEARISH"
            else:
                return "MIXED"
        except Exception as e:
            logger.warning(f"Error checking EMA alignment: {e}")
            return "NEUTRAL"

    def _check_momentum_divergence(self, closes, rsi, macd_hist):
        """Deteksi divergence antara price dan momentum"""
        try:
            if len(closes) < 20:
                return "NO_DIVERGENCE"
            
            # Price highs and lows
            price_highs = closes[-10:]
            price_lows = closes[-10:]
            
            # RSI highs and lows
            rsi_highs = rsi[-10:]
            rsi_lows = rsi[-10:]
            
            # Check for bearish divergence (price makes higher high, RSI makes lower high)
            if (np.argmax(price_highs) > np.argmax(rsi_highs) and 
                len(price_highs) > 5 and len(rsi_highs) > 5):
                return "BEARISH_DIVERGENCE"
            
            # Check for bullish divergence (price makes lower low, RSI makes higher low)
            if (np.argmin(price_lows) > np.argmin(rsi_lows) and 
                len(price_lows) > 5 and len(rsi_lows) > 5):
                return "BULLISH_DIVERGENCE"
            
            return "NO_DIVERGENCE"
        except Exception as e:
            logger.warning(f"Error checking momentum divergence: {e}")
            return "NO_DIVERGENCE"

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
                'ema_20': current_price * 0.998,
                'ema_50': current_price * 0.995,
                'ema_200': current_price * 0.990,
                'adx': 25,
                'trend_direction': 'BULLISH',
                'trend_strength': 'MODERATE',
                'ema_alignment': 'MIXED'
            },
            'momentum': {
                'rsi': 50,
                'macd': 0.001,
                'macd_signal': 0.0005,
                'macd_histogram': 0.0005,
                'stoch_k': 50,
                'stoch_d': 50,
                'williams_r': -50,
                'cci': 0,
                'price_change_pct': 0,
                'momentum_divergence': 'NO_DIVERGENCE'
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
            },
            'ema_data': {
                'ema_20': [current_price * 0.998] * 50,
                'ema_50': [current_price * 0.995] * 50,
                'ema_200': [current_price * 0.990] * 50
            }
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
            logger.info("TwelveData running in DEMO mode")
        else:
            logger.info("TwelveData running in LIVE mode")
    
    def get_real_time_price(self, pair: str) -> float:
        """Ambil current price real-time dari TwelveData atau simulasi"""
        cache_key = f"{pair}_{datetime.now().strftime('%Y%m%d%H%M')}"
        
        if pair in self.price_cache:
            cached_time, price = self.price_cache[pair]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                return price
        
        if self.demo_mode:
            return self._get_simulated_real_time_price(pair)
        
        try:
            formatted_pair = f"{pair[:3]}/{pair[3:]}"
            url = f"{self.base_url}/price?symbol={formatted_pair}&apikey={self.api_key}"
            
            logger.info(f"Fetching real-time price for {pair} from TwelveData...")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data and data['price'] is not None:
                    price = float(data['price'])
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
            base_prices = {
                'USDJPY': 147.25, 'GBPJPY': 198.50, 'EURJPY': 172.10, 'CHFJPY': 184.30, 'CADJPY': 108.50,
                'EURUSD': 1.0835, 'GBPUSD': 1.2640, 'USDCHF': 0.8840,
                'AUDUSD': 0.6545, 'USDCAD': 1.3510, 'NZDUSD': 0.6095
            }
            
            base_price = base_prices.get(pair, 150.0)
            variation = random.uniform(-0.001, 0.001)
            simulated_price = round(base_price * (1 + variation), 4)
            
            self.price_cache[pair] = (datetime.now(), simulated_price)
            
            logger.info(f"Simulated real-time price for {pair}: {simulated_price:.4f}")
            return simulated_price
            
        except Exception as e:
            logger.error(f"Error in simulated price for {pair}: {e}")
            return 150.0

# ==================== ADVANCED RISK MANAGEMENT SYSTEM YANG DIPERBAIKI ====================
class AdvancedRiskManager:
    def __init__(self, backtest_mode: bool = False):
        # PARAMETER YANG LEBIH KONSERVATIF
        self.max_daily_loss_pct = config.MAX_DAILY_LOSS
        self.max_drawdown_pct = config.MAX_DRAWDOWN
        self.max_position_size_pct = config.MAX_POSITION_SIZE_PCT
        self.daily_trade_limit = config.BACKTEST_DAILY_TRADE_LIMIT if backtest_mode else config.DAILY_TRADE_LIMIT
        self.correlation_threshold = config.CORRELATION_THRESHOLD
        
        # STANDARD YANG LEBIH KETAT
        self.backtest_min_confidence = config.BACKTEST_MIN_CONFIDENCE
        self.backtest_risk_score_threshold = config.BACKTEST_RISK_SCORE_THRESHOLD
        self.backtest_mode = backtest_mode
        
        # Trading session tracking
        self.today_trades = 0
        self.daily_pnl = 0.0
        self.peak_balance = 10000.0
        self.current_drawdown = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Correlation matrix
        self.correlation_matrix = {
            'USDJPY': {'EURUSD': -0.7, 'GBPUSD': -0.6, 'USDCHF': 0.9, 'EURJPY': 0.8, 'GBPJPY': 0.7, 'CADJPY': 0.6},
            'EURUSD': {'USDJPY': -0.7, 'GBPUSD': 0.8, 'USDCHF': -0.7, 'EURJPY': 0.9, 'GBPJPY': 0.6, 'CADJPY': -0.4},
            'GBPUSD': {'USDJPY': -0.6, 'EURUSD': 0.8, 'USDCHF': -0.6, 'EURJPY': 0.7, 'GBPJPY': 0.9, 'CADJPY': -0.3},
            'USDCHF': {'USDJPY': 0.9, 'EURUSD': -0.7, 'GBPUSD': -0.6, 'EURJPY': -0.6, 'GBPJPY': -0.5, 'CADJPY': 0.5},
            'EURJPY': {'USDJPY': 0.8, 'EURUSD': 0.9, 'GBPUSD': 0.7, 'USDCHF': -0.6, 'GBPJPY': 0.8, 'CADJPY': 0.7},
            'GBPJPY': {'USDJPY': 0.7, 'EURUSD': 0.6, 'GBPUSD': 0.9, 'USDCHF': -0.5, 'EURJPY': 0.8, 'CADJPY': 0.6},
            'CHFJPY': {'USDJPY': 0.6, 'EURJPY': 0.6, 'GBPJPY': 0.5, 'USDCHF': 0.8, 'EURUSD': -0.5, 'CADJPY': 0.4},
            'CADJPY': {'USDJPY': 0.6, 'EURJPY': 0.7, 'GBPJPY': 0.6, 'USDCAD': -0.8, 'USDCHF': 0.5, 'EURUSD': -0.4},
            'AUDUSD': {'USDJPY': -0.5, 'EURUSD': 0.6, 'GBPUSD': 0.5, 'NZDUSD': 0.8, 'USDCAD': -0.4, 'CADJPY': -0.3},
            'USDCAD': {'USDJPY': 0.4, 'EURUSD': -0.5, 'GBPUSD': -0.4, 'AUDUSD': -0.4, 'USDCHF': 0.6, 'CADJPY': -0.8},
            'NZDUSD': {'USDJPY': -0.5, 'EURUSD': 0.5, 'GBPUSD': 0.4, 'AUDUSD': 0.8, 'USDCAD': -0.3, 'CADJPY': -0.2}
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
        Validasi trade dengan risk assessment yang lebih ketat
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
        
        # PERBAIKAN 1: Auto-approve HOLD signals dengan risk rendah
        if signal == 'HOLD':
            validation_result['approved'] = True
            validation_result['risk_score'] = 1
            validation_result['warnings'].append("HOLD signal - monitoring market conditions")
            logger.info(f"Risk validation for {pair}-{signal}: AUTO-APPROVED - Score: 1")
            return validation_result
        
        # PERBAIKAN 2: Minimum confidence yang lebih tinggi
        min_confidence = self.backtest_min_confidence if self.backtest_mode else config.MIN_CONFIDENCE
        if confidence < min_confidence:
            validation_result['risk_score'] += 3
            validation_result['rejection_reasons'].append(f"Low confidence: {confidence}% < {min_confidence}%")
            risk_factors['confidence'] = 'HIGH'
            validation_result['approved'] = False
        
        # PERBAIKAN 3: Check daily trade limit
        if self.today_trades >= self.daily_trade_limit:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(
                f"Daily trade limit reached ({self.daily_trade_limit})"
            )
            risk_factors['daily_limit'] = 'HIGH'
            validation_result['risk_score'] += 3
        
        # PERBAIKAN 4: Check daily loss limit
        daily_loss_limit = account_balance * self.max_daily_loss_pct
        if self.daily_pnl <= -daily_loss_limit:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(
                f"Daily loss limit reached (${-self.daily_pnl:.2f})"
            )
            risk_factors['daily_loss'] = 'HIGH'
            validation_result['risk_score'] += 3
        
        # PERBAIKAN 5: Check drawdown limit
        if self.current_drawdown >= self.max_drawdown_pct:
            validation_result['approved'] = False
            validation_result['rejection_reasons'].append(
                f"Max drawdown reached ({self.current_drawdown:.1%})"
            )
            risk_factors['drawdown'] = 'HIGH'
            validation_result['risk_score'] += 3
        
        # PERBAIKAN 6: Position size validation yang lebih ketat
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
        
        # PERBAIKAN 7: Correlation risk assessment lebih ketat
        correlation_risk = self._check_correlation_risk(pair, signal, open_positions)
        if correlation_risk['high_risk']:
            validation_result['risk_score'] += 3
            validation_result['rejection_reasons'].append(
                f"High correlation with {correlation_risk['correlated_pairs']}"
            )
            risk_factors['correlation'] = 'HIGH'
            validation_result['approved'] = False
        elif correlation_risk['medium_risk']:
            validation_result['risk_score'] += 2
            validation_result['warnings'].append(
                f"Moderate correlation with {correlation_risk['correlated_pairs']}"
            )
            risk_factors['correlation'] = 'MEDIUM'
        
        # PERBAIKAN 8: Market volatility check lebih ketat
        volatility_risk = self._check_volatility_risk(pair, current_price)
        if volatility_risk['high_volatility']:
            validation_result['risk_score'] += 3
            validation_result['rejection_reasons'].append(
                f"High volatility: {volatility_risk['volatility_pct']:.1%}"
            )
            risk_factors['volatility'] = 'HIGH'
            validation_result['approved'] = False
        elif volatility_risk['medium_volatility']:
            validation_result['risk_score'] += 2
            validation_result['warnings'].append(
                f"Elevated volatility: {volatility_risk['volatility_pct']:.1%}"
            )
            risk_factors['volatility'] = 'MEDIUM'
        
        # PERBAIKAN 9: Time-based risk lebih ketat
        time_risk = self._check_time_risk()
        if time_risk['high_risk_period']:
            validation_result['risk_score'] += 2
            validation_result['warnings'].append(f"Trading during {time_risk['period_name']}")
            risk_factors['timing'] = 'MEDIUM'
        
        # PERBAIKAN 10: Liquidity check
        liquidity_risk = self._check_liquidity_risk()
        if liquidity_risk['low_liquidity']:
            validation_result['risk_score'] += 2
            validation_result['warnings'].append("Reduced liquidity period")
            risk_factors['liquidity'] = 'MEDIUM'
        
        # Final approval decision
        validation_result['risk_factors'] = risk_factors
        
        # PERBAIKAN 11: Risk threshold yang lebih ketat
        risk_threshold = self.backtest_risk_score_threshold if self.backtest_mode else 5
        
        if validation_result['risk_score'] >= risk_threshold:
            validation_result['approved'] = False
            if not validation_result['rejection_reasons']:
                validation_result['rejection_reasons'].append(
                    f"Overall risk score too high: {validation_result['risk_score']}/10"
                )
        
        status = "APPROVED" if validation_result['approved'] else "REJECTED"
        logger.info(f"Risk validation for {pair}-{signal}: {status} - Score: {validation_result['risk_score']}")
        
        # PERBAIKAN 12: Log detail risk factors untuk debugging
        if risk_factors:
            logger.info(f"Risk factors for {pair}: {risk_factors}")
        
        return validation_result

    def enhanced_risk_validation(self, pair: str, signal: str, confidence: int, 
                               technical_data: Dict, current_price: float) -> Dict:
        """Validasi risk yang lebih ketat dengan analisis teknikal"""
        
        validation_result = {
            'approved': False,
            'rejection_reasons': [],
            'risk_score': 0,
            'adjusted_confidence': confidence,
            'technical_issues': []
        }
        
        trend = technical_data['trend']
        momentum = technical_data['momentum'] 
        volatility = technical_data['volatility']
        
        # Filter 1: Minimum confidence
        min_confidence = config.MIN_CONFIDENCE
        if confidence < min_confidence:
            validation_result['rejection_reasons'].append(f"Confidence terlalu rendah: {confidence}% < {min_confidence}%")
            validation_result['risk_score'] += 3
        
        # Filter 2: Trend strength
        if trend['adx'] < config.MIN_ADX:
            validation_result['rejection_reasons'].append(f"Trend terlalu lemah (ADX: {trend['adx']:.1f} < {config.MIN_ADX})")
            validation_result['risk_score'] += 2
            validation_result['technical_issues'].append("weak_trend")
        
        # Filter 3: Volatility check
        if volatility['volatility_pct'] > config.VOLATILITY_THRESHOLD:
            validation_result['rejection_reasons'].append(f"Volatilitas terlalu tinggi: {volatility['volatility_pct']:.2f}% > {config.VOLATILITY_THRESHOLD*100}%")
            validation_result['risk_score'] += 2
            validation_result['technical_issues'].append("high_volatility")
        
        # Filter 4: RSI extreme check
        rsi = momentum['rsi']
        if signal == 'BUY' and rsi > config.MIN_RSI_BUY + 10:  # Allow some buffer
            validation_result['rejection_reasons'].append(f"RSI tidak mendukung BUY: {rsi:.1f} > {config.MIN_RSI_BUY + 10}")
            validation_result['risk_score'] += 2
            validation_result['adjusted_confidence'] = max(50, confidence - 20)
            validation_result['technical_issues'].append("rsi_conflict")
        
        if signal == 'SELL' and rsi < config.MAX_RSI_SELL - 10:  # Allow some buffer
            validation_result['rejection_reasons'].append(f"RSI tidak mendukung SELL: {rsi:.1f} < {config.MAX_RSI_SELL - 10}")
            validation_result['risk_score'] += 2
            validation_result['adjusted_confidence'] = max(50, confidence - 20)
            validation_result['technical_issues'].append("rsi_conflict")
        
        # Filter 5: EMA alignment conflict
        if signal == 'BUY' and trend['ema_alignment'] in ['STRONG_BEARISH', 'BEARISH']:
            validation_result['rejection_reasons'].append(f"EMA alignment tidak mendukung BUY: {trend['ema_alignment']}")
            validation_result['risk_score'] += 3
            validation_result['technical_issues'].append("ema_conflict")
            validation_result['approved'] = False
        
        if signal == 'SELL' and trend['ema_alignment'] in ['STRONG_BULLISH', 'BULLISH']:
            validation_result['rejection_reasons'].append(f"EMA alignment tidak mendukung SELL: {trend['ema_alignment']}")
            validation_result['risk_score'] += 3
            validation_result['technical_issues'].append("ema_conflict")
            validation_result['approved'] = False
        
        # Filter 6: Momentum divergence
        if momentum['momentum_divergence'] == 'BEARISH_DIVERGENCE' and signal == 'BUY':
            validation_result['rejection_reasons'].append("Bearish momentum divergence detected")
            validation_result['risk_score'] += 2
            validation_result['technical_issues'].append("bearish_divergence")
        
        if momentum['momentum_divergence'] == 'BULLISH_DIVERGENCE' and signal == 'SELL':
            validation_result['rejection_reasons'].append("Bullish momentum divergence detected")
            validation_result['risk_score'] += 2
            validation_result['technical_issues'].append("bullish_divergence")
        
        # Approval decision - LEBIH KETAT
        if (validation_result['risk_score'] <= 2 and 
            confidence >= min_confidence and 
            len(validation_result['technical_issues']) == 0):
            validation_result['approved'] = True
        
        logger.info(f"Enhanced risk validation for {pair}-{signal}: {'APPROVED' if validation_result['approved'] else 'REJECTED'} - Score: {validation_result['risk_score']}")
        
        return validation_result
    
    def _check_correlation_risk(self, pair: str, signal: str, open_positions: List[Dict]) -> Dict:
        """Check correlation risk yang lebih ketat"""
        high_risk = False
        medium_risk = False
        correlated_pairs = []
        
        for position in open_positions:
            open_pair = position['pair']
            open_signal = position['signal']
            
            if open_pair in self.correlation_matrix and pair in self.correlation_matrix[open_pair]:
                correlation = self.correlation_matrix[open_pair][pair]
                
                # LEBIH KETAT: Hanya flag sebagai high risk jika correlation sangat tinggi dan signal sama
                if abs(correlation) > 0.8 and open_signal == signal:
                    high_risk = True
                    correlated_pairs.append(f"{open_pair} (corr: {correlation:.2f})")
                elif abs(correlation) > 0.6 and open_signal == signal:
                    medium_risk = True
                    correlated_pairs.append(f"{open_pair} (corr: {correlation:.2f})")
        
        return {
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'correlated_pairs': correlated_pairs
        }
    
    def _check_volatility_risk(self, pair: str, current_price: float) -> Dict:
        """Check volatility risk yang lebih ketat"""
        try:
            price_data = data_manager.get_price_data(pair, '1H', days=5)
            if len(price_data) > 10:
                closes = price_data['close'].values
                returns = np.diff(closes) / closes[:-1]
                volatility = np.std(returns) * np.sqrt(24)
                
                return {
                    'high_volatility': volatility > config.VOLATILITY_THRESHOLD,
                    'medium_volatility': volatility > config.VOLATILITY_THRESHOLD * 0.7,
                    'volatility_pct': volatility
                }
        except Exception as e:
            logger.warning(f"Volatility calculation error for {pair}: {e}")
        
        return {'high_volatility': False, 'medium_volatility': False, 'volatility_pct': 0.01}
    
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
        
        low_liquidity_periods = [
            (21, 24), (0, 5), (23, 24), (0, 1)
        ]
        
        for start_hour, end_hour in low_liquidity_periods:
            if start_hour <= current_hour < end_hour:
                return {'low_liquidity': True}
        
        if now.weekday() >= 5:
            return {'low_liquidity': True}
        
        return {'low_liquidity': False}
    
    def update_trade_result(self, pnl: float, trade_success: bool):
        """Update risk manager dengan hasil trade"""
        self.daily_pnl += pnl
        self.today_trades += 1
        
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
        
        if self.today_trades >= self.daily_trade_limit * 0.7:
            warnings.append(f"Approaching daily trade limit ({self.today_trades}/{self.daily_trade_limit})")
        
        daily_loss_limit = self.peak_balance * self.max_daily_loss_pct
        if self.daily_pnl < 0 and abs(self.daily_pnl) > (daily_loss_limit * 0.6):
            warnings.append(f"Approaching daily loss limit (${self.daily_pnl:.2f}/${daily_loss_limit:.2f})")
        
        if self.current_drawdown > self.max_drawdown_pct * 0.7:
            warnings.append(f"Approaching maximum drawdown ({self.current_drawdown:.1%}/{self.max_drawdown_pct:.1%})")
        
        time_risk = self._check_time_risk()
        if time_risk['high_risk_period']:
            warnings.append(f"Trading during {time_risk['period_name']}")
        
        return warnings
    
    def _calculate_overall_risk_level(self) -> str:
        """Calculate overall risk level"""
        risk_score = 0
        
        trade_usage = self.today_trades / self.daily_trade_limit
        if trade_usage > 0.8:
            risk_score += 3
        elif trade_usage > 0.6:
            risk_score += 2
        elif trade_usage > 0.4:
            risk_score += 1
        
        daily_loss_limit = self.peak_balance * self.max_daily_loss_pct
        loss_usage = abs(self.daily_pnl) / daily_loss_limit if self.daily_pnl < 0 else 0
        if loss_usage > 0.8:
            risk_score += 3
        elif loss_usage > 0.6:
            risk_score += 2
        elif loss_usage > 0.4:
            risk_score += 1
        
        drawdown_usage = self.current_drawdown / self.max_drawdown_pct
        if drawdown_usage > 0.8:
            risk_score += 3
        elif drawdown_usage > 0.6:
            risk_score += 2
        elif drawdown_usage > 0.4:
            risk_score += 1
        
        time_risk = self._check_time_risk()
        if time_risk['high_risk_period']:
            risk_score += 2
        
        if risk_score >= 6:
            return "HIGH"
        elif risk_score >= 4:
            return "MEDIUM"
        else:
            return "LOW"

# ==================== DEEPSEEK AI ANALYZER YANG DIPERBAIKI ====================
class DeepSeekAnalyzer:
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.demo_mode = not self.api_key or self.api_key == "demo"
        
        if self.demo_mode:
            logger.info("DeepSeek AI running in DEMO mode")
        else:
            logger.info("DeepSeek AI running in LIVE mode")
    
    def analyze_market(self, pair: str, technical_data: Dict, fundamental_news: str) -> Dict:
        """Menganalisis market menggunakan DeepSeek AI dengan logika yang lebih akurat"""
        if self.demo_mode:
            logger.info("Using enhanced analysis (Demo mode)")
            return self._enhanced_analysis_with_strict_rules(technical_data, fundamental_news, pair)
        
        try:
            prompt = self._build_enhanced_analysis_prompt(pair, technical_data, fundamental_news)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": """Anda adalah analis forex profesional dengan pengalaman 10 tahun. 
                        Berikan analisis yang REALISTIS, PRAKTIS, dan KONSERVATIF. 
                        Fokus pada risk management dan peluang trading yang JELAS.
                        JANGAN berikan sinyal jika kondisi tidak optimal.
                        Accuracy lebih penting daripada frequency."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.2,  # Lower temperature untuk konsistensi
                "max_tokens": 2000
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                ai_response = response.json()["choices"][0]["message"]["content"]
                parsed_analysis = self._parse_ai_response(ai_response, technical_data)
                
                # Validasi tambahan untuk memastikan sinyal konsisten dengan kondisi market
                validated_analysis = self._validate_ai_analysis(parsed_analysis, technical_data)
                return validated_analysis
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return self._enhanced_analysis_with_strict_rules(technical_data, fundamental_news, pair)
                
        except Exception as e:
            logger.error(f"DeepSeek analysis failed: {e}")
            return self._enhanced_analysis_with_strict_rules(technical_data, fundamental_news, pair)
    
    def _build_enhanced_analysis_prompt(self, pair: str, technical_data: Dict, news: str) -> str:
        """Membangun prompt yang lebih detail untuk analisis AI yang akurat"""
        trend = technical_data['trend']
        momentum = technical_data['momentum']
        volatility = technical_data['volatility']
        levels = technical_data['levels']
        
        return f"""
ANALISIS FOREX UNTUK {pair} - PERHATIAN KRITIS UNTUK AKURASI

**DATA TEKNIKAL DETAIL:**
- Harga Saat Ini: {levels['current_price']:.4f}
- Support: {levels['support']:.4f}
- Resistance: {levels['resistance']:.4f}
- Trend: {trend['trend_direction']} ({trend['trend_strength']})
- EMA Alignment: {trend['ema_alignment']}
- EMA 20: {trend['ema_20']:.4f} | EMA 50: {trend['ema_50']:.4f} | EMA 200: {trend['ema_200']:.4f}

**INDIKATOR MOMENTUM:**
- RSI: {momentum['rsi']:.1f} ({'OVERSOLD' if momentum['rsi'] < 30 else 'OVERBOUGHT' if momentum['rsi'] > 70 else 'NETRAL'})
- MACD: {momentum['macd']:.5f} (Signal: {momentum['macd_signal']:.5f})
- MACD Histogram: {momentum['macd_histogram']:.5f} ({'BULLISH' if momentum['macd_histogram'] > 0 else 'BEARISH'})
- Williams %R: {momentum['williams_r']:.1f}
- CCI: {momentum['cci']:.1f}
- Momentum Divergence: {momentum['momentum_divergence']}

**VOLATILITAS & RISK:**
- Volatilitas: {volatility['volatility_pct']:.2f}%
- ATR: {volatility['atr']:.4f}
- ADX: {trend['adx']:.1f} ({'TREND KUAT' if trend['adx'] > 25 else 'TREND LEMAH'})
- Bollinger Bandwidth: {volatility['bollinger_bandwidth']:.2f}%

**BERITA FUNDAMENTAL:** {news}

**INSTRUKSI ANALISIS KRITIS:**
1. HANYA berikan sinyal BUY jika:
   - RSI < 35 DAN MACD histogram positif DAN harga di atas EMA 20
   - EMA alignment bullish (EMA 20 > EMA 50 > EMA 200)
   - Trend kuat (ADX > 20)
   - TIDAK ADA bearish divergence

2. HANYA berikan sinyal SELL jika:
   - RSI > 65 DAN MACD histogram negatif DAN harga di bawah EMA 20  
   - EMA alignment bearish (EMA 20 < EMA 50 < EMA 200)
   - Trend kuat (ADX > 20)
   - TIDAK ADA bullish divergence

3. BERI SINYAL HOLD jika:
   - RSI antara 40-60 (area netral)
   - EMA alignment mixed
   - Volatilitas tinggi (>2.5%)
   - Konflik antara indikator
   - ADX < 20 (trend lemah)

4. CONFIDENCE HARUS REALISTIS:
   - 80+ hanya untuk konfirmasi kuat semua indikator
   - 70-79 untuk konfirmasi mayoritas indikator  
   - <70 untuk kondisi kurang optimal

5. RISK LEVEL HARUS SESUAI:
   - HIGH: Volatilitas tinggi, konflik indikator, news penting
   - MEDIUM: Kondisi normal, beberapa konfirmasi
   - LOW: Semua indikator searah, volatilitas rendah

**FORMAT OUTPUT:**
{{
    "signal": "BUY/SELL/HOLD",
    "confidence": 0-100,
    "entry_price": "harga spesifik dengan reasoning",
    "stop_loss": "harga dengan calculation ATR",
    "take_profit_1": "harga target 1",
    "take_profit_2": "harga target 2", 
    "risk_level": "LOW/MEDIUM/HIGH",
    "analysis_summary": "analisis detail dalam Bahasa Indonesia dengan reasoning jelas",
    "key_levels": "level support/resistance kunci",
    "timeframe_suggestion": "timeframe optimal",
    "warning_notes": "catatan peringatan jika ada"
}}

**HARAP DIINGAT:** Accuracy lebih penting daripada frequency. Reject trade yang tidak memenuhi kriteria ketat.
"""
    
    def _parse_ai_response(self, ai_response: str, technical_data: Dict) -> Dict:
        """Parse response dari DeepSeek AI"""
        try:
            cleaned_response = ai_response.strip()
            
            if '```json' in cleaned_response:
                start_idx = cleaned_response.find('```json') + 7
                end_idx = cleaned_response.find('```', start_idx)
                if end_idx == -1:
                    end_idx = len(cleaned_response)
                json_str = cleaned_response[start_idx:end_idx].strip()
            elif '```' in cleaned_response:
                start_idx = cleaned_response.find('```') + 3
                end_idx = cleaned_response.find('```', start_idx)
                if end_idx == -1:
                    end_idx = len(cleaned_response)
                json_str = cleaned_response[start_idx:end_idx].strip()
            else:
                json_str = cleaned_response
            
            json_str = json_str.strip()
            if not json_str.startswith('{'):
                start_idx = json_str.find('{')
                if start_idx != -1:
                    json_str = json_str[start_idx:]
            
            if not json_str.endswith('}'):
                end_idx = json_str.rfind('}')
                if end_idx != -1:
                    json_str = json_str[:end_idx+1]
            
            analysis = json.loads(json_str)
            
            required_fields = ['signal', 'confidence', 'entry_price', 'stop_loss', 
                              'take_profit_1', 'risk_level', 'analysis_summary']
            
            for field in required_fields:
                if field not in analysis:
                    logger.warning(f"Missing field {field} in AI response, using enhanced analysis")
                    return self._enhanced_analysis_with_strict_rules(technical_data, "", "")
            
            analysis['ai_provider'] = 'DeepSeek AI'
            analysis['timestamp'] = datetime.now().isoformat()
            
            if 'confidence' in analysis:
                analysis['confidence'] = int(analysis['confidence'])
            
            logger.info("Successfully parsed AI response")
            return analysis
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse AI JSON response: {e}, using enhanced analysis")
            return self._enhanced_analysis_with_strict_rules(technical_data, "", "")
    
    def _validate_ai_analysis(self, ai_analysis: Dict, technical_data: Dict) -> Dict:
        """Validasi analisis AI untuk memastikan konsistensi dengan kondisi market"""
        trend = technical_data['trend']
        momentum = technical_data['momentum']
        volatility = technical_data['volatility']
        
        signal = ai_analysis.get('signal', 'HOLD')
        confidence = ai_analysis.get('confidence', 50)
        
        # Validasi 1: RSI vs Signal
        rsi = momentum['rsi']
        if signal == 'BUY' and rsi > 60:
            ai_analysis['signal'] = 'HOLD'
            ai_analysis['confidence'] = max(50, confidence - 20)
            ai_analysis['warning_notes'] = "RSI too high for BUY signal - overridden to HOLD"
            logger.warning("Overriding BUY signal due to high RSI")
        
        elif signal == 'SELL' and rsi < 40:
            ai_analysis['signal'] = 'HOLD'
            ai_analysis['confidence'] = max(50, confidence - 20)
            ai_analysis['warning_notes'] = "RSI too low for SELL signal - overridden to HOLD"
            logger.warning("Overriding SELL signal due to low RSI")
        
        # Validasi 2: EMA Alignment vs Signal
        ema_alignment = trend['ema_alignment']
        if signal == 'BUY' and ema_alignment in ['STRONG_BEARISH', 'BEARISH']:
            ai_analysis['signal'] = 'HOLD'
            ai_analysis['confidence'] = max(40, confidence - 30)
            ai_analysis['warning_notes'] = "EMA alignment contradicts BUY signal - overridden to HOLD"
            logger.warning("Overriding BUY signal due to bearish EMA alignment")
        
        elif signal == 'SELL' and ema_alignment in ['STRONG_BULLISH', 'BULLISH']:
            ai_analysis['signal'] = 'HOLD'
            ai_analysis['confidence'] = max(40, confidence - 30)
            ai_analysis['warning_notes'] = "EMA alignment contradicts SELL signal - overridden to HOLD"
            logger.warning("Overriding SELL signal due to bullish EMA alignment")
        
        # Validasi 3: Volatility check
        if volatility['volatility_pct'] > config.VOLATILITY_THRESHOLD and signal != 'HOLD':
            ai_analysis['signal'] = 'HOLD'
            ai_analysis['confidence'] = 45
            ai_analysis['warning_notes'] = "High volatility - avoiding trades"
            logger.warning("Overriding signal to HOLD due to high volatility")
        
        return ai_analysis
    
    def _enhanced_analysis_with_strict_rules(self, technical_data: Dict, news: str, pair: str) -> Dict:
        """Analisis enhanced dengan rules yang sangat ketat"""
        trend = technical_data['trend']
        momentum = technical_data['momentum']
        levels = technical_data['levels']
        volatility = technical_data['volatility']
        
        current_price = levels['current_price']
        rsi = momentum['rsi']
        macd_hist = momentum['macd_histogram']
        ema_alignment = trend.get('ema_alignment', 'MIXED')
        adx = trend['adx']
        volatility_pct = volatility['volatility_pct']
        momentum_divergence = momentum.get('momentum_divergence', 'NO_DIVERGENCE')
        
        # FILTER SANGAT KETAT
        signal = "HOLD"
        confidence = 50
        risk_level = "MEDIUM"
        analysis_summary = ""
        
        # Condition 1: Avoid trading in poor conditions
        if (volatility_pct > config.VOLATILITY_THRESHOLD or 
            adx < config.MIN_ADX or
            (rsi > 40 and rsi < 60) or
            momentum_divergence != 'NO_DIVERGENCE'):
            
            signal = "HOLD"
            confidence = 45
            risk_level = "HIGH" if volatility_pct > config.VOLATILITY_THRESHOLD else "MEDIUM"
            
            reasons = []
            if volatility_pct > config.VOLATILITY_THRESHOLD:
                reasons.append(f"volatilitas tinggi ({volatility_pct:.2f}%)")
            if adx < config.MIN_ADX:
                reasons.append(f"trend lemah (ADX: {adx:.1f})")
            if rsi > 40 and rsi < 60:
                reasons.append(f"RSI netral ({rsi:.1f})")
            if momentum_divergence != 'NO_DIVERGENCE':
                reasons.append(f"momentum divergence ({momentum_divergence})")
                
            analysis_summary = f"Kondisi tidak optimal: {', '.join(reasons)}"
        
        # Condition 2: Strict BUY conditions
        elif (rsi < config.MIN_RSI_BUY and 
              macd_hist > 0.0005 and 
              trend['trend_direction'] == 'BULLISH' and
              ema_alignment in ['STRONG_BULLISH', 'BULLISH'] and
              adx > config.MIN_ADX and
              momentum_divergence != 'BEARISH_DIVERGENCE'):
            
            signal = "BUY"
            base_confidence = 70
            
            # Confidence adjustments
            if rsi < 30: base_confidence += 10
            if ema_alignment == 'STRONG_BULLISH': base_confidence += 12
            if adx > 30: base_confidence += 8
            if macd_hist > 0.001: base_confidence += 5
            
            confidence = min(85, base_confidence)
            risk_level = "MEDIUM"
            analysis_summary = f"Strong bullish setup: RSI {rsi:.1f}, EMA {ema_alignment}, ADX {adx:.1f}"
        
        # Condition 3: Strict SELL conditions  
        elif (rsi > config.MAX_RSI_SELL and 
              macd_hist < -0.0005 and 
              trend['trend_direction'] == 'BEARISH' and
              ema_alignment in ['STRONG_BEARISH', 'BEARISH'] and
              adx > config.MIN_ADX and
              momentum_divergence != 'BULLISH_DIVERGENCE'):
            
            signal = "SELL"
            base_confidence = 70
            
            # Confidence adjustments
            if rsi > 70: base_confidence += 10
            if ema_alignment == 'STRONG_BEARISH': base_confidence += 12
            if adx > 30: base_confidence += 8
            if macd_hist < -0.001: base_confidence += 5
            
            confidence = min(85, base_confidence)
            risk_level = "MEDIUM"
            analysis_summary = f"Strong bearish setup: RSI {rsi:.1f}, EMA {ema_alignment}, ADX {adx:.1f}"
        
        else:
            # Default to HOLD with specific reasoning
            signal = "HOLD"
            confidence = 55
            risk_level = "LOW"
            analysis_summary = "Market conditions not meeting strict entry criteria"
        
        # Set stop loss dan take profit dengan risk management ketat
        atr = volatility['atr']
        if signal == "BUY":
            sl = current_price - (atr * 1.5)  # Use ATR for dynamic SL
            tp1 = current_price + (atr * 2)
            tp2 = current_price + (atr * 3)
        elif signal == "SELL":
            sl = current_price + (atr * 1.5)
            tp1 = current_price - (atr * 2)
            tp2 = current_price - (atr * 3)
        else:
            sl = current_price * 0.995
            tp1 = current_price * 1.005
            tp2 = current_price * 1.01

        return {
            "signal": signal,
            "confidence": confidence,
            "entry_price": f"{current_price:.4f}",
            "stop_loss": f"{sl:.4f}",
            "take_profit_1": f"{tp1:.4f}",
            "take_profit_2": f"{tp2:.4f}",
            "risk_level": risk_level,
            "analysis_summary": analysis_summary,
            "key_levels": f"Support: {levels['support']:.4f}, Resistance: {levels['resistance']:.4f}",
            "timeframe_suggestion": "4H-1D untuk konfirmasi",
            "warning_notes": "Strict rules applied - trade only if conditions optimal",
            "ai_provider": "Enhanced Technical Analysis with Strict Rules",
            "timestamp": datetime.now().isoformat()
        }

# ==================== DATA MANAGER YANG DIPERBAIKI ====================
class DataManager:
    def __init__(self):
        self.historical_data = {}
        self.load_historical_data()

    def get_price_data_with_timezone(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Dapatkan data harga dengan timezone awareness"""
        try:
            df = self.get_price_data(pair, timeframe, days)
            
            if df.empty:
                return df
                
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                if df['date'].dt.tz is None:
                    df['date'] = df['date'].dt.tz_localize('UTC')
            
            return df
            
        except Exception as e:
            logger.error(f"Error in get_price_data_with_timezone for {pair}-{timeframe}: {e}")
            return self.get_price_data(pair, timeframe, days)

    def ensure_fresh_data(self, pair: str, timeframe: str, min_records: int = 100):
        """Nonaktifkan validasi stale data"""
        try:
            if pair not in self.historical_data or timeframe not in self.historical_data[pair]:
                self._generate_sample_data(pair, timeframe)
                return
                
            df = self.historical_data[pair][timeframe]
            if df.empty or len(df) < min_records:
                self._generate_sample_data(pair, timeframe)
                return
                
        except Exception as e:
            logger.error(f"Error ensuring fresh data for {pair}-{timeframe}: {e}")

    def validate_and_fix_data(self, pair: str, timeframe: str):
        """Validasi dan perbaiki data yang rusak"""
        try:
            if pair in self.historical_data and timeframe in self.historical_data[pair]:
                df = self.historical_data[pair][timeframe]
                
                if df.empty or len(df) == 0:
                    logger.warning(f"Empty dataframe for {pair}-{timeframe}, regenerating data")
                    self._generate_sample_data(pair, timeframe)
                    return
                
                required_cols = ['open', 'high', 'low', 'close', 'date']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    logger.warning(f"Missing columns {missing_cols} for {pair}-{timeframe}, regenerating data")
                    self._generate_sample_data(pair, timeframe)
                    return
                    
                critical_cols = ['open', 'high', 'low', 'close']
                for col in critical_cols:
                    if df[col].isna().any():
                        logger.warning(f"NaN values found in {col} for {pair}-{timeframe}, regenerating data")
                        self._generate_sample_data(pair, timeframe)
                        return
                        
                logger.info(f"Data validation passed for {pair}-{timeframe}")
                
        except Exception as e:
            logger.error(f"Data validation error for {pair}-{timeframe}: {e}")
            self._generate_sample_data(pair, timeframe)
    
    def load_historical_data(self):
        """Load data historis dari file CSV"""
        try:
            data_dir = "historical_data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                logger.info("Created historical_data directory")
                self._create_sample_data()
                return
            
            loaded_count = 0
            for filename in os.listdir(data_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(data_dir, filename)
                    try:
                        df = pd.read_csv(file_path)
                        
                        df = self._standardize_columns(df)
                        
                        if 'date' not in df.columns:
                            logger.warning(f"File {filename} missing 'date' column, skipping")
                            continue
                        
                        try:
                            df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
                            df = df.dropna(subset=['date'])
                        except Exception as e:
                            logger.warning(f"Error parsing dates in {filename}: {e}")
                            continue
                        
                        required_cols = ['open', 'high', 'low', 'close']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            logger.warning(f"File {filename} missing columns: {missing_cols}, skipping")
                            continue
                            
                        name_parts = filename.replace('.csv', '').split('_')
                        if len(name_parts) >= 2:
                            pair = name_parts[0].upper()
                            timeframe = name_parts[1].upper()
                            
                            if pair not in config.FOREX_PAIRS:
                                continue
                                
                            if pair not in self.historical_data:
                                self.historical_data[pair] = {}
                            
                            self.historical_data[pair][timeframe] = df
                            loaded_count += 1
                            logger.info(f"Loaded {pair}-{timeframe}: {len(df)} records")
                            
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")
                        continue
            
            logger.info(f"Total loaded datasets: {loaded_count}")
            
            if loaded_count == 0:
                logger.warning("No valid data found, creating sample data...")
                self._create_sample_data()
            
        except Exception as e:
            logger.error(f"Error in load_historical_data: {e}")
            self._create_sample_data()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardisasi nama kolom"""
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close',
            'Date': 'date', 'TIME': 'date', 'Timestamp': 'date'
        }
        
        df.columns = [column_mapping.get(col, col.lower()) for col in df.columns]
        return df

    def _create_sample_data(self):
        """Buat sample data jika tidak ada data historis"""
        logger.info("Creating sample historical data...")
        
        for pair in config.FOREX_PAIRS[:6]:
            for timeframe in ['M30', '1H', '4H', '1D']:
                self._generate_sample_data(pair, timeframe)
    
    def _generate_sample_data(self, pair: str, timeframe: str):
        """Generate sample data yang realistis"""
        try:
            if timeframe == 'M30':
                periods = 2000
            elif timeframe == '1H':
                periods = 1500
            elif timeframe == '4H':
                periods = 1000
            else:
                periods = 500
                
            base_prices = {
                'USDJPY': 147.0, 'GBPJPY': 198.0, 'EURJPY': 172.0, 'CHFJPY': 184.0, 'CADJPY': 108.5,
                'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDCHF': 0.8850,
                'AUDUSD': 0.6550, 'USDCAD': 1.3500, 'NZDUSD': 0.6100
            }
            
            base_price = base_prices.get(pair, 150.0)
            prices = []
            current_price = base_price
            
            end_date = datetime.now().replace(tzinfo=None)
            
            if timeframe == 'M30':
                start_date = end_date - timedelta(hours=periods*0.5)
            elif timeframe == '1H':
                start_date = end_date - timedelta(hours=periods)
            elif timeframe == '4H':
                start_date = end_date - timedelta(hours=periods*4)
            else:
                start_date = end_date - timedelta(days=periods)
            
            current_date = start_date
            
            for i in range(periods):
                volatility = 0.0015
                drift = (base_price - current_price) * 0.001
                random_shock = np.random.normal(0, volatility)
                change = drift + random_shock
                current_price = current_price * (1 + change)
                
                open_price = current_price
                close_price = current_price * (1 + np.random.normal(0, volatility * 0.3))
                high = max(open_price, close_price) + abs(change) * base_price * 0.5
                low = min(open_price, close_price) - abs(change) * base_price * 0.5
                
                high = max(high, max(open_price, close_price) + 0.0001)
                low = min(low, min(open_price, close_price) - 0.0001)
                
                if timeframe == 'M30':
                    current_date = start_date + timedelta(minutes=30*i)
                elif timeframe == '1H':
                    current_date = start_date + timedelta(hours=i)
                elif timeframe == '4H':
                    hours_to_add = (i * 4) % 24
                    days_to_add = (i * 4) // 24
                    current_date = start_date + timedelta(days=days_to_add, hours=hours_to_add)
                else:
                    current_date = start_date + timedelta(days=i)
                
                current_date_utc = current_date.replace(tzinfo=None)
                
                prices.append({
                    'date': current_date_utc,
                    'open': round(float(open_price), 4),
                    'high': round(float(high), 4),
                    'low': round(float(low), 4),
                    'close': round(float(close_price), 4),
                    'volume': int(np.random.randint(10000, 50000))
                })
            
            df = pd.DataFrame(prices)
            
            data_dir = "historical_data"
            os.makedirs(data_dir, exist_ok=True)
            filename = f"{data_dir}/{pair}_{timeframe}.csv"
            df.to_csv(filename, index=False)
            
            if pair not in self.historical_data:
                self.historical_data[pair] = {}
            self.historical_data[pair][timeframe] = df
            
            logger.info(f"Created sample data: {filename} with {len(df)} records")
            
        except Exception as e:
            logger.error(f"Error generating sample data for {pair}-{timeframe}: {e}")

    def get_price_data(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Dapatkan data harga untuk backtesting"""
        try:
            if pair in self.historical_data and timeframe in self.historical_data[pair]:
                df = self.historical_data[pair][timeframe]
                if df.empty or 'date' not in df.columns:
                    logger.warning(f"Missing date column for {pair}-{timeframe}, regenerating data")
                    return self._generate_simple_data(pair, timeframe, days)
                
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.dropna(subset=['date'])
                except Exception as e:
                    logger.warning(f"Error parsing dates for {pair}-{timeframe}: {e}")
                    return self._generate_simple_data(pair, timeframe, days)
                
                if timeframe == 'M30':
                    required_points = min(len(df), days * 48)
                elif timeframe == '1H':
                    required_points = min(len(df), days * 24)
                elif timeframe == '4H':
                    required_points = min(len(df), days * 6)
                else:
                    required_points = min(len(df), days)
                    
                result_df = df.tail(required_points).copy()
                
                required_cols = ['open', 'high', 'low', 'close']
                missing_cols = [col for col in required_cols if col not in result_df.columns]
                if missing_cols:
                    logger.warning(f"Missing columns {missing_cols}, generating synthetic data")
                    return self._generate_simple_data(pair, timeframe, days)
                
                return result_df
            
            return self._generate_simple_data(pair, timeframe, days)
            
        except Exception as e:
            logger.error(f"Error getting price data for {pair}-{timeframe}: {e}")
            return self._generate_simple_data(pair, timeframe, days)

    def _generate_simple_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate simple synthetic data"""
        if timeframe == 'M30':
            points = days * 48
        elif timeframe == '1H':
            points = days * 24
        elif timeframe == '4H':
            points = days * 6
        else:
            points = days
            
        base_prices = {
            'USDJPY': 147.0, 'GBPJPY': 198.0, 'EURJPY': 172.0, 'CHFJPY': 184.0, 'CADJPY': 108.5,
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDCHF': 0.8850,
            'AUDUSD': 0.6550, 'USDCAD': 1.3500, 'NZDUSD': 0.6100
        }
        
        base_price = base_prices.get(pair, 150.0)
        prices = []
        current_price = base_price
        
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(points):
            change = np.random.normal(0, 0.001)
            current_price = current_price * (1 + change)
            
            open_price = current_price
            close_price = current_price * (1 + np.random.normal(0, 0.0005))
            high = max(open_price, close_price) + abs(change) * 0.1
            low = min(open_price, close_price) - abs(change) * 0.1
            
            if high <= low:
                high = low + 0.0001
            
            if timeframe == 'M30':
                current_date = start_date + timedelta(minutes=30*i)
            elif timeframe == '1H':
                current_date = start_date + timedelta(hours=i)
            elif timeframe == '4H':
                current_date = start_date + timedelta(hours=4*i)
            else:
                current_date = start_date + timedelta(days=i)
            
            prices.append({
                'date': current_date,
                'open': round(float(open_price), 4),
                'high': round(float(high), 4),
                'low': round(float(low), 4),
                'close': round(float(close_price), 4),
                'volume': int(np.random.randint(10000, 50000))
            })
        
        df = pd.DataFrame(prices)
        df['date'] = pd.to_datetime(df['date'])
        
        return df

# ==================== FUNDAMENTAL ANALYSIS ENGINE ====================
class FundamentalAnalysisEngine:
    def __init__(self):
        self.news_cache = {}
        self.cache_duration = timedelta(minutes=30)
        self.demo_mode = not config.NEWS_API_KEY or config.NEWS_API_KEY == "demo"
        logger.info(f"Fundamental Analysis Engine initialized - {'DEMO MODE' if self.demo_mode else 'LIVE MODE'}")
    
    def get_forex_news(self, pair: str) -> str:
        """Mendapatkan berita fundamental untuk pair forex"""
        cache_key = f"{pair}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.news_cache:
            cached_time, news = self.news_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return news
        
        try:
            country_map = {
                'USDJPY': 'Japan United States economy Bank of Japan Federal Reserve',
                'GBPJPY': 'Japan UK economy Brexit Bank of England',
                'EURJPY': 'Japan Europe ECB economy European Central Bank',
                'CHFJPY': 'Japan Switzerland economy SNB Swiss National Bank',
                'CADJPY': 'Japan Canada economy Bank of Canada Bank of Japan',
                'EURUSD': 'Europe United States Fed ECB Federal Reserve European Central Bank',
                'GBPUSD': 'UK United States Bank of England Fed Brexit',
                'USDCHF': 'United States Switzerland SNB Fed Swiss National Bank',
                'AUDUSD': 'Australia United States RBA Fed Reserve Bank of Australia',
                'USDCAD': 'United States Canada Fed Bank of Canada BOC',
                'NZDUSD': 'New Zealand United States RBNZ Fed Reserve Bank of New Zealand'
            }
            
            query = country_map.get(pair, 'forex market news')
            
            if not self.demo_mode:
                url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=3&apiKey={config.NEWS_API_KEY}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    if articles:
                        news_items = []
                        for article in articles[:2]:
                            title = article.get('title', '')
                            source = article.get('source', {}).get('name', 'Unknown')
                            title = title.encode('ascii', 'ignore').decode('ascii')[:100]
                            news_items.append(f"{title} (Source: {source})")
                        
                        news_text = " | ".join(news_items)
                        self.news_cache[cache_key] = (datetime.now(), news_text)
                        logger.info(f"Retrieved live news for {pair}")
                        return news_text
            
            news_text = self._get_fallback_news(pair)
            self.news_cache[cache_key] = (datetime.now(), news_text)
            logger.info(f"Using demo news for {pair}")
            return news_text
            
        except Exception as e:
            logger.error(f"Error fetching news for {pair}: {e}")
            return self._get_fallback_news(pair)
    
    def _get_fallback_news(self, pair: str) -> str:
        """Berita fallback ketika API tidak tersedia"""
        news_templates = {
            'USDJPY': [
                "Bank of Japan maintains ultra-loose monetary policy. Fed signals potential rate cuts in 2024.",
                "Yen weakness continues as BOJ sticks to yield curve control. USD strength persists.",
                "USD/JPY approaches intervention levels as interest rate differential widens."
            ],
            'GBPJPY': [
                "Bank of England holds rates steady amid inflation concerns. GBP shows volatility.",
                "UK economic data mixed, GBP/JPY influenced by risk sentiment and carry trades.",
                "Brexit aftermath continues to impact GBP crosses with Japanese Yen."
            ],
            'EURJPY': [
                "ECB monitoring inflation closely. Euro area growth shows signs of stabilization.",
                "EUR/JPY influenced by ECB policy outlook and Japanese economic recovery.",
                "European inflation data key for EUR direction against safe-haven JPY."
            ],
            'CHFJPY': [
                "Swiss National Bank maintains focus on currency interventions amid global uncertainty.",
                "CHF/JPY influenced by safe-haven flows and Bank of Japan policy decisions.",
                "Switzerland inflation stays within target range, supporting CHF stability."
            ],
            'CADJPY': [
                "CAD/JPY influenced by oil prices and Bank of Canada policy decisions.",
                "Canadian dollar shows volatility against yen amid commodity price fluctuations.",
                "Bank of Canada and Bank of Japan policy divergence impacts CAD/JPY cross."
            ],
            'EURUSD': [
                "EUR/USD trades in tight range ahead of key economic data releases.",
                "European and US economic indicators driving EUR/USD direction.",
                "Central bank policy divergence continues to influence EUR/USD movements."
            ],
            'GBPUSD': [
                "GBP/USD volatile amid mixed UK economic data and dollar strength.",
                "Cable influenced by Bank of England and Federal Reserve policy expectations.",
                "Brexit-related developments continue to impact GBP/USD trading."
            ],
            'USDCHF': [
                "USD/CHF supported by safe-haven flows and interest rate differentials.",
                "Swiss National Bank interventions influence USD/CHF price action.",
                "USD/CHF reacts to global risk sentiment and US economic data."
            ]
        }
        
        templates = news_templates.get(pair, ["Market analysis ongoing. Monitor economic indicators for trading opportunities."])
        return random.choice(templates)

# ==================== TRADING SIGNAL GENERATOR YANG DIPERBAIKI ====================
def get_higher_timeframe(timeframe: str) -> str:
    """Dapatkan timeframe yang lebih tinggi untuk konfirmasi"""
    timeframe_map = {
        'M30': '1H',
        '1H': '4H', 
        '4H': '1D',
        '1D': '1W',
        '1W': '1M'
    }
    return timeframe_map.get(timeframe, '4H')

def check_trend_alignment(current_tf: Dict, higher_tf: Dict) -> str:
    """Cek alignment trend antara timeframe"""
    current_trend = current_tf['trend']['trend_direction']
    higher_trend = higher_tf['trend']['trend_direction']
    
    if current_trend == higher_trend:
        return current_trend
    elif current_tf['trend']['adx'] > higher_tf['trend']['adx'] + 10:
        return current_trend  # Trend current TF lebih kuat
    else:
        return higher_trend  # Trend higher TF lebih kuat

def generate_enhanced_trading_signals(price_data: pd.DataFrame, pair: str, timeframe: str) -> List[Dict]:
    """Generate sinyal trading dengan konfirmasi berlapis yang lebih akurat"""
    signals = []
    
    try:
        if len(price_data) < 100:  # Minimum data points
            logger.warning(f"Insufficient data for {pair}-{timeframe}: {len(price_data)} points")
            return signals
        
        tech_engine = TechnicalAnalysisEngine()
        
        # Analisis multi-timeframe
        higher_timeframe = get_higher_timeframe(timeframe)
        higher_tf_data = data_manager.get_price_data(pair, higher_timeframe, days=60)
        
        if higher_tf_data.empty:
            logger.warning(f"No higher timeframe data for {pair}-{higher_timeframe}")
            higher_tf_analysis = None
        else:
            higher_tf_analysis = tech_engine.calculate_all_indicators(higher_tf_data)
        
        logger.info(f"Generating ENHANCED signals for {pair}-{timeframe} with higher TF: {higher_timeframe}")
        
        signal_count = 0
        # Less frequent sampling untuk quality over quantity
        step_size = max(10, len(price_data) // 50)
        
        for i in range(100, len(price_data), step_size):
            try:
                window_data = price_data.iloc[:i+1]
                
                if len(window_data) < 100:
                    continue
                    
                tech_analysis = tech_engine.calculate_all_indicators(window_data)
                current_price = tech_analysis['levels']['current_price']
                
                # Konfirmasi dari higher timeframe
                if higher_tf_analysis is not None:
                    trend_alignment = check_trend_alignment(tech_analysis, higher_tf_analysis)
                else:
                    trend_alignment = "NEUTRAL"
                
                # Generate qualified signal dengan rules ketat
                signal = generate_qualified_signal(tech_analysis, trend_alignment, pair)
                
                if signal and signal['confidence'] >= config.MIN_CONFIDENCE:
                    current_date = window_data.iloc[-1]['date']
                    signals.append({
                        'date': current_date,
                        'pair': pair,
                        'action': signal['action'],
                        'confidence': signal['confidence'],
                        'price': float(current_price),
                        'rsi': float(tech_analysis['momentum']['rsi']),
                        'macd_hist': float(tech_analysis['momentum']['macd_histogram']),
                        'trend': tech_analysis['trend']['trend_direction'],
                        'adx': float(tech_analysis['trend']['adx']),
                        'ema_alignment': tech_analysis['trend'].get('ema_alignment', 'MIXED'),
                        'volatility': float(tech_analysis['volatility']['volatility_pct']),
                        'trend_alignment': trend_alignment,
                        'reason': signal.get('reason', 'Multiple conditions met')
                    })
                    signal_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing data point {i}: {e}")
                continue
        
        logger.info(f"Generated {signal_count} QUALITY trading signals for {pair}-{timeframe}")
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating enhanced trading signals: {e}")
        return []

def generate_qualified_signal(tech_analysis: Dict, trend_alignment: str, pair: str) -> Optional[Dict]:
    """Generate sinyal yang memenuhi multiple kondisi ketat"""
    trend = tech_analysis['trend']
    momentum = tech_analysis['momentum']
    levels = tech_analysis['levels']
    volatility = tech_analysis['volatility']
    
    current_price = levels['current_price']
    rsi = momentum['rsi']
    macd_hist = momentum['macd_histogram']
    adx = trend['adx']
    ema_alignment = trend.get('ema_alignment', 'MIXED')
    volatility_pct = volatility['volatility_pct']
    momentum_divergence = momentum.get('momentum_divergence', 'NO_DIVERGENCE')
    
    # Filter 1: Hindari trading saat kondisi tidak optimal
    if (volatility_pct > config.VOLATILITY_THRESHOLD or 
        adx < config.MIN_ADX or
        (rsi > 40 and rsi < 60) or
        momentum_divergence != 'NO_DIVERGENCE'):
        return None
    
    # Filter 2: Konfirmasi trend alignment
    if trend_alignment == "BEARISH" and trend['trend_direction'] == 'BULLISH':
        return None
    if trend_alignment == "BULLISH" and trend['trend_direction'] == 'BEARISH':
        return None
    
    # Kondisi BUY yang sangat ketat
    buy_conditions = [
        rsi < config.MIN_RSI_BUY and macd_hist > 0.0005 and trend['trend_direction'] == 'BULLISH',
        rsi < config.MIN_RSI_BUY - 3 and ema_alignment in ['STRONG_BULLISH', 'BULLISH'] and adx > 25,
        rsi < 30 and current_price < trend['ema_20'] and trend['ema_20'] > trend['ema_50'],
        rsi < 32 and momentum_divergence == 'BULLISH_DIVERGENCE' and trend['trend_direction'] == 'BULLISH'
    ]
    
    # Kondisi SELL yang sangat ketat  
    sell_conditions = [
        rsi > config.MAX_RSI_SELL and macd_hist < -0.0005 and trend['trend_direction'] == 'BEARISH',
        rsi > config.MAX_RSI_SELL + 3 and ema_alignment in ['STRONG_BEARISH', 'BEARISH'] and adx > 25,
        rsi > 70 and current_price > trend['ema_20'] and trend['ema_20'] < trend['ema_50'],
        rsi > 68 and momentum_divergence == 'BEARISH_DIVERGENCE' and trend['trend_direction'] == 'BEARISH'
    ]
    
    buy_score = sum(1 for condition in buy_conditions if condition)
    sell_score = sum(1 for condition in sell_conditions if condition)
    
    # PERBAIKAN: Hanya eksekusi jika minimal 2 kondisi terpenuhi
    if buy_score >= config.REQUIRED_CONDITIONS:
        confidence = 65 + (buy_score * 8)
        if rsi < 30: confidence += 10
        if ema_alignment == 'STRONG_BULLISH': confidence += 12
        if adx > 30: confidence += 8
        if trend_alignment == "BULLISH": confidence += 10
        if momentum_divergence == 'BULLISH_DIVERGENCE': confidence += 8
        
        return {
            'action': 'BUY',
            'confidence': min(90, confidence),
            'reason': f"Strong bullish confluence: {buy_score} conditions met"
        }
    
    elif sell_score >= config.REQUIRED_CONDITIONS:
        confidence = 65 + (sell_score * 8)
        if rsi > 70: confidence += 10
        if ema_alignment == 'STRONG_BEARISH': confidence += 12
        if adx > 30: confidence += 8
        if trend_alignment == "BEARISH": confidence += 10
        if momentum_divergence == 'BEARISH_DIVERGENCE': confidence += 8
        
        return {
            'action': 'SELL', 
            'confidence': min(90, confidence),
            'reason': f"Strong bearish confluence: {sell_score} conditions met"
        }
    
    return None

# ==================== ENHANCED BACKTESTING ENGINE ====================
class AdvancedBacktestingEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.risk_manager = AdvancedRiskManager(backtest_mode=True)
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.portfolio_history = []
        self.trade_history = []
        self.winning_trades = 0
        self.losing_trades = 0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        self.total_long_trades = 0
        self.total_short_trades = 0
        self.winning_long_trades = 0
        self.winning_short_trades = 0
        
        self.risk_adjusted_return = 0.0
        self.sharpe_ratio = 0.0
        self.calmar_ratio = 0.0
        
        self.risk_manager = AdvancedRiskManager(backtest_mode=True)
        
        logger.info("Backtesting engine reset")
    
    def run_comprehensive_backtest(self, signals: List[Dict], price_data: pd.DataFrame, 
                                 pair: str, timeframe: str) -> Dict:
        """Menjalankan backtest komprehensif dengan rules ketat"""
        self.reset()
        
        logger.info(f"Running comprehensive backtest for {pair}-{timeframe} with {len(signals)} signals")
        
        if not signals or price_data.empty:
            return self._empty_backtest_result(pair)
        
        multi_timeframe_analysis = self._analyze_multiple_timeframes(pair)
        
        executed_trades = 0
        for signal in signals:
            if self._execute_trade_with_enhanced_risk_management(signal, price_data, multi_timeframe_analysis):
                executed_trades += 1
        
        logger.info(f"Backtest completed: {executed_trades} trades executed")
        
        return self._generate_comprehensive_report(pair, timeframe, multi_timeframe_analysis)
    
    def _analyze_multiple_timeframes(self, pair: str) -> Dict:
        """Analisis multiple timeframe untuk konfirmasi trend"""
        timeframe_analysis = {}
        
        for tf in ['M30', '1H', '4H', '1D']:
            try:
                data = data_manager.get_price_data(pair, tf, days=30)
                if not data.empty and len(data) > 20:
                    tech = tech_engine.calculate_all_indicators(data)
                    timeframe_analysis[tf] = {
                        'trend': tech['trend']['trend_direction'],
                        'rsi': tech['momentum']['rsi'],
                        'signal': 'BULLISH' if tech['trend']['ema_20'] > tech['trend']['ema_50'] else 'BEARISH',
                        'strength': self._calculate_trend_strength(tech),
                        'adx': tech['trend']['adx'],
                        'ema_alignment': tech['trend'].get('ema_alignment', 'MIXED')
                    }
                else:
                    timeframe_analysis[tf] = {'trend': 'UNKNOWN', 'rsi': 50, 'signal': 'HOLD', 'strength': 0, 'adx': 25, 'ema_alignment': 'MIXED'}
            except Exception as e:
                logger.warning(f"Error analyzing {pair}-{tf}: {e}")
                timeframe_analysis[tf] = {'trend': 'UNKNOWN', 'rsi': 50, 'signal': 'HOLD', 'strength': 0, 'adx': 25, 'ema_alignment': 'MIXED'}
        
        return timeframe_analysis
    
    def _calculate_trend_strength(self, technical_data: Dict) -> float:
        """Hitung strength trend dari 0-1"""
        trend = technical_data['trend']
        momentum = technical_data['momentum']
        
        strength_score = 0
        
        ema_alignment = trend.get('ema_alignment', 'MIXED')
        if ema_alignment == 'STRONG_BULLISH' or ema_alignment == 'STRONG_BEARISH':
            strength_score += 0.3
        elif ema_alignment == 'BULLISH' or ema_alignment == 'BEARISH':
            strength_score += 0.15
        
        adx = trend['adx']
        if adx > 40:
            strength_score += 0.3
        elif adx > 25:
            strength_score += 0.15
        
        rsi = momentum['rsi']
        if (rsi > 50 and trend['trend_direction'] == 'BULLISH') or (rsi < 50 and trend['trend_direction'] == 'BEARISH'):
            strength_score += 0.2
        
        if momentum['macd_histogram'] > 0 and trend['trend_direction'] == 'BULLISH':
            strength_score += 0.2
        elif momentum['macd_histogram'] < 0 and trend['trend_direction'] == 'BEARISH':
            strength_score += 0.2
        
        return min(1.0, strength_score)
    
    def _execute_trade_with_enhanced_risk_management(self, signal: Dict, price_data: pd.DataFrame, 
                                               mtf_analysis: Dict) -> bool:
        """Eksekusi trade dengan risk management yang ditingkatkan"""
        try:
            signal_date = signal['date']
            action = signal['action']
            confidence = signal['confidence']
            
            if confidence < config.BACKTEST_MIN_CONFIDENCE:
                return False
            
            try:
                if hasattr(signal_date, 'date'):
                    signal_date_date = signal_date.date()
                else:
                    signal_date_date = pd.to_datetime(signal_date).date()
                
                price_data_dates = pd.to_datetime(price_data['date']).dt.date
                trade_data = price_data[price_data_dates == signal_date_date]
                
                if trade_data.empty:
                    if len(price_data) > 0:
                        trade_data = price_data.iloc[-1:]
                    else:
                        return False
                        
                entry_price = float(trade_data['close'].iloc[0])
                
            except Exception as e:
                logger.warning(f"Date processing error: {e}, using latest price")
                if len(price_data) > 0:
                    entry_price = float(price_data['close'].iloc[-1])
                else:
                    return False
            
            mtf_confirmation = self._get_mtf_confirmation(action, mtf_analysis)
            
            if not mtf_confirmation['confirmed']:
                logger.info(f"MTF confirmation weak for {signal['pair']}-{action}, rejecting trade")
                return False
            
            # Enhanced risk validation dengan technical analysis
            current_tech = tech_engine.calculate_all_indicators(price_data)
            enhanced_risk_validation = self.risk_manager.enhanced_risk_validation(
                pair=signal.get('pair', 'UNKNOWN'),
                signal=action,
                confidence=confidence,
                technical_data=current_tech,
                current_price=entry_price
            )
            
            if not enhanced_risk_validation['approved']:
                logger.info(f"Enhanced risk validation failed: {enhanced_risk_validation['rejection_reasons']}")
                return False
            
            lot_size = 0.1  # Default lot size
            trade_result = self._simulate_enhanced_trade_execution(
                action=action,
                entry_price=entry_price,
                lot_size=lot_size,
                confidence=confidence,
                mtf_strength=mtf_confirmation['strength'],
                risk_score=enhanced_risk_validation['risk_score'],
                technical_data=current_tech
            )
            
            self._update_portfolio(signal, trade_result, entry_price, lot_size, enhanced_risk_validation)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in enhanced trade execution: {e}")
            return False
    
    def _get_mtf_confirmation(self, action: str, mtf_analysis: Dict) -> Dict:
        """Dapatkan konfirmasi multi-timeframe"""
        confirming_timeframes = 0
        total_timeframes = len(mtf_analysis)
        total_strength = 0
        
        for tf, analysis in mtf_analysis.items():
            if analysis['signal'] == action:
                confirming_timeframes += 1
                total_strength += analysis['strength']
        
        confirmation_score = confirming_timeframes / total_timeframes if total_timeframes > 0 else 0
        avg_strength = total_strength / confirming_timeframes if confirming_timeframes > 0 else 0
        
        return {
            'confirmed': confirmation_score >= 0.75,
            'score': confirmation_score,
            'strength': avg_strength,
            'confirming_tf': confirming_timeframes,
            'total_tf': total_timeframes
        }
    
    def _simulate_enhanced_trade_execution(self, action: str, entry_price: float, 
                                         lot_size: float, confidence: int, mtf_strength: float,
                                         risk_score: int, technical_data: Dict) -> Dict:
        """Simulasi eksekusi trade yang lebih realistis dengan analisis teknikal"""
        
        base_probability = confidence / 100.0
        mtf_bonus = mtf_strength * 0.3
        risk_penalty = risk_score * 0.1
        
        # Additional factors dari technical analysis
        rsi = technical_data['momentum']['rsi']
        adx = technical_data['trend']['adx']
        ema_alignment = technical_data['trend']['ema_alignment']
        
        # Technical bonuses
        tech_bonus = 0
        if action == 'BUY' and rsi < 30:
            tech_bonus += 0.1
        elif action == 'SELL' and rsi > 70:
            tech_bonus += 0.1
        
        if adx > 30:
            tech_bonus += 0.1
        
        if (action == 'BUY' and ema_alignment == 'STRONG_BULLISH') or (action == 'SELL' and ema_alignment == 'STRONG_BEARISH'):
            tech_bonus += 0.15
        
        enhanced_probability = base_probability + mtf_bonus + tech_bonus - risk_penalty
        enhanced_probability = max(0.4, min(0.85, enhanced_probability))
        
        if np.random.random() < enhanced_probability:
            if action == 'BUY':
                price_change_pct = np.random.uniform(0.01, 0.04)
                profit = entry_price * price_change_pct * lot_size * 100000
            else:
                price_change_pct = np.random.uniform(-0.04, -0.01)
                profit = abs(entry_price * price_change_pct * lot_size * 100000)
            
            outcome = 'WIN'
            close_reason = 'TAKE_PROFIT'
            
        else:
            if action == 'BUY':
                price_change_pct = np.random.uniform(-0.01, -0.003)
                profit = entry_price * price_change_pct * lot_size * 100000
            else:
                price_change_pct = np.random.uniform(0.003, 0.01)
                profit = -abs(entry_price * price_change_pct * lot_size * 100000)
            
            outcome = 'LOSS'
            close_reason = 'STOP_LOSS'
        
        return {
            'profit': profit,
            'outcome': outcome,
            'close_reason': close_reason,
            'probability_used': enhanced_probability,
            'price_change_pct': price_change_pct
        }
    
    def _get_current_positions(self) -> List[Dict]:
        return []
    
    def _update_portfolio(self, signal: Dict, trade_result: Dict, entry_price: float, 
                         lot_size: float, risk_validation: Dict):
        profit = trade_result['profit']
        
        self.balance += profit
        
        trade_record = {
            'entry_date': signal['date'].strftime('%Y-%m-%d') if hasattr(signal['date'], 'strftime') else str(signal['date']),
            'pair': signal.get('pair', 'UNKNOWN'),
            'action': signal['action'],
            'entry_price': round(entry_price, 4),
            'lot_size': round(lot_size, 2),
            'profit': round(profit, 2),
            'outcome': trade_result['outcome'],
            'close_reason': trade_result['close_reason'],
            'confidence': signal.get('confidence', 50),
            'balance_after': round(self.balance, 2),
            'risk_score': risk_validation['risk_score'],
            'probability': round(trade_result['probability_used'], 3),
            'price_change_pct': round(trade_result['price_change_pct'] * 100, 2)
        }
        
        self.trade_history.append(trade_record)
        
        self.portfolio_history.append({
            'date': trade_record['entry_date'],
            'balance': self.balance,
            'drawdown': self._calculate_current_drawdown()
        })
        
        if trade_result['outcome'] == 'WIN':
            self.winning_trades += 1
            self.consecutive_losses = 0
            
            if signal['action'] == 'BUY':
                self.winning_long_trades += 1
            else:
                self.winning_short_trades += 1
                
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        if signal['action'] == 'BUY':
            self.total_long_trades += 1
        else:
            self.total_short_trades += 1
        
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        current_drawdown = self._calculate_current_drawdown()
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        self.risk_manager.update_trade_result(profit, trade_result['outcome'] == 'WIN')
    
    def _calculate_current_drawdown(self) -> float:
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance
    
    def _generate_comprehensive_report(self, pair: str, timeframe: str, mtf_analysis: Dict) -> Dict:
        total_trades = len(self.trade_history)
        
        if total_trades == 0:
            return self._empty_backtest_result(pair)
        
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(trade['profit'] for trade in self.trade_history)
        average_profit = total_profit / total_trades if total_trades > 0 else 0
        
        winning_trades = [t for t in self.trade_history if t['outcome'] == 'WIN']
        losing_trades = [t for t in self.trade_history if t['outcome'] == 'LOSS']
        
        avg_win = sum(t['profit'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['profit'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        long_win_rate = (self.winning_long_trades / self.total_long_trades * 100) if self.total_long_trades > 0 else 0
        short_win_rate = (self.winning_short_trades / self.total_short_trades * 100) if self.total_short_trades > 0 else 0
        
        total_return_pct = ((self.balance - self.initial_balance) / self.initial_balance * 100)
        
        returns = [t['profit'] / self.initial_balance for t in self.trade_history]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                self.sharpe_ratio = (avg_return - 0.02/252) / std_return * np.sqrt(252)
            else:
                self.sharpe_ratio = 0
        else:
            self.sharpe_ratio = 0
        
        if self.max_drawdown > 0:
            self.calmar_ratio = total_return_pct / (self.max_drawdown * 100)
        else:
            self.calmar_ratio = float('inf')
        
        recovery_factor = total_profit / (self.max_drawdown * self.initial_balance) if self.max_drawdown > 0 else 0
        
        self.risk_adjusted_return = total_return_pct / (self.max_drawdown * 100 + 1) if self.max_drawdown > 0 else total_return_pct
        
        avg_risk_score = np.mean([t.get('risk_score', 0) for t in self.trade_history]) if self.trade_history else 0
        
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
                'max_drawdown': round(self.max_drawdown * 100, 2),
                'profit_factor': round(profit_factor, 2),
                'sharpe_ratio': round(self.sharpe_ratio, 2),
                'calmar_ratio': round(self.calmar_ratio, 2),
                'recovery_factor': round(recovery_factor, 2),
                'risk_adjusted_return': round(self.risk_adjusted_return, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(avg_loss, 2),
                'avg_trade': round(average_profit, 2),
                'max_consecutive_losses': self.max_consecutive_losses,
                'avg_risk_score': round(avg_risk_score, 1)
            },
            'trade_analysis': {
                'long_trades': self.total_long_trades,
                'short_trades': self.total_short_trades,
                'long_win_rate': round(long_win_rate, 2),
                'short_win_rate': round(short_win_rate, 2),
                'best_trade': max(self.trade_history, key=lambda x: x['profit']) if self.trade_history else {},
                'worst_trade': min(self.trade_history, key=lambda x: x['profit']) if self.trade_history else {},
                'avg_confidence': round(np.mean([t['confidence'] for t in self.trade_history]), 1) if self.trade_history else 0,
                'avg_probability': round(np.mean([t.get('probability', 0.5) for t in self.trade_history]), 3) if self.trade_history else 0
            },
            'multi_timeframe_analysis': mtf_analysis,
            'risk_report': self.risk_manager.get_risk_report(),
            'trade_history': self.trade_history[-20:],
            'performance_grade': self._calculate_performance_grade(win_rate, profit_factor, self.sharpe_ratio),
            'metadata': {
                'pair': pair,
                'timeframe': timeframe,
                'initial_balance': self.initial_balance,
                'testing_date': datetime.now().isoformat(),
                'total_days': len(set(t['entry_date'] for t in self.trade_history)),
                'note': 'Enhanced backtest with strict risk management and multi-timeframe analysis'
            }
        }
    
    def _calculate_performance_grade(self, win_rate: float, profit_factor: float, sharpe_ratio: float) -> str:
        score = 0
        
        if win_rate >= 60:
            score += 3
        elif win_rate >= 55:
            score += 2
        elif win_rate >= 50:
            score += 1
        
        if profit_factor >= 2.0:
            score += 3
        elif profit_factor >= 1.5:
            score += 2
        elif profit_factor >= 1.2:
            score += 1
        
        if sharpe_ratio >= 2.0:
            score += 3
        elif sharpe_ratio >= 1.5:
            score += 2
        elif sharpe_ratio >= 1.0:
            score += 1
        elif sharpe_ratio >= 0.5:
            score += 0
        else:
            score -= 1
        
        if score >= 8:
            return "A+"
        elif score >= 7:
            return "A"
        elif score >= 6:
            return "A-"
        elif score >= 5:
            return "B+"
        elif score >= 4:
            return "B"
        elif score >= 3:
            return "B-"
        elif score >= 2:
            return "C+"
        elif score >= 1:
            return "C"
        else:
            return "D"
    
    def _empty_backtest_result(self, pair: str) -> Dict:
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
                'max_drawdown': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'calmar_ratio': 0,
                'recovery_factor': 0,
                'risk_adjusted_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_trade': 0,
                'max_consecutive_losses': 0,
                'avg_risk_score': 0
            },
            'trade_analysis': {
                'long_trades': 0,
                'short_trades': 0,
                'long_win_rate': 0,
                'short_win_rate': 0
            },
            'risk_report': self.risk_manager.get_risk_report(),
            'trade_history': [],
            'performance_grade': 'N/A',
            'metadata': {
                'pair': pair,
                'initial_balance': self.initial_balance,
                'testing_date': datetime.now().isoformat(),
                'message': 'No trades executed during backtest period - strategy too conservative'
            }
        }

# ==================== INITIALIZE SYSTEM ====================
logger.info("Initializing Enhanced Forex Analysis System...")

tech_engine = TechnicalAnalysisEngine()
fundamental_engine = FundamentalAnalysisEngine()
deepseek_analyzer = DeepSeekAnalyzer()
data_manager = DataManager()
twelve_data_client = TwelveDataClient()

risk_manager = AdvancedRiskManager()
advanced_backtester = AdvancedBacktestingEngine()

logger.info(f"Supported pairs: {config.FOREX_PAIRS}")
logger.info(f"Historical data: {len(data_manager.historical_data)} pairs loaded")
logger.info(f"DeepSeek AI: {'LIVE MODE' if not deepseek_analyzer.demo_mode else 'DEMO MODE'}")
logger.info(f"TwelveData Real-time: {'LIVE MODE' if not twelve_data_client.demo_mode else 'DEMO MODE'}")
logger.info(f"News API: {'LIVE MODE' if not fundamental_engine.demo_mode else 'DEMO MODE'}")
logger.info(f"Advanced Risk Management: ENABLED")
logger.info(f"Enhanced Backtesting: ENABLED")
logger.info(f"Backtesting Parameters: Daily Trade Limit: {config.BACKTEST_DAILY_TRADE_LIMIT}, Min Confidence: {config.BACKTEST_MIN_CONFIDENCE}")

logger.info("All system components initialized successfully")

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', 
                         pairs=config.FOREX_PAIRS,
                         timeframes=config.TIMEFRAMES,
                         initial_balance=config.INITIAL_BALANCE)

@app.route('/api/analyze')
def api_analyze():
    """Endpoint untuk analisis market real-time"""
    try:
        pair = request.args.get('pair', 'USDJPY').upper()
        timeframe = request.args.get('timeframe', '4H').upper()
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        real_time_price = twelve_data_client.get_real_time_price(pair)
        
        price_data = data_manager.get_price_data_with_timezone(pair, timeframe, days=60)
        if price_data.empty:
            logger.warning(f"No price data for {pair}-{timeframe}, generating sample data")
            data_manager._generate_sample_data(pair, timeframe)
            price_data = data_manager.get_price_data_with_timezone(pair, timeframe, days=60)
        
        technical_analysis = tech_engine.calculate_all_indicators(price_data)
        
        technical_analysis['levels']['current_price'] = real_time_price
        
        fundamental_news = fundamental_engine.get_forex_news(pair)
        
        ai_analysis = deepseek_analyzer.analyze_market(pair, technical_analysis, fundamental_news)
        
        # Enhanced risk validation
        enhanced_risk_validation = risk_manager.enhanced_risk_validation(
            pair=pair,
            signal=ai_analysis.get('signal', 'HOLD'),
            confidence=ai_analysis.get('confidence', 50),
            technical_data=technical_analysis,
            current_price=real_time_price
        )
        
        price_series = []
        try:
            hist_df = data_manager.get_price_data_with_timezone(pair, timeframe, days=200)
            if not hist_df.empty:
                hist_df = hist_df.sort_values('date')
                
                for _, row in hist_df.iterrows():
                    date_value = row['date']
                    
                    if hasattr(date_value, 'isoformat'):
                        if date_value.tzinfo is None:
                            date_str = date_value.isoformat() + 'Z'
                        else:
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
        
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'fundamental_analysis': fundamental_news,
            'ai_analysis': ai_analysis,
            'enhanced_risk_validation': enhanced_risk_validation,
            'price_data': {
                'current': real_time_price,
                'support': technical_analysis.get('levels', {}).get('support'),
                'resistance': technical_analysis.get('levels', {}).get('resistance'),
                'change_pct': technical_analysis.get('momentum', {}).get('price_change_pct', 0)
            },
            'price_series': price_series,
            'analysis_summary': f"{pair} currently trading at {real_time_price:.4f}",
            'ai_provider': ai_analysis.get('ai_provider', 'DeepSeek AI'),
            'data_source': 'TwelveData Live' if not twelve_data_client.demo_mode else 'TwelveData Demo',
            'timezone_info': 'UTC'
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Endpoint untuk basic backtesting"""
    try:
        data = request.get_json()
        pair = data.get('pair', 'USDJPY')
        timeframe = data.get('timeframe', '4H')
        days = int(data.get('days', 30))
        initial_balance = float(data.get('initial_balance', config.INITIAL_BALANCE))
        
        logger.info(f"Basic backtest request: {pair}-{timeframe} for {days} days")
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        price_data = data_manager.get_price_data(pair, timeframe, days)
        
        if price_data.empty:
            return jsonify({'error': 'No price data available for backtesting'}), 400
        
        signals = generate_enhanced_trading_signals(price_data, pair, timeframe)
        
        simple_backtester = AdvancedBacktestingEngine(initial_balance)
        result = simple_backtester.run_comprehensive_backtest(signals, price_data, pair, timeframe)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500

@app.route('/api/advanced_backtest', methods=['POST'])
def api_advanced_backtest():
    """Endpoint untuk advanced backtesting"""
    try:
        data = request.get_json()
        pair = data.get('pair', 'USDJPY')
        timeframe = data.get('timeframe', '4H')
        days = int(data.get('days', 30))
        initial_balance = float(data.get('initial_balance', config.INITIAL_BALANCE))
        
        logger.info(f"Advanced backtest request: {pair}-{timeframe} for {days} days")
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        price_data = data_manager.get_price_data(pair, timeframe, days)
        
        if price_data.empty:
            return jsonify({'error': 'No price data available for backtesting'}), 400
        
        logger.info(f"Price data loaded: {len(price_data)} records for {pair}-{timeframe}")
        
        signals = generate_enhanced_trading_signals(price_data, pair, timeframe)
        
        logger.info(f"Generated {len(signals)} trading signals for backtesting")
        
        advanced_backtester.initial_balance = initial_balance
        result = advanced_backtester.run_comprehensive_backtest(signals, price_data, pair, timeframe)
        
        result['backtest_parameters'] = {
            'pair': pair,
            'timeframe': timeframe,
            'days': days,
            'initial_balance': initial_balance,
            'signals_generated': len(signals),
            'trades_executed': result['summary']['total_trades'],
            'backtest_mode': True,
            'risk_parameters': {
                'daily_trade_limit': config.BACKTEST_DAILY_TRADE_LIMIT,
                'min_confidence': config.BACKTEST_MIN_CONFIDENCE,
                'risk_score_threshold': config.BACKTEST_RISK_SCORE_THRESHOLD
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Advanced backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Advanced backtest failed: {str(e)}'}), 500

@app.route('/api/market_overview')
def api_market_overview():
    """Overview market untuk semua pair dengan REAL-TIME prices"""
    overview = {}
    
    for pair in config.FOREX_PAIRS:
        try:
            real_time_price = twelve_data_client.get_real_time_price(pair)
            
            price_data = data_manager.get_price_data(pair, '1D', days=5)
            
            if not price_data.empty:
                tech = tech_engine.calculate_all_indicators(price_data)
                
                change_pct = 0
                if len(price_data) > 1 and 'close' in price_data.columns:
                    try:
                        prev_price = float(price_data['close'].iloc[-2])
                        change_pct = ((real_time_price - prev_price) / prev_price) * 100
                    except:
                        change_pct = 0
                
                rsi = tech['momentum']['rsi']
                trend = tech['trend']['trend_direction']
                ema_alignment = tech['trend'].get('ema_alignment', 'MIXED')
                adx = tech['trend']['adx']
                volatility = tech['volatility']['volatility_pct']
                
                # Enhanced recommendation logic
                if volatility > config.VOLATILITY_THRESHOLD:
                    recommendation = 'AVOID'
                    confidence = 'HIGH'
                elif adx < config.MIN_ADX:
                    recommendation = 'HOLD'
                    confidence = 'MEDIUM'
                elif (rsi < config.MIN_RSI_BUY and 
                      trend == 'BULLISH' and 
                      ema_alignment in ['STRONG_BULLISH', 'BULLISH']):
                    recommendation = 'BUY'
                    confidence = 'HIGH'
                elif (rsi > config.MAX_RSI_SELL and 
                      trend == 'BEARISH' and 
                      ema_alignment in ['STRONG_BEARISH', 'BEARISH']):
                    recommendation = 'SELL'
                    confidence = 'HIGH'
                elif rsi < config.MIN_RSI_BUY + 5 and trend == 'BULLISH':
                    recommendation = 'BUY'
                    confidence = 'MEDIUM'
                elif rsi > config.MAX_RSI_SELL - 5 and trend == 'BEARISH':
                    recommendation = 'SELL'
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
                    'ema_alignment': ema_alignment,
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
                    'ema_alignment': 'MIXED',
                    'volatility': 0,
                    'recommendation': 'HOLD',
                    'confidence': 'LOW',
                    'error': 'No historical data available',
                    'data_source': 'TwelveData Live' if not twelve_data_client.demo_mode else 'TwelveData Demo'
                }
        except Exception as e:
            logger.error(f"Error getting overview for {pair}: {e}")
            overview[pair] = {
                'price': 0,
                'change': 0,
                'rsi': 50,
                'trend': 'UNKNOWN',
                'trend_strength': 'UNKNOWN',
                'ema_alignment': 'MIXED',
                'volatility': 0,
                'recommendation': 'HOLD',
                'confidence': 'LOW',
                'error': f'Processing error: {str(e)}'
            }
    
    return jsonify(overview)

@app.route('/api/risk_dashboard')
def api_risk_dashboard():
    """Dashboard risk management yang komprehensif"""
    try:
        risk_report = risk_manager.get_risk_report()
        
        market_analysis = {}
        for pair in config.FOREX_PAIRS[:4]:
            try:
                real_time_price = twelve_data_client.get_real_time_price(pair)
                
                data = data_manager.get_price_data(pair, '1D', days=5)
                if not data.empty:
                    tech = tech_engine.calculate_all_indicators(data)
                    volatility = tech['volatility']['volatility_pct']
                    
                    if volatility > config.VOLATILITY_THRESHOLD:
                        risk_level = 'HIGH'
                    elif volatility > config.VOLATILITY_THRESHOLD * 0.7:
                        risk_level = 'MEDIUM'
                    else:
                        risk_level = 'LOW'
                        
                    market_analysis[pair] = {
                        'current_price': real_time_price,
                        'volatility': round(volatility, 3),
                        'trend': tech['trend']['trend_direction'],
                        'ema_alignment': tech['trend'].get('ema_alignment', 'MIXED'),
                        'rsi': round(tech['momentum']['rsi'], 1),
                        'risk_level': risk_level,
                        'atr': round(tech['volatility']['atr'], 4),
                        'data_source': 'TwelveData Live' if not twelve_data_client.demo_mode else 'TwelveData Demo'
                    }
            except Exception as e:
                logger.error(f"Error analyzing {pair} for risk dashboard: {e}")
                market_analysis[pair] = {'error': str(e)}
        
        trading_recommendations = []
        for pair in config.FOREX_PAIRS[:4]:
            try:
                data = data_manager.get_price_data(pair, '4H', days=10)
                if not data.empty:
                    tech = tech_engine.calculate_all_indicators(data)
                    
                    rsi = tech['momentum']['rsi']
                    trend = tech['trend']['trend_direction']
                    volatility = tech['volatility']['volatility_pct']
                    adx = tech['trend']['adx']
                    ema_alignment = tech['trend'].get('ema_alignment', 'MIXED')
                    
                    if (rsi < config.MIN_RSI_BUY and 
                        trend == 'BULLISH' and 
                        adx > config.MIN_ADX and 
                        ema_alignment in ['STRONG_BULLISH', 'BULLISH']):
                        recommendation = 'BUY'
                        confidence = 'HIGH'
                        reason = 'Oversold with strong bullish alignment and momentum'
                    elif (rsi > config.MAX_RSI_SELL and 
                          trend == 'BEARISH' and 
                          adx > config.MIN_ADX and 
                          ema_alignment in ['STRONG_BEARISH', 'BEARISH']):
                        recommendation = 'SELL'
                        confidence = 'HIGH'
                        reason = 'Overbought with strong bearish alignment and momentum'
                    elif volatility > config.VOLATILITY_THRESHOLD:
                        recommendation = 'AVOID'
                        confidence = 'HIGH'
                        reason = 'High volatility market conditions'
                    elif adx < config.MIN_ADX:
                        recommendation = 'AVOID'
                        confidence = 'MEDIUM'
                        reason = 'Weak trend momentum'
                    else:
                        recommendation = 'HOLD'
                        confidence = 'LOW'
                        reason = 'Neutral market conditions'
                    
                    trading_recommendations.append({
                        'pair': pair,
                        'recommendation': recommendation,
                        'confidence': confidence,
                        'reason': reason,
                        'current_rsi': round(rsi, 1),
                        'trend': trend,
                        'ema_alignment': ema_alignment,
                        'volatility': round(volatility, 3),
                        'adx': round(adx, 1)
                    })
                    
            except Exception as e:
                logger.error(f"Error generating recommendation for {pair}: {e}")
                continue
        
        return jsonify({
            'risk_management': risk_report,
            'market_conditions': market_analysis,
            'trading_recommendations': trading_recommendations,
            'system_status': {
                'total_pairs_monitored': len(config.FOREX_PAIRS),
                'risk_manager_status': 'ACTIVE',
                'last_update': datetime.now().isoformat(),
                'data_provider': 'TwelveData Live' if not twelve_data_client.demo_mode else 'TwelveData Demo'
            }
        })
        
    except Exception as e:
        logger.error(f"Risk dashboard error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_status')
def api_system_status():
    """Status sistem dan ketersediaan API"""
    return jsonify({
        'system': 'RUNNING',
        'historical_data': f"{len(data_manager.historical_data)} pairs loaded",
        'supported_pairs': config.FOREX_PAIRS,
        'deepseek_ai': 'LIVE MODE' if not deepseek_analyzer.demo_mode else 'DEMO MODE',
        'news_api': 'LIVE MODE' if not fundamental_engine.demo_mode else 'DEMO MODE',
        'twelve_data': 'LIVE MODE' if not twelve_data_client.demo_mode else 'DEMO MODE',
        'risk_management': 'ADVANCED',
        'backtesting_engine': 'ENHANCED',
        'server_time': datetime.now().isoformat(),
        'version': '3.2',
        'features': [
            'Enhanced EMA Analysis (20, 50, 200)',
            'Advanced Risk Management', 
            'Multi-Timeframe Analysis',
            'AI-Powered Analysis',
            'Comprehensive Backtesting',
            'Real-time Market Overview',
            'Risk Dashboard',
            'TwelveData Real-time Integration',
            'Fundamental News Analysis'
        ]
    })

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("Starting Enhanced Forex Analysis System v3.2...")
    logger.info(f"Supported pairs: {config.FOREX_PAIRS}")
    logger.info(f"Historical data: {len(data_manager.historical_data)} pairs loaded")
    logger.info(f"DeepSeek AI: {'LIVE MODE' if not deepseek_analyzer.demo_mode else 'DEMO MODE'}")
    logger.info(f"TwelveData Real-time: {'LIVE MODE' if not twelve_data_client.demo_mode else 'DEMO MODE'}")
    logger.info(f"News API: {'LIVE MODE' if not fundamental_engine.demo_mode else 'DEMO MODE'}")
    logger.info(f"Advanced Risk Management: ENABLED")
    logger.info(f"Enhanced Backtesting: ENABLED")
    logger.info(f"Backtesting Parameters:")
    logger.info(f"  - Daily Trade Limit: {config.BACKTEST_DAILY_TRADE_LIMIT}")
    logger.info(f"  - Min Confidence: {config.BACKTEST_MIN_CONFIDENCE}")
    logger.info(f"  - Risk Score Threshold: {config.BACKTEST_RISK_SCORE_THRESHOLD}")
    
    os.makedirs('historical_data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Forex Analysis System is ready and running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
