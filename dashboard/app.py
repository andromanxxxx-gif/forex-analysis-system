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
    
    # Hapus handler lama jika ada
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Stream handler untuk console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    # File handler untuk log file
    file_handler = logging.FileHandler('enhanced_forex_trading.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Add handlers ke root logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger

# Setup logging
logger = setup_logging()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'forex-secure-key-2024')

# ==================== KONFIGURASI SISTEM ====================
@dataclass
class SystemConfig:
    # API Configuration
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "demo")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "demo") 
    TWELVE_DATA_KEY: str = os.environ.get("TWELVE_DATA_KEY", "demo")
    
    # Enhanced Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    MAX_DAILY_LOSS: float = 0.02
    MAX_DRAWDOWN: float = 0.08
    MAX_POSITIONS: int = 3
    STOP_LOSS_PCT: float = 0.01
    TAKE_PROFIT_PCT: float = 0.02
    
    # Risk Management Parameters
    CORRELATION_THRESHOLD: float = 0.7
    VOLATILITY_THRESHOLD: float = 0.02
    DAILY_TRADE_LIMIT: int = 10
    MAX_POSITION_SIZE_PCT: float = 0.05
    
    # Enhanced Backtesting Parameters
    BACKTEST_DAILY_TRADE_LIMIT: int = 20
    BACKTEST_MIN_CONFIDENCE: int = 70
    BACKTEST_RISK_SCORE_THRESHOLD: int = 6
    
    # Optimized Parameters untuk 1H dan 4H
    TIMEFRAME_OPTIMIZED_PARAMS: Dict = field(default_factory=lambda: {
        '1H': {
            'RSI_PERIOD': 16,
            'MACD_FAST': 10,
            'MACD_SLOW': 21,
            'MACD_SIGNAL': 7,
            'STOCH_K': 12,
            'STOCH_D': 3,
            'MIN_CONFIDENCE': 70,
            'STOP_LOSS_PCT': 0.008,
            'TAKE_PROFIT_PCT': 0.016,
            'VOLATILITY_FILTER': 0.015,
            'MIN_ADX': 20,
            'MAX_VOLATILITY': 0.035
        },
        '4H': {
            'RSI_PERIOD': 18,
            'MACD_FAST': 12,
            'MACD_SLOW': 26,
            'MACD_SIGNAL': 9,
            'STOCH_K': 14,
            'STOCH_D': 3,
            'MIN_CONFIDENCE': 75,
            'STOP_LOSS_PCT': 0.012,
            'TAKE_PROFIT_PCT': 0.024,
            'VOLATILITY_FILTER': 0.020,
            'MIN_ADX': 22,
            'MAX_VOLATILITY': 0.040
        },
        '1D': {
            'RSI_PERIOD': 14,
            'MACD_FAST': 12,
            'MACD_SLOW': 26,
            'MACD_SIGNAL': 9,
            'STOCH_K': 14,
            'STOCH_D': 3,
            'MIN_CONFIDENCE': 80,
            'STOP_LOSS_PCT': 0.015,
            'TAKE_PROFIT_PCT': 0.030,
            'VOLATILITY_FILTER': 0.025,
            'MIN_ADX': 25,
            'MAX_VOLATILITY': 0.050
        }
    })
    
    # Supported Instruments
    FOREX_PAIRS: List[str] = field(default_factory=lambda: [
        "USDJPY", "GBPJPY", "EURJPY", "CHFJPY", "CADJPY",
        "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
    ])
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M30", "1H", "4H", "1D"])
    
    # Backtesting
    DEFAULT_BACKTEST_DAYS: int = 90
    MIN_DATA_POINTS: int = 100
    
    # Trading Hours (UTC)
    HIGH_IMPACT_HOURS: List[Tuple[int, int]] = field(default_factory=lambda: [(8, 10), (13, 15)])

config = SystemConfig()

# ==================== ENGINE ANALISIS TEKNIKAL ====================
class TechnicalAnalysisEngine:
    def __init__(self):
        self.indicators = {}
        logger.info("Enhanced Technical Analysis Engine initialized")
    
    def calculate_optimized_indicators(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Menghitung indikator dengan parameter yang dioptimasi per timeframe"""
        try:
            if df.empty:
                return self._fallback_indicators(df)
                
            required_columns = ['open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return self._fallback_indicators(df)
            
            if len(df) < 20:
                return self._fallback_indicators(df)
                
            # Dapatkan parameter optimized
            params = config.TIMEFRAME_OPTIMIZED_PARAMS.get(timeframe, config.TIMEFRAME_OPTIMIZED_PARAMS['4H'])
            
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            opens = df['open'].values
            
            # Handle NaN values
            closes = np.nan_to_num(closes, nan=closes[-1] if len(closes) > 0 else 150.0)
            highs = np.nan_to_num(highs, nan=closes[-1] if len(closes) > 0 else 150.0)
            lows = np.nan_to_num(lows, nan=closes[-1] if len(closes) > 0 else 150.0)
            opens = np.nan_to_num(opens, nan=closes[-1] if len(closes) > 0 else 150.0)
            
            try:
                # Trend Indicators dengan parameter optimized
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
                # Momentum Indicators dengan parameter optimized
                rsi_period = params['RSI_PERIOD']
                rsi = talib.RSI(closes, timeperiod=rsi_period)
                
                macd_fast = params['MACD_FAST']
                macd_slow = params['MACD_SLOW'] 
                macd_signal = params['MACD_SIGNAL']
                macd, macd_signal_line, macd_hist = talib.MACD(closes, 
                                                            fastperiod=macd_fast,
                                                            slowperiod=macd_slow, 
                                                            signalperiod=macd_signal)
                
                stoch_k_period = params['STOCH_K']
                stoch_d_period = params['STOCH_D']
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes,
                                            fastk_period=stoch_k_period,
                                            slowk_period=3,
                                            slowd_period=stoch_d_period)
                
                williams_r = talib.WILLR(highs, lows, closes, timeperiod=14)
            except Exception as e:
                logger.warning(f"Error calculating momentum indicators: {e}")
                rsi = np.full_like(closes, 50)
                macd, macd_signal_line, macd_hist = np.zeros_like(closes), np.zeros_like(closes), np.zeros_like(closes)
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
            
            # Tentukan trend strength
            trend_strength = self._calculate_trend_strength(
                sma_20[-1], sma_50[-1], adx[-1], macd_hist[-1], current_price
            )
            
            return {
                'trend': {
                    'sma_20': safe_float(sma_20[-1], current_price),
                    'sma_50': safe_float(sma_50[-1], current_price),
                    'ema_12': safe_float(ema_12[-1], current_price),
                    'ema_26': safe_float(ema_26[-1], current_price),
                    'adx': safe_float(adx[-1], 25),
                    'trend_direction': 'BULLISH' if safe_float(sma_20[-1], current_price) > safe_float(sma_50[-1], current_price) else 'BEARISH',
                    'trend_strength': trend_strength['strength'],
                    'trend_score': trend_strength['score']
                },
                'momentum': {
                    'rsi': safe_float(rsi[-1], 50),
                    'macd': safe_float(macd[-1], 0),
                    'macd_signal': safe_float(macd_signal_line[-1], 0),
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
                },
                'timeframe_params': params
            }
        except Exception as e:
            logger.error(f"Error calculating optimized indicators: {e}")
            return self._fallback_indicators(df)

    def _calculate_trend_strength(self, sma_20: float, sma_50: float, adx: float, macd_hist: float, price: float) -> Dict:
        """Hitung strength trend dengan scoring system"""
        score = 0
        strength = "WEAK"
        
        # SMA Alignment
        if (sma_20 > sma_50 and sma_20 > sma_50 * 1.005) or (sma_20 < sma_50 and sma_20 < sma_50 * 0.995):
            score += 2
        
        # ADX Strength
        if adx > 40:
            score += 3
            strength = "STRONG"
        elif adx > 25:
            score += 2
            strength = "MODERATE"
        elif adx > 20:
            score += 1
        
        # MACD Momentum
        if abs(macd_hist) > 0.002:
            score += 2
        elif abs(macd_hist) > 0.001:
            score += 1
        
        # Price position relative to SMA
        if (price > sma_20 > sma_50) or (price < sma_20 < sma_50):
            score += 1
        
        # Adjust strength based on final score
        if score >= 6:
            strength = "VERY STRONG"
        elif score >= 4:
            strength = "STRONG"
        elif score >= 2:
            strength = "MODERATE"
        
        return {'strength': strength, 'score': score}

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Backward compatibility - gunakan parameter default"""
        return self.calculate_optimized_indicators(df, '4H')

    def multi_timeframe_confirmation(self, pair: str, current_timeframe: str) -> Dict:
        """Konfirmasi sinyal dengan timeframe yang lebih tinggi"""
        confirmation = {
            'confirmed': False,
            'strength': 0,
            'higher_tf_trend': 'NEUTRAL',
            'confidence_boost': 0,
            'higher_tf_adx': 0
        }
        
        try:
            # Mapping timeframe untuk konfirmasi
            tf_mapping = {
                'M30': '1H',
                '1H': '4H',
                '4H': '1D'
            }
            
            higher_tf = tf_mapping.get(current_timeframe, '1D')
            
            higher_data = data_manager.get_price_data(pair, higher_tf, days=60)
            
            if len(higher_data) > 20:
                higher_tech = self.calculate_optimized_indicators(higher_data, higher_tf)
                current_data = data_manager.get_price_data(pair, current_timeframe, days=30)
                current_tech = self.calculate_optimized_indicators(current_data, current_timeframe)
                
                # Konfirmasi trend alignment
                higher_trend = higher_tech['trend']['trend_direction']
                current_trend = current_tech['trend']['trend_direction']
                
                confirmation['higher_tf_trend'] = higher_trend
                confirmation['higher_tf_adx'] = higher_tech['trend']['adx']
                
                if higher_trend == current_trend:
                    confirmation['confirmed'] = True
                    confirmation['confidence_boost'] = 20
                    
                    # Tambah boost berdasarkan strength trend higher TF
                    if higher_tech['trend']['adx'] > 30:
                        confirmation['confidence_boost'] += 15
                        confirmation['strength'] = 3
                    elif higher_tech['trend']['adx'] > 20:
                        confirmation['confidence_boost'] += 10
                        confirmation['strength'] = 2
                    else:
                        confirmation['strength'] = 1
                        
                logger.info(f"MTF Confirmation for {pair}-{current_timeframe}: {higher_tf} trend={higher_trend}, confirmed={confirmation['confirmed']}, boost={confirmation['confidence_boost']}")
                    
        except Exception as e:
            logger.error(f"Multi-timeframe confirmation error for {pair}: {e}")
        
        return confirmation

    def advanced_market_filters(self, pair: str, timeframe: str) -> bool:
        """Filter kondisi market yang tidak ideal untuk trading"""
        try:
            data = data_manager.get_price_data(pair, timeframe, days=20)
            if len(data) < 15:
                return True
                
            tech = self.calculate_optimized_indicators(data, timeframe)
            params = config.TIMEFRAME_OPTIMIZED_PARAMS.get(timeframe, config.TIMEFRAME_OPTIMIZED_PARAMS['4H'])
            
            # Filter 1: Hindari extreme volatility
            if tech['volatility']['volatility_pct'] > params['MAX_VOLATILITY']:
                logger.info(f"Skip {pair}-{timeframe}: Extreme volatility {tech['volatility']['volatility_pct']:.3f}%")
                return False
                
            # Filter 2: Hindari market yang terlalu flat
            if tech['volatility']['bollinger_bandwidth'] < 2.0:
                logger.info(f"Skip {pair}-{timeframe}: Low volatility (boring market)")
                return False
                
            # Filter 3: Pastikan ada trend yang jelas
            if tech['trend']['adx'] < params['MIN_ADX']:
                logger.info(f"Skip {pair}-{timeframe}: Weak trend (ADX: {tech['trend']['adx']:.1f})")
                return False
                
            # Filter 4: Hindari kondisi overbought/oversold extreme
            rsi = tech['momentum']['rsi']
            if rsi < 15 or rsi > 85:
                logger.info(f"Skip {pair}-{timeframe}: Extreme RSI ({rsi:.1f})")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Market filter error for {pair}: {e}")
            return True

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
                'trend_strength': 'MODERATE',
                'trend_score': 3
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
            },
            'timeframe_params': config.TIMEFRAME_OPTIMIZED_PARAMS['4H']
        }

# ==================== DATA MANAGER TANPA SUPPORT 1W ====================
class EnhancedDataManager:
    def __init__(self):
        self.historical_data = {}
        self.load_historical_data()

    def has_timeframe_data(self, pair: str, timeframe: str) -> bool:
        """Cek apakah data untuk timeframe tersedia"""
        return pair in self.historical_data and timeframe in self.historical_data[pair]

    def load_historical_data(self):
        """Load data historis tanpa resample 1W"""
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
                        
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'], errors='coerce', format='mixed')
                            df = df.dropna(subset=['date'])
                        
                        required_cols = ['open', 'high', 'low', 'close']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
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
                            
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")
                        continue
            
            logger.info(f"Loaded {loaded_count} datasets")
            
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
        """Buat sample data untuk semua timeframe tanpa 1W"""
        logger.info("Creating enhanced sample historical data...")
        
        for pair in config.FOREX_PAIRS:
            for timeframe in ['M30', '1H', '4H', '1D']:
                self._generate_sample_data(pair, timeframe)

    def _generate_sample_data(self, pair: str, timeframe: str):
        """Generate sample data yang realistis"""
        try:
            # Tentukan periods berdasarkan timeframe
            if timeframe == 'M30':
                periods = 2000
            elif timeframe == '1H':
                periods = 1500
            elif timeframe == '4H':
                periods = 1000
            else:  # 1D
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
            else:  # 1D
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
                else:  # 1D
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
        """Dapatkan data harga dengan fallback"""
        try:
            if pair in self.historical_data and timeframe in self.historical_data[pair]:
                df = self.historical_data[pair][timeframe]
                if df.empty:
                    return self._generate_simple_data(pair, timeframe, days)
                
                if timeframe == 'M30':
                    required_points = min(len(df), days * 48)
                elif timeframe == '1H':
                    required_points = min(len(df), days * 24)
                elif timeframe == '4H':
                    required_points = min(len(df), days * 6)
                else:  # 1D
                    required_points = min(len(df), days)
                    
                result_df = df.tail(required_points).copy()
                
                required_cols = ['date', 'open', 'high', 'low', 'close']
                for col in required_cols:
                    if col not in result_df.columns:
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
        else:  # 1D
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
            else:  # 1D
                current_date = start_date + timedelta(days=i)
            
            prices.append({
                'date': current_date,
                'open': round(float(open_price), 4),
                'high': round(float(high), 4),
                'low': round(float(low), 4),
                'close': round(float(close_price), 4),
                'volume': int(np.random.randint(10000, 50000))
            })
        
        return pd.DataFrame(prices)

# ==================== ENHANCED SIGNAL GENERATOR ====================
def generate_enhanced_signals(price_data: pd.DataFrame, pair: str, timeframe: str) -> List[Dict]:
    """Generate sinyal dengan filter dan konfirmasi yang lebih baik"""
    signals = []
    
    try:
        if len(price_data) < 50:
            return signals
            
        tech_engine = TechnicalAnalysisEngine()
        
        # Filter kondisi market terlebih dahulu
        if not tech_engine.advanced_market_filters(pair, timeframe):
            logger.info(f"Market filters blocked signals for {pair}-{timeframe}")
            return signals
        
        current_tech = tech_engine.calculate_optimized_indicators(price_data, timeframe)
        
        # Dapatkan parameter optimized
        params = config.TIMEFRAME_OPTIMIZED_PARAMS.get(timeframe, config.TIMEFRAME_OPTIMIZED_PARAMS['4H'])
        
        # Multi-timeframe confirmation
        mtf_confirmation = tech_engine.multi_timeframe_confirmation(pair, timeframe)
        
        # Generate signals hanya dengan kondisi kuat
        base_confidence = 0
        
        # BUY Conditions dengan konfirmasi kuat
        buy_conditions = [
            current_tech['momentum']['rsi'] < 35,
            current_tech['momentum']['macd_histogram'] > 0.001,
            current_tech['trend']['trend_direction'] == 'BULLISH',
            current_tech['momentum']['stoch_k'] < 25,
            mtf_confirmation['confirmed'] and mtf_confirmation['higher_tf_trend'] == 'BULLISH',
            current_tech['trend']['adx'] > params['MIN_ADX'],
            current_tech['volatility']['volatility_pct'] < params['MAX_VOLATILITY']
        ]
        
        # SELL Conditions dengan konfirmasi kuat  
        sell_conditions = [
            current_tech['momentum']['rsi'] > 65,
            current_tech['momentum']['macd_histogram'] < -0.001,
            current_tech['trend']['trend_direction'] == 'BEARISH',
            current_tech['momentum']['stoch_k'] > 75,
            mtf_confirmation['confirmed'] and mtf_confirmation['higher_tf_trend'] == 'BEARISH',
            current_tech['trend']['adx'] > params['MIN_ADX'],
            current_tech['volatility']['volatility_pct'] < params['MAX_VOLATILITY']
        ]
        
        buy_score = sum(1 for condition in buy_conditions if condition)
        sell_score = sum(1 for condition in sell_conditions if condition)
        
        # Hanya eksekusi jika minimal 4 kondisi terpenuhi
        if buy_score >= 4:
            confidence = 60 + (buy_score * 5) + mtf_confirmation['confidence_boost']
            if confidence >= params['MIN_CONFIDENCE']:
                signals.append({
                    'date': price_data.iloc[-1]['date'],
                    'pair': pair,
                    'action': 'BUY',
                    'confidence': min(95, confidence),
                    'price': current_tech['levels']['current_price'],
                    'mtf_strength': mtf_confirmation['strength'],
                    'conditions_met': buy_score,
                    'rsi': current_tech['momentum']['rsi'],
                    'adx': current_tech['trend']['adx'],
                    'macd_hist': current_tech['momentum']['macd_histogram']
                })
                
        elif sell_score >= 4:
            confidence = 60 + (sell_score * 5) + mtf_confirmation['confidence_boost']
            if confidence >= params['MIN_CONFIDENCE']:
                signals.append({
                    'date': price_data.iloc[-1]['date'],
                    'pair': pair,
                    'action': 'SELL', 
                    'confidence': min(95, confidence),
                    'price': current_tech['levels']['current_price'],
                    'mtf_strength': mtf_confirmation['strength'],
                    'conditions_met': sell_score,
                    'rsi': current_tech['momentum']['rsi'],
                    'adx': current_tech['trend']['adx'],
                    'macd_hist': current_tech['momentum']['macd_histogram']
                })
        
        logger.info(f"Enhanced signals for {pair}-{timeframe}: {len(signals)} signals with confidence {[s['confidence'] for s in signals]}")
        
    except Exception as e:
        logger.error(f"Enhanced signal generation error: {e}")
        
    return signals

# Backward compatibility
def generate_trading_signals(price_data: pd.DataFrame, pair: str, timeframe: str) -> List[Dict]:
    return generate_enhanced_signals(price_data, pair, timeframe)

# ==================== INISIALISASI KOMPONEN SISTEM ====================
logger.info("Initializing Enhanced Forex Analysis System...")

tech_engine = TechnicalAnalysisEngine()
data_manager = EnhancedDataManager()

# Inisialisasi komponen lainnya
class TwelveDataClient:
    def __init__(self):
        self.api_key = config.TWELVE_DATA_KEY
        self.demo_mode = not self.api_key or self.api_key == "demo"
        
    def get_real_time_price(self, pair: str) -> float:
        if self.demo_mode:
            base_prices = {'USDJPY': 147.25, 'EURUSD': 1.0835}
            return base_prices.get(pair, 150.0)
        return 150.0

class FundamentalAnalysisEngine:
    def get_forex_news(self, pair: str) -> str:
        return "Market analysis ongoing."

class DeepSeekAnalyzer:
    def analyze_market(self, pair: str, technical_data: Dict, fundamental_news: str) -> Dict:
        return {
            "signal": "HOLD",
            "confidence": 50,
            "analysis_summary": "Demo analysis"
        }

class AdvancedRiskManager:
    def validate_trade(self, **kwargs) -> Dict:
        return {
            'approved': True,
            'risk_score': 0,
            'adjusted_lot_size': 0.1
        }

twelve_data_client = TwelveDataClient()
fundamental_engine = FundamentalAnalysisEngine()
deepseek_analyzer = DeepSeekAnalyzer()
risk_manager = AdvancedRiskManager()

logger.info("Enhanced Forex Analysis System initialized successfully")

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', 
                         pairs=config.FOREX_PAIRS,
                         timeframes=config.TIMEFRAMES,
                         initial_balance=config.INITIAL_BALANCE)

@app.route('/api/analyze')
def api_analyze():
    try:
        pair = request.args.get('pair', 'USDJPY').upper()
        timeframe = request.args.get('timeframe', '4H').upper()
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        real_time_price = twelve_data_client.get_real_time_price(pair)
        
        price_data = data_manager.get_price_data(pair, timeframe, days=60)
        if price_data.empty:
            return jsonify({'error': 'No price data available'}), 400
        
        technical_analysis = tech_engine.calculate_optimized_indicators(price_data, timeframe)
        technical_analysis['levels']['current_price'] = real_time_price
        
        fundamental_news = fundamental_engine.get_forex_news(pair)
        ai_analysis = deepseek_analyzer.analyze_market(pair, technical_analysis, fundamental_news)
        
        risk_assessment = risk_manager.validate_trade(
            pair=pair,
            signal=ai_analysis.get('signal', 'HOLD'),
            confidence=ai_analysis.get('confidence', 50)
        )
        
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'fundamental_analysis': fundamental_news,
            'ai_analysis': ai_analysis,
            'risk_assessment': risk_assessment,
            'price_data': {
                'current': real_time_price,
                'support': technical_analysis.get('levels', {}).get('support'),
                'resistance': technical_analysis.get('levels', {}).get('resistance')
            },
            'analysis_summary': f"{pair} analysis completed"
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    try:
        data = request.get_json()
        pair = data.get('pair', 'USDJPY')
        timeframe = data.get('timeframe', '4H')
        days = int(data.get('days', 30))
        
        logger.info(f"Backtest request: {pair}-{timeframe} for {days} days")
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        price_data = data_manager.get_price_data(pair, timeframe, days)
        
        if price_data.empty:
            return jsonify({'error': 'No price data available for backtesting'}), 400
        
        signals = generate_enhanced_signals(price_data, pair, timeframe)
        
        result = {
            'status': 'success',
            'signals_generated': len(signals),
            'signals': signals[:10],  # Limit to first 10 signals
            'metadata': {
                'pair': pair,
                'timeframe': timeframe,
                'days': days,
                'testing_date': datetime.now().isoformat()
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500

@app.route('/api/system_status')
def api_system_status():
    return jsonify({
        'system': 'RUNNING',
        'historical_data': f"{len(data_manager.historical_data)} pairs loaded",
        'supported_pairs': config.FOREX_PAIRS,
        'version': '4.0',
        'enhanced_features': [
            'Multi-Timeframe Analysis',
            'Optimized Parameters for 1H/4H',
            'Advanced Market Filters', 
            'Enhanced Signal Generation'
        ]
    })

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("Starting Enhanced Forex Analysis System v4.0...")
    logger.info(f"Supported pairs: {config.FOREX_PAIRS}")
    logger.info(f"Historical data: {len(data_manager.historical_data)} pairs loaded")
    
    os.makedirs('historical_data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Enhanced Forex Analysis System is ready and running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
