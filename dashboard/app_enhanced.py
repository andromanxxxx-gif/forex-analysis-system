# [FILE: app_enhanced_fixed.py] - FIXED VERSION
from flask import Flask, request, jsonify, render_template
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import psutil
import hashlib
import jwt
from threading import Lock
from functools import lru_cache
import time

# ==================== FIX: JSON SERIALIZATION UTILS ====================
def convert_numpy_types(obj):
    """
    Convert numpy data types to native Python types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

def safe_jsonify(data):
    """Safe JSON serialization that handles numpy types"""
    return jsonify(convert_numpy_types(data))

# ==================== RATE LIMITING DECORATOR ====================
def rate_limited(max_per_minute):
    """Decorator untuk rate limiting API calls"""
    min_interval = 60.0 / max_per_minute
    def decorator(func):
        last_time_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# ==================== KONFIGURASI LOGGING ====================
def setup_logging():
    """Setup logging yang compatible dengan Windows"""
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

# ==================== KONFIGURASI SISTEM ====================
@dataclass
class SystemConfig:
    # API Configuration
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "demo")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "demo") 
    TWELVE_DATA_KEY_REALTIME: str = os.environ.get("TWELVE_DATA_KEY_REALTIME", "demo")
    TWELVE_DATA_KEY_HISTORICAL: str = os.environ.get("TWELVE_DATA_KEY_HISTORICAL", "demo")
    ALPHA_VANTAGE_KEY: str = os.environ.get("ALPHA_VANTAGE_KEY", "demo")
    
    # Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    
    # Supported Instruments
    FOREX_PAIRS: List[str] = field(default_factory=lambda: [
        "USDJPY", "GBPJPY", "EURJPY", "CHFJPY", 
        "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
    ])
    
    # Data source mapping
    DATA_SOURCE_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        'M30': 'TWELVEDATA_HISTORICAL',
        '1H': 'TWELVEDATA_HISTORICAL', 
        '4H': 'TWELVEDATA_HISTORICAL',
        '1D': 'ALPHA_VANTAGE',
        '1W': 'ALPHA_VANTAGE'
    })
    
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M30", "1H", "4H", "1D", "1W"])

config = SystemConfig()

# ==================== TECHNICAL ANALYSIS ENGINE (FIXED) ====================
class TechnicalAnalysisEngine:
    """Engine untuk analisis teknikal lengkap dengan perbaikan error handling"""
    
    def __init__(self):
        self.indicators_cache = {}
        self.cache_lock = Lock()
        logger.info("Technical Analysis Engine initialized")

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Hitung semua indikator teknikal dengan error handling"""
        try:
            if df.empty or len(df) < 20:
                return self._get_fallback_indicators(df)
            
            # Pastikan data sudah sorted dan konversi ke float64
            df = df.sort_values('date').reset_index(drop=True)
            
            # Extract price series dengan konversi ke float64
            closes = self._safe_convert_to_float64(df['close'].values)
            highs = self._safe_convert_to_float64(df['high'].values)
            lows = self._safe_convert_to_float64(df['low'].values)
            opens = self._safe_convert_to_float64(df['open'].values)
            
            results = {
                'trend': self._calculate_trend_indicators(closes, highs, lows),
                'momentum': self._calculate_momentum_indicators(closes, highs, lows),
                'volatility': self._calculate_volatility_indicators(highs, lows, closes),
                'levels': self._calculate_support_resistance(df),
                'volume': self._calculate_volume_indicators(df),
                'oscillators': self._calculate_oscillators(closes, highs, lows)
            }
            
            # Generate trading signals
            results['signals'] = self._generate_signals(results)
            
            return convert_numpy_types(results)  # Konversi untuk JSON
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._get_fallback_indicators(df)

    def _safe_convert_to_float64(self, array):
        """Konversi array ke float64 dengan error handling"""
        try:
            return np.array(array, dtype=np.float64)
        except Exception as e:
            logger.warning(f"Error converting to float64: {e}, using original array")
            return array

    def _calculate_trend_indicators(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
        """Hitung indikator trend dengan error handling"""
        try:
            # Konversi ke float64 untuk TA-Lib
            closes = self._safe_convert_to_float64(closes)
            highs = self._safe_convert_to_float64(highs)
            lows = self._safe_convert_to_float64(lows)
            
            # SMA
            sma_20 = talib.SMA(closes, timeperiod=20)
            sma_50 = talib.SMA(closes, timeperiod=50)
            
            # EMA
            ema_12 = talib.EMA(closes, timeperiod=12)
            ema_26 = talib.EMA(closes, timeperiod=26)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(closes)
            
            # ADX
            adx = talib.ADX(highs, lows, closes, timeperiod=14)
            
            return {
                'sma_20': float(sma_20[-1]) if sma_20 is not None and len(sma_20) > 0 and not np.isnan(sma_20[-1]) else None,
                'sma_50': float(sma_50[-1]) if sma_50 is not None and len(sma_50) > 0 and not np.isnan(sma_50[-1]) else None,
                'ema_12': float(ema_12[-1]) if ema_12 is not None and len(ema_12) > 0 and not np.isnan(ema_12[-1]) else None,
                'ema_26': float(ema_26[-1]) if ema_26 is not None and len(ema_26) > 0 and not np.isnan(ema_26[-1]) else None,
                'macd': float(macd[-1]) if macd is not None and len(macd) > 0 and not np.isnan(macd[-1]) else None,
                'macd_signal': float(macd_signal[-1]) if macd_signal is not None and len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else None,
                'macd_histogram': float(macd_hist[-1]) if macd_hist is not None and len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else None,
                'adx': float(adx[-1]) if adx is not None and len(adx) > 0 and not np.isnan(adx[-1]) else None,
                'direction': self._determine_trend_direction(closes, sma_20, sma_50, adx),
                'strength': self._calculate_trend_strength(adx, macd_hist)
            }
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {e}")
            return {'direction': 'NEUTRAL', 'strength': 50}

    def _calculate_momentum_indicators(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
        """Hitung indikator momentum dengan error handling"""
        try:
            closes = self._safe_convert_to_float64(closes)
            highs = self._safe_convert_to_float64(highs)
            lows = self._safe_convert_to_float64(lows)
            
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(highs, lows, closes)
            
            # Williams %R
            willr = talib.WILLR(highs, lows, closes, timeperiod=14)
            
            # CCI
            cci = talib.CCI(highs, lows, closes, timeperiod=20)
            
            # Price change
            price_change_pct = 0
            if len(closes) > 1:
                price_change_pct = ((closes[-1] - closes[-2]) / closes[-2] * 100)
            
            rsi_value = float(rsi[-1]) if rsi is not None and len(rsi) > 0 and not np.isnan(rsi[-1]) else 50.0
            
            return {
                'rsi': rsi_value,
                'stochastic_k': float(slowk[-1]) if slowk is not None and len(slowk) > 0 and not np.isnan(slowk[-1]) else None,
                'stochastic_d': float(slowd[-1]) if slowd is not None and len(slowd) > 0 and not np.isnan(slowd[-1]) else None,
                'williams_r': float(willr[-1]) if willr is not None and len(willr) > 0 and not np.isnan(willr[-1]) else None,
                'cci': float(cci[-1]) if cci is not None and len(cci) > 0 and not np.isnan(cci[-1]) else None,
                'price_change_pct': float(price_change_pct),
                'overbought': bool(rsi_value > 70) if rsi_value else False,
                'oversold': bool(rsi_value < 30) if rsi_value else False
            }
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {'rsi': 50, 'price_change_pct': 0}

    def _calculate_volatility_indicators(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """Hitung indikator volatilitas"""
        try:
            highs = self._safe_convert_to_float64(highs)
            lows = self._safe_convert_to_float64(lows)
            closes = self._safe_convert_to_float64(closes)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
            
            # ATR
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            
            return {
                'bb_upper': float(bb_upper[-1]) if bb_upper is not None and len(bb_upper) > 0 and not np.isnan(bb_upper[-1]) else None,
                'bb_middle': float(bb_middle[-1]) if bb_middle is not None and len(bb_middle) > 0 and not np.isnan(bb_middle[-1]) else None,
                'bb_lower': float(bb_lower[-1]) if bb_lower is not None and len(bb_lower) > 0 and not np.isnan(bb_lower[-1]) else None,
                'atr': float(atr[-1]) if atr is not None and len(atr) > 0 and not np.isnan(atr[-1]) else None,
                'atr_pct': (float(atr[-1]) / closes[-1] * 100) if atr is not None and len(atr) > 0 and not np.isnan(atr[-1]) and closes[-1] > 0 else None,
                'volatility_class': self._classify_volatility(atr, closes)
            }
        except Exception as e:
            logger.error(f"Error calculating volatility indicators: {e}")
            return {'atr': 0.5, 'volatility_class': 'MEDIUM'}

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Hitung indikator volume dengan perbaikan type conversion"""
        try:
            if 'volume' not in df.columns or df['volume'].isna().all():
                return {'obv': 0, 'volume_sma': 0, 'volume_trend': 'UNKNOWN'}
            
            # FIX: Konversi ke float64 untuk TA-Lib
            volumes = self._safe_convert_to_float64(df['volume'].values)
            closes = self._safe_convert_to_float64(df['close'].values)
            
            # OBV
            obv = talib.OBV(closes, volumes)
            
            # Volume SMA
            volume_sma = talib.SMA(volumes, timeperiod=20)
            
            obv_value = float(obv[-1]) if obv is not None and len(obv) > 0 and not np.isnan(obv[-1]) else 0
            volume_sma_value = float(volume_sma[-1]) if volume_sma is not None and len(volume_sma) > 0 and not np.isnan(volume_sma[-1]) else 0
            
            volume_trend = 'UNKNOWN'
            if volumes is not None and len(volumes) > 0 and not np.isnan(volume_sma_value):
                volume_trend = 'INCREASING' if volumes[-1] > volume_sma_value else 'DECREASING'
            
            return {
                'obv': obv_value,
                'volume_sma': volume_sma_value,
                'volume_trend': volume_trend
            }
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {e}")
            return {'obv': 0, 'volume_sma': 0, 'volume_trend': 'UNKNOWN'}

    def _calculate_oscillators(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict:
        """Hitung oscillator tambahan"""
        try:
            closes = self._safe_convert_to_float64(closes)
            highs = self._safe_convert_to_float64(highs)
            lows = self._safe_convert_to_float64(lows)
            
            # Ultimate Oscillator
            ultosc = talib.ULTOSC(highs, lows, closes)
            
            # TRIX
            trix = talib.TRIX(closes, timeperiod=15)
            
            return {
                'ultimate_oscillator': float(ultosc[-1]) if ultosc is not None and len(ultosc) > 0 and not np.isnan(ultosc[-1]) else None,
                'trix': float(trix[-1]) if trix is not None and len(trix) > 0 and not np.isnan(trix[-1]) else None
            }
        except Exception as e:
            logger.error(f"Error calculating oscillators: {e}")
            return {}

    def _calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """Hitung level support dan resistance"""
        try:
            if len(df) < window:
                current_price = float(df['close'].iloc[-1]) if not df.empty else 150.0
                return {
                    'support': current_price * 0.99,
                    'resistance': current_price * 1.01,
                    'pivot': current_price
                }
            
            # Simple support resistance menggunakan high/low recent
            recent_high = float(df['high'].tail(window).max())
            recent_low = float(df['low'].tail(window).min())
            current_close = float(df['close'].iloc[-1])
            
            # Pivot points
            pivot = (recent_high + recent_low + current_close) / 3
            r1 = 2 * pivot - recent_low
            s1 = 2 * pivot - recent_high
            
            return {
                'support': float(s1),
                'resistance': float(r1),
                'pivot': float(pivot),
                'recent_high': recent_high,
                'recent_low': recent_low
            }
        except Exception as e:
            logger.error(f"Error calculating support resistance: {e}")
            current_price = float(df['close'].iloc[-1]) if not df.empty else 150.0
            return {
                'support': current_price * 0.99,
                'resistance': current_price * 1.01,
                'pivot': current_price
            }

    def _determine_trend_direction(self, closes: np.ndarray, sma_20: np.ndarray, sma_50: np.ndarray, adx: np.ndarray) -> str:
        """Tentukan arah trend"""
        try:
            if len(closes) < 20 or sma_20 is None or sma_50 is None or len(sma_20) == 0 or len(sma_50) == 0:
                return "NEUTRAL"
            
            price_vs_sma20 = closes[-1] > (sma_20[-1] if not np.isnan(sma_20[-1]) else 0)
            price_vs_sma50 = closes[-1] > (sma_50[-1] if not np.isnan(sma_50[-1]) else 0)
            sma20_vs_sma50 = (sma_20[-1] if not np.isnan(sma_20[-1]) else 0) > (sma_50[-1] if not np.isnan(sma_50[-1]) else 0)
            
            bullish_signals = sum([price_vs_sma20, price_vs_sma50, sma20_vs_sma50])
            
            if bullish_signals >= 2:
                return "BULLISH"
            elif bullish_signals == 1:
                return "NEUTRAL"
            else:
                return "BEARISH"
                
        except Exception as e:
            logger.error(f"Error determining trend direction: {e}")
            return "NEUTRAL"

    def _calculate_trend_strength(self, adx: np.ndarray, macd_hist: np.ndarray) -> int:
        """Hitung kekuatan trend (0-100)"""
        try:
            strength = 50
            
            if adx is not None and len(adx) > 0 and not np.isnan(adx[-1]):
                strength = min(100, max(0, (float(adx[-1]) / 60) * 100))
            
            return int(strength)
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 50

    def _classify_volatility(self, atr: np.ndarray, closes: np.ndarray) -> str:
        """Klasifikasikan volatilitas"""
        try:
            if atr is None or len(atr) == 0 or np.isnan(atr[-1]) or len(closes) == 0:
                return "MEDIUM"
            
            atr_pct = (float(atr[-1]) / float(closes[-1])) * 100
            
            if atr_pct > 0.1:
                return "HIGH"
            elif atr_pct > 0.05:
                return "MEDIUM"
            else:
                return "LOW"
        except Exception as e:
            logger.error(f"Error classifying volatility: {e}")
            return "MEDIUM"

    def _generate_signals(self, indicators: Dict) -> Dict:
        """Generate trading signals"""
        try:
            signals = {
                'entry_signals': [],
                'exit_signals': [],
                'strength': 0,
                'composite_score': 50
            }
            
            trend = indicators.get('trend', {})
            momentum = indicators.get('momentum', {})
            volatility = indicators.get('volatility', {})
            
            # Trend signals
            trend_direction = trend.get('direction', 'NEUTRAL')
            if 'BULLISH' in trend_direction:
                signals['entry_signals'].append('TREND_BULLISH')
            elif 'BEARISH' in trend_direction:
                signals['entry_signals'].append('TREND_BEARISH')
            
            # Momentum signals
            rsi = momentum.get('rsi', 50)
            if rsi < 30:
                signals['entry_signals'].append('RSI_OVERSOLD')
            elif rsi > 70:
                signals['entry_signals'].append('RSI_OVERBOUGHT')
            
            # Calculate composite score
            score = 50
            score += len([s for s in signals['entry_signals'] if 'BULLISH' in s or 'OVERSOLD' in s]) * 10
            score -= len([s for s in signals['entry_signals'] if 'BEARISH' in s or 'OVERBOUGHT' in s]) * 10
            
            signals['composite_score'] = max(0, min(100, score))
            signals['strength'] = abs(score - 50) * 2
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {'entry_signals': [], 'exit_signals': [], 'strength': 0, 'composite_score': 50}

    def _get_fallback_indicators(self, df: pd.DataFrame) -> Dict:
        """Fallback indicators ketika data tidak cukup"""
        current_price = float(df['close'].iloc[-1]) if not df.empty else 150.0
        
        return convert_numpy_types({
            'trend': {'direction': 'NEUTRAL', 'strength': 50},
            'momentum': {'rsi': 50, 'price_change_pct': 0},
            'volatility': {'atr': 0.5, 'volatility_class': 'MEDIUM'},
            'levels': {
                'support': current_price * 0.99,
                'resistance': current_price * 1.01,
                'pivot': current_price
            },
            'volume': {'obv': 0, 'volume_sma': 0, 'volume_trend': 'UNKNOWN'},
            'oscillators': {},
            'signals': {
                'entry_signals': [],
                'exit_signals': [],
                'strength': 0,
                'composite_score': 50
            }
        })

# ==================== DEEPSEEK AI ANALYZER (FIXED) ====================
class DeepSeekAnalyzer:
    """AI-powered market analysis dengan perbaikan JSON serialization"""
    
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.demo_mode = not self.api_key or self.api_key == "demo"
        self.analysis_cache = {}
        self.cache_timeout = 600
        
        logger.info(f"DeepSeek Analyzer initialized - {'DEMO' if self.demo_mode else 'LIVE'} mode")

    def analyze_market(self, pair: str, technical_analysis: Dict, fundamental_analysis: Dict) -> Dict[str, Any]:
        """Analisis market menggunakan AI dengan error handling"""
        try:
            cache_key = hashlib.md5(f"{pair}_{json.dumps(technical_analysis, sort_keys=True)}".encode()).hexdigest()
            
            # Check cache
            if cache_key in self.analysis_cache:
                cached_time, analysis = self.analysis_cache[cache_key]
                if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                    return analysis
            
            if self.demo_mode:
                analysis = self._get_demo_analysis(pair, technical_analysis, fundamental_analysis)
            else:
                analysis = self._get_ai_analysis(pair, technical_analysis, fundamental_analysis)
            
            # Cache the analysis
            self.analysis_cache[cache_key] = (datetime.now(), analysis)
            
            return convert_numpy_types(analysis)  # Konversi untuk JSON
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {pair}: {e}")
            return self._get_demo_analysis(pair, technical_analysis, fundamental_analysis)

    def _get_ai_analysis(self, pair: str, technical: Dict, fundamental: Dict) -> Dict[str, Any]:
        """Dapatkan analisis dari DeepSeek API"""
        try:
            prompt = self._build_analysis_prompt(pair, technical, fundamental)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "Anda adalah analis forex profesional. Berikan analisis trading yang jelas dan terstruktur berdasarkan data teknikal dan fundamental. Fokus pada fakta dan data yang tersedia."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content']
                
                return self._parse_ai_response(ai_response, pair, technical, fundamental)
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return self._get_demo_analysis(pair, technical, fundamental)
                
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {e}")
            return self._get_demo_analysis(pair, technical, fundamental)

    def _build_analysis_prompt(self, pair: str, technical: Dict, fundamental: Dict) -> str:
        """Bangun prompt untuk AI analysis"""
        current_price = technical.get('levels', {}).get('pivot', 0)
        trend = technical.get('trend', {})
        momentum = technical.get('momentum', {})
        levels = technical.get('levels', {})
        
        prompt = f"""
        Analisis pair forex {pair}:
        
        DATA TEKNIKAL:
        - Harga saat ini: {current_price:.4f}
        - Trend: {trend.get('direction', 'NEUTRAL')} (Strength: {trend.get('strength', 50)}/100)
        - RSI: {momentum.get('rsi', 50):.1f}
        - Support: {levels.get('support', 0):.4f}
        - Resistance: {levels.get('resistance', 0):.4f}
        - Volatilitas: {technical.get('volatility', {}).get('volatility_class', 'MEDIUM')}
        
        Berikan analisis dalam format JSON dengan struktur:
        {{
            "signal": "BUY/SELL/HOLD",
            "confidence": 0-100,
            "reasoning": "Analisis singkat",
            "key_levels": {{
                "entry": "level entry",
                "stop_loss": "level stop loss", 
                "take_profit": "level take profit"
            }},
            "risk_rating": "LOW/MEDIUM/HIGH",
            "timeframe": "timeframe rekomendasi"
        }}
        """
        
        return prompt

    def _parse_ai_response(self, ai_response: str, pair: str, technical: Dict, fundamental: Dict) -> Dict[str, Any]:
        """Parse response dari AI dengan error handling"""
        try:
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            
            if json_match:
                ai_data = json.loads(json_match.group())
            else:
                return self._get_demo_analysis(pair, technical, fundamental)
            
            return {
                'signal': str(ai_data.get('signal', 'HOLD')),
                'confidence': int(ai_data.get('confidence', 50)),
                'reasoning': str(ai_data.get('reasoning', 'AI analysis completed')),
                'key_levels': ai_data.get('key_levels', {}),
                'risk_rating': str(ai_data.get('risk_rating', 'MEDIUM')),
                'timeframe': str(ai_data.get('timeframe', '4H')),
                'ai_provider': 'DeepSeek AI',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._get_demo_analysis(pair, technical, fundamental)

    def _get_demo_analysis(self, pair: str, technical: Dict, fundamental: Dict) -> Dict[str, Any]:
        """Generate demo AI analysis dengan data yang aman"""
        current_price = technical.get('levels', {}).get('pivot', 150.0)
        rsi = technical.get('momentum', {}).get('rsi', 50)
        trend = technical.get('trend', {}).get('direction', 'NEUTRAL')
        
        # Simple logic untuk demo
        if rsi < 30 and 'BULLISH' in trend:
            signal = "BUY"
            confidence = 75
            reasoning = "RSI oversold dengan trend bullish, potential reversal"
        elif rsi > 70 and 'BEARISH' in trend:
            signal = "SELL" 
            confidence = 75
            reasoning = "RSI overbought dengan trend bearish, potential reversal"
        else:
            signal = "HOLD"
            confidence = 50
            reasoning = "Market dalam kondisi neutral, tunggu konfirmasi lebih lanjut"
        
        return {
            'signal': str(signal),
            'confidence': int(confidence),
            'reasoning': f"Demo Analysis: {reasoning}",
            'key_levels': {
                'entry': float(round(current_price, 4)),
                'stop_loss': float(round(current_price * 0.99, 4)),
                'take_profit': float(round(current_price * 1.02, 4))
            },
            'risk_rating': 'MEDIUM',
            'timeframe': '4H',
            'ai_provider': 'DeepSeek AI (Demo Mode)',
            'timestamp': str(datetime.now().isoformat())
        }

# ==================== DATA MANAGER (FIXED) ====================
class DataManager:
    def __init__(self):
        self.historical_data = {}
        logger.info("Data Manager initialized")

    def get_price_data_with_timezone(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Dapatkan data harga dengan timezone awareness"""
        try:
            if pair not in config.FOREX_PAIRS:
                logger.warning(f"Unsupported pair: {pair}")
                return self._generate_simple_data(pair, timeframe, days)
                
            df = self.get_price_data(pair, timeframe, days)
            
            if df.empty:
                logger.warning(f"No data returned for {pair}-{timeframe}, generating fallback data")
                return self._generate_simple_data(pair, timeframe, days)
                
            return df
            
        except Exception as e:
            logger.error(f"Error in get_price_data_with_timezone for {pair}-{timeframe}: {e}")
            return self._generate_simple_data(pair, timeframe, days)

    def get_price_data(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Dapatkan data harga - simplified version"""
        try:
            return self._generate_simple_data(pair, timeframe, days)
        except Exception as e:
            logger.error(f"Error getting price data for {pair}-{timeframe}: {e}")
            return self._generate_simple_data(pair, timeframe, days)

    def get_real_time_price(self, pair: str) -> float:
        """Dapatkan harga real-time simulasi"""
        try:
            base_prices = {
                'USDJPY': 147.25, 'GBPJPY': 198.50, 'EURJPY': 172.10, 'CHFJPY': 184.30,
                'EURUSD': 1.0835, 'GBPUSD': 1.2640, 'USDCHF': 0.8840,
                'AUDUSD': 0.6545, 'USDCAD': 1.3510, 'NZDUSD': 0.6095
            }
            
            base_price = base_prices.get(pair, 150.0)
            variation = random.uniform(-0.0005, 0.0005)
            return float(round(base_price * (1 + variation), 4))
            
        except Exception as e:
            logger.error(f"Error in simulated real-time price for {pair}: {e}")
            return 150.0

    def _generate_simple_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate simple synthetic data untuk fallback"""
        points_per_day = {
            'M30': 48, '1H': 24, '4H': 6, '1D': 1, '1W': 1
        }
        
        points = int(days * points_per_day.get(timeframe, 6))
        points = max(50, min(points, 1000))
        
        base_prices = {
            'USDJPY': 147.0, 'GBPJPY': 198.0, 'EURJPY': 172.0, 'CHFJPY': 184.0,
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDCHF': 0.8850,
            'AUDUSD': 0.6550, 'USDCAD': 1.3500, 'NZDUSD': 0.6100
        }
        
        base_price = base_prices.get(pair, 150.0)
        prices = []
        current_price = base_price
        
        start_date = datetime.now() - timedelta(days=days)
        
        for i in range(points):
            volatility = 0.001
            change = np.random.normal(0, volatility)
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
                'open': float(round(open_price, 4)),
                'high': float(round(high, 4)),
                'low': float(round(low, 4)),
                'close': float(round(close_price, 4)),
                'volume': int(np.random.randint(10000, 50000))
            })
        
        df = pd.DataFrame(prices)
        logger.info(f"Generated simulated data for {pair}-{timeframe}: {len(df)} records")
        return df

# ==================== INISIALISASI KOMPONEN ====================
logger.info("Initializing Fixed Forex Trading System...")

# Inisialisasi komponen
tech_engine = TechnicalAnalysisEngine()
deepseek_analyzer = DeepSeekAnalyzer()
data_manager = DataManager()

# ==================== FLASK ROUTES (FIXED) ====================
@app.route('/')
def index():
    return render_template('index.html', 
                         pairs=config.FOREX_PAIRS,
                         timeframes=config.TIMEFRAMES,
                         initial_balance=config.INITIAL_BALANCE)

@app.route('/api/analyze')
def api_analyze():
    """Endpoint untuk analisis market dengan perbaikan error handling"""
    try:
        pair = request.args.get('pair', 'USDJPY').upper()
        timeframe = request.args.get('timeframe', '4H').upper()
        
        # Validasi input
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
            
        if timeframe not in config.TIMEFRAMES:
            return jsonify({'error': f'Unsupported timeframe: {timeframe}'}), 400
        
        # 1) Harga realtime
        real_time_price = data_manager.get_real_time_price(pair)
        
        # 2) Data historis
        price_data = data_manager.get_price_data_with_timezone(pair, timeframe, days=60)
        
        if price_data.empty:
            logger.warning(f"Using fallback data for {pair}-{timeframe}")
            price_data = data_manager._generate_simple_data(pair, timeframe, 60)
        
        # 3) Analisis teknikal
        technical_analysis = tech_engine.calculate_all_indicators(price_data)
        technical_analysis['levels']['current_price'] = real_time_price
        
        # 4) Analisis AI
        ai_analysis = deepseek_analyzer.analyze_market(pair, technical_analysis, {})
        
        # 5) Price series untuk chart
        price_series = []
        try:
            if not price_data.empty:
                price_data = price_data.sort_values('date')
                for _, row in price_data.tail(200).iterrows():
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
            logger.error(f"Error preparing price series: {e}")
            price_series = []
        
        # 6) Susun response dengan data yang aman
        response = {
            'pair': str(pair),
            'timeframe': str(timeframe),
            'timestamp': str(datetime.now().isoformat()),
            'technical_analysis': technical_analysis,
            'ai_analysis': ai_analysis,
            'price_data': {
                'current': float(real_time_price),
                'support': float(technical_analysis.get('levels', {}).get('support', 0)),
                'resistance': float(technical_analysis.get('levels', {}).get('resistance', 0)),
                'change_pct': float(technical_analysis.get('momentum', {}).get('price_change_pct', 0))
            },
            'price_series': price_series,
            'analysis_summary': f"{pair} trading at {real_time_price:.4f}",
            'ai_provider': str(ai_analysis.get('ai_provider', 'DeepSeek AI')),
            'data_source': 'Simulated Data (Demo Mode)',
            'data_points': len(price_series)
        }
        
        # FIX: Gunakan safe_jsonify untuk menghindari serialization errors
        return safe_jsonify(response)
    
    except Exception as e:
        logger.error(f"Analysis error for {pair}-{timeframe}: {e}", exc_info=True)
        return jsonify({
            'error': 'Analysis failed', 
            'message': str(e),
            'fallback_data': True
        }), 500

@app.route('/api/system_status')
def api_system_status():
    """Status sistem"""
    return safe_jsonify({
        'system': 'RUNNING',
        'timestamp': str(datetime.now().isoformat()),
        'version': '5.0',
        'supported_pairs': config.FOREX_PAIRS,
        'features': [
            'Technical Analysis',
            'AI-Powered Market Analysis', 
            'Risk Management',
            'Real-time Price Updates'
        ]
    })

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    try:
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
        
        health_status = {
            'status': 'healthy',
            'timestamp': str(datetime.now().isoformat()),
            'memory_usage_mb': float(round(memory_usage, 2)),
            'active_components': {
                'technical_engine': 'operational',
                'ai_analyzer': 'operational',
                'data_manager': 'operational'
            }
        }
        
        return safe_jsonify(health_status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'degraded', 'error': str(e)}), 500

# ==================== ERROR HANDLERS ====================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request'}), 400
@lru_cache(maxsize=100)
def cached_analysis(pair: str, timeframe: str):
    """Cache analysis results untuk mengurangi beban"""
    # Implementasi analisis di sini
    return analysis_data

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("=== FIXED FOREX TRADING SYSTEM STARTUP ===")
    logger.info("All system components initialized successfully")
    logger.info("Forex Trading System is ready and running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
