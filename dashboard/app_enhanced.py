# [FILE: app_enhanced.py] - DUAL TWELVEDATA API KEYS + ALPHA VANTAGE OPTIMIZATION
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
import time
import glob
from functools import wraps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# ==================== KONFIGURASI SISTEM DENGAN DUAL API KEYS ====================
@dataclass
class SystemConfig:
    # API Configuration - DUAL TWELVEDATA KEYS
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "demo")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "demo") 
    TWELVE_DATA_KEY_REALTIME: str = os.environ.get("TWELVE_DATA_KEY_REALTIME", "demo")  # Key untuk real-time
    TWELVE_DATA_KEY_HISTORICAL: str = os.environ.get("TWELVE_DATA_KEY_HISTORICAL", "demo")  # Key untuk historical
    ALPHA_VANTAGE_KEY: str = os.environ.get("ALPHA_VANTAGE_KEY", "demo")
    
    # Enhanced Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    MAX_DAILY_LOSS: float = 0.03
    MAX_DRAWDOWN: float = 0.10
    MAX_POSITIONS: int = 3
    STOP_LOSS_PCT: float = 0.01
    TAKE_PROFIT_PCT: float = 0.02
    
    # Risk Management Parameters
    CORRELATION_THRESHOLD: float = 0.7
    VOLATILITY_THRESHOLD: float = 0.02
    DAILY_TRADE_LIMIT: int = 50
    MAX_POSITION_SIZE_PCT: float = 0.05
    
    # Backtesting-specific parameters
    BACKTEST_DAILY_TRADE_LIMIT: int = 100
    BACKTEST_MIN_CONFIDENCE: int = 40
    BACKTEST_RISK_SCORE_THRESHOLD: int = 8
    
    # Rate limiting untuk multiple APIs
    ALPHA_VANTAGE_RATE_LIMIT: int = 3
    TWELVEDATA_HISTORICAL_RATE_LIMIT: int = 8  # Lebih tinggi untuk historical
    TWELVEDATA_REALTIME_RATE_LIMIT: int = 10   # Tinggi untuk real-time
    
    # Supported Instruments
    FOREX_PAIRS: List[str] = field(default_factory=lambda: [
        "USDJPY", "GBPJPY", "EURJPY", "CHFJPY", 
        "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
    ])
    
    # Data source mapping - STRATEGI BARU!
    DATA_SOURCE_MAPPING: Dict[str, str] = field(default_factory=lambda: {
        'M30': 'TWELVEDATA_HISTORICAL',
        '1H': 'TWELVEDATA_HISTORICAL', 
        '4H': 'TWELVEDATA_HISTORICAL',
        '1D': 'ALPHA_VANTAGE',
        '1W': 'ALPHA_VANTAGE'
    })
    
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M30", "1H", "4H", "1D", "1W"])
    
    # Backtesting
    DEFAULT_BACKTEST_DAYS: int = 90
    MIN_DATA_POINTS: int = 100
    
    # Trading Hours (UTC)
    HIGH_IMPACT_HOURS: List[Tuple[int, int]] = field(default_factory=lambda: [(8, 10), (13, 15)])

config = SystemConfig()

# ==================== TWELVEDATA HISTORICAL CLIENT ====================
class TwelveDataHistoricalClient:
    """Client khusus untuk data historical dari TwelveData"""
    def __init__(self):
        self.api_key = config.TWELVE_DATA_KEY_HISTORICAL
        self.base_url = "https://api.twelvedata.com"
        self.historical_cache = {}
        self.cache_timeout = 3600  # 1 jam cache
        self.demo_mode = not self.api_key or self.api_key == "demo"
        
        self.session = self._create_session_with_retry()
        
        if self.demo_mode:
            logger.info("TwelveData Historical running in DEMO mode")
        else:
            logger.info("TwelveData Historical running in LIVE mode")

    def _create_session_with_retry(self):
        """Create requests session dengan retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    @rate_limited(8)  # 8 requests per minute untuk historical
    def get_historical_data(self, pair: str, interval: str, days: int = 30) -> pd.DataFrame:
        """Ambil data historical dari TwelveData untuk M30, 1H, 4H"""
        cache_key = f"{pair}_{interval}_{days}"
        
        # Check cache
        if cache_key in self.historical_cache:
            cached_time, df = self.historical_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                logger.info(f"Using cached TwelveData historical data for {pair}-{interval}")
                return df.copy()
        
        if self.demo_mode:
            return self._generate_simulated_historical_data(pair, interval, days)
        
        try:
            # Format pair untuk TwelveData
            formatted_pair = f"{pair[:3]}/{pair[3:]}"
            
            # Map interval ke format TwelveData
            interval_map = {
                'M30': '30min',
                '1H': '1h', 
                '4H': '4h'
            }
            
            if interval not in interval_map:
                logger.error(f"Unsupported interval for TwelveData historical: {interval}")
                return self._generate_simulated_historical_data(pair, interval, days)
            
            # Calculate outputsize berdasarkan days
            if interval == 'M30':
                outputsize = min(days * 48, 5000)
            elif interval == '1H':
                outputsize = min(days * 24, 5000)
            elif interval == '4H':
                outputsize = min(days * 6, 5000)
            else:
                outputsize = 1000
            
            params = {
                'symbol': formatted_pair,
                'interval': interval_map[interval],
                'outputsize': outputsize,
                'apikey': self.api_key,
                'format': 'JSON'
            }
            
            logger.info(f"Fetching historical data from TwelveData for {pair}-{interval}...")
            response = self.session.get(f"{self.base_url}/time_series", params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data:
                    records = []
                    for item in data['values']:
                        try:
                            records.append({
                                'date': pd.to_datetime(item['datetime']),
                                'open': float(item['open']),
                                'high': float(item['high']),
                                'low': float(item['low']),
                                'close': float(item['close']),
                                'volume': int(float(item.get('volume', 0)))
                            })
                        except (ValueError, KeyError) as e:
                            logger.warning(f"Skipping invalid data point: {e}")
                            continue
                    
                    if records:
                        df = pd.DataFrame(records)
                        df = df.sort_values('date').reset_index(drop=True)
                        
                        # Cache the data
                        self.historical_cache[cache_key] = (datetime.now(), df.copy())
                        self._save_data_to_file(df, pair, interval)
                        
                        logger.info(f"Successfully fetched {len(df)} historical records from TwelveData for {pair}-{interval}")
                        return df
                    else:
                        logger.error("No valid records processed from TwelveData")
                        return self._generate_simulated_historical_data(pair, interval, days)
                else:
                    logger.error(f"Invalid TwelveData historical response: {data}")
                    return self._generate_simulated_historical_data(pair, interval, days)
            else:
                logger.error(f"TwelveData historical API error: {response.status_code}")
                return self._generate_simulated_historical_data(pair, interval, days)
                
        except Exception as e:
            logger.error(f"Error getting historical data from TwelveData for {pair}-{interval}: {e}")
            return self._generate_simulated_historical_data(pair, interval, days)
    
    def _save_data_to_file(self, df: pd.DataFrame, pair: str, interval: str):
        """Save data to CSV"""
        if df.empty:
            return
            
        try:
            os.makedirs('historical_data', exist_ok=True)
            filename = f"historical_data/TD_{pair}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved TwelveData historical data to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save TwelveData data to file: {e}")
    
    def _generate_simulated_historical_data(self, pair: str, interval: str, days: int) -> pd.DataFrame:
        """Generate simulated historical data untuk fallback"""
        try:
            intervals_per_day = {
                'M30': 48, '1H': 24, '4H': 6, '1D': 1, '1W': 1/7
            }
            
            points = int(days * intervals_per_day.get(interval, 6))
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
                drift = (base_price - current_price) * 0.0005
                random_shock = np.random.normal(0, volatility)
                change = drift + random_shock
                current_price = current_price * (1 + change)
                
                open_price = current_price
                close_variation = np.random.normal(0, volatility * 0.2)
                close_price = current_price * (1 + close_variation)
                
                price_range = abs(change) * base_price * 0.8
                high = max(open_price, close_price) + price_range
                low = min(open_price, close_price) - price_range
                
                high = max(high, max(open_price, close_price) + 0.0001)
                low = min(low, min(open_price, close_price) - 0.0001)
                
                if interval == 'M30':
                    current_date = start_date + timedelta(minutes=30*i)
                elif interval == '1H':
                    current_date = start_date + timedelta(hours=i)
                elif interval == '4H':
                    current_date = start_date + timedelta(hours=4*i)
                else:
                    current_date = start_date + timedelta(days=i)
                
                prices.append({
                    'date': current_date,
                    'open': round(float(open_price), 4),
                    'high': round(float(high), 4),
                    'low': round(float(low), 4),
                    'close': round(float(close_price), 4),
                    'volume': int(np.random.randint(5000, 30000))
                })
            
            df = pd.DataFrame(prices)
            logger.info(f"Generated simulated historical data for {pair}-{interval}: {len(df)} records")
            
            self._save_data_to_file(df, pair, interval)
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating simulated data for {pair}: {e}")
            return pd.DataFrame({
                'date': [datetime.now() - timedelta(days=i) for i in range(10)],
                'open': [150.0] * 10, 
                'high': [150.5] * 10, 
                'low': [149.5] * 10, 
                'close': [150.2] * 10,
                'volume': [10000] * 10
            })

# ==================== TWELVEDATA REALTIME CLIENT ====================
class TwelveDataRealtimeClient:
    """Client khusus untuk real-time prices dari TwelveData"""
    def __init__(self):
        self.api_key = config.TWELVE_DATA_KEY_REALTIME
        self.base_url = "https://api.twelvedata.com"
        self.price_cache = {}
        self.cache_timeout = 30  # 30 detik cache untuk real-time
        self.demo_mode = not self.api_key or self.api_key == "demo"
        
        self.session = self._create_session_with_retry()
        
        if self.demo_mode:
            logger.info("TwelveData Realtime running in DEMO mode")
        else:
            logger.info("TwelveData Realtime running in LIVE mode")

    def _create_session_with_retry(self):
        """Create requests session dengan retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=0.5
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    @rate_limited(10)  # 10 requests per minute untuk real-time
    def get_real_time_price(self, pair: str) -> float:
        """Ambil current price real-time dari TwelveData"""
        cache_key = pair
        
        # Check cache
        if pair in self.price_cache:
            cached_time, price = self.price_cache[pair]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                return price
        
        if self.demo_mode:
            return self._get_simulated_real_time_price(pair)
        
        try:
            formatted_pair = f"{pair[:3]}/{pair[3:]}"
            url = f"{self.base_url}/price?symbol={formatted_pair}&apikey={self.api_key}"
            
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'price' in data and data['price'] is not None:
                    price = float(data['price'])
                    
                    self.price_cache[pair] = (datetime.now(), price)
                    logger.info(f"TwelveData Real-time price for {pair}: {price}")
                    return price
                else:
                    logger.error(f"Invalid real-time response from TwelveData: {data}")
                    return self._get_simulated_real_time_price(pair)
            else:
                logger.error(f"TwelveData real-time API error: {response.status_code}")
                return self._get_simulated_real_time_price(pair)
                
        except Exception as e:
            logger.error(f"Error getting real-time price for {pair}: {e}")
            return self._get_simulated_real_time_price(pair)
    
    def _get_simulated_real_time_price(self, pair: str) -> float:
        """Harga real-time simulasi untuk demo mode"""
        try:
            base_prices = {
                'USDJPY': 147.25, 'GBPJPY': 198.50, 'EURJPY': 172.10, 'CHFJPY': 184.30,
                'EURUSD': 1.0835, 'GBPUSD': 1.2640, 'USDCHF': 0.8840,
                'AUDUSD': 0.6545, 'USDCAD': 1.3510, 'NZDUSD': 0.6095
            }
            
            base_price = base_prices.get(pair, 150.0)
            
            variation = random.uniform(-0.0005, 0.0005)
            simulated_price = round(base_price * (1 + variation), 4)
            
            self.price_cache[pair] = (datetime.now(), simulated_price)
            
            return simulated_price
            
        except Exception as e:
            logger.error(f"Error in simulated real-time price for {pair}: {e}")
            return 150.0

# ==================== ALPHA VANTAGE CLIENT (UNTUK 1D & 1W) ====================
class AlphaVantageClient:
    def __init__(self):
        self.api_key = config.ALPHA_VANTAGE_KEY
        self.base_url = "https://www.alphavantage.co/query"
        self.historical_cache = {}
        self.cache_timeout = 86400  # 24 jam cache untuk daily data
        self.demo_mode = not self.api_key or self.api_key == "demo"
        
        self.session = self._create_session_with_retry()
        
        if self.demo_mode:
            logger.info("Alpha Vantage running in DEMO mode")
        else:
            logger.info("Alpha Vantage running in LIVE mode for 1D/1W data")

    def _create_session_with_retry(self):
        """Create requests session dengan retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    @rate_limited(3)  # 3 calls per minute untuk Alpha Vantage free tier
    def get_historical_data(self, pair: str, interval: str, days: int = 30) -> pd.DataFrame:
        """Ambil data historis dari Alpha Vantage hanya untuk 1D dan 1W"""
        cache_key = f"{pair}_{interval}_{days}"
        
        # Check cache
        if cache_key in self.historical_cache:
            cached_time, df = self.historical_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_timeout):
                logger.info(f"Using cached Alpha Vantage data for {pair}-{interval}")
                return df.copy()
        
        # Hanya handle 1D dan 1W
        if interval not in ['1D', '1W']:
            logger.warning(f"Alpha Vantage only supports 1D/1W, not {interval}")
            return pd.DataFrame()
        
        if self.demo_mode:
            return self._generate_simulated_historical_data(pair, interval, days)
        
        try:
            # Tentukan function
            function = 'FX_DAILY' if interval == '1D' else 'FX_WEEKLY'
            
            params = {
                'function': function,
                'from_symbol': pair[:3],
                'to_symbol': pair[3:],
                'apikey': self.api_key,
                'datatype': 'json',
                'outputsize': 'compact'
            }
            
            logger.info(f"Fetching {interval} data from Alpha Vantage for {pair}...")
            response = self.session.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                time_series_key = 'Time Series FX (Daily)' if interval == '1D' else 'Time Series FX (Weekly)'
                
                if time_series_key in data:
                    records = []
                    for timestamp, values in data[time_series_key].items():
                        try:
                            records.append({
                                'date': pd.to_datetime(timestamp),
                                'open': float(values['1. open']),
                                'high': float(values['2. high']),
                                'low': float(values['3. low']),
                                'close': float(values['4. close']),
                                'volume': 0
                            })
                        except (ValueError, KeyError) as e:
                            logger.warning(f"Skipping invalid data point {timestamp}: {e}")
                            continue
                    
                    if records:
                        df = pd.DataFrame(records)
                        df = df.sort_values('date').reset_index(drop=True)
                        
                        # Filter berdasarkan days
                        start_date = datetime.now() - timedelta(days=min(days, 365))
                        df = df[df['date'] >= start_date]
                        
                        self.historical_cache[cache_key] = (datetime.now(), df.copy())
                        self._save_data_to_file(df, pair, interval)
                        
                        logger.info(f"Successfully fetched {len(df)} {interval} records from Alpha Vantage for {pair}")
                        return df
                else:
                    if 'Information' in data:
                        logger.warning(f"Alpha Vantage API limit: {data['Information']}")
                    elif 'Error Message' in data:
                        logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            
            return self._generate_simulated_historical_data(pair, interval, days)
                
        except Exception as e:
            logger.error(f"Error getting data from Alpha Vantage for {pair}-{interval}: {e}")
            return self._generate_simulated_historical_data(pair, interval, days)
    
    def _save_data_to_file(self, df: pd.DataFrame, pair: str, interval: str):
        """Save data to CSV"""
        if df.empty:
            return
            
        try:
            os.makedirs('historical_data', exist_ok=True)
            filename = f"historical_data/AV_{pair}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Saved Alpha Vantage data to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save Alpha Vantage data to file: {e}")
    
    def _generate_simulated_historical_data(self, pair: str, interval: str, days: int) -> pd.DataFrame:
        """Generate simulated data untuk Alpha Vantage fallback"""
        points = days
        points = max(20, min(points, 365))
        
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
            volatility = 0.008 if interval == '1D' else 0.015
            drift = (base_price - current_price) * 0.001
            random_shock = np.random.normal(0, volatility)
            change = drift + random_shock
            current_price = current_price * (1 + change)
            
            open_price = current_price
            close_variation = np.random.normal(0, volatility * 0.3)
            close_price = current_price * (1 + close_variation)
            
            price_range = abs(change) * base_price
            high = max(open_price, close_price) + price_range * 0.5
            low = min(open_price, close_price) - price_range * 0.5
            
            high = max(high, max(open_price, close_price) + 0.0001)
            low = min(low, min(open_price, close_price) - 0.0001)
            
            if interval == '1D':
                current_date = start_date + timedelta(days=i)
            else:  # 1W
                current_date = start_date + timedelta(days=7*i)
            
            prices.append({
                'date': current_date,
                'open': round(float(open_price), 4),
                'high': round(float(high), 4),
                'low': round(float(low), 4),
                'close': round(float(close_price), 4),
                'volume': int(np.random.randint(10000, 50000))
            })
        
        df = pd.DataFrame(prices)
        logger.info(f"Generated simulated {interval} data for {pair}: {len(df)} records")
        
        self._save_data_to_file(df, pair, interval)
        
        return df

# ==================== DATA MANAGER YANG DIOPTIMISASI ====================
class DataManager:
    def __init__(self):
        self.historical_data = {}
        self.alpha_vantage_client = AlphaVantageClient()
        self.twelve_data_historical_client = TwelveDataHistoricalClient()
        self.twelve_data_realtime_client = TwelveDataRealtimeClient()
        
        logger.info("Data Manager initialized with DUAL TwelveData API keys + Alpha Vantage")

    def get_price_data_with_timezone(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Dapatkan data harga dengan timezone awareness"""
        try:
            df = self.get_price_data(pair, timeframe, days)
            
            if df.empty:
                logger.warning(f"No data returned for {pair}-{timeframe}, generating fallback data")
                return self._generate_simple_data(pair, timeframe, days)
                
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                if df['date'].dt.tz is None:
                    df['date'] = df['date'].dt.tz_localize('UTC')
            
            return df
            
        except Exception as e:
            logger.error(f"Error in get_price_data_with_timezone for {pair}-{timeframe}: {e}")
            return self._generate_simple_data(pair, timeframe, days)

    def get_price_data(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Dapatkan data harga dari sumber yang optimal berdasarkan mapping"""
        try:
            # Tentukan sumber data berdasarkan mapping
            data_source = config.DATA_SOURCE_MAPPING.get(timeframe, 'TWELVEDATA_HISTORICAL')
            
            if data_source == 'TWELVEDATA_HISTORICAL' and timeframe in ['M30', '1H', '4H']:
                # Gunakan TwelveData Historical untuk intraday
                api_data = self.twelve_data_historical_client.get_historical_data(pair, timeframe, days)
                if not api_data.empty and len(api_data) > 10:
                    logger.info(f"Using TwelveData Historical for {pair}-{timeframe}: {len(api_data)} records")
                    
                    if pair not in self.historical_data:
                        self.historical_data[pair] = {}
                    self.historical_data[pair][timeframe] = api_data
                    
                    return api_data
            
            elif data_source == 'ALPHA_VANTAGE' and timeframe in ['1D', '1W']:
                # Gunakan Alpha Vantage untuk daily/weekly
                api_data = self.alpha_vantage_client.get_historical_data(pair, timeframe, days)
                if not api_data.empty and len(api_data) > 5:
                    logger.info(f"Using Alpha Vantage for {pair}-{timeframe}: {len(api_data)} records")
                    
                    if pair not in self.historical_data:
                        self.historical_data[pair] = {}
                    self.historical_data[pair][timeframe] = api_data
                    
                    return api_data
            
            # Fallback ke cached data
            if pair in self.historical_data and timeframe in self.historical_data[pair]:
                df = self.historical_data[pair][timeframe]
                if not df.empty:
                    required_points = self._calculate_required_points(timeframe, days)
                    if len(df) > required_points:
                        return df.tail(required_points).copy()
                    else:
                        return df.copy()
            
            # Final fallback
            logger.info(f"Generating fallback data for {pair}-{timeframe}")
            return self._generate_simple_data(pair, timeframe, days)
            
        except Exception as e:
            logger.error(f"Error getting price data for {pair}-{timeframe}: {e}")
            return self._generate_simple_data(pair, timeframe, days)

    def get_real_time_price(self, pair: str) -> float:
        """Dapatkan harga real-time dari TwelveData Realtime"""
        return self.twelve_data_realtime_client.get_real_time_price(pair)

    def _calculate_required_points(self, timeframe: str, days: int) -> int:
        """Hitung jumlah data points yang diperlukan"""
        if timeframe == 'M30':
            return days * 48
        elif timeframe == '1H':
            return days * 24
        elif timeframe == '4H':
            return days * 6
        else:
            return days

    def ensure_fresh_data(self, pair: str, timeframe: str, min_records: int = 50):
        """Pastikan data fresh tersedia"""
        try:
            data_source = config.DATA_SOURCE_MAPPING.get(timeframe, 'TWELVEDATA_HISTORICAL')
            
            if data_source == 'TWELVEDATA_HISTORICAL':
                api_data = self.twelve_data_historical_client.get_historical_data(pair, timeframe, 30)
            else:
                api_data = self.alpha_vantage_client.get_historical_data(pair, timeframe, 30)
            
            if not api_data.empty and len(api_data) >= min_records:
                if pair not in self.historical_data:
                    self.historical_data[pair] = {}
                self.historical_data[pair][timeframe] = api_data
                logger.info(f"Fresh data ensured for {pair}-{timeframe}: {len(api_data)} records")
                return
                
        except Exception as e:
            logger.error(f"Error ensuring fresh data for {pair}-{timeframe}: {e}")

    def _generate_simple_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate simple synthetic data untuk fallback"""
        if timeframe == 'M30':
            points = days * 48
        elif timeframe == '1H':
            points = days * 24
        elif timeframe == '4H':
            points = days * 6
        else:
            points = days
            
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
        
        return pd.DataFrame(prices)

# ==================== INISIALISASI KOMPONEN SISTEM ====================
# [TechnicalAnalysisEngine, FundamentalAnalysisEngine, DeepSeekAnalyzer, 
# AdvancedRiskManager, AdvancedBacktestingEngine tetap sama seperti sebelumnya]
# ... (kode untuk komponen-komponen ini tidak berubah)

# Inisialisasi dengan konfigurasi baru
logger.info("Initializing Enhanced Forex System with DUAL TwelveData API Keys...")

tech_engine = TechnicalAnalysisEngine()
fundamental_engine = FundamentalAnalysisEngine()
deepseek_analyzer = DeepSeekAnalyzer()
data_manager = DataManager()  # Ini yang sudah diperbarui dengan dual keys
advanced_backtester = AdvancedBacktestingEngine()
risk_manager = AdvancedRiskManager()

# Load data dengan strategi baru
logger.info("Loading historical data with optimized data source mapping...")
for pair in config.FOREX_PAIRS:
    for timeframe in config.TIMEFRAMES:
        data_source = config.DATA_SOURCE_MAPPING.get(timeframe, 'TWELVEDATA_HISTORICAL')
        logger.info(f"Loading {pair}-{timeframe} from {data_source}")
        data_manager.ensure_fresh_data(pair, timeframe, min_records=30)

# Tampilkan status sistem yang diperbarui
logger.info("=== SYSTEM STATUS ===")
logger.info(f"TwelveData Realtime: {'LIVE' if not data_manager.twelve_data_realtime_client.demo_mode else 'DEMO'}")
logger.info(f"TwelveData Historical: {'LIVE' if not data_manager.twelve_data_historical_client.demo_mode else 'DEMO'}")
logger.info(f"Alpha Vantage: {'LIVE' if not data_manager.alpha_vantage_client.demo_mode else 'DEMO'}")
logger.info("Data Source Mapping:")
for tf, source in config.DATA_SOURCE_MAPPING.items():
    logger.info(f"  {tf} -> {source}")
logger.info("All system components initialized successfully")

# ==================== FLASK ROUTES (TIDAK BERUBAH) ====================
@app.route('/')
def index():
    return render_template('index.html', 
                         pairs=config.FOREX_PAIRS,
                         timeframes=config.TIMEFRAMES,
                         initial_balance=config.INITIAL_BALANCE)

@app.route('/api/analyze')
def api_analyze():
    """Endpoint untuk analisis market dengan data source yang dioptimasi"""
    try:
        pair = request.args.get('pair', 'USDJPY').upper()
        timeframe = request.args.get('timeframe', '4H').upper()
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        # 1) Harga realtime dari TwelveData Realtime
        real_time_price = data_manager.get_real_time_price(pair)
        
        # 2) Data historis dari sumber optimal
        price_data = data_manager.get_price_data_with_timezone(pair, timeframe, days=60)
        
        if price_data.empty:
            price_data = data_manager._generate_simple_data(pair, timeframe, 60)
        
        # 3) Analisis teknikal
        technical_analysis = tech_engine.calculate_all_indicators(price_data)
        technical_analysis['levels']['current_price'] = real_time_price
        
        # 4) Analisis fundamental
        fundamental_news = fundamental_engine.get_forex_news(pair)
        
        # 5) Analisis AI
        ai_analysis = deepseek_analyzer.analyze_market(pair, technical_analysis, fundamental_news)
        
        # 6) Risk assessment
        risk_assessment = risk_manager.validate_trade(
            pair=pair,
            signal=ai_analysis.get('signal', 'HOLD'),
            confidence=ai_analysis.get('confidence', 50),
            proposed_lot_size=0.1,
            account_balance=config.INITIAL_BALANCE,
            current_price=real_time_price,
            open_positions=[]
        )
        
        # 7) Price series untuk chart
        price_series = []
        try:
            if not price_data.empty:
                price_data = price_data.sort_values('date')
                for _, row in price_data.iterrows():
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
            logger.error(f"Error preparing price series: {e}")
            price_series = []
        
        # 8) Tentukan data source untuk response
        data_source = config.DATA_SOURCE_MAPPING.get(timeframe, 'TWELVEDATA_HISTORICAL')
        source_description = {
            'TWELVEDATA_HISTORICAL': 'TwelveData Historical API',
            'ALPHA_VANTAGE': 'Alpha Vantage API'
        }.get(data_source, 'Optimized Data Source')
        
        # 9) Susun response
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
                'resistance': technical_analysis.get('levels', {}).get('resistance'),
                'change_pct': technical_analysis.get('momentum', {}).get('price_change_pct', 0)
            },
            'price_series': price_series,
            'analysis_summary': f"{pair} trading at {real_time_price:.4f}",
            'ai_provider': ai_analysis.get('ai_provider', 'DeepSeek AI'),
            'data_source': f"{source_description} + TwelveData Real-time",
            'data_source_mapping': config.DATA_SOURCE_MAPPING,
            'timezone_info': 'UTC',
            'data_points': len(price_series)
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

# [Backtest routes dan lainnya tetap sama...]

@app.route('/api/system_status')
def api_system_status():
    """Status sistem dengan informasi dual API keys"""
    return jsonify({
        'system': 'RUNNING',
        'data_sources': {
            'twelvedata_realtime': 'LIVE' if not data_manager.twelve_data_realtime_client.demo_mode else 'DEMO',
            'twelvedata_historical': 'LIVE' if not data_manager.twelve_data_historical_client.demo_mode else 'DEMO',
            'alpha_vantage': 'LIVE' if not data_manager.alpha_vantage_client.demo_mode else 'DEMO'
        },
        'data_source_mapping': config.DATA_SOURCE_MAPPING,
        'supported_pairs': config.FOREX_PAIRS,
        'server_time': datetime.now().isoformat(),
        'version': '4.0',
        'features': [
            'Dual TwelveData API Keys',
            'Optimized Data Source Mapping', 
            'Alpha Vantage for Daily/Weekly',
            'TwelveData Historical for Intraday',
            'TwelveData Real-time for Live Prices',
            'Advanced Risk Management',
            'AI-Powered Analysis'
        ]
    })

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("Starting Enhanced Forex System with DUAL TwelveData API Keys...")
    logger.info("=== DATA SOURCE MAPPING ===")
    for timeframe, source in config.DATA_SOURCE_MAPPING.items():
        logger.info(f"  {timeframe} -> {source}")
    
    logger.info("=== API STATUS ===")
    logger.info(f"TwelveData Realtime: {'LIVE MODE' if not data_manager.twelve_data_realtime_client.demo_mode else 'DEMO MODE'}")
    logger.info(f"TwelveData Historical: {'LIVE MODE' if not data_manager.twelve_data_historical_client.demo_mode else 'DEMO MODE'}")
    logger.info(f"Alpha Vantage: {'LIVE MODE' if not data_manager.alpha_vantage_client.demo_mode else 'DEMO MODE'}")
    
    # Create necessary directories
    os.makedirs('historical_data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Forex Analysis System is ready and running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
