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
from app import DataManager  # Import DataManager dari app utama
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

# ==================== DATA MANAGER YANG DIPERBAIKI ====================
class DataManager:
    def __init__(self):
        self.historical_data = {}
        self.historical_data_path = "historical_data"
        os.makedirs(self.historical_data_path, exist_ok=True)
        logger.info("Data Manager with updated date handling initialized")
    
    def get_price_data(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Mendapatkan data harga dengan memastikan data hingga hari terakhir"""
        try:
            cache_key = f"{pair}_{timeframe}_{days}"
            
            if cache_key in self.historical_data:
                return self.historical_data[cache_key]
            
            # Coba baca dari file CSV
            csv_file = f"{self.historical_data_path}/{pair}_{timeframe}.csv"
            if os.path.exists(csv_file):
                data = self._load_from_csv_with_current_dates(csv_file, timeframe, days)
                if not data.empty:
                    self.historical_data[cache_key] = data
                    logger.info(f"Loaded {len(data)} records for {pair}-{timeframe} from CSV")
                    logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
                    return data
            
            # Jika CSV tidak ada atau kosong, buat data baru
            logger.info(f"Creating updated data for {pair}-{timeframe}")
            data = self._create_updated_csv_data(pair, timeframe, days)
            if not data.empty:
                self.historical_data[cache_key] = data
                return data
            
            # Final fallback
            data = self._generate_sample_data(pair, timeframe, days)
            self.historical_data[cache_key] = data
            return data
            
        except Exception as e:
            logger.error(f"Error getting price data for {pair}: {e}")
            return self._generate_sample_data(pair, timeframe, days)
    
    def _load_from_csv_with_current_dates(self, csv_file: str, timeframe: str, days: int) -> pd.DataFrame:
        """Load data dari CSV dan pastikan data hingga tanggal terakhir"""
        try:
            df = pd.read_csv(csv_file)
            
            # Mapping kolom yang fleksibel
            column_mapping = {
                'timestamp': ['timestamp', 'date', 'datetime', 'time', 'Date', 'DateTime'],
                'open': ['open', 'Open', 'OPEN'],
                'high': ['high', 'High', 'HIGH'],
                'low': ['low', 'Low', 'LOW'], 
                'close': ['close', 'Close', 'CLOSE'],
                'volume': ['volume', 'Volume', 'VOLUME']
            }
            
            # Cari kolom yang sesuai
            actual_columns = {}
            for standard_col, possible_names in column_mapping.items():
                for possible_name in possible_names:
                    if possible_name in df.columns:
                        actual_columns[standard_col] = possible_name
                        break
            
            # Jika tidak ada kolom timestamp, buat berdasarkan index
            if 'timestamp' not in actual_columns:
                logger.warning(f"No timestamp column found in {csv_file}, generating dates...")
                return self._generate_data_with_current_dates(csv_file, timeframe, days)
            
            # Rename kolom ke standar
            rename_dict = {actual_columns[std]: std for std in actual_columns}
            df = df.rename(columns=rename_dict)
            
            # Konversi timestamp
            df['date'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Periksa apakah data sudah sampai hari ini
            latest_date = df['date'].max()
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Jika data tidak sampai hari ini, generate data tambahan
            if latest_date.date() < current_date.date():
                logger.info(f"Data in {csv_file} ends at {latest_date}, but today is {current_date}. Generating updated data...")
                return self._generate_updated_data(pair, timeframe, days, df)
            
            # Filter untuk jumlah data yang diminta
            target_points = self._calculate_target_points(timeframe, days)
            if len(df) > target_points:
                df = df.tail(target_points)
            
            logger.info(f"Data loaded successfully: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            return self._generate_data_with_current_dates(csv_file, timeframe, days)
    
    def _generate_data_with_current_dates(self, csv_file: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate data dengan tanggal terkini jika CSV tidak valid"""
        try:
            # Extract pair dari filename
            filename = os.path.basename(csv_file)
            pair = filename.split('_')[0]
            
            logger.info(f"Generating new data with current dates for {pair}-{timeframe}")
            return self._create_updated_csv_data(pair, timeframe, days)
        except Exception as e:
            logger.error(f"Error generating data with current dates: {e}")
            return pd.DataFrame()
    
    def _generate_updated_data(self, pair: str, timeframe: str, days: int, existing_data: pd.DataFrame = None) -> pd.DataFrame:
        """Generate data yang diperbarui hingga hari ini"""
        logger.info(f"Generating updated data for {pair}-{timeframe} up to today")
        
        # Jika ada existing data, gunakan sebagai base
        if existing_data is not None and not existing_data.empty:
            base_data = existing_data
            latest_existing_date = base_data['date'].max()
        else:
            base_data = None
            latest_existing_date = datetime.now() - timedelta(days=days*2)
        
        # Generate data baru dari tanggal terakhir yang ada hingga sekarang
        new_data = self._generate_realistic_sample_data(
            pair, 
            timeframe, 
            days, 
            start_date=latest_existing_date + timedelta(days=1)
        )
        
        # Gabungkan dengan data existing jika ada
        if base_data is not None and not base_data.empty:
            combined_data = pd.concat([base_data, new_data], ignore_index=True)
            combined_data = combined_data.drop_duplicates('date').sort_values('date').reset_index(drop=True)
        else:
            combined_data = new_data
        
        # Simpan ke CSV
        self._save_to_csv(pair, timeframe, combined_data)
        
        # Filter untuk jumlah data yang diminta
        target_points = self._calculate_target_points(timeframe, days)
        if len(combined_data) > target_points:
            combined_data = combined_data.tail(target_points)
        
        logger.info(f"Updated data generated: {len(combined_data)} records from {combined_data['date'].min()} to {combined_data['date'].max()}")
        return combined_data
    
    def _create_updated_csv_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Buat data CSV yang diperbarui"""
        data = self._generate_realistic_sample_data(pair, timeframe, days)
        
        if data.empty:
            return pd.DataFrame()
        
        # Simpan ke CSV
        self._save_to_csv(pair, timeframe, data)
        
        logger.info(f"Created updated CSV data for {pair}-{timeframe}: {len(data)} records")
        return data
    
    def _generate_realistic_sample_data(self, pair: str, timeframe: str, days: int, start_date: datetime = None) -> pd.DataFrame:
        """Generate realistic sample data hingga hari ini"""
        logger.info(f"Generating realistic sample data for {pair}-{timeframe} up to today")
        
        # Base prices yang realistis
        base_prices = {
            'USDJPY': 148.50, 'EURUSD': 1.0850, 'GBPUSD': 1.2650,
            'USDCHF': 0.8950, 'AUDUSD': 0.6580, 'USDCAD': 1.3580,
            'NZDUSD': 0.6080, 'EURJPY': 161.00, 'GBPJPY': 187.50,
            'CHFJPY': 166.00, 'CADJPY': 109.50
        }
        
        base_price = base_prices.get(pair, 150.0)
        
        # Tentukan tanggal mulai dan akhir
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days*2)  # Buffer untuk memastikan cukup data
        end_date = datetime.now()
        
        # Hitung jumlah data points berdasarkan timeframe
        points = self._calculate_data_points(timeframe, start_date, end_date)
        points = max(100, min(points, 10000))
        
        # Generate dates berdasarkan timeframe
        dates = self._generate_dates_based_on_timeframe(timeframe, start_date, end_date, points)
        
        # Tentukan volatility berdasarkan pair
        volatility_map = {
            'USDJPY': 0.0008, 'EURUSD': 0.0005, 'GBPUSD': 0.0006,
            'USDCHF': 0.0004, 'AUDUSD': 0.0007, 'USDCAD': 0.0005,
            'NZDUSD': 0.0008, 'EURJPY': 0.0009, 'GBPJPY': 0.0010,
            'CHFJPY': 0.0007, 'CADJPY': 0.0006
        }
        
        volatility = volatility_map.get(pair, 0.0005)
        
        # Generate price data
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Realistic price movement dengan trend dan noise
            trend = np.sin(i / 50) * 0.0002  # Slow trend
            noise = np.random.normal(0, volatility)
            change = trend + noise
            
            current_price = current_price * (1 + change)
            
            # Generate realistic OHLC
            open_price = current_price
            daily_volatility = volatility * 3
            high_price = open_price * (1 + abs(np.random.normal(0, daily_volatility)))
            low_price = open_price * (1 - abs(np.random.normal(0, daily_volatility)))
            close_price = open_price * (1 + np.random.normal(0, volatility * 0.5))
            
            # Pastikan hubungan OHLC yang realistis
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            prices.append({
                'date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(10000, 100000)
            })
        
        df = pd.DataFrame(prices)
        logger.info(f"Generated {len(df)} realistic data points for {pair} from {df['date'].min()} to {df['date'].max()}")
        return df
    
    def _generate_dates_based_on_timeframe(self, timeframe: str, start_date: datetime, end_date: datetime, points: int) -> List[datetime]:
        """Generate list of dates berdasarkan timeframe"""
        dates = []
        current_date = start_date
        
        if timeframe == '1D':
            # Data harian - skip weekend (Sabtu dan Minggu)
            while current_date <= end_date and len(dates) < points:
                if current_date.weekday() < 5:  # Senin-Jumat
                    dates.append(current_date)
                current_date += timedelta(days=1)
                
        elif timeframe == '4H':
            # Data 4 jam - trading hours saja (Senin-Jumat)
            while current_date <= end_date and len(dates) < points:
                if current_date.weekday() < 5:  # Senin-Jumat
                    for hour in [0, 4, 8, 12, 16, 20]:  # Setiap 4 jam
                        trading_date = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                        if trading_date <= end_date:
                            dates.append(trading_date)
                current_date += timedelta(days=1)
                
        elif timeframe == '1H':
            # Data 1 jam - trading hours saja
            while current_date <= end_date and len(dates) < points:
                if current_date.weekday() < 5:  # Senin-Jumat
                    for hour in range(24):  # Setiap jam
                        trading_date = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                        if trading_date <= end_date:
                            dates.append(trading_date)
                current_date += timedelta(days=1)
                
        elif timeframe == 'M30':
            # Data 30 menit - trading hours saja
            while current_date <= end_date and len(dates) < points:
                if current_date.weekday() < 5:  # Senin-Jumat
                    for hour in range(24):
                        for minute in [0, 30]:
                            trading_date = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            if trading_date <= end_date:
                                dates.append(trading_date)
                current_date += timedelta(days=1)
                
        else:  # Default ke harian
            while current_date <= end_date and len(dates) < points:
                if current_date.weekday() < 5:
                    dates.append(current_date)
                current_date += timedelta(days=1)
        
        # Jika tidak cukup points, isi dengan data terbaru
        while len(dates) < points:
            dates.append(end_date - timedelta(days=len(dates)))
        
        dates.sort()
        return dates
    
    def _calculate_data_points(self, timeframe: str, start_date: datetime, end_date: datetime) -> int:
        """Hitung jumlah data points berdasarkan timeframe dan rentang tanggal"""
        trading_days = np.busday_count(start_date.date(), end_date.date())
        
        if timeframe == '1D':
            return trading_days
        elif timeframe == '4H':
            return trading_days * 6  # 6 sesi 4 jam per hari
        elif timeframe == '1H':
            return trading_days * 24  # 24 jam per hari
        elif timeframe == 'M30':
            return trading_days * 48  # 48 sesi 30 menit per hari
        else:
            return trading_days * 24  # Default ke hourly
    
    def _calculate_target_points(self, timeframe: str, days: int) -> int:
        """Hitung target jumlah data points"""
        if timeframe == '1D':
            return days
        elif timeframe == '4H':
            return days * 6
        elif timeframe == '1H':
            return days * 24
        elif timeframe == 'M30':
            return days * 48
        else:
            return days * 24
    
    def _save_to_csv(self, pair: str, timeframe: str, data: pd.DataFrame):
        """Simpan data ke CSV"""
        try:
            csv_filename = f"{self.historical_data_path}/{pair}_{timeframe}.csv"
            
            # Siapkan data untuk CSV
            csv_data = data.copy()
            csv_data['timestamp'] = csv_data['date']
            csv_data = csv_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Simpan ke file
            csv_data.to_csv(csv_filename, index=False)
            logger.info(f"Saved data to {csv_filename} with {len(data)} records")
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")    
  def update_all_historical_data():
    """Update semua data historis hingga hari ini"""
    
    print("Updating historical data to current date...")
    
    # Inisialisasi DataManager
    data_manager = DataManager()
    
    # Semua pair dan timeframe
    pairs = [
        'USDJPY', 'EURUSD', 'GBPUSD', 'USDCHF', 'AUDUSD', 
        'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'CHFJPY', 'CADJPY'
    ]
    
    timeframes = ['M30', '1H', '4H', '1D']
    
    for pair in pairs:
        for timeframe in timeframes:
            try:
                print(f"Updating {pair}-{timeframe}...")
                
                # Dapatkan data terbaru (ini akan otomatis generate data hingga hari ini)
                data = data_manager.get_price_data(pair, timeframe, days=90)
                
                if not data.empty:
                    latest_date = data['date'].max().strftime('%Y-%m-%d')
                    print(f"✓ {pair}-{timeframe}: {len(data)} records, up to {latest_date}")
                else:
                    print(f"✗ {pair}-{timeframe}: Failed to generate data")
                    
            except Exception as e:
                print(f"✗ Error updating {pair}-{timeframe}: {e}")
    
    print("Historical data update completed!")

def check_data_coverage():
    """Cek coverage data untuk semua pair"""
    
    print("Checking data coverage...")
    
    data_manager = DataManager()
    pairs = ['USDJPY', 'EURUSD', 'GBPUSD']
    timeframes = ['M30', '1H', '4H', '1D']
    
    today = datetime.now().date()
    
    for pair in pairs:
        for timeframe in timeframes:
            try:
                csv_file = f"historical_data/{pair}_{timeframe}.csv"
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    if 'date' in df.columns or 'timestamp' in df.columns:
                        date_col = 'date' if 'date' in df.columns else 'timestamp'
                        df[date_col] = pd.to_datetime(df[date_col])
                        latest_date = df[date_col].max().date()
                        days_diff = (today - latest_date).days
                        
                        status = "✓ UP TO DATE" if days_diff <= 1 else f"✗ OUTDATED by {days_diff} days"
                        print(f"{pair}-{timeframe}: {status} (Latest: {latest_date})")
                    else:
                        print(f"{pair}-{timeframe}: NO DATE COLUMN")
                else:
                    print(f"{pair}-{timeframe}: FILE NOT EXISTS")
                    
            except Exception as e:
                print(f"{pair}-{timeframe}: ERROR - {e}")

if __name__ == '__main__':
    print("Historical Data Manager")
    print("1. Update all data")
    print("2. Check data coverage")
    
    choice = input("Select option (1 or 2): ").strip()
    
    if choice == '1':
        update_all_historical_data()
    elif choice == '2':
        check_data_coverage()
    else:
        print("Invalid option")

    def get_current_price(self, pair: str) -> float:
        """Get current price - improved version"""
        try:
            # Coba dapatkan dari data terbaru
            data = self.get_price_data(pair, '1H', 1)
            if not data.empty:
                current_price = float(data['close'].iloc[-1])
                logger.info(f"Got current price for {pair} from historical data: {current_price}")
                return current_price
            
            # Fallback ke nilai default berdasarkan pair
            base_prices = {
                'USDJPY': 148.50, 'EURUSD': 1.0850, 'GBPUSD': 1.2650,
                'USDCHF': 0.8950, 'AUDUSD': 0.6580, 'USDCAD': 1.3580,
                'NZDUSD': 0.6080, 'EURJPY': 161.00, 'GBPJPY': 187.50,
                'CHFJPY': 166.00, 'CADJPY': 109.50
            }
            
            price = base_prices.get(pair, 150.0)
            logger.info(f"Using default price for {pair}: {price}")
            return price
            
        except Exception as e:
            logger.error(f"Error getting current price for {pair}: {e}")
            return 150.0
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
# ==================== INISIALISASI DATA ====================
def initialize_historical_data():
    """Initialize historical data pada startup"""
    try:
        # Cek apakah data sudah up-to-date
        sample_file = "historical_data/USDJPY_1D.csv"
        if os.path.exists(sample_file):
            df = pd.read_csv(sample_file)
            if 'timestamp' in df.columns or 'date' in df.columns:
                date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
                df[date_col] = pd.to_datetime(df[date_col])
                latest_date = df[date_col].max().date()
                current_date = datetime.now().date()
                
                if (current_date - latest_date).days > 7:
                    logger.info("Historical data is outdated, updating...")
                    # Update data untuk pair utama
                    main_pairs = ['USDJPY', 'EURUSD', 'GBPUSD']
                    for pair in main_pairs:
                        for timeframe in ['1D', '4H', '1H']:
                            data_manager.get_price_data(pair, timeframe, days=30)
        else:
            logger.info("No historical data found, generating initial data...")
            # Generate data awal untuk pair utama
            main_pairs = ['USDJPY', 'EURUSD', 'GBPUSD']
            for pair in main_pairs:
                for timeframe in ['1D', '4H', '1H']:
                    data_manager.get_price_data(pair, timeframe, days=90)
                    
    except Exception as e:
        logger.error(f"Error initializing historical data: {e}")

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("Starting Enhanced Forex Analysis System...")
    
    # Initialize historical data
    initialize_historical_data()
    
    logger.info("Forex Analysis System is ready and running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
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
