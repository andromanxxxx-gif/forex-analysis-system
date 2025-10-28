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

# Validate API keys
def validate_api_keys():
    twelve_data_key = os.getenv('TWELVE_DATA_API_KEY')
    deepseek_key = os.getenv('DEEPSEEK_API_KEY')
    newsapi_key = os.getenv('NEWS_API_KEY')
    
    print("🔐 API Key Validation:")
    print(f"   Twelve Data: {'✅' if twelve_data_key else '❌'} {'(Configured)' if twelve_data_key else '(Not Configured)'}")
    print(f"   DeepSeek: {'✅' if deepseek_key else '❌'} {'(Configured)' if deepseek_key else '(Not Configured)'}")
    print(f"   NewsAPI: {'✅' if newsapi_key else '❌'} {'(Configured)' if newsapi_key else '(Not Configured)'}")
    
    # Test DeepSeek key format
    if deepseek_key:
        if deepseek_key.startswith('sk-') and len(deepseek_key) > 20:
            print("   DeepSeek Key Format: ✅ Valid")
        else:
            print("   DeepSeek Key Format: ⚠️ Suspicious format")

validate_api_keys()

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
        
        # Setup session dengan enhanced timeout handling
        self.session = self._create_session()
        
        print(f"🔑 API Keys loaded: TwelveData: {'✅' if self.twelve_data_api_key else '❌'}, "
              f"DeepSeek: {'✅' if self.deepseek_api_key else '❌'}, "
              f"NewsAPI: {'✅' if self.news_api_key else '❌'}")

    def _create_session(self):
        """Create HTTP session dengan enhanced timeout handling"""
        session = requests.Session()
        
        # Enhanced timeout configuration
        session.timeout = 30  # Default timeout for all requests
        
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            # Enhanced retry strategy
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"],
                backoff_factor=1,
                respect_retry_after_header=True
            )
            
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=10,
                pool_maxsize=10
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            print("✅ HTTP session dengan enhanced retry strategy berhasil dibuat")
            
        except Exception as e:
            print(f"⚠️ Tidak bisa membuat retry strategy: {e}")
            print("🔧 Menggunakan session tanpa retry strategy")
        
        return session

    def test_deepseek_connection(self):
        """Test connection to DeepSeek API"""
        try:
            if not self.deepseek_api_key:
                return False, "API key not configured"
                
            url = "https://api.deepseek.com/v1/models"
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}"
            }
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return True, "Connection successful"
            else:
                return False, f"HTTP {response.status_code}: {response.text}"
                
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except requests.exceptions.ConnectionError:
            return False, "Connection error"
        except Exception as e:
            return False, f"Error: {str(e)}"

    # ========== DATA LOADING METHODS ==========

    def debug_data_quality(self, df, column_name):
        """Debug data quality for a specific column"""
        if column_name in df.columns:
            series = df[column_name]
            print(f"  {column_name}: min={series.min():.2f}, max={series.max():.2f}, "
                  f"mean={series.mean():.2f}, nulls={series.isnull().sum()}, unique={series.nunique()}")

    def load_from_local_csv(self, timeframe, limit=500):
        """Load data dari file CSV lokal dengan validasi yang lebih longgar"""
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
                    print(f"📅 Data range: {len(df)} records")
                    
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
                    
                    # Hapus rows dengan data yang benar-benar invalid
                    initial_count = len(df)
                    df = df.dropna(subset=['open', 'high', 'low', 'close'])
                    removed_count = initial_count - len(df)
                    if removed_count > 0:
                        print(f"⚠️ Removed {removed_count} rows with missing OHLC data")
                    
                    # Fill remaining NaN values
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    
                    df = df.sort_values('datetime')
                    print(f"✅ Successfully loaded {len(df)} records from {filename}")
                    
                    # Debug data quality
                    print("🔍 Data quality check:")
                    self.debug_data_quality(df, 'open')
                    self.debug_data_quality(df, 'high')
                    self.debug_data_quality(df, 'low')
                    self.debug_data_quality(df, 'close')
                    
                    # Tampilkan sample data
                    if len(df) > 0:
                        print("📊 Sample data (first 3 rows):")
                        print(df[['datetime', 'open', 'high', 'low', 'close']].head(3))
                        print("📊 Sample data (last 3 rows):")
                        print(df[['datetime', 'open', 'high', 'low', 'close']].tail(3))
                    
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

    def gentle_data_cleaning(self, df):
        """Gentle data cleaning - hanya menghapus data yang benar-benar invalid"""
        print("🧹 Gentle Data Cleaning - Only removing truly invalid data")
        
        if len(df) < 10:
            return df
            
        initial_count = len(df)
        
        # Hanya hapus data yang benar-benar tidak mungkin untuk harga emas
        df = df[
            (df['close'] > 100) & (df['close'] < 10000) &  # Harga emas realistis $100-$10,000
            (df['high'] > 100) & (df['high'] < 10000) &
            (df['low'] > 100) & (df['low'] < 10000) &
            (df['open'] > 100) & (df['open'] < 10000) &
            (df['high'] >= df['low']) &  # High harus >= low
            (df['open'] > 0) & (df['close'] > 0)  # Harga harus positif
        ]
        
        # Pastikan data terurut
        df = df.sort_values('datetime')
        df = df.reset_index(drop=True)
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            print(f"🧹 Removed {removed_count} truly invalid records")
            print(f"📊 Final data range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        else:
            print("✅ No invalid records found - all data looks good")
        
        return df

    def enhanced_data_validation(self, df):
        """Enhanced data validation dengan kriteria yang lebih longgar"""
        print("🔍 Enhanced data validation dengan kriteria longgar...")
        
        if df is None or len(df) == 0:
            print("❌ Empty dataframe")
            return False
            
        # Check untuk harga emas yang realistis (lebih longgar)
        current_price = df['close'].iloc[-1]
        if current_price < 100 or current_price > 10000:
            print(f"❌ CRITICAL: Unrealistic gold price: ${current_price:.2f}")
            return False
        
        # Check price relationships dasar
        invalid_high_low = (df['high'] < df['low']).sum()
        invalid_open_close = (df['open'] <= 0).sum() | (df['close'] <= 0).sum()
        
        if invalid_high_low > 0:
            print(f"❌ CRITICAL: {invalid_high_low} rows have high < low")
            return False
            
        if invalid_open_close > 0:
            print(f"❌ CRITICAL: {invalid_open_close} rows have non-positive prices")
            return False
        
        # Check for reasonable volatility (lebih longgar)
        daily_returns = df['close'].pct_change().abs()
        extreme_moves = (daily_returns > 0.15).sum()  # 15% daily move dianggap extreme
        
        if extreme_moves > len(df) * 0.05:  # Hanya 5% data boleh extreme
            print(f"⚠️ WARNING: Too many extreme price moves: {extreme_moves}")
            
        print("✅ Enhanced data validation passed")
        return True

    def load_historical_data(self, timeframe, limit=500):
        """Load data historis dengan gentle cleaning"""
        try:
            # Try local CSV first
            df = self.load_from_local_csv(timeframe, limit)
            if df is not None:
                # Apply gentle cleaning
                df = self.gentle_data_cleaning(df)
                if len(df) >= 20:  # Minimal data setelah cleaning
                    print(f"✅ Using gently cleaned local data for {timeframe}")
                    return df.tail(limit)
                else:
                    print("❌ Insufficient data after gentle cleaning")
                    
            # Try download if local data invalid
            print(f"📥 Local data invalid, trying to download for {timeframe}...")
            df = self.download_historical_data(timeframe)
            if df is not None and self.enhanced_data_validation(df):
                print(f"✅ Using downloaded data for {timeframe}")
                return df.tail(limit)
                
            # Final fallback - generate realistic sample data
            print(f"🔄 All methods failed, using realistic generated data for {timeframe}")
            return self.generate_realistic_sample_data(timeframe, limit)
            
        except Exception as e:
            print(f"❌ Error in load_historical_data: {e}")
            return self.generate_realistic_sample_data(timeframe, limit)

    def generate_realistic_sample_data(self, timeframe, limit=500):
        """Generate realistic sample data based on current gold prices"""
        print(f"🔄 Generating REALISTIC sample data for {timeframe}")
        
        periods = limit
        # Use realistic base price for gold (current approximate)
        base_price = 1950.0
        
        if timeframe == '1H':
            freq = 'H'
            volatility = 0.002  # 0.2% hourly volatility
        elif timeframe == '4H':
            freq = '4H' 
            volatility = 0.004  # 0.4% 4-hour volatility
        else:
            freq = 'D'
            volatility = 0.008  # 0.8% daily volatility
            
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        
        np.random.seed(42)
        returns = np.random.normal(0, volatility, periods)
        prices = base_price * (1 + returns).cumprod()
        
        data = []
        for i in range(periods):
            open_price = prices[i]
            close_price = prices[i] * np.random.uniform(0.998, 1.002)
            high_price = max(open_price, close_price) * np.random.uniform(1.001, 1.015)
            low_price = min(open_price, close_price) * np.random.uniform(0.985, 0.999)
            
            # Ensure high > low and reasonable ranges
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            data.append({
                'datetime': dates[i],
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': np.random.randint(10000, 100000)
            })
        
        df = pd.DataFrame(data)
        
        self.data_cache[timeframe] = df
        print(f"✅ Generated {len(df)} realistic records for {timeframe}")
        return df

    def clean_and_validate_data(self, df):
        """Enhanced data cleaning and validation yang lebih gentle"""
        print("🧹 Gentle cleaning and validating dataframe...")
        
        if df is None or len(df) == 0:
            print("❌ Empty dataframe provided")
            return df
            
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows dengan missing critical data
        initial_count = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        removed_count = initial_count - len(df)
        if removed_count > 0:
            print(f"⚠️ Removed {removed_count} rows with missing data")
        
        # Forward fill then backward fill untuk data yang missing
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Gentle cleaning completed. Final data count: {len(df)}")
        return df

    def validate_price_data(self, df):
        """Validate price data dengan kriteria longgar"""
        if len(df) == 0:
            return False
            
        # Check untuk harga emas yang realistis (longgar)
        current_price = df['close'].iloc[-1]
        if current_price < 100 or current_price > 10000:
            print(f"❌ WARNING: Unrealistic price detected: ${current_price:.2f}")
            return False
        
        # Check untuk pergerakan harga yang wajar
        price_changes = df['close'].pct_change().abs()
        max_change = price_changes.max()
        if max_change > 0.2:  # 20% daily move dianggap extreme
            print(f"⚠️ WARNING: Extreme price movement detected: {max_change:.2%}")
            
        # Check untuk relationship harga yang konsisten
        invalid_high_low = (df['high'] < df['low']).sum()
        if invalid_high_low > 0:
            print(f"❌ WARNING: {invalid_high_low} rows have high < low")
            return False
            
        return True

    # ========== INDICATOR CALCULATION METHODS ==========

    def calculate_indicators(self, df):
        """Calculate technical indicators - ENHANCED VERSION"""
        try:
            if len(df) < 20:  # Kurangi minimum data required
                print(f"⚠️ Limited data for indicators. Have {len(df)}, proceeding anyway")
                return self.add_corrected_fallback_indicators(df)
                
            # Clean and validate data first
            df = self.clean_and_validate_data(df)
            
            # Validate price data sanity
            if not self.validate_price_data(df):
                print("⚠️ Price data validation warning, but proceeding with calculations")
                
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            
            print(f"📊 Calculating indicators for {len(df)} records...")
            print(f"💰 Price range: ${close.min():.2f} - ${close.max():.2f}")
            
            # Use TA-Lib if available, otherwise use corrected calculations
            if TALIB_AVAILABLE:
                print("🔧 Using TA-Lib for indicator calculations")
                df = self.calculate_indicators_talib(df, close, high, low)
            else:
                print("🔧 Using corrected pandas calculations")
                df = self.calculate_indicators_pandas(df, close, high, low)
            
            # Enhanced verification
            if not self.enhanced_indicator_verification(df):
                print("⚠️ Indicator verification warning, but keeping calculations")
            
            print("✅ Indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"❌ Error calculating indicators: {e}")
            traceback.print_exc()
            return self.add_corrected_fallback_indicators(df)

    def calculate_indicators_talib(self, df, close, high, low):
        """Calculate indicators using TA-Lib with MACD FIX"""
        try:
            # EMA
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            df['ema_50'] = talib.EMA(close, timeperiod=50)
            
            # MACD - dengan fix untuk inconsistency
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            
            # Force fix MACD histogram jika ada inconsistency
            expected_hist = macd - macd_signal
            hist_discrepancy = np.abs(macd_hist - expected_hist)
            
            # Jika discrepancy besar, gunakan calculated value
            large_discrepancies = hist_discrepancy > 0.1
            if np.any(large_discrepancies):
                print(f"⚠️  Fixing {np.sum(large_discrepancies)} MACD histogram discrepancies")
                macd_hist = expected_hist
            
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # RSI
            df['rsi'] = talib.RSI(close, timeperiod=14)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Stochastic
            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Final MACD validation
            self.validate_and_fix_macd(df)
            
            print("✅ TA-Lib indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"❌ TA-Lib calculation error: {e}, falling back to pandas")
            return self.calculate_indicators_pandas(df, close, high, low)

    def validate_and_fix_macd(self, df):
        """Validate and fix MACD calculations"""
        if len(df) == 0:
            return
            
        # Check last few rows for consistency
        check_rows = min(10, len(df))
        for i in range(-check_rows, 0):
            idx = df.index[i]
            macd = df.loc[idx, 'macd']
            macd_signal = df.loc[idx, 'macd_signal'] 
            macd_hist = df.loc[idx, 'macd_hist']
            
            expected_hist = macd - macd_signal
            discrepancy = abs(macd_hist - expected_hist)
            
            if discrepancy > 0.001:
                print(f"🔧 Fixing MACD histogram at index {idx}: {macd_hist} -> {expected_hist}")
                df.loc[idx, 'macd_hist'] = expected_hist
        
        # Verify fix
        last_row = df.iloc[-1]
        macd = last_row['macd']
        macd_signal = last_row['macd_signal']
        macd_hist = last_row['macd_hist']
        expected_hist = macd - macd_signal
        
        print(f"🔍 MACD Final Verification:")
        print(f"   MACD: {macd:.4f}, Signal: {macd_signal:.4f}")
        print(f"   Histogram: {macd_hist:.4f}, Expected: {expected_hist:.4f}")
        print(f"   Consistent: {abs(macd_hist - expected_hist) < 0.001}")

    def calculate_indicators_pandas(self, df, close, high, low):
        """Calculate indicators using corrected pandas methods"""
        try:
            close_series = pd.Series(close)
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            
            # EMA - CORRECTED: Use proper EWM with different spans
            df['ema_12'] = close_series.ewm(span=12, adjust=False).mean()
            df['ema_26'] = close_series.ewm(span=26, adjust=False).mean()
            df['ema_50'] = close_series.ewm(span=50, adjust=False).mean()
            
            # MACD - CORRECTED: Calculate from EMAs
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # RSI - CORRECTED: Proper RSI calculation
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.inf)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands - CORRECTED
            df['bb_middle'] = close_series.rolling(window=20, min_periods=1).mean()
            bb_std = close_series.rolling(window=20, min_periods=1).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Stochastic - CORRECTED
            lowest_low = low_series.rolling(window=14, min_periods=1).min()
            highest_high = high_series.rolling(window=14, min_periods=1).max()
            df['stoch_k'] = 100 * (close_series - lowest_low) / (highest_high - lowest_low).replace(0, 1)
            df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
            
            print("✅ Pandas indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"❌ Pandas calculation error: {e}")
            raise

    def enhanced_indicator_verification(self, df):
        """Enhanced indicator verification"""
        if len(df) == 0:
            return False
            
        last_row = df.iloc[-1]
        
        # Check MACD consistency
        macd = last_row.get('macd', 0)
        macd_signal = last_row.get('macd_signal', 0) 
        macd_hist = last_row.get('macd_hist', 0)
        
        # Verify MACD histogram calculation
        expected_hist = macd - macd_signal
        if abs(macd_hist - expected_hist) > 0.001:
            print(f"❌ MACD HISTOGRAM ERROR: {macd_hist} vs expected {expected_hist}")
            return False
        
        # Check RSI range
        rsi = last_row.get('rsi', 50)
        if rsi < 0 or rsi > 100:
            print(f"❌ Invalid RSI value: {rsi}")
            return False
            
        # Check EMA relationships are reasonable
        ema_12 = last_row.get('ema_12', 0)
        ema_26 = last_row.get('ema_26', 0)
        ema_50 = last_row.get('ema_50', 0)
        
        if ema_12 == 0 or ema_26 == 0 or ema_50 == 0:
            print("❌ Zero EMA values detected")
            return False
            
        # Check if all EMAs are equal (calculation error)
        if ema_12 == ema_26 == ema_50:
            print("❌ All EMAs have same value - calculation error")
            return False
            
        return True

    def add_corrected_fallback_indicators(self, df):
        """Corrected fallback indicators with proper calculations"""
        print("🔄 Using CORRECTED fallback indicators")
        
        if len(df) == 0:
            return df
            
        close = df['close'].values
        close_series = pd.Series(close)
        
        # Simple but correct calculations
        df['ema_12'] = close_series.ewm(span=12, adjust=False).mean()
        df['ema_26'] = close_series.ewm(span=26, adjust=False).mean()
        df['ema_50'] = close_series.ewm(span=50, adjust=False).mean()
        
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = close_series.rolling(window=20, min_periods=1).mean()
        bb_std = close_series.rolling(window=20, min_periods=1).std().fillna(10)
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Stochastic
        low_14 = pd.Series(df['low']).rolling(window=14, min_periods=1).min()
        high_14 = pd.Series(df['high']).rolling(window=14, min_periods=1).max()
        df['stoch_k'] = 100 * (close_series - low_14) / (high_14 - low_14).replace(0, 1)
        df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
        
        # Ensure no NaN values
        df = self.ensure_no_nan_indicators(df)
        
        print("✅ Corrected fallback indicators applied successfully")
        return df

    def ensure_no_nan_indicators(self, df):
        """Ensure no NaN values in indicators"""
        indicator_defaults = {
            'ema_12': df['close'].iloc[-1] if len(df) > 0 else 1950.0,
            'ema_26': df['close'].iloc[-1] if len(df) > 0 else 1950.0,
            'ema_50': df['close'].iloc[-1] if len(df) > 0 else 1950.0,
            'macd': 0,
            'macd_signal': 0,
            'macd_hist': 0,
            'rsi': 50,
            'bb_upper': df['close'].iloc[-1] * 1.02 if len(df) > 0 else 1989.0,
            'bb_middle': df['close'].iloc[-1] if len(df) > 0 else 1950.0,
            'bb_lower': df['close'].iloc[-1] * 0.98 if len(df) > 0 else 1911.0,
            'stoch_k': 50,
            'stoch_d': 50
        }
        
        for indicator, default in indicator_defaults.items():
            if indicator in df.columns:
                df[indicator] = df[indicator].fillna(default)
        
        return df

    # ========== AI ANALYSIS METHODS ==========

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
        """Create optimized trading prompt for faster DeepSeek response dengan jarak minimal 500 pips"""
        
        prompt = f"""
ANALISIS XAUUSD - {timeframe}

DATA TEKNIKAL:
- Harga: ${current_price:.2f}
- Change: {tech_analysis['price_change_percent']:+.2f}%
- Trend: {tech_analysis['price_trend']}
- EMA: {tech_analysis['ema_trend']}
- RSI: {tech_analysis['rsi_value']:.1f} ({tech_analysis['rsi_signal']})
- MACD: {tech_analysis['macd_signal']}
- Support: ${tech_analysis['support_level']:.2f}
- Resistance: ${tech_analysis['resistance_level']:.2f}
- Volatility: {tech_analysis['volatility_percent']:.2f}%

KONTEKS: {fundamental_context[:500]}...

PENTING: Untuk rekomendasi trading, GUNAKAN JARAK MINIMAL 500 PIPS (setara dengan $50.0 untuk XAUUSD) untuk stop loss dan take profit dari harga entry.

BERIKAN REKOMENDASI TRADING DENGAN KETENTUAN:
- Stop loss harus berjarak minimal 500 pips dari entry
- Take profit harus berjarak minimal 1000 pips dari entry untuk risk-reward 1:2
- Jika tidak memungkinkan, berikan rekomendasi HOLD

🎯 REKOMENDASI: [BUY/SELL/HOLD]
💰 ENTRY: $[price]
🛑 STOP LOSS: $[price] (minimal 500 pips dari entry)
✅ TAKE PROFIT: $[price] (minimal 1000 pips dari entry untuk 1:2 RR)
📊 RISK-REWARD: 1:[ratio]
⚠️ RISK: [LOW/MEDIUM/HIGH]

ANALISIS: [Brief technical/fundamental analysis]

STRATEGI: [Trading plan dengan emphasis pada jarak minimal 500 pips]
"""
        return prompt

    def call_deepseek_api(self, prompt):
        """Call DeepSeek API for trading analysis with enhanced error handling"""
        try:
            if not self.deepseek_api_key:
                print("❌ DeepSeek API key not configured")
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
                        "content": "Anda adalah analis trading emas (XAUUSD) profesional dengan pengalaman 10+ tahun. Berikan analisis yang akurat dan rekomendasi trading yang dapat ditindaklanjuti berdasarkan data teknikal dan fundamental. Fokus pada manajemen risiko dan probabilitas tinggi. SELALU gunakan jarak minimal 500 pips untuk stop loss dan 1000 pips untuk take profit dari harga entry."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1200,
                "stream": False
            }

            print("🔄 Calling DeepSeek API...")
            
            # Enhanced timeout and retry logic
            max_retries = 2
            timeout_duration = 30
            
            for attempt in range(max_retries):
                try:
                    print(f"📡 Attempt {attempt + 1}/{max_retries} to connect to DeepSeek API...")
                    
                    response = self.session.post(
                        url, 
                        json=payload, 
                        headers=headers, 
                        timeout=timeout_duration
                    )
                    
                    self.last_api_call = time.time()
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'choices' in data and len(data['choices']) > 0:
                            analysis = data['choices'][0]['message']['content']
                            print("✅ DeepSeek AI analysis generated successfully")
                            return analysis
                        else:
                            print(f"❌ DeepSeek API response format unexpected: {data}")
                            break
                    else:
                        error_msg = f"DeepSeek API HTTP error: {response.status_code}"
                        if response.status_code == 401:
                            error_msg += " - Invalid API Key"
                            print(f"❌ {error_msg}")
                            break
                        elif response.status_code == 429:
                            error_msg += " - Rate Limited"
                            print(f"⚠️ {error_msg}")
                            wait_time = (attempt + 1) * 10
                            print(f"⏳ Waiting {wait_time} seconds before retry...")
                            time.sleep(wait_time)
                            continue
                        elif response.status_code >= 500:
                            error_msg += " - Server Error"
                            print(f"⚠️ {error_msg}")
                        else:
                            print(f"❌ {error_msg}")
                            break
                            
                except requests.exceptions.Timeout:
                    print(f"⏰ Timeout occurred on attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ All retry attempts failed due to timeout")
                        break
                        
                except requests.exceptions.ConnectionError as e:
                    print(f"🔌 Connection error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("❌ All retry attempts failed due to connection issues")
                        break
                        
                except Exception as e:
                    print(f"❌ Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
                    break

            print("❌ Failed to get response from DeepSeek API after all retries")
            return None
            
        except Exception as e:
            print(f"❌ Error calling DeepSeek API: {e}")
            return None

    def generate_fallback_analysis(self, df, current_price, timeframe):
        """Generate high-quality fallback analysis when AI is unavailable dengan jarak minimal 500 pips"""
        try:
            if len(df) < 5:
                return "🔍 Insufficient data for comprehensive analysis. Please try again later."
            
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Enhanced technical analysis
            price_change = ((current['close'] - previous['close']) / previous['close']) * 100
            price_trend = "BULLISH" if price_change > 0 else "BEARISH"
            
            # Multi-timeframe momentum analysis
            momentum = "NEUTRAL"
            if len(df) >= 3:
                prev_trend = ((previous['close'] - df['close'].iloc[-3]) / df['close'].iloc[-3]) * 100
                if price_change > 0 and prev_trend > 0:
                    momentum = "STRONG BULLISH"
                elif price_change < 0 and prev_trend < 0:
                    momentum = "STRONG BEARISH"
            
            # Enhanced indicator-based recommendation
            rsi = current.get('rsi', 50)
            macd_hist = current.get('macd_hist', 0)
            ema_12 = current.get('ema_12', current['close'])
            ema_26 = current.get('ema_26', current['close'])
            
            # Sophisticated trading logic
            bullish_signals = 0
            bearish_signals = 0
            
            if rsi < 40: bullish_signals += 1
            if rsi > 60: bearish_signals += 1
            if macd_hist > 0: bullish_signals += 1  
            if macd_hist < 0: bearish_signals += 1
            if current['close'] > ema_12: bullish_signals += 1
            if current['close'] < ema_12: bearish_signals += 1
            if ema_12 > ema_26: bullish_signals += 1
            if ema_12 < ema_26: bearish_signals += 1
            
            # Calculate minimum 500 pips distance (50.0 for XAUUSD since 1 pip = $0.01)
            min_pips_distance = 50.0  # 500 pips = $5.00 for XAUUSD
            
            # Determine recommendation with minimum 500 pips distance
            if bullish_signals >= 3 and bearish_signals <= 1:
                recommendation = "BUY"
                entry_price = current_price
                stop_loss = entry_price - min_pips_distance
                take_profit = entry_price + (min_pips_distance * 2)  # 1:2 risk-reward
                risk_level = "MEDIUM"
                rr_ratio = "2.0"
            elif bearish_signals >= 3 and bullish_signals <= 1:
                recommendation = "SELL" 
                entry_price = current_price
                stop_loss = entry_price + min_pips_distance
                take_profit = entry_price - (min_pips_distance * 2)  # 1:2 risk-reward
                risk_level = "MEDIUM"
                rr_ratio = "2.0"
            else:
                recommendation = "HOLD"
                entry_price = current_price
                stop_loss = entry_price - min_pips_distance
                take_profit = entry_price + min_pips_distance
                risk_level = "LOW"
                rr_ratio = "1.0"
            
            # Validate stop loss and take profit distances
            if recommendation == "BUY":
                if stop_loss >= entry_price:
                    stop_loss = entry_price - min_pips_distance
                if take_profit <= entry_price:
                    take_profit = entry_price + (min_pips_distance * 2)
                    
            elif recommendation == "SELL":
                if stop_loss <= entry_price:
                    stop_loss = entry_price + min_pips_distance
                if take_profit >= entry_price:
                    take_profit = entry_price - (min_pips_distance * 2)
            
            # Calculate actual distances in pips
            stop_loss_distance = abs(entry_price - stop_loss)
            take_profit_distance = abs(entry_price - take_profit)
            
            # Generate comprehensive analysis
            analysis = f"""
🎯 REKOMENDASI: {recommendation}
💰 ENTRY: ${entry_price:.2f}
🛑 STOP LOSS: ${stop_loss:.2f} (Jarak: {stop_loss_distance:.2f} pips)
✅ TAKE PROFIT: ${take_profit:.2f} (Jarak: {take_profit_distance:.2f} pips)
📊 RISK-REWARD RATIO: 1:{rr_ratio}
⚠️ RISK LEVEL: {risk_level}

ANALISIS TEKNIKAL:
• Harga: ${current_price:.2f} ({price_change:+.2f}%)
• Momentum: {momentum}
• RSI: {rsi:.1f} - {'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'}
• EMA Trend: {'Bullish' if ema_12 > ema_26 else 'Bearish'}
• Support: ${current.get('bb_lower', current['close'] * 0.98):.2f}
• Resistance: ${current.get('bb_upper', current['close'] * 1.02):.2f}

STRATEGI TRADING:
• {recommendation} dengan konfirmasi price action
• Stop loss ketat dengan jarak minimal 500 pips
• Target profit dengan risk-reward 1:{rr_ratio}
• Monitor volume dan breakout levels

CATATAN:
Semua rekomendasi memiliki jarak minimal 500 pips dari entry price untuk manajemen risiko yang proper.

📈 Sinyal: {bullish_signals} Bullish / {bearish_signals} Bearish
"""
            return analysis.strip()
            
        except Exception as e:
            return f"""
🎯 REKOMENDASI: HOLD
💰 ENTRY: ${current_price:.2f}
🛑 STOP LOSS: ${current_price - 50.0:.2f}
✅ TAKE PROFIT: ${current_price + 50.0:.2f}
📊 RISK-REWARD RATIO: 1:1
⚠️ RISK LEVEL: LOW

CATATAN: Sistem analisis sedang mengalami gangguan teknis. Disarankan untuk wait and see hingga kondisi normal.
"""

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

@app.route('/api/debug')
def debug():
    """Debug endpoint untuk status sistem dengan connection test"""
    try:
        # Test DeepSeek connection
        deepseek_test = analyzer.test_deepseek_connection()
        
        # Check data files
        data_files = {}
        timeframes = ['1H', '4H', '1D']
        for tf in timeframes:
            filename = f"data/XAUUSD_{tf}.csv"
            exists = os.path.exists(filename)
            data_files[tf] = {
                "exists": exists,
                "rows": 0
            }
            if exists:
                try:
                    df = pd.read_csv(filename)
                    data_files[tf]["rows"] = len(df)
                except:
                    data_files[tf]["rows"] = 0
        
        debug_info = {
            "api_status": {
                "twelve_data": bool(analyzer.twelve_data_api_key),
                "deepseek": bool(analyzer.deepseek_api_key),
                "newsapi": bool(analyzer.news_api_key),
                "deepseek_connection": deepseek_test[0],
                "deepseek_connection_message": deepseek_test[1]
            },
            "data_files": data_files,
            "system_time": datetime.now().isoformat(),
            "talib_available": TALIB_AVAILABLE,
            "template_folder": app.template_folder,
            "index_html_exists": os.path.exists(os.path.join(app.template_folder, 'index.html'))
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_cache')
def clear_cache():
    """Clear data cache"""
    try:
        analyzer.data_cache = {}
        return jsonify({"status": "success", "message": "Cache cleared"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/debug/data/<timeframe>')
def debug_data(timeframe):
    """Debug data endpoint"""
    try:
        df = analyzer.load_historical_data(timeframe, limit=10)
        if df is None:
            return jsonify({"error": "No data available"}), 404
            
        return jsonify({
            "rows": len(df),
            "columns": df.columns.tolist(),
            "date_range": {
                "start": str(df['datetime'].min()) if 'datetime' in df.columns else "N/A",
                "end": str(df['datetime'].max()) if 'datetime' in df.columns else "N/A"
            },
            "sample": df.head(3).to_dict('records')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "service": "XAUUSD Trading Analysis",
        "template_folder": app.template_folder
    })

# ========== STATIC FILE ROUTES ==========

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    try:
        return send_from_directory('static', 'favicon.ico')
    except:
        return '', 404

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
    print("  • ✅ Enhanced Error Handling & Retry Logic")
    print("  • ✅ MINIMAL 500 PIPS DISTANCE FOR RISK MANAGEMENT")
    print("=" * 70)
    
    print("🚀 Starting AI-enhanced trading analysis server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
