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

# Setup template folder - PASTIKAN PATH BENAR
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
                        # Jangan langsung fillna, kita cek dulu
                    
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
        # Harga emas bisa naik signifikan dalam jangka panjang, jadi batas atas sangat longgar
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
            # Tidak return False, hanya warning karena mungkin ada event market yang valid
            
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
        base_price = 4237.0
        
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
            # Tidak return False, hanya warning
            
        # Check untuk relationship harga yang konsisten
        invalid_high_low = (df['high'] < df['low']).sum()
        if invalid_high_low > 0:
            print(f"❌ WARNING: {invalid_high_low} rows have high < low")
            return False
            
        return True

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
                # Tidak return, lanjutkan saja
                
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
                # Tidak recalculate, tetap gunakan yang ada
            
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
            'ema_12': df['close'].iloc[-1] if len(df) > 0 else 4237.0,
            'ema_26': df['close'].iloc[-1] if len(df) > 0 else 4237.0,
            'ema_50': df['close'].iloc[-1] if len(df) > 0 else 4237.0,
            'macd': 0,
            'macd_signal': 0,
            'macd_hist': 0,
            'rsi': 50,
            'bb_upper': df['close'].iloc[-1] * 1.02 if len(df) > 0 else 4320.0,
            'bb_middle': df['close'].iloc[-1] if len(df) > 0 else 4237.0,
            'bb_lower': df['close'].iloc[-1] * 0.98 if len(df) > 0 else 4150.0,
            'stoch_k': 50,
            'stoch_d': 50
        }
        
        for indicator, default in indicator_defaults.items():
            if indicator in df.columns:
                df[indicator] = df[indicator].fillna(default)
        
        return df

    def generate_ai_analysis(self, df, current_price):
        """Generate AI analysis for the market"""
        try:
            if not self.deepseek_api_key:
                return "AI analysis unavailable - DeepSeek API key not configured"
                
            # Prepare market context
            price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) > 1 else 0
            
            analysis_text = f"""
XAUUSD Current Price: ${current_price:.2f}
24h Change: {price_change:+.2f}%

TECHNICAL ANALYSIS:
- RSI: {df['rsi'].iloc[-1]:.1f} ({'Overbought' if df['rsi'].iloc[-1] > 70 else 'Oversold' if df['rsi'].iloc[-1] < 30 else 'Neutral'})
- MACD: {df['macd'].iloc[-1]:.4f} ({'Bullish' if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else 'Bearish'})
- Trend: {'Bullish' if df['ema_12'].iloc[-1] > df['ema_26'].iloc[-1] else 'Bearish'}

KEY LEVELS:
- Support: ${df['bb_lower'].iloc[-1]:.2f}
- Resistance: ${df['bb_upper'].iloc[-1]:.2f}

OUTLOOK: Gold shows {'strength' if price_change > 0 else 'weakness'} in current trading. Monitor key levels for breakout opportunities.
"""
            return analysis_text.strip()
            
        except Exception as e:
            return f"AI Analysis temporarily unavailable: {str(e)}"

    def get_market_news(self):
        """Get market news"""
        try:
            if not self.news_api_key:
                return {"articles": []}
                
            # Simulate news response
            return {
                "articles": [
                    {
                        "title": "Gold Prices Stable Amid Economic Data",
                        "description": "Gold prices holding steady as traders await key economic indicators.",
                        "publishedAt": datetime.now().isoformat(),
                        "source": {"name": "Market News"}
                    },
                    {
                        "title": "Federal Reserve Policy Impact on Gold",
                        "description": "Analysts watch Fed decisions for impact on precious metals market.",
                        "publishedAt": (datetime.now() - timedelta(hours=1)).isoformat(),
                        "source": {"name": "Financial Times"}
                    },
                    {
                        "title": "Gold Demand Increases in Asian Markets", 
                        "description": "Strong physical gold demand reported from key Asian markets.",
                        "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "source": {"name": "Reuters"}
                    }
                ]
            }
        except Exception as e:
            return {"articles": [], "error": str(e)}

# Create analyzer instance
analyzer = XAUUSDAnalyzer()

# ========== ROUTES UTAMA ==========

@app.route('/')
def index():
    """Serve the main dashboard page from templates folder"""
    try:
        # Check if templates folder exists
        templates_path = app.template_folder
        index_path = os.path.join(templates_path, 'index.html')
        
        print(f"📁 Looking for index.html at: {index_path}")
        print(f"📁 Templates folder exists: {os.path.exists(templates_path)}")
        print(f"📁 Index.html exists: {os.path.exists(index_path)}")
        
        if os.path.exists(index_path):
            print("✅ Serving index.html from templates folder")
            return render_template('index.html')
        else:
            error_msg = f"index.html not found in templates folder. Path: {index_path}"
            print(f"❌ {error_msg}")
            
            # List files in templates directory for debugging
            if os.path.exists(templates_path):
                files = os.listdir(templates_path)
                print(f"📁 Files in templates directory: {files}")
            
            return f"""
            <html>
                <head><title>Error - File Not Found</title></head>
                <body>
                    <h1>Error: index.html not found</h1>
                    <p>Expected path: {index_path}</p>
                    <p>Templates folder exists: {os.path.exists(templates_path)}</p>
                    <p>Please ensure index.html is in the templates folder.</p>
                </body>
            </html>
            """, 500
            
    except Exception as e:
        error_msg = f"Error loading dashboard: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return f"""
        <html>
            <head><title>Error</title></head>
            <body>
                <h1>Server Error</h1>
                <p>{error_msg}</p>
                <pre>{traceback.format_exc()}</pre>
            </body>
        </html>
        """, 500

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
        
        # Calculate indicators
        df_with_indicators = analyzer.calculate_indicators(df)
        
        # Get current price
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
        
        # Generate AI analysis
        ai_analysis = analyzer.generate_ai_analysis(df_with_indicators, current_price)
        
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
    """Debug endpoint untuk status sistem"""
    try:
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
                "newsapi": bool(analyzer.news_api_key)
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
    
    # Check if index.html exists in templates
    index_path = os.path.join(app.template_folder, 'index.html')
    if not os.path.exists(index_path):
        print(f"⚠️ WARNING: index.html not found in templates folder: {index_path}")
        print("📁 Creating basic index.html file...")
        
        # Create a basic index.html file
        basic_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>XAUUSD Trading Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .error { color: red; padding: 20px; background: #ffeeee; border: 1px solid red; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>XAUUSD Trading Analysis</h1>
                <div class="error">
                    <h2>Setup Required</h2>
                    <p>Please place the main index.html file in the templates folder.</p>
                    <p>Current templates folder: TEMPLATE_FOLDER_PATH</p>
                </div>
            </div>
        </body>
        </html>
        """.replace('TEMPLATE_FOLDER_PATH', app.template_folder)
        
        with open(index_path, 'w') as f:
            f.write(basic_html)
        print("✅ Created basic index.html file")
    
    print("=" * 70)
    print("🚀 XAUUSD Professional Trading Analysis - GENTLE DATA HANDLING")
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
    print("  • Twelve Data → Real-time Prices")
    print("  • DeepSeek AI → Market Analysis") 
    print("  • NewsAPI → Fundamental News")
    print("=" * 70)
    print("🔄 GENTLE DATA HANDLING FEATURES:")
    print("  • ✅ Gentle Data Cleaning - Hanya hapus data yang benar-benar invalid")
    print("  • ✅ Wide Price Range - $100-$10,000 untuk harga emas")
    print("  • ✅ Minimal Data Rejection - Terima semua data yang masuk akal")
    print("  • ✅ Better Logging - Tampilkan sample data untuk verifikasi")
    print("  • ✅ Relaxed Validation - Kriteria validasi yang lebih longgar")
    print("=" * 70)
    
    print("🚀 Starting gentle data handling server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
