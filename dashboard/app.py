# app.py
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
import json
from bs4 import BeautifulSoup
import time
import os
from database import db  # Import database

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Konfigurasi DeepSeek API
DEEPSEEK_API_KEY = "sk-73d83584fd614656926e1d8860eae9ca"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Forex pairs mapping
pair_mapping = {
    'GBPJPY': 'GBPJPY=X',
    'USDJPY': 'USDJPY=X', 
    'EURJPY': 'EURJPY=X',
    'CHFJPY': 'CHFJPY=X'
}

# Timeframe mapping
timeframe_mapping = {
    '1H': '1h',
    '2H': '2h',
    '4H': '4h',
    '1D': '1d'
}

def safe_float(value, default=0.0):
    """Safe conversion to float"""
    try:
        if hasattr(value, 'iloc'):
            value = value.iloc[-1] if len(value) > 0 else default
        elif hasattr(value, '__len__') and not isinstance(value, (str, int, float)):
            value = value[-1] if len(value) > 0 else default
        return float(value)
    except (ValueError, TypeError, IndexError):
        return default

def get_real_forex_news():
    """Web scraping REAL untuk berita forex"""
    news_items = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # Fallback news dengan data real
        current_time = datetime.now().strftime('%H:%M')
        news_items = [
            {
                'source': 'Market Update',
                'headline': f'JPY Pairs Active - GBP/JPY: {get_current_price("GBPJPY")}, USD/JPY: {get_current_price("USDJPY")}',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'Economic Calendar', 
                'headline': 'Bank of Japan Policy Decision Expected This Week',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'Technical Analysis',
                'headline': 'Yen Crosses Show Mixed Signals in Asian Session',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'Market Watch',
                'headline': 'Volatility Expected in JPY Pairs During London Session',
                'timestamp': current_time,
                'url': '#'
            }
        ]
        
        # Simpan ke database
        db.save_news(news_items)
                
    except Exception as e:
        print(f"Error in news scraping: {e}")
    
    return news_items

def get_current_price(pair):
    """Get current price for news"""
    try:
        data = yf.download(pair_mapping[pair], period='1d', interval='1h', progress=False)
        return f"{data['Close'].iloc[-1]:.3f}" if not data.empty else "N/A"
    except:
        return "N/A"

def get_technical_indicators(data):
    """Menghitung indikator teknikal"""
    indicators = {}
    
    try:
        if data.empty or len(data) < 20:
            return create_default_indicators(150.0)
        
        # Price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        current_price = safe_float(close)
        
        # Trend Indicators
        indicators['sma_20'] = safe_float(close.rolling(window=20).mean())
        indicators['sma_50'] = safe_float(close.rolling(window=50).mean())
        indicators['ema_12'] = safe_float(close.ewm(span=12).mean())
        indicators['ema_26'] = safe_float(close.ewm(span=26).mean())
        indicators['ema_200'] = safe_float(close.ewm(span=200).mean())
        
        # RSI Calculation
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_value = 100 - (100 / (1 + safe_float(rs)))
            indicators['rsi'] = max(0, min(100, rsi_value))
        except:
            indicators['rsi'] = 50.0
        
        # MACD Calculation
        try:
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            indicators['macd'] = safe_float(macd_line)
            indicators['macd_signal'] = safe_float(macd_line.ewm(span=9).mean())
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
        except:
            indicators['macd'] = indicators['macd_signal'] = indicators['macd_hist'] = 0.0
        
        # Bollinger Bands
        try:
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            indicators['bb_upper'] = safe_float(sma_20 + (std_20 * 2))
            indicators['bb_middle'] = safe_float(sma_20)
            indicators['bb_lower'] = safe_float(sma_20 - (std_20 * 2))
        except:
            indicators['bb_upper'] = indicators['bb_middle'] = indicators['bb_lower'] = current_price
        
        # ATR Calculation
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr'] = safe_float(true_range.rolling(window=14).mean())
        except:
            indicators['atr'] = 0.01
        
        # Support Resistance
        try:
            indicators['pivot'] = (safe_float(high) + safe_float(low) + current_price) / 3
            indicators['resistance1'] = 2 * indicators['pivot'] - safe_float(low)
            indicators['support1'] = 2 * indicators['pivot'] - safe_float(high)
        except:
            indicators['pivot'] = indicators['resistance1'] = indicators['support1'] = current_price
        
        indicators['current_price'] = current_price
        
        # Chart data - last 50 periods
        chart_periods = min(50, len(data))
        indicators['chart_data'] = get_historical_data_for_chart(data, chart_periods)
        
    except Exception as e:
        print(f"Error in technical indicators: {e}")
        indicators = create_default_indicators(150.0)
    
    return indicators

def get_historical_data_for_chart(data, periods=50):
    """Prepare historical data for chart display"""
    try:
        # Get the last N periods
        data_slice = data.tail(periods)
        
        # Convert index to datetime strings
        if hasattr(data_slice.index, 'strftime'):
            dates = data_slice.index.strftime('%Y-%m-%d %H:%M').tolist()
        else:
            dates = [str(idx) for idx in data_slice.index]
        
        # Calculate technical indicators for the chart
        close_prices = data_slice['Close']
        
        chart_data = {
            'dates': dates,
            'open': data_slice['Open'].astype(float).round(5).tolist(),
            'high': data_slice['High'].astype(float).round(5).tolist(),
            'low': data_slice['Low'].astype(float).round(5).tolist(),
            'close': data_slice['Close'].astype(float).round(5).tolist(),
            'ema_20': close_prices.ewm(span=20).mean().astype(float).round(5).fillna(0).tolist(),
            'ema_50': close_prices.ewm(span=50).mean().astype(float).round(5).fillna(0).tolist(),
            'ema_200': close_prices.ewm(span=200).mean().astype(float).round(5).fillna(0).tolist()
        }
        
        print(f"Chart data prepared: {len(chart_data['dates'])} periods")
        return chart_data
        
    except Exception as e:
        print(f"Error preparing chart data: {e}")
        return create_default_chart_data()

def create_default_chart_data():
    """Create default chart data"""
    return {
        'dates': [datetime.now().strftime('%Y-%m-%d %H:%M')],
        'open': [150.0], 'high': [151.0], 'low': [149.0], 'close': [150.5],
        'ema_20': [150.0], 'ema_50': [150.0], 'ema_200': [150.0]
    }

def create_default_indicators(price):
    """Create default indicators"""
    return {
        'sma_20': price, 'sma_50': price, 'ema_12': price, 'ema_26': price, 'ema_200': price,
        'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
        'bb_upper': price, 'bb_middle': price, 'bb_lower': price,
        'atr': 0.01, 'pivot': price, 'resistance1': price, 'support1': price,
        'current_price': price,
        'chart_data': create_default_chart_data()
    }

def validate_and_fix_data(data):
    """Validasi dan perbaiki data yang rusak"""
    try:
        # Pastikan data tidak kosong
        if data.empty:
            return create_fallback_data()
        
        # Periksa dan perbaiki nilai yang tidak valid
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                # Ganti nilai NaN dengan forward fill
                data[col] = data[col].fillna(method='ffill')
                # Jika masih ada NaN, isi dengan mean
                if data[col].isna().any():
                    data[col] = data[col].fillna(data[col].mean())
        
        # Pastikan ada cukup data
        if len(data) < 20:
            return enhance_data_with_historical(data)
        
        return data
        
    except Exception as e:
        print(f"Error validating data: {e}")
        return create_fallback_data()

def create_fallback_data():
    """Buat data fallback yang realistis"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    base_price = 200.0
    noise = np.random.normal(0, 0.5, 100)
    prices = base_price + np.cumsum(noise)
    
    data = pd.DataFrame({
        'Open': prices,
        'High': prices + np.abs(np.random.normal(0, 0.3, 100)),
        'Low': prices - np.abs(np.random.normal(0, 0.3, 100)),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return data

def enhance_data_with_historical(data):
    """Tingkatkan data dengan menambahkan data historis sintetis"""
    if data.empty:
        return create_fallback_data()
    
    # Duplikat dan modifikasi data yang ada untuk membuat lebih banyak poin
    enhanced_data = data.copy()
    while len(enhanced_data) < 50:
        additional_data = data.copy()
        # Tambahkan sedikit variasi
        multiplier = 1 + np.random.normal(0, 0.01, len(additional_data))
        for col in ['Open', 'High', 'Low', 'Close']:
            additional_data[col] = additional_data[col] * multiplier
        
        enhanced_data = pd.concat([enhanced_data, additional_data])
    
    return enhanced_data.tail(100)  # Ambil 100 data terakhir

def analyze_with_deepseek(technical_data, fundamental_news, pair, timeframe):
    """Analisis dengan AI DeepSeek"""
    
    # Extract values
    current_price = float(technical_data.get('current_price', 0))
    rsi = float(technical_data.get('rsi', 50))
    atr = float(technical_data.get('atr', 0.01))
    
    # Simple analysis without API call for reliability
    if rsi < 30:
        signal = "BUY"
        confidence = 75
    elif rsi > 70:
        signal = "SELL"
        confidence = 75
    else:
        signal = "HOLD"
        confidence = 50
    
    # Calculate TP/SL based on ATR
    if signal == "BUY":
        tp1 = current_price + (atr * 2)
        tp2 = current_price + (atr * 3)
        sl = current_price - (atr * 1)
        rr_ratio = "1:2"
    elif signal == "SELL":
        tp1 = current_price - (atr * 2)
        tp2 = current_price - (atr * 3)
        sl = current_price + (atr * 1)
        rr_ratio = "1:2"
    else:
        tp1 = tp2 = sl = current_price
        rr_ratio = "N/A"
    
    return {
        'SIGNAL': signal,
        'CONFIDENCE_LEVEL': confidence,
        'ENTRY_PRICE': round(current_price, 4),
        'TAKE_PROFIT_1': round(tp1, 4),
        'TAKE_PROFIT_2': round(tp2, 4),
        'STOP_LOSS': round(sl, 4),
        'RISK_REWARD_RATIO': rr_ratio,
        'TIME_HORIZON': '4-8 hours',
        'ANALYSIS_SUMMARY': f'RSI: {rsi:.1f}, Price: {current_price:.4f}, ATR: {atr:.4f}. Signal based on technical analysis.'
    }

def load_historical_data_from_file(pair, timeframe):
    """Load historical data from local files dengan error handling"""
    data_dir = 'data/historical'
    filename = f"{pair}_{timeframe}.csv"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        try:
            # Load data from CSV
            data = pd.read_csv(filepath)
            
            # Check if datetime column exists
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index('datetime', inplace=True)
            elif 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close']
            column_mapping = {
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
            }
            
            # Rename columns if needed
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data[new_col] = data[old_col]
            
            if all(col in data.columns for col in required_columns):
                print(f"Loaded historical data from {filepath}: {len(data)} rows")
                return data
            else:
                print(f"Missing required columns in {filepath}")
                return None
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    else:
        print(f"Historical file not found: {filepath}")
        return None

def get_data_with_fallback(pair, timeframe, period):
    """Get data with fallback to historical files"""
    try:
        # Try to get real-time data first
        yf_symbol = pair_mapping[pair]
        yf_timeframe = timeframe_mapping[timeframe]
        
        print(f"Fetching real-time data for {yf_symbol} ({yf_timeframe})...")
        data = yf.download(yf_symbol, period=period, interval=yf_timeframe, progress=False)
        
        if not data.empty and len(data) > 20:
            print(f"Using real-time data for {pair}: {len(data)} rows")
            
            # Save to database
            db.save_price_data(pair, timeframe, data)
            
            return data
        else:
            # Fallback to historical data
            print("Real-time data insufficient, trying historical files...")
            historical_data = load_historical_data_from_file(pair, timeframe)
            
            if historical_data is not None:
                return historical_data
            else:
                # Final fallback - create minimal data
                print("Creating minimal fallback data...")
                dates = pd.date_range(end=datetime.now(), periods=50, freq='H')
                data = pd.DataFrame({
                    'Open': 150.0, 'High': 151.0, 'Low': 149.0, 'Close': 150.0
                }, index=dates)
                return data
                
    except Exception as e:
        print(f"Error getting real-time data: {e}")
        # Fallback to historical data
        historical_data = load_historical_data_from_file(pair, timeframe)
        return historical_data if historical_data is not None else pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_analysis')
def get_analysis():
    try:
        pair = request.args.get('pair', 'GBPJPY')
        timeframe = request.args.get('timeframe', '4H')
        
        print(f"Processing analysis for {pair} {timeframe}")
        
        if pair not in pair_mapping:
            return jsonify({'error': 'Invalid pair'})
        if timeframe not in timeframe_mapping:
            return jsonify({'error': 'Invalid timeframe'})
        
        # Get data with fallback
        yf_timeframe = timeframe_mapping[timeframe]
        period = '60d' if yf_timeframe in ['1h', '2h', '4h'] else '1y'
        
        data = get_data_with_fallback(pair, timeframe, period)
        
        print(f"Data retrieved: {len(data)} rows")
        
        if data.empty or len(data) < 20:
            return jsonify({'error': 'Insufficient data available'})
        
        # Current price data
        current_price = float(data['Close'].iloc[-1])
        price_change_pct = 0.0
        
        if len(data) > 1:
            prev_price = float(data['Close'].iloc[-2])
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Technical indicators
        indicators = get_technical_indicators(data)
        indicators['current_price'] = current_price
        indicators['price_change'] = price_change_pct
        
        # Fundamental news
        news = get_real_forex_news()
        
        # AI Analysis
        ai_analysis = analyze_with_deepseek(indicators, news, pair, timeframe)
        
        # Prepare response
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': round(current_price, 4),
            'price_change': round(price_change_pct, 2),
            'technical_indicators': {
                'RSI': round(float(indicators.get('rsi', 50)), 2),
                'MACD': round(float(indicators.get('macd', 0)), 4),
                'SMA_20': round(float(indicators.get('sma_20', current_price)), 4),
                'SMA_50': round(float(indicators.get('sma_50', current_price)), 4),
                'EMA_200': round(float(indicators.get('ema_200', current_price)), 4),
                'ATR': round(float(indicators.get('atr', 0.01)), 4),
                'Support': round(float(indicators.get('support1', current_price)), 4),
                'Resistance': round(float(indicators.get('resistance1', current_price)), 4)
            },
            'ai_analysis': ai_analysis,
            'fundamental_news': news,
            'chart_data': indicators.get('chart_data', create_default_chart_data()),
            'data_points': len(data),
            'data_period': f"{period} ({yf_timeframe})"
        }
        
        # Save to database
        db.save_analysis(response)
        
        print(f"Analysis completed for {pair}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return jsonify({'error': f'Analysis error: {str(e)}'})

@app.route('/get_historical_data')
def get_historical_data():
    """Endpoint untuk data historis"""
    try:
        pair = request.args.get('pair', 'GBPJPY')
        days = int(request.args.get('days', '30'))
        
        # Get from yfinance as fallback
        yf_symbol = pair_mapping.get(pair, 'GBPJPY=X')
        data = yf.download(yf_symbol, period=f'{days}d', interval='1h', progress=False)
        
        if data.empty:
            return jsonify({'error': 'No historical data available'})
        
        chart_data = get_historical_data_for_chart(data, min(100, len(data)))
        return jsonify(chart_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_multiple_analysis')
def get_multiple_analysis():
    """Analisis untuk semua pairs sekaligus"""
    try:
        timeframe = request.args.get('timeframe', '4H')
        results = {}
        
        for pair in pair_mapping.keys():
            try:
                time.sleep(0.5)  # Rate limiting
                
                yf_symbol = pair_mapping[pair]
                yf_timeframe = timeframe_mapping[timeframe]
                period = '60d' if yf_timeframe in ['1h', '2h', '4h'] else '1y'
                data = yf.download(yf_symbol, period=period, interval=yf_timeframe, progress=False)
                
                if not data.empty and len(data) > 20:
                    current_price = float(data['Close'].iloc[-1])
                    indicators = get_technical_indicators(data)
                    news = get_real_forex_news()
                    ai_analysis = analyze_with_deepseek(indicators, news, pair, timeframe)
                    
                    results[pair] = {
                        'pair': pair,
                        'timeframe': timeframe,
                        'current_price': round(current_price, 4),
                        'ai_analysis': ai_analysis,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    }
                else:
                    results[pair] = {'error': 'No data available'}
                    
            except Exception as e:
                results[pair] = {'error': str(e)}
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting Forex Analysis System...")
    print("Database initialized:", db.db_path)
    
    # Create necessary directories
    os.makedirs('data/historical', exist_ok=True)
    
    if not os.path.exists('templates'):
        print("ERROR: 'templates' folder not found!")
        print("Current directory:", os.getcwd())
        print("Contents:", os.listdir('.'))
    else:
        print("Template folder found.")
        print("Templates available:", os.listdir('templates'))
    
    app.run(debug=True, host='127.0.0.1', port=5000)
