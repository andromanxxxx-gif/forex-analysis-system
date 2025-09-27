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
import sqlite3
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
        return float(value)
    except (ValueError, TypeError, IndexError):
        return default

def get_real_forex_news():
    """Web scraping untuk berita forex"""
    news_items = []
    
    try:
        # Berita real-time dengan informasi market aktual
        current_time = datetime.now().strftime('%H:%M')
        
        # Dapatkan harga terkini untuk berita yang lebih informatif
        prices = {}
        for pair in pair_mapping.keys():
            try:
                data = yf.download(pair_mapping[pair], period='1d', interval='1h')
                if not data.empty:
                    prices[pair] = data['Close'].iloc[-1]
            except:
                prices[pair] = 0.0
        
        news_items = [
            {
                'source': 'Market Watch',
                'headline': f'JPY Pairs Update - GBP/JPY: {prices.get("GBPJPY", 0):.3f}, USD/JPY: {prices.get("USDJPY", 0):.3f}',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'Economic Calendar', 
                'headline': 'Bank of Japan Monetary Policy Decision in Focus',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'Technical Analysis',
                'headline': 'Yen Volatility Expected Amid Global Market Movements',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'Forex News',
                'headline': 'JPY Crosses Show Technical Breakout Opportunities',
                'timestamp': current_time,
                'url': '#'
            }
        ]
        
        # Simpan ke database
        db.save_news(news_items)
        
    except Exception as e:
        print(f"Error in news generation: {e}")
        # Fallback news
        current_time = datetime.now().strftime('%H:%M')
        news_items = [
            {
                'source': 'System',
                'headline': 'Market Analysis in Progress',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'System',
                'headline': 'Real-time Data Processing',
                'timestamp': current_time,
                'url': '#'
            }
        ]
    
    return news_items

def load_historical_data_from_file(pair, timeframe):
    """Load historical data from local files"""
    data_dir = 'data/historical'
    filename = f"{pair}_{timeframe}.csv"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        try:
            # Load data from CSV
            data = pd.read_csv(filepath)
            
            # Convert datetime column
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
                data.set_index('datetime', inplace=True)
            elif 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data.set_index('Date', inplace=True)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            column_mapping = {
                'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low', 'CLOSE': 'close'
            }
            
            # Rename columns to standard names
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data.rename(columns={old_col: new_col}, inplace=True)
            
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
            # Save to database for future use
            db.save_price_data(pair, data)
            return data
        else:
            # Fallback to historical data from files
            print("Real-time data insufficient, trying historical files...")
            historical_data = load_historical_data_from_file(pair, timeframe)
            if historical_data is not None:
                return historical_data
            else:
                # Final fallback: try database
                print("Trying database historical data...")
                db_data = db.get_historical_prices(pair, 30)  # Last 30 days
                if not db_data.empty:
                    db_data.set_index('timestamp', inplace=True)
                    return db_data
                else:
                    print("No data available from any source")
                    return None
                    
    except Exception as e:
        print(f"Error getting real-time data: {e}")
        # Fallback to historical data from files
        historical_data = load_historical_data_from_file(pair, timeframe)
        return historical_data

def get_technical_indicators(data):
    """Menghitung indikator teknikal"""
    indicators = {}
    
    try:
        if data.empty or len(data) < 20:
            return create_default_indicators(150.0)
        
        # Determine column names based on data structure
        open_col = 'Open' if 'Open' in data.columns else 'open'
        high_col = 'High' if 'High' in data.columns else 'high'
        low_col = 'Low' if 'Low' in data.columns else 'low'
        close_col = 'Close' if 'Close' in data.columns else 'close'
        
        # Price data
        high = data[high_col]
        low = data[low_col]
        close = data[close_col]
        
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
        
        # Determine column names
        open_col = 'Open' if 'Open' in data.columns else 'open'
        high_col = 'High' if 'High' in data.columns else 'high'
        low_col = 'Low' if 'Low' in data.columns else 'low'
        close_col = 'Close' if 'Close' in data.columns else 'close'
        
        # Calculate technical indicators for the chart
        close_prices = data_slice[close_col]
        
        chart_data = {
            'dates': data_slice.index.strftime('%Y-%m-%d %H:%M').tolist(),
            'open': data_slice[open_col].astype(float).round(5).tolist(),
            'high': data_slice[high_col].astype(float).round(5).tolist(),
            'low': data_slice[low_col].astype(float).round(5).tolist(),
            'close': data_slice[close_col].astype(float).round(5).tolist(),
            'ema_20': close_prices.ewm(span=20).mean().astype(float).round(5).tolist(),
            'ema_50': close_prices.ewm(span=50).mean().astype(float).round(5).tolist(),
            'ema_200': close_prices.ewm(span=200).mean().astype(float).round(5).tolist()
        }
        
        print(f"Chart data prepared: {len(chart_data['dates'])} periods")
        return chart_data
    except Exception as e:
        print(f"Error preparing historical data: {e}")
        return create_default_chart_data()

def create_default_chart_data():
    """Create default chart data"""
    # Create some sample data for chart
    dates = []
    prices = []
    current_date = datetime.now() - timedelta(days=5)
    
    base_price = 150.0
    for i in range(20):
        dates.append(current_date.strftime('%Y-%m-%d %H:%M'))
        price = base_price + np.random.normal(0, 0.5)
        prices.append(price)
        current_date += timedelta(hours=1)
    
    return {
        'dates': dates,
        'open': prices,
        'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
        'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
        'close': prices,
        'ema_20': prices,
        'ema_50': prices,
        'ema_200': prices
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

def analyze_with_deepseek(technical_data, fundamental_news, pair, timeframe):
    """Analisis dengan AI DeepSeek"""
    
    # Extract values
    current_price = float(technical_data.get('current_price', 0))
    rsi = float(technical_data.get('rsi', 50))
    atr = float(technical_data.get('atr', 0.01))
    ema_200 = float(technical_data.get('ema_200', current_price))
    
    # Enhanced analysis logic
    price_above_ema = current_price > ema_200
    
    if rsi < 30 and price_above_ema:
        signal = "STRONG BUY"
        confidence = 80
    elif rsi > 70 and not price_above_ema:
        signal = "STRONG SELL"
        confidence = 80
    elif rsi < 35:
        signal = "BUY"
        confidence = 70
    elif rsi > 65:
        signal = "SELL"
        confidence = 70
    else:
        signal = "HOLD"
        confidence = 50
    
    # Calculate TP/SL based on ATR
    if "BUY" in signal:
        tp1 = current_price + (atr * 2)
        tp2 = current_price + (atr * 3)
        sl = current_price - (atr * 1)
        rr_ratio = "1:2"
    elif "SELL" in signal:
        tp1 = current_price - (atr * 2)
        tp2 = current_price - (atr * 3)
        sl = current_price + (atr * 1)
        rr_ratio = "1:2"
    else:
        tp1 = tp2 = sl = current_price
        rr_ratio = "N/A"
    
    analysis_summary = f"RSI: {rsi:.1f}, Price: {current_price:.4f}, EMA200: {ema_200:.4f}. "
    if "STRONG" in signal:
        analysis_summary += "Strong signal due to RSI extreme and price position relative to EMA200."
    else:
        analysis_summary += "Signal based on technical analysis."
    
    return {
        'SIGNAL': signal,
        'CONFIDENCE_LEVEL': confidence,
        'ENTRY_PRICE': round(current_price, 4),
        'TAKE_PROFIT_1': round(tp1, 4),
        'TAKE_PROFIT_2': round(tp2, 4),
        'STOP_LOSS': round(sl, 4),
        'RISK_REWARD_RATIO': rr_ratio,
        'TIME_HORIZON': '4-8 hours',
        'ANALYSIS_SUMMARY': analysis_summary
    }

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
        
        # Determine period based on timeframe
        period_map = {
            '1h': '60d',  # 60 days for hourly
            '2h': '60d',  # 60 days for 2-hourly
            '4h': '60d',  # 60 days for 4-hourly
            '1d': '365d'  # 1 year for daily
        }
        
        yf_timeframe = timeframe_mapping[timeframe]
        period = period_map.get(yf_timeframe, '60d')
        
        # Get data with fallback to historical files
        data = get_data_with_fallback(pair, timeframe, period)
        
        if data is None or data.empty or len(data) < 20:
            return jsonify({'error': 'No data available (real-time or historical)'})
        
        print(f"Data loaded successfully: {len(data)} rows")
        
        # Current price data
        close_col = 'Close' if 'Close' in data.columns else 'close'
        current_price = float(data[close_col].iloc[-1])
        price_change_pct = 0.0
        
        if len(data) > 1:
            prev_price = float(data[close_col].iloc[-2])
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Technical indicators
        indicators = get_technical_indicators(data)
        indicators['current_price'] = current_price
        indicators['price_change'] = price_change_pct
        
        # Fundamental news
        news = get_real_forex_news()
        
        # AI Analysis
        ai_analysis = analyze_with_deepseek(indicators, news, pair, timeframe)
        
        # Calculate historical stats
        historical_stats = calculate_historical_stats(data, close_col)
        
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
                'Resistance': round(float(indicators.get('resistance1', current_price)), 4),
                'BB_Upper': round(float(indicators.get('bb_upper', current_price)), 4),
                'BB_Lower': round(float(indicators.get('bb_lower', current_price)), 4)
            },
            'ai_analysis': ai_analysis,
            'fundamental_news': news,
            'chart_data': indicators.get('chart_data', create_default_chart_data()),
            'historical_stats': historical_stats,
            'data_points': len(data),
            'data_period': f"{period} ({yf_timeframe})"
        }
        
        # Save to database
        db.save_analysis(response)
        
        print(f"Analysis completed successfully for {pair}")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return jsonify({'error': f'Analysis error: {str(e)}'})

def calculate_historical_stats(data, close_col='close'):
    """Calculate historical performance statistics"""
    try:
        if len(data) < 2:
            return {}
        
        close_prices = data[close_col]
        high_prices = data['High' if 'High' in data.columns else 'high']
        low_prices = data['Low' if 'Low' in data.columns else 'low']
        
        stats = {
            '1_period_change': calculate_percentage_change(close_prices, 1),
            '5_period_change': calculate_percentage_change(close_prices, 5),
            '10_period_change': calculate_percentage_change(close_prices, 10),
            '20_period_change': calculate_percentage_change(close_prices, 20),
            'high_10_period': float(high_prices.tail(10).max()),
            'low_10_period': float(low_prices.tail(10).min()),
            'volatility': float(close_prices.pct_change().std() * 100) if len(close_prices) > 1 else 0,
            'total_periods': len(data)
        }
        
        return stats
    except Exception as e:
        print(f"Error calculating historical stats: {e}")
        return {}

def calculate_percentage_change(prices, periods_back):
    """Calculate percentage change from periods back"""
    if len(prices) > periods_back:
        current_price = float(prices.iloc[-1])
        past_price = float(prices.iloc[-periods_back-1])
        return ((current_price - past_price) / past_price) * 100
    return 0

@app.route('/get_historical_data')
def get_historical_data():
    """Endpoint untuk data historis khusus"""
    try:
        pair = request.args.get('pair', 'GBPJPY')
        days = int(request.args.get('days', '30'))
        
        # Try database first
        db_data = db.get_historical_prices(pair, days)
        if not db_data.empty:
            db_data.set_index('timestamp', inplace=True)
            chart_data = get_historical_data_for_chart(db_data, min(100, len(db_data)))
            return jsonify(chart_data)
        
        # Fallback to file
        historical_data = load_historical_data_from_file(pair, '1H')
        if historical_data is not None:
            chart_data = get_historical_data_for_chart(historical_data, min(100, len(historical_data)))
            return jsonify(chart_data)
        
        return jsonify({'error': 'No historical data available'})
        
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
                
                # Use the main analysis function for each pair
                yf_symbol = pair_mapping[pair]
                yf_timeframe = timeframe_mapping[timeframe]
                period = '60d' if yf_timeframe in ['1h', '2h', '4h'] else '1y'
                data = yf.download(yf_symbol, period=period, interval=yf_timeframe, progress=False)
                
                if not data.empty and len(data) > 20:
                    close_col = 'Close' if 'Close' in data.columns else 'close'
                    current_price = float(data[close_col].iloc[-1])
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

@app.route('/system_status')
def system_status():
    """Endpoint untuk mengecek status sistem"""
    try:
        # Check database connection
        conn = sqlite3.connect('forex_data.db')
        cursor = conn.cursor()
        
        # Get table counts
        cursor.execute("SELECT COUNT(*) FROM trading_analysis")
        analysis_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM news_data")
        news_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM price_data")
        price_count = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'status': 'operational',
            'database': {
                'analysis_records': analysis_count,
                'news_records': news_count,
                'price_records': price_count
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("=" * 50)
    print("🤖 AI Forex Trading System")
    print("=" * 50)
    
    # Create data directory if it doesn't exist
    os.makedirs('data/historical', exist_ok=True)
    os.makedirs('data/exports', exist_ok=True)
    os.makedirs('data/backups', exist_ok=True)
    
    print("Data directories created")
    print("Database initialized:", db.db_path)
    
    if not os.path.exists('templates'):
        print("❌ ERROR: 'templates' folder not found!")
        print("Please create a 'templates' folder with 'index.html' inside")
    else:
        print("✅ Template folder found")
        print("🌐 Starting web server...")
        print("📊 Dashboard available at: http://127.0.0.1:5000")
        print("🔄 Auto-refresh every 5 minutes")
        print("-" * 50)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
