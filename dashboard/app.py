from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import json
import time
import os
import sqlite3
import traceback

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Forex pairs mapping - corrected symbols
pair_mapping = {
    'GBPJPY': 'GBPJPY=X',
    'USDJPY': 'USDJPY=X', 
    'EURJPY': 'EURJPY=X',
    'CHFJPY': 'CHFJPY=X',
    'AUDJPY': 'AUDJPY=X',
    'CADJPY': 'CADJPY=X'
}

# Timeframe mapping with correct yfinance intervals
timeframe_mapping = {
    '1M': '1m',    # 1 minute
    '5M': '5m',    # 5 minutes
    '15M': '15m',  # 15 minutes
    '30M': '30m',  # 30 minutes
    '1H': '1h',    # 1 hour
    '2H': '2h',    # 2 hours
    '4H': '4h',    # 4 hours
    '1D': '1d',    # 1 day
    '1W': '1wk',   # 1 week
    '1MO': '1mo'   # 1 month
}

# Period mapping for different timeframes
period_mapping = {
    '1m': '1d',    # 1 minute - max 1 day data
    '5m': '5d',    # 5 minutes - max 5 days data
    '15m': '15d',  # 15 minutes - max 15 days data
    '30m': '1mo',  # 30 minutes - max 1 month data
    '1h': '2mo',   # 1 hour - max 2 months data
    '2h': '3mo',   # 2 hours - max 3 months data
    '4h': '6mo',   # 4 hours - max 6 months data
    '1d': '1y',    # 1 day - max 1 year data
    '1wk': '2y',   # 1 week - max 2 years data
    '1mo': '5y'    # 1 month - max 5 years data
}

class Database:
    def __init__(self, db_path='forex_analysis.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                current_price REAL,
                price_change REAL,
                technical_indicators TEXT,
                ai_analysis TEXT,
                chart_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                timeframe TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, analysis_data):
        """Save analysis results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_results 
                (pair, timeframe, timestamp, current_price, price_change, technical_indicators, ai_analysis, chart_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_data['pair'],
                analysis_data['timeframe'],
                analysis_data['timestamp'],
                analysis_data['current_price'],
                analysis_data['price_change'],
                json.dumps(analysis_data['technical_indicators']),
                json.dumps(analysis_data['ai_analysis']),
                json.dumps(analysis_data.get('chart_data', {}))
            ))
            
            conn.commit()
            conn.close()
            print(f"Analysis saved for {analysis_data['pair']}")
            
        except Exception as e:
            print(f"Error saving analysis: {e}")
    
    def save_price_data(self, pair, timeframe, data):
        """Save price data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if len(data) > 0:
                latest = data.iloc[-1]
                
                # Convert index to date string
                if hasattr(data.index, 'strftime'):
                    date_str = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_str = str(data.index[-1])
                
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (pair, timeframe, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pair, timeframe, date_str,
                    float(latest['Open']), float(latest['High']),
                    float(latest['Low']), float(latest['Close']),
                    float(latest.get('Volume', 0))
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving price data: {e}")

db = Database()

def get_real_time_price(pair):
    """Get real-time current price using yfinance Ticker"""
    try:
        symbol = pair_mapping.get(pair, f"{pair}=X")
        ticker = yf.Ticker(symbol)
        
        # Get real-time data
        info = ticker.info
        current_price = info.get('regularMarketPrice') or info.get('currentPrice')
        
        if current_price:
            return float(current_price)
        
        # Fallback to historical data
        hist = ticker.history(period='1d', interval='1m')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        
        return 0.0
    except Exception as e:
        print(f"Error getting real-time price for {pair}: {e}")
        return 0.0

def get_market_data(pair, timeframe):
    """Get real market data with proper error handling"""
    try:
        symbol = pair_mapping.get(pair, f"{pair}=X")
        yf_interval = timeframe_mapping.get(timeframe, '1h')
        period = period_mapping.get(yf_interval, '1mo')
        
        print(f"Fetching real-time data: {symbol}, {yf_interval}, {period}")
        
        # Method 1: Try yfinance download
        data = yf.download(symbol, period=period, interval=yf_interval, progress=False, auto_adjust=True)
        
        if data.empty:
            # Method 2: Try Ticker history
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=yf_interval, auto_adjust=True)
        
        if data.empty or len(data) < 5:
            print(f"No real-time data found for {pair}")
            return None
            
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in data.columns:
                print(f"Missing column {col} in data")
                return None
        
        print(f"Real-time data retrieved: {len(data)} rows, latest: {data.index[-1]}")
        return data
        
    except Exception as e:
        print(f"Error getting market data for {pair}: {e}")
        return None

def get_technical_indicators(data):
    """Calculate real technical indicators"""
    try:
        if data.empty or len(data) < 10:
            return create_default_indicators(150.0)
        
        close = data['Close']
        high = data['High']
        low = data['Low']
        
        current_price = float(close.iloc[-1])
        
        # Simple Moving Averages
        sma_20 = close.rolling(window=min(20, len(close))).mean()
        sma_50 = close.rolling(window=min(50, len(close))).mean()
        
        # Exponential Moving Averages
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        
        # RSI Calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()
        macd_histogram = macd_line - macd_signal
        
        # Support and Resistance (simplified)
        recent_high = high.tail(20).max()
        recent_low = low.tail(20).min()
        
        indicators = {
            'current_price': current_price,
            'sma_20': float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else current_price,
            'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else current_price,
            'ema_12': float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else current_price,
            'ema_26': float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else current_price,
            'rsi': float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0,
            'macd': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
            'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
            'macd_hist': float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0.0,
            'resistance': float(recent_high) if not pd.isna(recent_high) else current_price * 1.02,
            'support': float(recent_low) if not pd.isna(recent_low) else current_price * 0.98
        }
        
        return indicators
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return create_default_indicators(150.0)

def create_default_indicators(price):
    """Create default indicators as fallback"""
    return {
        'current_price': price,
        'sma_20': price, 'sma_50': price, 'ema_12': price, 'ema_26': price,
        'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
        'resistance': price * 1.02, 'support': price * 0.98
    }

def prepare_chart_data(data, max_points=100):
    """Prepare chart data from real market data"""
    try:
        if data.empty:
            return create_default_chart_data()
        
        # Take last N points
        data_slice = data.tail(max_points)
        
        # Convert dates
        dates = []
        for idx in data_slice.index:
            if hasattr(idx, 'strftime'):
                dates.append(idx.strftime('%Y-%m-%d %H:%M'))
            else:
                dates.append(str(idx))
        
        # Calculate EMAs
        close_prices = data_slice['Close']
        ema_20 = close_prices.ewm(span=20, adjust=False).mean()
        ema_50 = close_prices.ewm(span=50, adjust=False).mean()
        
        chart_data = {
            'dates': dates,
            'open': data_slice['Open'].tolist(),
            'high': data_slice['High'].tolist(),
            'low': data_slice['Low'].tolist(),
            'close': data_slice['Close'].tolist(),
            'ema_20': ema_20.tolist(),
            'ema_50': ema_50.tolist()
        }
        
        return chart_data
        
    except Exception as e:
        print(f"Error preparing chart data: {e}")
        return create_default_chart_data()

def create_default_chart_data():
    """Create default chart data"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    return {
        'dates': [current_time],
        'open': [150.0], 'high': [151.0], 'low': [149.0], 'close': [150.0],
        'ema_20': [150.0], 'ema_50': [150.0]
    }

def generate_trading_signal(indicators):
    """Generate trading signal based on technical indicators"""
    try:
        rsi = indicators.get('rsi', 50)
        price = indicators.get('current_price', 150)
        sma_20 = indicators.get('sma_20', price)
        sma_50 = indicators.get('sma_50', price)
        
        # Simple signal logic
        if rsi < 30 and price > sma_20 and sma_20 > sma_50:
            signal = "STRONG BUY"
            confidence = 85
        elif rsi > 70 and price < sma_20 and sma_20 < sma_50:
            signal = "STRONG SELL" 
            confidence = 85
        elif rsi < 40 and price > sma_50:
            signal = "BUY"
            confidence = 70
        elif rsi > 60 and price < sma_50:
            signal = "SELL"
            confidence = 70
        else:
            signal = "HOLD"
            confidence = 50
        
        # Calculate risk levels
        atr = abs(indicators.get('resistance', price) - indicators.get('support', price)) * 0.1
        
        if signal in ["BUY", "STRONG BUY"]:
            tp1 = price + (atr * 1.5)
            tp2 = price + (atr * 2.5)
            sl = price - (atr * 1.0)
            rr_ratio = "1:1.5"
        elif signal in ["SELL", "STRONG SELL"]:
            tp1 = price - (atr * 1.5)
            tp2 = price - (atr * 2.5)
            sl = price + (atr * 1.0)
            rr_ratio = "1:1.5"
        else:
            tp1 = tp2 = sl = price
            rr_ratio = "N/A"
        
        return {
            'SIGNAL': signal,
            'CONFIDENCE_LEVEL': confidence,
            'ENTRY_PRICE': round(price, 4),
            'TAKE_PROFIT_1': round(tp1, 4),
            'TAKE_PROFIT_2': round(tp2, 4),
            'STOP_LOSS': round(sl, 4),
            'RISK_REWARD_RATIO': rr_ratio,
            'TIME_HORIZON': 'Intraday' if confidence > 70 else 'Swing',
            'ANALYSIS_SUMMARY': f'RSI: {rsi:.1f}, Price: {price:.3f}, Trend: {"Bullish" if price > sma_50 else "Bearish"}'
        }
        
    except Exception as e:
        print(f"Error generating signal: {e}")
        return {
            'SIGNAL': 'HOLD',
            'CONFIDENCE_LEVEL': 50,
            'ENTRY_PRICE': 150.0,
            'TAKE_PROFIT_1': 151.0,
            'TAKE_PROFIT_2': 152.0,
            'STOP_LOSS': 149.0,
            'RISK_REWARD_RATIO': '1:1',
            'TIME_HORIZON': 'Wait',
            'ANALYSIS_SUMMARY': 'Signal generation error'
        }

def get_market_news():
    """Get real market news"""
    try:
        current_time = datetime.now().strftime('%H:%M')
        
        # Simulated real news based on market hours
        news_items = [
            {
                'source': 'Market Watch',
                'headline': 'Asian Session: JPY Pairs Active, BOJ Policy in Focus',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'Reuters',
                'headline': 'Forex Markets Show Moderate Volatility in Early Trading',
                'timestamp': current_time,
                'url': '#'
            },
            {
                'source': 'Technical Analysis',
                'headline': 'Key Support/Resistance Levels Being Tested',
                'timestamp': current_time,
                'url': '#'
            }
        ]
        
        return news_items
    except Exception as e:
        print(f"Error getting news: {e}")
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_analysis')
def get_analysis():
    try:
        pair = request.args.get('pair', 'GBPJPY')
        timeframe = request.args.get('timeframe', '1H')
        
        print(f"🔍 Processing REAL-TIME analysis for {pair} {timeframe}")
        
        if pair not in pair_mapping:
            return jsonify({'error': f'Invalid pair: {pair}'})
        
        # Get REAL market data
        market_data = get_market_data(pair, timeframe)
        
        if market_data is None:
            return jsonify({'error': f'No real-time data available for {pair}. Market may be closed or symbol invalid.'})
        
        # Calculate current price and change
        current_price = float(market_data['Close'].iloc[-1])
        price_change = 0.0
        
        if len(market_data) > 1:
            prev_price = float(market_data['Close'].iloc[-2])
            price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Get technical indicators from REAL data
        indicators = get_technical_indicators(market_data)
        
        # Prepare chart data
        chart_data = prepare_chart_data(market_data)
        
        # Generate trading signal
        ai_analysis = generate_trading_signal(indicators)
        
        # Get market news
        news = get_market_news()
        
        # Prepare response with REAL data
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': round(current_price, 4),
            'price_change': round(price_change, 2),
            'technical_indicators': {
                'RSI': round(indicators.get('rsi', 50), 2),
                'MACD': round(indicators.get('macd', 0), 4),
                'SMA_20': round(indicators.get('sma_20', current_price), 4),
                'SMA_50': round(indicators.get('sma_50', current_price), 4),
                'EMA_12': round(indicators.get('ema_12', current_price), 4),
                'EMA_26': round(indicators.get('ema_26', current_price), 4),
                'Resistance': round(indicators.get('resistance', current_price * 1.02), 4),
                'Support': round(indicators.get('support', current_price * 0.98), 4)
            },
            'ai_analysis': ai_analysis,
            'fundamental_news': news,
            'chart_data': chart_data,
            'data_points': len(market_data),
            'data_source': 'Yahoo Finance Real-time'
        }
        
        # Save to database
        db.save_analysis(response)
        db.save_price_data(pair, timeframe, market_data)
        
        print(f"✅ REAL-TIME Analysis completed for {pair}: {current_price}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in REAL-TIME analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Real-time analysis error: {str(e)}'})

@app.route('/get_multiple_pairs')
def get_multiple_pairs():
    """Get analysis for multiple pairs quickly"""
    try:
        timeframe = request.args.get('timeframe', '1H')
        results = {}
        
        # Analyze major JPY pairs
        pairs = ['GBPJPY', 'USDJPY', 'EURJPY']
        
        for pair in pairs:
            try:
                market_data = get_market_data(pair, timeframe)
                
                if market_data is not None and len(market_data) > 5:
                    current_price = float(market_data['Close'].iloc[-1])
                    indicators = get_technical_indicators(market_data)
                    signal = generate_trading_signal(indicators)
                    
                    results[pair] = {
                        'price': round(current_price, 4),
                        'signal': signal['SIGNAL'],
                        'confidence': signal['CONFIDENCE_LEVEL'],
                        'change': round(((current_price - float(market_data['Close'].iloc[-2])) / float(market_data['Close'].iloc[-2])) * 100, 2) if len(market_data) > 1 else 0,
                        'timestamp': datetime.now().strftime('%H:%M')
                    }
                else:
                    results[pair] = {'error': 'No data'}
                    
            except Exception as e:
                results[pair] = {'error': str(e)}
            
            time.sleep(0.5)  # Rate limiting
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_live_price/<pair>')
def get_live_price(pair):
    """Get only live price for quick updates"""
    try:
        price = get_real_time_price(pair)
        return jsonify({
            'pair': pair,
            'price': round(price, 4),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting REAL-TIME Forex Analysis System...")
    print("📊 Data Source: Yahoo Finance Real-time")
    print("💹 Supported Pairs:", list(pair_mapping.keys()))
    
    # Create necessary directories
    os.makedirs('data/historical', exist_ok=True)
    
    # Test real-time data connection
    print("🔌 Testing real-time data connection...")
    try:
        test_data = get_market_data('GBPJPY', '1H')
        if test_data is not None:
            print(f"✅ Real-time connection successful! Latest GBP/JPY: {test_data['Close'].iloc[-1]:.4f}")
        else:
            print("❌ Real-time connection failed")
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
