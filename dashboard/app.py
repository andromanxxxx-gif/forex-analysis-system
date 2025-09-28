# app.py
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os
import sqlite3
import json
import traceback
import time
from datetime import datetime, timedelta
import random

# NOTE:
# - Place this app.py in the same folder as index.html and the CSV files:
#   USDJPY_1D.csv, GBPJPY_1D.csv, EURJPY_1D.csv, CHFJPY_1D.csv (or similar)
# - To use a real AI key set environment variable DEEPSEEK_API_KEY.
#   e.g. export DEEPSEEK_API_KEY="sk-xxxxx"
# - Run: python app.py
# - Open http://127.0.0.1:5000 in your browser.

app = Flask(__name__, static_folder='.', static_url_path='/static')

# Optional: set your real key in environment variable
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'sk-73d83584fd614656926e1d8860eae9ca')  # keep empty for fallback
DEEPSEEK_API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1/chat/completions')

# Basic pair base prices (fallback)
pair_base_prices = {
    'GBPJPY': 187.50,
    'USDJPY': 149.50,
    'EURJPY': 174.80,
    'CHFJPY': 170.20,
    'AUDJPY': 105.30,
    'CADJPY': 108.90
}

# In-memory store for historical CSV data (loaded at startup)
HISTORICAL = {}  # keys: 'USDJPY', 'GBPJPY', etc -> DataFrame with columns ['date','open','high','low','close','volume']

DB_PATH = 'forex_analysis.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            current_price REAL,
            price_change REAL,
            technical_indicators TEXT,
            ai_analysis TEXT,
            chart_data TEXT,
            data_source TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_analysis(analysis_data):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO analysis_results 
                (pair, timeframe, timestamp, current_price, price_change, technical_indicators, ai_analysis, chart_data, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_data['pair'],
            analysis_data['timeframe'],
            analysis_data['timestamp'],
            analysis_data['current_price'],
            analysis_data['price_change'],
            json.dumps(analysis_data['technical_indicators']),
            json.dumps(analysis_data['ai_analysis']),
            json.dumps(analysis_data.get('chart_data', {})),
            analysis_data.get('data_source', 'simulated')
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Error saving analysis:", e)

def load_csv_data():
    """Load CSV files found in current directory that match pattern *_1D.csv"""
    files = [f for f in os.listdir('.') if f.endswith('_1D.csv')]
    for f in files:
        try:
            # try read with pandas and normalize columns
            df = pd.read_csv(f)
            # Try to detect datetime column
            if 'date' not in (c.lower() for c in df.columns):
                # find first column that looks like date or index
                df.columns = [c.strip() for c in df.columns]
                if 'Date' in df.columns:
                    df = df.rename(columns={'Date': 'date'})
                elif 'datetime' in (c.lower() for c in df.columns):
                    cols = {c: c for c in df.columns}
                else:
                    # fallback: assume first column is date-like
                    df = df.rename(columns={df.columns[0]: 'date'})
            # standardize column names to lower-case
            df_cols = {c: c.lower() for c in df.columns}
            df = df.rename(columns=df_cols)
            # ensure required columns exist
            for col in ['date','open','high','low','close']:
                if col not in df.columns:
                    # if missing, try to infer from common names
                    pass
            # convert date
            try:
                df['date'] = pd.to_datetime(df['date'])
            except Exception:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            # sort by date ascending
            df = df.sort_values('date').reset_index(drop=True)
            # construct pair name from filename: USDJPY_1D.csv -> USDJPY
            pair = os.path.basename(f).split('_')[0].upper()
            HISTORICAL[pair] = df
            print(f"Loaded historical: {f} -> pair {pair}, {len(df)} rows")
        except Exception as e:
            print(f"Failed to load {f}: {e}")

def get_realtime_price_sim(pair):
    """Return a realistic simulated realtime price (fallback if no scraping)."""
    base = pair_base_prices.get(pair, 150.0)
    hour = datetime.now().hour
    # volatility depending on session
    if 0 <= hour < 5:
        vol = 0.001
    elif 5 <= hour < 13:
        vol = 0.002
    else:
        vol = 0.0015
    change = random.normalvariate(0, vol)
    price = round(base*(1+change), 4)
    # clamp to reasonable range for known pairs
    if pair == 'GBPJPY':
        price = max(180.0, min(195.0, price))
    if pair == 'USDJPY':
        price = max(147.0, min(152.0, price))
    return price

def series_from_dfpair(pair, points=200):
    """Return chart data dict from HISTORICAL[pair] (last n points)."""
    if pair not in HISTORICAL:
        # fallback to generated series
        chart = generate_realistic_chart_data(pair, periods=points)
        return chart
    df = HISTORICAL[pair].copy()
    if 'close' not in df.columns:
        # fallback: try second column
        df['close'] = df.iloc[:,1]
    df = df.dropna(subset=['close'])
    df = df.tail(points)
    dates = df['date'].dt.strftime('%Y-%m-%d %H:%M').tolist()
    close = df['close'].astype(float).round(4).tolist()
    high = (df['high'].astype(float).round(4).tolist() if 'high' in df.columns else [c*1.001 for c in close])
    low = (df['low'].astype(float).round(4).tolist() if 'low' in df.columns else [c*0.999 for c in close])
    open_ = (df['open'].astype(float).round(4).tolist() if 'open' in df.columns else close)
    s = pd.Series(close)
    ema20 = s.ewm(span=20).mean().round(4).tolist()
    ema50 = s.ewm(span=50).mean().round(4).tolist()
    return {
        'dates': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'ema_20': ema20,
        'ema_50': ema50
    }

def generate_realistic_chart_data(pair, periods=100):
    # similar to earlier code but simpler: random walk around base price
    base = pair_base_prices.get(pair, 150.0)
    prices = []
    for i in range(periods):
        noise = random.normalvariate(0, 0.002)
        prices.append(round(base*(1+noise), 4))
    dates = [(datetime.now() - timedelta(minutes=(periods-i))).strftime('%Y-%m-%d %H:%M') for i in range(periods)]
    s = pd.Series(prices)
    ema20 = s.ewm(span=20).mean().round(4).tolist()
    ema50 = s.ewm(span=50).mean().round(4).tolist()
    return {'dates': dates, 'open': prices, 'high':[p*1.001 for p in prices], 'low':[p*0.999 for p in prices], 'close': prices, 'ema_20': ema20, 'ema_50': ema50}

def calculate_indicators_from_chart(chart_data):
    """Return simplified technical indicators from chart_data dict."""
    close = pd.Series(chart_data['close'])
    current_price = float(close.iloc[-1])
    base = pair_base_prices.get('USDJPY', current_price)
    # RSI simple approximation
    delta = close.diff().fillna(0)
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = (up / (down.replace(0, np.nan))).fillna(0)
    rsi = (100 - (100 / (1 + rs))).fillna(50).iloc[-1]
    sma20 = close.rolling(20, min_periods=1).mean().iloc[-1]
    sma50 = close.rolling(50, min_periods=1).mean().iloc[-1]
    macd = (close.ewm(span=12).mean() - close.ewm(span=26).mean()).iloc[-1]
    vol = int(random.randint(5000, 50000))
    price_change_pct = ((current_price - close.iloc[0]) / close.iloc[0]) * 100 if close.iloc[0] != 0 else 0
    return {
        'current_price': round(current_price,4),
        'price_change': round(price_change_pct, 4),
        'rsi': round(float(rsi), 2),
        'macd': round(float(macd), 4),
        'sma_20': round(float(sma20), 4),
        'sma_50': round(float(sma50), 4),
        'ema_12': round(float(close.ewm(span=12).mean().iloc[-1]), 4),
        'ema_26': round(float(close.ewm(span=26).mean().iloc[-1]), 4),
        'resistance': round(current_price * 1.004, 4),
        'support': round(current_price * 0.996, 4),
        'volume': vol
    }

def generate_ai_fallback(tech, pair):
    """Simple AI fallback (no external call)."""
    rsi = tech.get('rsi', 50)
    cp = tech.get('current_price', pair_base_prices.get(pair, 150.0))
    if rsi < 30:
        signal = "BUY"
    elif rsi > 70:
        signal = "SELL"
    else:
        signal = "HOLD"
    confidence = 70 if signal in ["BUY","SELL"] else 50
    # simple TP/SL
    atr = cp * 0.002
    if signal == "BUY":
        tp1 = cp + atr*2; tp2 = cp + atr*3; sl = cp - atr
    elif signal == "SELL":
        tp1 = cp - atr*2; tp2 = cp - atr*3; sl = cp + atr
    else:
        tp1 = tp2 = sl = cp
    return {
        'SIGNAL': signal,
        'SIGNAL_TYPE': 'LONG' if signal=='BUY' else 'SHORT' if signal=='SELL' else 'NEUTRAL',
        'CONFIDENCE_LEVEL': confidence,
        'ENTRY_PRICE': round(cp,4),
        'TAKE_PROFIT_1': round(tp1,4),
        'TAKE_PROFIT_2': round(tp2,4),
        'STOP_LOSS': round(sl,4),
        'RISK_REWARD_RATIO': '1:2' if signal in ['BUY','SELL'] else 'N/A',
        'TIME_HORIZON': '4-12 hours',
        'PIPS_RISK': round(abs(cp-sl)*100,1),
        'ANALYSIS_SUMMARY': f'Fallback analysis based on RSI {tech.get("rsi")}',
        'RAW_AI_RESPONSE': 'fallback',
        'TRADING_ADVICE': 'Use risk management. This is fallback advice.'
    }

@app.route('/')
def index():
    # Serve the index.html from current directory
    return send_from_directory('.', 'index.html')

@app.route('/get_history')
def get_history():
    pair = request.args.get('pair', 'GBPJPY').upper()
    points = int(request.args.get('points', 200))
    try:
        chart = series_from_dfpair(pair, points=points)
        return jsonify({'pair': pair, 'chart_data': chart, 'data_source': 'historical_csv' if pair in HISTORICAL else 'generated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_multiple_pairs')
def get_multiple_pairs():
    """Quick overview for pair cards"""
    pairs = ['GBPJPY','USDJPY','EURJPY','CHFJPY']
    results = {}
    for p in pairs:
        try:
            chart = series_from_dfpair(p, points=60)
            tech = calculate_indicators_from_chart(chart)
            ai = generate_ai_fallback(tech,p)
            results[p] = {
                'price': tech['current_price'],
                'change': tech['price_change'],
                'signal': ai['SIGNAL'],
                'confidence': ai['CONFIDENCE_LEVEL'],
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
        except Exception as e:
            results[p] = {'error': str(e)}
    return jsonify(results)

@app.route('/get_analysis')
def get_analysis():
    """
    Query params:
      pair (e.g. GBPJPY)
      timeframe (e.g. 1H, 4H, 1D)
      use_history=1 -> include historical CSV as baseline for chart
      history_points=200 -> number of historical points
    """
    pair = request.args.get('pair', 'GBPJPY').upper()
    timeframe = request.args.get('timeframe', '1H')
    use_history = request.args.get('use_history', '0') == '1'
    history_points = int(request.args.get('history_points', 200))
    try:
        # 1) chart data: either from history + realtime appended, or generated
        if use_history and pair in HISTORICAL:
            chart = series_from_dfpair(pair, points=history_points)
            # append a last realtime point
            last_price = get_realtime_price_sim(pair)
            chart['dates'].append(datetime.now().strftime('%Y-%m-%d %H:%M'))
            chart['close'].append(round(last_price,4))
            chart['open'].append(chart['close'][-2] if len(chart['close'])>1 else last_price)
            chart['high'].append(max(chart['close'][-1], chart['open'][-1])*1.001)
            chart['low'].append(min(chart['close'][-1], chart['open'][-1])*0.999)
            # recompute emas on updated closes
            s = pd.Series(chart['close'])
            chart['ema_20'] = s.ewm(span=20).mean().round(4).tolist()
            chart['ema_50'] = s.ewm(span=50).mean().round(4).tolist()
            data_source = 'historical_csv + realtime'
        else:
            chart = generate_realistic_chart_data(pair, periods=history_points if history_points>50 else 100)
            data_source = 'simulated_realtime'
        # 2) technical indicators
        tech = calculate_indicators_from_chart(chart)
        # 3) AI analysis (use fallback if no real key)
        if DEEPSEEK_API_KEY:
            # if user has key, you could call external model here.
            ai_analysis = generate_ai_fallback(tech, pair)  # placeholder for real call
        else:
            ai_analysis = generate_ai_fallback(tech, pair)
        # 4) news (simple simulated)
        news = [
            {'source':'Reuters','headline':'BoJ watches JPY crosses', 'timestamp': datetime.now().strftime('%H:%M'), 'impact':'High','sentiment':'neutral'},
            {'source':'MarketWatch','headline': f'{pair} shows range', 'timestamp': datetime.now().strftime('%H:%M'), 'impact':'Medium','sentiment':'cautious'}
        ]
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': tech['current_price'],
            'price_change': tech['price_change'],
            'technical_indicators': {
                'RSI': tech['rsi'],
                'MACD': tech['macd'],
                'SMA_20': tech['sma_20'],
                'SMA_50': tech['sma_50'],
                'EMA_12': tech['ema_12'],
                'EMA_26': tech['ema_26'],
                'Resistance': tech['resistance'],
                'Support': tech['support'],
                'Volume': tech['volume']
            },
            'ai_analysis': ai_analysis,
            'fundamental_news': news,
            'chart_data': chart,
            'data_points': len(chart.get('close', [])),
            'data_source': data_source
        }
        # Save asynchronously-lite (not blocking much)
        try:
            save_analysis(response)
        except Exception as e:
            print("Warning: failed to save analysis:", e)
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status':'ok', 'time': datetime.now().isoformat(), 'historical_pairs': list(HISTORICAL.keys())})

if __name__ == '__main__':
    print("Starting backend...")
    init_db()
    load_csv_data()
    print("Historical pairs loaded:", list(HISTORICAL.keys()))
    app.run(debug=True, host='127.0.0.1', port=5000)
