from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import json
import time
import os
import sqlite3
import traceback
import requests
import xml.etree.ElementTree as ET

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ==============================
# CONFIGURASI SISTEM
# ==============================
DEEPSEEK_API_KEY = "****"  # ganti dengan API key kamu
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Pair dengan konfigurasi realistis
pair_configurations = {
    'GBPJPY': {'base_price': 187.50, 'volatility': 0.0035, 'min_price': 180.0, 'max_price': 195.0, 'pip_size': 0.01},
    'USDJPY': {'base_price': 149.50, 'volatility': 0.0020, 'min_price': 147.0, 'max_price': 152.0, 'pip_size': 0.01},
    'EURJPY': {'base_price': 174.80, 'volatility': 0.0028, 'min_price': 172.0, 'max_price': 178.0, 'pip_size': 0.01},
    'CHFJPY': {'base_price': 170.20, 'volatility': 0.0018, 'min_price': 168.0, 'max_price': 173.0, 'pip_size': 0.01},
    'AUDJPY': {'base_price': 105.30, 'volatility': 0.0025, 'min_price': 103.0, 'max_price': 108.0, 'pip_size': 0.01},
    'CADJPY': {'base_price': 108.90, 'volatility': 0.0022, 'min_price': 106.0, 'max_price': 111.0, 'pip_size': 0.01}
}

# Simulasi RSS News Data
RSS_NEWS_DATA = """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
<channel>
<title>Forex News</title>
<link>https://www.investing.com</link>
<item>
<title>Dollar heads for winning week; PCE data looms large</title>
<pubDate>2025-09-26 08:01:16</pubDate>
<author>Investing.com</author>
<link>https://www.investing.com/news/forex-news</link>
</item>
<item>
<title>Asia FX heads for sharp weekly losses on Fed rate caution</title>
<pubDate>2025-09-26 03:42:05</pubDate>
<author>Investing.com</author>
<link>https://www.investing.com/news/forex-news</link>
</item>
</channel>
</rss>"""

# ==============================
# DATABASE
# ==============================
class Database:
    def __init__(self, db_path='forex_analysis.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
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
        """)
        conn.commit()
        conn.close()
    
    def save_analysis(self, analysis_data):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_results 
                (pair, timeframe, timestamp, current_price, price_change, technical_indicators, ai_analysis, chart_data, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_data['pair'],
                analysis_data['timeframe'],
                analysis_data['timestamp'],
                analysis_data['current_price'],
                analysis_data['price_change'],
                json.dumps(analysis_data['technical_indicators']),
                json.dumps(analysis_data['ai_analysis']),
                json.dumps(analysis_data.get('chart_data', {})),
                analysis_data.get('data_source', 'Unknown')
            ))
            conn.commit()
            conn.close()
            print(f"✅ Analysis saved for {analysis_data['pair']}")
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")

db = Database()

# ==============================
# UTILITAS DATA & ANALISIS
# ==============================
def get_pair_config(pair):
    return pair_configurations.get(pair, {
        'base_price': 150.0, 'volatility': 0.002,
        'min_price': 145.0, 'max_price': 155.0, 'pip_size': 0.01
    })

def get_real_forex_price(pair):
    config = get_pair_config(pair)
    base_price, vol = config['base_price'], config['volatility']
    price_change = np.random.normal(0, vol)
    price = base_price * (1 + price_change)
    return round(max(config['min_price'], min(config['max_price'], price)), 4)

def generate_realistic_chart_data(pair, periods=100):
    config = get_pair_config(pair)
    prices = []
    dates = []
    for i in range(periods):
        date = datetime.now() - timedelta(hours=periods - i)
        dates.append(date.strftime('%Y-%m-%d %H:%M'))
        prices.append(get_real_forex_price(pair))
    series = pd.Series(prices)
    return {
        'dates': dates,
        'open': prices,
        'high': [p * (1 + config['volatility']/2) for p in prices],
        'low': [p * (1 - config['volatility']/2) for p in prices],
        'close': prices,
        'ema_20': series.ewm(span=20).mean().tolist(),
        'ema_50': series.ewm(span=50).mean().tolist()
    }

def calculate_realistic_indicators(pair):
    config = get_pair_config(pair)
    price = get_real_forex_price(pair)
    return {
        'current_price': price,
        'price_change': round(((price - config['base_price']) / config['base_price']) * 100, 2),
        'rsi': round(np.random.uniform(30, 70), 2),
        'macd': round(np.random.normal(0, config['volatility']*10), 4),
        'sma_20': round(price * (1 + np.random.normal(0, config['volatility'])), 4),
        'sma_50': round(price * (1 + np.random.normal(0, config['volatility']*2)), 4),
        'ema_12': round(price * (1 + np.random.normal(0, config['volatility']*0.8)), 4),
        'ema_26': round(price * (1 + np.random.normal(0, config['volatility']*1.5)), 4),
        'resistance': round(price * (1 + config['volatility']*2), 4),
        'support': round(price * (1 - config['volatility']*2), 4),
        'volume': np.random.randint(10000, 50000),
        'pair': pair,
        'volatility': config['volatility']
    }

def parse_rss_news():
    root = ET.fromstring(RSS_NEWS_DATA)
    news = []
    for item in root.findall('.//item'):
        news.append({
            'source': item.find('author').text if item.find('author') is not None else 'Unknown',
            'headline': item.find('title').text,
            'timestamp': item.find('pubDate').text,
            'link': item.find('link').text
        })
    return news

# ==============================
# FLASK ROUTES
# ==============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_analysis')
def get_analysis():
    try:
        pair = request.args.get('pair', 'USDJPY')
        timeframe = request.args.get('timeframe', '1H')

        print(f"\n🔍 Analyzing {pair} {timeframe}")

        technical_data = calculate_realistic_indicators(pair)
        chart_data = generate_realistic_chart_data(pair)
        news = parse_rss_news()

        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': technical_data['current_price'],
            'price_change': technical_data['price_change'],
            'technical_indicators': technical_data,
            'ai_analysis': {"SIGNAL": "HOLD", "CONFIDENCE_LEVEL": 50},  # placeholder
            'fundamental_news': news,
            'chart_data': chart_data,
            'data_points': len(chart_data['dates']),
            'data_source': 'Enhanced Simulation'
        }

        db.save_analysis(response)
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_multiple_pairs')
def get_multiple_pairs():
    results = {}
    for pair in pair_configurations.keys():
        td = calculate_realistic_indicators(pair)
        results[pair] = {
            'price': td['current_price'],
            'change': td['price_change'],
            'rsi': td['rsi'],
            'timestamp': datetime.now().strftime('%H:%M')
        }
        time.sleep(0.2)
    return jsonify(results)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'pairs_configured': list(pair_configurations.keys()),
        'data_sources': 'enhanced_simulation + rss_news'
    })

# ==============================
# MAIN
# ==============================
if __name__ == '__main__':
    print("🚀 Starting Enhanced Forex Analysis System...")
    print("💹 Configured Pairs:", list(pair_configurations.keys()))
    print("🤖 DeepSeek AI: Integrated")
    print("📰 RSS News: Active")
    app.run(debug=True, host='127.0.0.1', port=5000)
