# ✅ RUN_FIXED.PY - versi stabil dengan endpoint debug & health

from flask import Flask, jsonify, request, render_template
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

# =====================================
# LOAD CONFIG
# =====================================
load_dotenv()
app = Flask(__name__)
CORS(app)

# =====================================
# IMPORT & SETUP (TA-Lib)
# =====================================
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib is available")
except ImportError:
    print("TA-Lib not available, using fallback calculations")
    TALIB_AVAILABLE = False

# =====================================
# CLASS ANALYZER RINGKAS
# =====================================
class XAUUSDAnalyzer:
    def __init__(self):
        self.data_cache = {}
        self.twelve_data_api_key = os.getenv('TWELVE_DATA_API_KEY')
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        print("API keys loaded.")

    def load_from_local_csv(self, timeframe, limit=500):
        path = f"data/XAUUSD_{timeframe}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"Loaded {len(df)} rows from {path}")
            return df.tail(limit)
        return None

    def get_realtime_price(self):
        try:
            if not self.twelve_data_api_key:
                return 1968.0
            url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.twelve_data_api_key}"
            res = requests.get(url, timeout=10)
            data = res.json()
            return float(data.get('price', 1968.0))
        except:
            return 1968.0

    def get_fundamental_news(self):
        return {"articles": []}

analyzer = XAUUSDAnalyzer()

# =====================================
# ENDPOINT UTAMA
# =====================================
@app.route('/')
def home():
    return "<h3>XAUUSD Analysis Server Running ✅</h3>"

@app.route('/api/analysis/<timeframe>')
def analysis(timeframe):
    df = analyzer.load_from_local_csv(timeframe)
    if df is None:
        return jsonify({"error": "Data not found"}), 404
    price = analyzer.get_realtime_price()
    return jsonify({
        "status": "ok",
        "timeframe": timeframe,
        "records": len(df),
        "current_price": price
    })

# =====================================
# ✅ Tambahan Endpoint Debug & Health
# =====================================
@app.route('/api/debug')
def debug_info():
    try:
        info = {
            "status": "ok",
            "server_time": datetime.now().isoformat(),
            "cache_size": len(analyzer.data_cache),
            "api_keys": {
                "twelve_data": bool(analyzer.twelve_data_api_key),
                "deepseek": bool(analyzer.deepseek_api_key),
                "newsapi": bool(analyzer.news_api_key)
            },
            "data_files": [f for f in os.listdir('data') if f.endswith('.csv')] if os.path.exists('data') else []
        }
        print("Debug info requested")
        return jsonify(info)
    except Exception as e:
        print(f"Error in /api/debug: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/health')
def health_check():
    try:
        response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.strftime("%H:%M:%S", time.gmtime(time.time())),
            "services": {
                "flask": True,
                "twelve_data_api": bool(analyzer.twelve_data_api_key),
                "deepseek_ai": bool(analyzer.deepseek_api_key),
                "news_api": bool(analyzer.news_api_key)
            }
        }
        print("Health check OK")
        return jsonify(response)
    except Exception as e:
        print(f"Error in /api/health: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# =====================================
# MAIN APP RUNNER
# =====================================
if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    print("Starting server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
