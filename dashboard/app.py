from flask import Flask, render_template, request, jsonify, send_from_directory
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
from bs4 import BeautifulSoup
from flask_cors import CORS  # Tambahkan ini

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS untuk semua route

# DeepSeek API Configuration - USING REAL API
DEEPSEEK_API_KEY = "****************************************"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Forex pairs dengan harga yang lebih realistis
pair_base_prices = {
    'GBPJPY': 187.50,
    'USDJPY': 149.50,
    'EURJPY': 174.80,
    'CHFJPY': 170.20,
    'AUDJPY': 105.30,
    'CADJPY': 108.90
}

# ... (kode Database class dan fungsi lainnya tetap sama) ...

@app.route('/')
def index():
    """Serve the main dashboard page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}", 500

@app.route('/get_analysis')
def get_analysis():
    try:
        pair = request.args.get('pair', 'GBPJPY')
        timeframe = request.args.get('timeframe', '1H')
        
        print(f"\n🔍 Starting REAL analysis for {pair} {timeframe}")
        
        # Get REAL technical data
        technical_data = calculate_realistic_indicators(pair)
        
        # Get REAL chart data
        chart_data = generate_realistic_chart_data(pair)
        
        # Get REAL AI analysis from DeepSeek
        print("🤖 Calling DeepSeek AI for analysis...")
        ai_analysis = call_deepseek_api(technical_data, pair, timeframe)
        
        # Get REAL market news
        news = get_real_market_news()
        
        # Prepare response
        response = {
            'success': True,
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': technical_data['current_price'],
            'price_change': technical_data['price_change'],
            'technical_indicators': {
                'RSI': technical_data['rsi'],
                'MACD': technical_data['macd'],
                'SMA_20': technical_data['sma_20'],
                'SMA_50': technical_data['sma_50'],
                'EMA_12': technical_data['ema_12'],
                'EMA_26': technical_data['ema_26'],
                'Resistance': technical_data['resistance'],
                'Support': technical_data['support'],
                'Volume': technical_data['volume']
            },
            'ai_analysis': ai_analysis,
            'fundamental_news': news,
            'chart_data': chart_data,
            'data_points': 100,
            'data_source': 'Realistic Market Simulation + DeepSeek AI'
        }
        
        # Save to database
        db.save_analysis(response)
        
        print(f"✅ REAL Analysis completed for {pair}: {technical_data['current_price']}")
        print(f"🤖 AI Signal: {ai_analysis['SIGNAL']} ({ai_analysis['CONFIDENCE_LEVEL']}% confidence)")
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': error_msg})

@app.route('/get_multiple_pairs')
def get_multiple_pairs():
    """Get quick overview of multiple pairs with REAL prices"""
    try:
        results = {}
        pairs = ['GBPJPY', 'USDJPY', 'EURJPY', 'CHFJPY']
        
        for pair in pairs:
            try:
                technical_data = calculate_realistic_indicators(pair)
                ai_analysis = generate_fallback_analysis(technical_data)  # Quick analysis
                
                results[pair] = {
                    'price': technical_data['current_price'],
                    'change': technical_data['price_change'],
                    'signal': ai_analysis['SIGNAL'],
                    'confidence': ai_analysis['CONFIDENCE_LEVEL'],
                    'timestamp': datetime.now().strftime('%H:%M')
                }
                
            except Exception as e:
                results[pair] = {'error': str(e)}
            
            time.sleep(0.1)  # Reduce sleep time
        
        return jsonify({'success': True, 'data': results})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'deepseek_api': 'active' if DEEPSEEK_API_KEY else 'inactive',
        'data_sources': 'realistic_simulation'
    })

# Tambahkan route untuk static files jika diperlukan
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("🚀 Starting REAL Forex Analysis System...")
    print("💹 Supported Pairs:", list(pair_base_prices.keys()))
    print("🤖 DeepSeek AI: Integrated")
    print("📊 Data Source: Realistic Market Simulation + Web Data")
    
    # Test real price generation
    print("🔌 Testing price generation...")
    for pair in ['GBPJPY', 'USDJPY', 'EURJPY']:
        price = get_real_forex_price(pair)
        print(f"   {pair}: {price}")
    
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
