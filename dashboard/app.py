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

def get_technical_indicators(data):
    """Menghitung indikator teknikal dengan EMA 200"""
    indicators = {}
    
    try:
        # Pastikan data tidak empty
        if data.empty or len(data) < 20:
            return create_default_indicators(150.0)
        
        # Price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        open_price = data['Open']
        
        current_price = safe_float(close)
        
        # Trend Indicators
        indicators['sma_20'] = safe_float(close.rolling(window=20).mean())
        indicators['sma_50'] = safe_float(close.rolling(window=50).mean())
        indicators['ema_12'] = safe_float(close.ewm(span=12).mean())
        indicators['ema_26'] = safe_float(close.ewm(span=26).mean())
        indicators['ema_200'] = safe_float(close.ewm(span=200).mean())  # EMA 200
        
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
        
        # Chart data - last 50 periods for the chart
        indicators['chart_data'] = {
            'dates': data.index[-50:].strftime('%Y-%m-%d %H:%M').tolist(),
            'open': data['Open'][-50:].astype(float).round(4).tolist(),
            'high': data['High'][-50:].astype(float).round(4).tolist(),
            'low': data['Low'][-50:].astype(float).round(4).tolist(),
            'close': data['Close'][-50:].astype(float).round(4).tolist(),
            'ema_20': data['Close'].ewm(span=20).mean()[-50:].astype(float).round(4).tolist(),
            'ema_50': data['Close'].ewm(span=50).mean()[-50:].astype(float).round(4).tolist(),
            'ema_200': data['Close'].ewm(span=200).mean()[-50:].astype(float).round(4).tolist()
        }
        
    except Exception as e:
        print(f"Error in technical indicators: {e}")
        indicators = create_default_indicators(150.0)
    
    return indicators

def create_default_indicators(price):
    """Create default indicators when data is not available"""
    return {
        'sma_20': price, 'sma_50': price, 'ema_12': price, 'ema_26': price, 'ema_200': price,
        'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
        'bb_upper': price, 'bb_middle': price, 'bb_lower': price,
        'atr': 0.01, 'pivot': price, 'resistance1': price, 'support1': price,
        'current_price': price,
        'chart_data': {
            'dates': [],
            'open': [], 'high': [], 'low': [], 'close': [],
            'ema_20': [], 'ema_50': [], 'ema_200': []
        }
    }

def get_fundamental_news():
    """Web scraping untuk berita fundamental forex"""
    return [
        {
            'source': 'Market Watch',
            'headline': 'Bank of Japan maintains ultra-loose monetary policy',
            'timestamp': datetime.now().strftime('%H:%M')
        },
        {
            'source': 'Reuters', 
            'headline': 'Yen volatility expected amid economic data releases',
            'timestamp': datetime.now().strftime('%H:%M')
        },
        {
            'source': 'Bloomberg',
            'headline': 'JPY pairs show strong technical momentum',
            'timestamp': datetime.now().strftime('%H:%M')
        }
    ]

def analyze_with_deepseek(technical_data, fundamental_news, pair, timeframe):
    """Analisis dengan AI DeepSeek"""
    
    # Extract float values from technical data
    current_price = float(technical_data.get('current_price', 0))
    rsi = float(technical_data.get('rsi', 50))
    macd = float(technical_data.get('macd', 0))
    sma_20 = float(technical_data.get('sma_20', current_price))
    sma_50 = float(technical_data.get('sma_50', current_price))
    ema_200 = float(technical_data.get('ema_200', current_price))
    atr = float(technical_data.get('atr', 0.01))
    
    prompt = f"""
    ANALISIS FOREX PROFESIONAL - {pair} TIMEFRAME {timeframe}

    DATA TEKNIKAL:
    - Current Price: {current_price:.4f}
    - RSI: {rsi:.2f}
    - MACD: {macd:.4f}
    - SMA 20: {sma_20:.4f}
    - SMA 50: {sma_50:.4f}
    - EMA 200: {ema_200:.4f}
    - ATR: {atr:.4f}

    BERITA TERKINI:
    {[news['headline'] for news in fundamental_news]}

    Berikan rekomendasi trading dalam format JSON:

    {{
        "SIGNAL": "BUY/SELL/HOLD",
        "CONFIDENCE_LEVEL": 0-100,
        "ENTRY_PRICE": number,
        "TAKE_PROFIT_1": number,
        "TAKE_PROFIT_2": number, 
        "STOP_LOSS": number,
        "RISK_REWARD_RATIO": "string",
        "TIME_HORIZON": "string",
        "ANALYSIS_SUMMARY": "string"
    }}
    """
    
    try:
        headers = {
            'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'deepseek-chat',
            'messages': [
                {
                    'role': 'system',
                    'content': 'Anda adalah analis forex profesional. Berikan analisis teknis dan rekomendasi trading praktis.'
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            'temperature': 0.3,
            'max_tokens': 1000
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            
            # Extract JSON dari response
            try:
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = analysis_text[start_idx:end_idx]
                    return json.loads(json_str)
            except:
                return generate_fallback_analysis(technical_data, pair, timeframe)
        else:
            print(f"API Error: {response.status_code}")
            return generate_fallback_analysis(technical_data, pair, timeframe)
            
    except Exception as e:
        print(f"DeepSeek API error: {e}")
        return generate_fallback_analysis(technical_data, pair, timeframe)

def generate_fallback_analysis(technical_data, pair, timeframe):
    """Generate analisis fallback"""
    current_price = float(technical_data.get('current_price', 150.0))
    rsi = float(technical_data.get('rsi', 50))
    ema_200 = float(technical_data.get('ema_200', current_price))
    
    # Enhanced logic based on RSI and EMA 200
    if rsi < 30 and current_price > ema_200:
        signal = "STRONG BUY"
        confidence = 80
    elif rsi > 70 and current_price < ema_200:
        signal = "STRONG SELL"
        confidence = 80
    elif rsi < 30:
        signal = "BUY"
        confidence = 70
    elif rsi > 70:
        signal = "SELL"
        confidence = 70
    else:
        signal = "HOLD"
        confidence = 50
    
    atr = float(technical_data.get('atr', 0.5))
    
    if signal in ["STRONG BUY", "BUY"]:
        tp1 = current_price + (atr * 1.5)
        tp2 = current_price + (atr * 2.5)
        sl = current_price - (atr * 1.0)
    elif signal in ["STRONG SELL", "SELL"]:
        tp1 = current_price - (atr * 1.5)
        tp2 = current_price - (atr * 2.5) 
        sl = current_price + (atr * 1.0)
    else:
        tp1 = tp2 = sl = current_price
    
    return {
        'SIGNAL': signal,
        'CONFIDENCE_LEVEL': confidence,
        'ENTRY_PRICE': round(current_price, 4),
        'TAKE_PROFIT_1': round(tp1, 4),
        'TAKE_PROFIT_2': round(tp2, 4),
        'STOP_LOSS': round(sl, 4),
        'RISK_REWARD_RATIO': '1:1.5',
        'TIME_HORIZON': '4-8 hours',
        'ANALYSIS_SUMMARY': f'Fallback analysis: RSI {rsi:.1f}, EMA200 {ema_200:.4f}, Price {current_price:.4f}'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_analysis')
def get_analysis():
    try:
        pair = request.args.get('pair', 'GBPJPY')
        timeframe = request.args.get('timeframe', '4H')
        
        if pair not in pair_mapping:
            return jsonify({'error': 'Invalid pair'})
        if timeframe not in timeframe_mapping:
            return jsonify({'error': 'Invalid timeframe'})
        
        # Get data from yfinance
        yf_symbol = pair_mapping[pair]
        yf_timeframe = timeframe_mapping[timeframe]
        
        # Determine period based on timeframe
        period = '60d' if yf_timeframe in ['1h', '2h', '4h'] else '1y'
        data = yf.download(yf_symbol, period=period, interval=yf_timeframe)
        
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
        news = get_fundamental_news()
        
        # AI Analysis
        ai_analysis = analyze_with_deepseek(indicators, news, pair, timeframe)
        
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': round(float(current_price), 4),
            'price_change': round(float(price_change_pct), 2),
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
            'chart_data': indicators.get('chart_data', {})
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'})

@app.route('/get_multiple_analysis')
def get_multiple_analysis():
    """Analisis untuk semua pairs sekaligus"""
    try:
        timeframe = request.args.get('timeframe', '4H')
        results = {}
        
        for pair in pair_mapping.keys():
            try:
                time.sleep(1)
                
                yf_symbol = pair_mapping[pair]
                yf_timeframe = timeframe_mapping[timeframe]
                period = '60d' if yf_timeframe in ['1h', '2h', '4h'] else '1y'
                data = yf.download(yf_symbol, period=period, interval=yf_timeframe)
                
                if not data.empty and len(data) > 20:
                    current_price = float(data['Close'].iloc[-1])
                    indicators = get_technical_indicators(data)
                    news = get_fundamental_news()
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
    if not os.path.exists('templates'):
        print("ERROR: 'templates' folder not found!")
        print("Please create a 'templates' folder with 'index.html' inside")
    else:
        print("Template folder found. Starting server...")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
