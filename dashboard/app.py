from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
import talib
import json
from bs4 import BeautifulSoup
import time

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

def get_technical_indicators(data):
    """Menghitung indikator teknikal"""
    indicators = {}
    
    try:
        # Price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Trend Indicators
        indicators['sma_20'] = close.rolling(window=20).mean().iloc[-1]
        indicators['sma_50'] = close.rolling(window=50).mean().iloc[-1]
        indicators['ema_12'] = close.ewm(span=12).mean().iloc[-1]
        indicators['ema_26'] = close.ewm(span=26).mean().iloc[-1]
        
        # Momentum Indicators - RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        indicators['macd'] = (ema_12 - ema_26).iloc[-1]
        indicators['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
        indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
        
        # Volatility Indicators - Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
        indicators['bb_middle'] = sma_20.iloc[-1]
        indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
        
        # ATR (Average True Range)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(window=14).mean().iloc[-1]
        
        # Support Resistance
        indicators['pivot'] = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
        indicators['resistance1'] = 2 * indicators['pivot'] - low.iloc[-1]
        indicators['support1'] = 2 * indicators['pivot'] - high.iloc[-1]
        
    except Exception as e:
        print(f"Error in technical indicators: {e}")
        # Set default values
        current_price = close.iloc[-1] if 'close' in locals() else 0
        indicators = {
            'sma_20': current_price, 'sma_50': current_price,
            'ema_12': current_price, 'ema_26': current_price,
            'rsi': 50, 'macd': 0, 'macd_signal': 0, 'macd_hist': 0,
            'bb_upper': current_price, 'bb_middle': current_price, 'bb_lower': current_price,
            'atr': 0.01, 'pivot': current_price, 'resistance1': current_price, 'support1': current_price
        }
    
    return indicators

def get_fundamental_news():
    """Web scraping untuk berita fundamental forex"""
    news_items = []
    
    try:
        # Contoh berita statis (bisa diganti dengan web scraping real)
        news_items = [
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
                
    except Exception as e:
        print(f"Error fetching news: {e}")
        news_items = [
            {'source': 'System', 'headline': 'Market Analysis in Progress', 'timestamp': 'Now'},
            {'source': 'System', 'headline': 'Technical Analysis Completed', 'timestamp': 'Now'}
        ]
    
    return news_items[:3]

def analyze_with_deepseek(technical_data, fundamental_news, pair, timeframe):
    """Analisis dengan AI DeepSeek"""
    
    prompt = f"""
    ANALISIS FOREX PROFESIONAL - {pair} TIMEFRAME {timeframe}

    DATA TEKNIKAL:
    - Current Price: {technical_data.get('current_price', 0)}
    - RSI: {technical_data.get('rsi', 0):.2f}
    - MACD: {technical_data.get('macd', 0):.4f}
    - Signal: {technical_data.get('macd_signal', 0):.4f}
    - SMA 20: {technical_data.get('sma_20', 0):.4f}
    - SMA 50: {technical_data.get('sma_50', 0):.4f}
    - Support: {technical_data.get('support1', 0):.4f}
    - Resistance: {technical_data.get('resistance1', 0):.4f}
    - ATR: {technical_data.get('atr', 0):.4f}

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
    current_price = technical_data.get('current_price', 0)
    rsi = technical_data.get('rsi', 50)
    
    # Simple logic based on RSI
    if rsi < 30:
        signal = "BUY"
        confidence = 70
    elif rsi > 70:
        signal = "SELL" 
        confidence = 70
    else:
        signal = "HOLD"
        confidence = 50
    
    atr = technical_data.get('atr', 0.01)
    
    if signal == "BUY":
        tp1 = current_price + (atr * 1.5)
        tp2 = current_price + (atr * 2.5)
        sl = current_price - (atr * 1.0)
    elif signal == "SELL":
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
        'ANALYSIS_SUMMARY': f'Fallback analysis: RSI {rsi:.1f}, Price {current_price:.4f}'
    }

@app.route('/')
def index():
    return render_template('index.html', pairs=list(pair_mapping.keys()))

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
        current_price = data['Close'].iloc[-1]
        price_change_pct = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
        
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
            'current_price': round(current_price, 4),
            'price_change': round(price_change_pct, 2),
            'technical_indicators': {
                'RSI': round(indicators['rsi'], 2),
                'MACD': round(indicators['macd'], 4),
                'SMA_20': round(indicators['sma_20'], 4),
                'SMA_50': round(indicators['sma_50'], 4),
                'ATR': round(indicators['atr'], 4),
                'Support': round(indicators['support1'], 4),
                'Resistance': round(indicators['resistance1'], 4)
            },
            'ai_analysis': ai_analysis,
            'fundamental_news': news
        }
        
        return jsonify(response)
        
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
                time.sleep(0.5)  # Small delay to avoid rate limiting
                
                # Simulate API call for each pair
                yf_symbol = pair_mapping[pair]
                yf_timeframe = timeframe_mapping[timeframe]
                period = '60d' if yf_timeframe in ['1h', '2h', '4h'] else '1y'
                data = yf.download(yf_symbol, period=period, interval=yf_timeframe)
                
                if not data.empty and len(data) > 20:
                    current_price = data['Close'].iloc[-1]
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
    app.run(debug=True, host='127.0.0.1', port=5000)
