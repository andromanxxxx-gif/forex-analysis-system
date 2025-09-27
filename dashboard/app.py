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

# Konfigurasi
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
    
    # Price data
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume'] if 'Volume' in data else pd.Series([1] * len(data))
    
    # Trend Indicators
    indicators['sma_20'] = talib.SMA(close, timeperiod=20).iloc[-1]
    indicators['sma_50'] = talib.SMA(close, timeperiod=50).iloc[-1]
    indicators['ema_12'] = talib.EMA(close, timeperiod=12).iloc[-1]
    indicators['ema_26'] = talib.EMA(close, timeperiod=26).iloc[-1]
    
    # Momentum Indicators
    indicators['rsi'] = talib.RSI(close, timeperiod=14).iloc[-1]
    indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
    indicators['macd'] = indicators['macd'].iloc[-1]
    indicators['macd_signal'] = indicators['macd_signal'].iloc[-1]
    indicators['macd_hist'] = indicators['macd_hist'].iloc[-1]
    
    # Volatility Indicators
    indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close, timeperiod=20)
    indicators['bb_upper'] = indicators['bb_upper'].iloc[-1]
    indicators['bb_middle'] = indicators['bb_middle'].iloc[-1]
    indicators['bb_lower'] = indicators['bb_lower'].iloc[-1]
    indicators['atr'] = talib.ATR(high, low, close, timeperiod=14).iloc[-1]
    
    # Support Resistance
    indicators['pivot'] = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
    indicators['resistance1'] = 2 * indicators['pivot'] - low.iloc[-1]
    indicators['support1'] = 2 * indicators['pivot'] - high.iloc[-1]
    
    return indicators

def get_fundamental_news():
    """Web scraping untuk berita fundamental forex"""
    news_items = []
    
    try:
        # Bloomberg Forex News
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Contoh sumber berita (bisa ditambah lebih banyak)
        sources = [
            'https://www.bloomberg.com/markets/currencies',
            'https://www.reuters.com/finance/currencies'
        ]
        
        for source in sources:
            try:
                response = requests.get(source, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Scraping sederhana - sesuaikan dengan struktur website
                headlines = soup.find_all('h1', limit=3) + soup.find_all('h2', limit=3)
                for headline in headlines:
                    text = headline.get_text().strip()
                    if text and len(text) > 20:
                        news_items.append({
                            'source': source.split('//')[-1].split('/')[0],
                            'headline': text,
                            'timestamp': datetime.now().strftime('%H:%M')
                        })
            except:
                continue
                
    except Exception as e:
        print(f"Error fetching news: {e}")
        # Fallback news
        news_items = [
            {'source': 'System', 'headline': 'Market Analysis in Progress', 'timestamp': 'Now'},
            {'source': 'System', 'headline': 'Technical Analysis Completed', 'timestamp': 'Now'}
        ]
    
    return news_items[:5]  # Return max 5 berita

def analyze_with_deepseek(technical_data, fundamental_news, pair, timeframe):
    """Analisis dengan AI DeepSeek"""
    
    prompt = f"""
    Sebagai analis forex profesional, analisalah data berikut untuk pair {pair} pada timeframe {timeframe}:

    DATA TEKNIKAL:
    {json.dumps(technical_data, indent=2)}

    BERITA FUNDAMENTAL TERKINI:
    {json.dumps(fundamental_news, indent=2)}

    Berikan rekomendasi trading yang jelas dengan format:
    1. SIGNAL (BUY/SELL/HOLD)
    2. CONFIDENCE_LEVEL (0-100%)
    3. ENTRY_PRICE (rekomendasi)
    4. TAKE_PROFIT_1, TAKE_PROFIT_2
    5. STOP_LOSS
    6. RISK_REWARD_RATIO
    7. TIME_HORIZON (jam/hari)
    8. ANALYSIS_SUMMARY (penjelasan singkat)

    Jawab dengan format JSON yang valid.
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
                    'content': 'Anda adalah analis forex profesional. Berikan analisis yang akurat dan rekomendasi trading yang praktis.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.3,
            'max_tokens': 1500
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            analysis_text = result['choices'][0]['message']['content']
            
            # Extract JSON dari response
            try:
                # Cari bagian JSON dalam response
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = analysis_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    # Fallback parsing manual
                    analysis = parse_analysis_manual(analysis_text)
            except:
                analysis = parse_analysis_manual(analysis_text)
                
            return analysis
        else:
            return generate_fallback_analysis(technical_data, pair, timeframe)
            
    except Exception as e:
        print(f"DeepSeek API error: {e}")
        return generate_fallback_analysis(technical_data, pair, timeframe)

def parse_analysis_manual(analysis_text):
    """Parsing manual jika JSON extraction gagal"""
    analysis = {
        'SIGNAL': 'HOLD',
        'CONFIDENCE_LEVEL': 50,
        'ENTRY_PRICE': 0,
        'TAKE_PROFIT_1': 0,
        'TAKE_PROFIT_2': 0,
        'STOP_LOSS': 0,
        'RISK_REWARD_RATIO': '1:1',
        'TIME_HORIZON': '4-8 jam',
        'ANALYSIS_SUMMARY': 'Analisis default - periksa koneksi API'
    }
    
    # Simple keyword matching
    if 'BUY' in analysis_text.upper():
        analysis['SIGNAL'] = 'BUY'
        analysis['CONFIDENCE_LEVEL'] = 65
    elif 'SELL' in analysis_text.upper():
        analysis['SIGNAL'] = 'SELL' 
        analysis['CONFIDENCE_LEVEL'] = 65
        
    return analysis

def generate_fallback_analysis(technical_data, pair, timeframe):
    """Generate analisis fallback jika API tidak available"""
    current_price = technical_data['current_price']
    rsi = technical_data['rsi']
    trend = "BULLISH" if technical_data['sma_20'] > technical_data['sma_50'] else "BEARISH"
    
    if rsi < 30 and trend == "BULLISH":
        signal = "BUY"
        confidence = 70
    elif rsi > 70 and trend == "BEARISH":
        signal = "SELL"
        confidence = 70
    else:
        signal = "HOLD"
        confidence = 50
    
    atr = technical_data['atr']
    tp1 = current_price + (atr * 1.5) if signal == "BUY" else current_price - (atr * 1.5)
    tp2 = current_price + (atr * 2.5) if signal == "BUY" else current_price - (atr * 2.5)
    sl = current_price - (atr * 1) if signal == "BUY" else current_price + (atr * 1)
    
    return {
        'SIGNAL': signal,
        'CONFIDENCE_LEVEL': confidence,
        'ENTRY_PRICE': round(current_price, 4),
        'TAKE_PROFIT_1': round(tp1, 4),
        'TAKE_PROFIT_2': round(tp2, 4),
        'STOP_LOSS': round(sl, 4),
        'RISK_REWARD_RATIO': '1:1.5',
        'TIME_HORIZON': '4-8 jam',
        'ANALYSIS_SUMMARY': f'Analisis teknikal: RSI {rsi:.1f}, Trend {trend}'
    }

@app.route('/')
def index():
    return render_template('index.html', pairs=list(pair_mapping.keys()))

@app.route('/get_analysis')
def get_analysis():
    try:
        pair = request.args.get('pair', 'GBPJPY')
        timeframe = request.args.get('timeframe', '4H')
        
        # Validasi input
        if pair not in pair_mapping:
            return jsonify({'error': 'Invalid pair'})
        if timeframe not in timeframe_mapping:
            return jsonify({'error': 'Invalid timeframe'})
        
        # Get data from yfinance
        yf_symbol = pair_mapping[pair]
        yf_timeframe = timeframe_mapping[timeframe]
        
        # Download data dengan periode yang sesuai
        period = '60d' if yf_timeframe in ['1h', '2h', '4h'] else '1y'
        data = yf.download(yf_symbol, period=period, interval=yf_timeframe, auto_adjust=True)
        
        if data.empty or len(data) < 50:
            return jsonify({'error': 'Insufficient data'})
        
        # Current price data
        current_price = data['Close'].iloc[-1]
        price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
        
        # Technical indicators
        indicators = get_technical_indicators(data)
        indicators['current_price'] = current_price
        indicators['price_change'] = price_change
        
        # Fundamental news
        news = get_fundamental_news()
        
        # AI Analysis dengan DeepSeek
        ai_analysis = analyze_with_deepseek(indicators, news, pair, timeframe)
        
        # Prepare response
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': round(current_price, 4),
            'price_change': round(price_change, 2),
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
                # Delay kecil untuk avoid rate limiting
                time.sleep(1)
                
                analysis_response = get_analysis()
                if analysis_response and hasattr(analysis_response, 'get_json'):
                    results[pair] = analysis_response.get_json()
            except Exception as e:
                results[pair] = {'error': str(e)}
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
