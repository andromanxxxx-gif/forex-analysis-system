# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import talib
import os

app = Flask(__name__)
CORS(app)

class XAUUSDAnalyzer:
    def __init__(self):
        self.twelve_data_api_key = "your_twelve_data_api_key"
        self.deepseek_api_key = "your_deepseek_api_key"
        self.news_api_key = "your_news_api_key"
        
    def load_historical_data(self, timeframe):
        """Load data historis dari CSV"""
        try:
            filename = f"data/XAUUSD_{timeframe}.csv"
            df = pd.read_csv(filename)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def get_realtime_price(self):
        """Ambil harga realtime dari Twelve Data"""
        try:
            url = f"https://api.twelvedata.com/price?symbol=XAU/USD&apikey={self.twelve_data_api_key}"
            response = requests.get(url)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"Error getting realtime price: {e}")
            return None

    def get_fundamental_news(self):
        """Ambil berita fundamental"""
        try:
            url = f"https://newsapi.org/v2/everything?q=gold+XAU+USD+Federal+Reserve&apiKey={self.news_api_key}"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            print(f"Error getting news: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Hitung indikator teknikal untuk XAUUSD"""
        # Price data
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values if 'volume' in df.columns else None
        
        indicators = {}
        
        # Trend Indicators
        indicators['sma_20'] = talib.SMA(close, timeperiod=20)
        indicators['sma_50'] = talib.SMA(close, timeperiod=50)
        indicators['sma_200'] = talib.SMA(close, timeperiod=200)
        indicators['ema_12'] = talib.EMA(close, timeperiod=12)
        indicators['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # Momentum Indicators
        indicators['rsi'] = talib.RSI(close, timeperiod=14)
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
        indicators['stoch_k'], indicators['stoch_d'] = talib.STOCH(high, low, close)
        
        # Volatility Indicators
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # Support/Resistance
        indicators['pivot'] = (high + low + close) / 3
        
        return indicators

    def analyze_with_deepseek(self, technical_data, news_data):
        """Analisis dengan AI DeepSeek"""
        prompt = f"""
        Sebagai analis profesional XAUUSD, lakukan analisis komprehensif berdasarkan data berikut:
        
        DATA TEKNIKAL:
        {json.dumps(technical_data, indent=2)}
        
        DATA FUNDAMENTAL:
        {json.dumps(news_data, indent=2)}
        
        Berikan analisis dalam format:
        1. TREND: (Bullish/Bearish/Sideways)
        2. SUPPORT: (level support utama)
        3. RESISTANCE: (level resistance utama)  
        4. SIGNAL: (Buy/Sell/Hold)
        5. RISK LEVEL: (High/Medium/Low)
        6. ANALYSIS: (analisis mendetail minimal 200 kata)
        7. KEY LEVELS: (level-level kunci untuk trading)
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Error in AI analysis: {str(e)}"

@app.route('/api/analysis/<timeframe>')
def get_analysis(timeframe):
    analyzer = XAUUSDAnalyzer()
    
    # Load historical data
    df = analyzer.load_historical_data(timeframe)
    if df is None:
        return jsonify({"error": "Data not found"}), 404
    
    # Get latest 600 data points
    df = df.tail(600)
    
    # Calculate technical indicators
    indicators = analyzer.calculate_technical_indicators(df)
    
    # Get realtime price and update last candle
    realtime_price = analyzer.get_realtime_price()
    if realtime_price:
        df.iloc[-1, df.columns.get_loc('close')] = realtime_price
        df.iloc[-1, df.columns.get_loc('high')] = max(df.iloc[-1]['high'], realtime_price)
        df.iloc[-1, df.columns.get_loc('low')] = min(df.iloc[-1]['low'], realtime_price)
    
    # Get fundamental news
    news_data = analyzer.get_fundamental_news()
    
    # Prepare technical data for AI
    latest_indicators = {}
    for key, values in indicators.items():
        if len(values) > 0 and not np.isnan(values[-1]):
            latest_indicators[key] = float(values[-1])
    
    technical_data = {
        "current_price": realtime_price,
        "indicators": latest_indicators,
        "price_action": {
            "open": float(df.iloc[-1]['open']),
            "high": float(df.iloc[-1]['high']), 
            "low": float(df.iloc[-1]['low']),
            "close": float(df.iloc[-1]['close'])
        }
    }
    
    # AI Analysis
    ai_analysis = analyzer.analyze_with_deepseek(technical_data, news_data)
    
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "timeframe": timeframe,
        "current_price": realtime_price,
        "technical_indicators": latest_indicators,
        "ai_analysis": ai_analysis,
        "chart_data": df.tail(100).to_dict('records'),
        "news": news_data
    })

@app.route('/api/chart/data/<timeframe>')
def get_chart_data(timeframe):
    analyzer = XAUUSDAnalyzer()
    df = analyzer.load_historical_data(timeframe)
    
    if timeframe == '1D':
        df = df.tail(730)  # 2 years
    elif timeframe == '4H':
        df = df.tail(2190)  # 1 year
    elif timeframe == '1H': 
        df = df.tail(4320)  # 6 months
    
    return jsonify(df.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
