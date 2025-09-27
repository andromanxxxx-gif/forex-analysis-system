from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import random

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Forex pairs mapping
pair_mapping = {
    'EURUSD': 'EURUSD=X',
    'GBPJPY': 'GBPJPY=X',
    'USDJPY': 'USDJPY=X',
    'GBPUSD': 'GBPUSD=X'
}

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    if len(data) < 20:
        return {}
    
    # SMA
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (simplified)
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    data['MACD'] = exp1 - exp2
    
    return {
        'sma_20': round(data['SMA_20'].iloc[-1], 5) if not pd.isna(data['SMA_20'].iloc[-1]) else None,
        'sma_50': round(data['SMA_50'].iloc[-1], 5) if not pd.isna(data['SMA_50'].iloc[-1]) else None,
        'rsi': round(data['RSI'].iloc[-1], 2) if not pd.isna(data['RSI'].iloc[-1]) else None,
        'macd': round(data['MACD'].iloc[-1], 5) if not pd.isna(data['MACD'].iloc[-1]) else None,
        'volume': int(data['Volume'].iloc[-1]) if 'Volume' in data.columns else None
    }

def generate_trading_signals(data, indicators):
    """Generate trading signals with TP/SL"""
    if len(data) < 2:
        return 'HOLD', 50.0, None, None, '1:1'
    
    current_price = data['Close'].iloc[-1]
    previous_price = data['Close'].iloc[-2]
    price_change = ((current_price - previous_price) / previous_price) * 100
    
    # Simple signal logic based on price movement and RSI
    rsi = indicators.get('rsi', 50)
    
    if rsi < 30 and price_change > 0:
        signal = "BUY"
        confidence = min(90, (30 - rsi) * 2 + 50)
    elif rsi > 70 and price_change < 0:
        signal = "SELL"
        confidence = min(90, (rsi - 70) * 2 + 50)
    else:
        signal = "HOLD"
        confidence = 50.0
    
    # Calculate TP/SL based on volatility
    volatility = data['Close'].pct_change().std() * 100
    
    if signal == "BUY":
        take_profit = round(current_price * (1 + volatility * 0.02), 5)
        stop_loss = round(current_price * (1 - volatility * 0.01), 5)
    elif signal == "SELL":
        take_profit = round(current_price * (1 - volatility * 0.02), 5)
        stop_loss = round(current_price * (1 + volatility * 0.01), 5)
    else:
        take_profit = stop_loss = None
    
    risk_reward = "1:2" if take_profit and stop_loss else "1:1"
    
    return signal, confidence, take_profit, stop_loss, risk_reward

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    pair = request.args.get('pair', 'EURUSD')
    timeframe = request.args.get('timeframe', '1h')
    
    try:
        # Map timeframe to yfinance period
        period_map = {'1h': '7d', '2h': '15d', '4h': '30d', '1d': '3mo'}
        period = period_map.get(timeframe, '7d')
        
        # Download data
        data = yf.download(
            pair_mapping[pair], 
            period=period, 
            interval=timeframe, 
            auto_adjust=True
        )
        
        if data.empty:
            return jsonify({'error': 'No data retrieved'})
        
        # Process data
        data = data.reset_index()
        latest_close = data['Close'].iloc[-1]
        
        # Calculate technical indicators
        indicators = calculate_technical_indicators(data)
        
        # Generate trading signals
        signal, confidence, tp, sl, rr = generate_trading_signals(data, indicators)
        
        # Prepare chart data
        chart_data = {
            'prices': [round(float(p), 5) for p in data['Close'].tail(24)],
            'timestamps': [ts.strftime('%H:%M') for ts in data['Date'].tail(24)] if 'Date' in data.columns else 
                         [ts.strftime('%H:%M') for ts in data['Datetime'].tail(24)]
        }
        
        # News sentiment (mock data)
        sentiment = {
            'positive': random.randint(60, 75),
            'negative': random.randint(25, 40)
        }
        
        # AI prediction (mock data)
        predictions = [
            "Model Machine Learning memprediksi tren naik dalam 24 jam ke depan.",
            "AI menganalisis momentum bullish untuk pair ini.",
            "Prediksi: Sideways movement dengan bias positif.",
            "Algorithm memperkirakan breakout pattern segera."
        ]
        
        return jsonify({
            'pair': pair,
            'signal': signal,
            'confidence': round(confidence, 1),
            'current_price': round(latest_close, 5),
            'take_profit': tp,
            'stop_loss': sl,
            'risk_reward': rr,
            'indicators': indicators,
            'chart_data': chart_data,
            'sentiment': sentiment,
            'prediction': random.choice(predictions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
