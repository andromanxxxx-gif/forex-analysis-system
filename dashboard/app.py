from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suppress FutureWarnings saja (hapus yf.pdr_override())
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Forex pairs mapping
pair_mapping = {
    'GBPJPY': 'GBPJPY=X',
    'EURUSD': 'EURUSD=X',
    'USDJPY': 'USDJPY=X',
    'GBPUSD': 'GBPUSD=X'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['POST'])
def get_data():
    pair = request.json.get('pair', 'GBPJPY')
    
    try:
        # Download data dengan parameter explicit
        data = yf.download(
            pair_mapping[pair], 
            period='7d', 
            interval='1h', 
            auto_adjust=True  # Explicitly set to avoid warning
        )
        
        if data.empty:
            return jsonify({'error': 'No data retrieved'})
        
        # Process data
        data = data.reset_index()
        data['Timestamp'] = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate simple indicators (replace ta library)
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate RSI manually
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Prepare chart data
        chart_data = {
            'timestamps': data['Timestamp'].tolist(),
            'prices': data['Close'].tolist(),
            'sma_20': data['SMA_20'].tolist(),
            'sma_50': data['SMA_50'].tolist(),
            'rsi': data['RSI'].tolist()
        }
        
        # Generate trading signal
        latest_close = data['Close'].iloc[-1]
        latest_rsi = data['RSI'].iloc[-1] if not pd.isna(data['RSI'].iloc[-1]) else 50
        
        if latest_rsi > 70:
            signal = "SELL"
            confidence = min(90, (latest_rsi - 70) * 3 + 50)
        elif latest_rsi < 30:
            signal = "BUY" 
            confidence = min(90, (30 - latest_rsi) * 3 + 50)
        else:
            signal = "HOLD"
            confidence = 50.0
            
        return jsonify({
            'chart_data': chart_data,
            'signal': signal,
            'confidence': round(confidence, 1),
            'current_price': round(latest_close, 4),
            'rsi': round(latest_rsi, 2) if not pd.isna(latest_rsi) else 50
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
