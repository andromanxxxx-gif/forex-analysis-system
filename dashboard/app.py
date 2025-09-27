from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

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

@app.route('/get_data')
def get_data():
    pair = request.args.get('pair', 'GBPJPY')
    
    try:
        # Download data
        data = yf.download(pair_mapping[pair], period='1d', interval='1h', auto_adjust=True)
        
        if data.empty:
            return jsonify({'error': 'No data retrieved'})
        
        # Simple data processing
        data = data.reset_index()
        latest_close = data['Close'].iloc[-1]
        
        # Generate simple signal based on price movement
        price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
        
        if abs(price_change) > 0.1:
            signal = "BUY" if price_change > 0 else "SELL"
            confidence = min(80, abs(price_change) * 100)
        else:
            signal = "HOLD"
            confidence = 50.0
        
        # Prepare chart data (simplified)
        chart_data = {
            'prices': data['Close'].tolist(),
            'timestamps': data['Datetime'].dt.strftime('%H:%M').tolist() if 'Datetime' in data.columns else []
        }
        
        return jsonify({
            'signal': signal,
            'confidence': round(confidence, 1),
            'current_price': round(latest_close, 4),
            'chart_data': chart_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
