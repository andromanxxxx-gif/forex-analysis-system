from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Forex pairs mapping
pair_mapping = {
    'EURUSD': 'EURUSD=X',
    'GBPJPY': 'GBPJPY=X', 
    'USDJPY': 'USDJPY=X',
    'GBPUSD': 'GBPUSD=X'
}

@app.route('/')
def index():
    pair = request.args.get('pair', 'EURUSD')
    return render_template('index.html', pair=pair)

@app.route('/get_data')
def get_data():
    try:
        pair = request.args.get('pair', 'EURUSD')
        
        # Get data from yfinance
        data = yf.download(pair_mapping[pair], period='1d', interval='1h', auto_adjust=True)
        
        if data.empty:
            return jsonify({'error': 'No data available'})
        
        # Simple analysis
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].pct_change().iloc[-1] * 100
        
        # Generate signal
        if price_change > 0.1:
            signal = "BUY"
            confidence = min(80, 60 + abs(price_change) * 10)
        elif price_change < -0.1:
            signal = "SELL"
            confidence = min(80, 60 + abs(price_change) * 10)
        else:
            signal = "HOLD"
            confidence = 50.0
        
        # Calculate TP/SL
        volatility = data['Close'].std()
        take_profit = round(current_price + volatility * 0.02, 4)
        stop_loss = round(current_price - volatility * 0.01, 4)
        
        return jsonify({
            'pair': pair,
            'current_price': round(current_price, 4),
            'signal': signal,
            'confidence': confidence,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'risk_reward': '1:2',
            'price_change': round(price_change, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
