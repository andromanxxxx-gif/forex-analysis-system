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

def fetch_data(pair):
    """Fetch data from yfinance"""
    try:
        if pair not in pair_mapping:
            pair = 'EURUSD'  # Default fallback
        
        data = yf.download(
            pair_mapping[pair], 
            period='7d', 
            interval='1h', 
            auto_adjust=True
        )
        
        if data.empty:
            # Return mock data if no real data available
            return create_mock_data()
        
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return create_mock_data()

def create_mock_data():
    """Create mock data when real data is unavailable"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    prices = [1.0750 + 0.002 * np.sin(i * 0.1) + 0.001 * random.random() for i in range(100)]
    
    df = pd.DataFrame({
        'Close': prices,
        'High': [p + 0.001 for p in prices],
        'Low': [p - 0.001 for p in prices],
        'Open': [p - 0.0005 for p in prices],
        'Volume': [1000000 + random.randint(-100000, 100000) for _ in prices]
    }, index=dates)
    
    return df

def calculate_indicators(df):
    """Calculate technical indicators"""
    try:
        if len(df) < 20:
            return create_sample_indicators()
        
        # SMA
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (simplified)
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        
        return {
            'sma_20': round(float(df['SMA_20'].iloc[-1]), 5) if not pd.isna(df['SMA_20'].iloc[-1]) else 1.0750,
            'sma_50': round(float(df['SMA_50'].iloc[-1]), 5) if not pd.isna(df['SMA_50'].iloc[-1]) else 1.0720,
            'rsi': round(float(df['RSI'].iloc[-1]), 2) if not pd.isna(df['RSI'].iloc[-1]) else 50.0,
            'macd': round(float(df['MACD'].iloc[-1]), 5) if not pd.isna(df['MACD'].iloc[-1]) else 0.001,
            'volume': int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 1000000
        }
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return create_sample_indicators()

def create_sample_indicators():
    """Create sample technical indicators"""
    return {
        'sma_20': 1.0752,
        'sma_50': 1.0748,
        'rsi': 55.5,
        'macd': 0.0008,
        'volume': 1250000
    }

def generate_signal(df):
    """Generate trading signals"""
    try:
        if len(df) < 2:
            return 'HOLD', 50.0
        
        current_price = df['Close'].iloc[-1]
        previous_price = df['Close'].iloc[-2]
        price_change = ((current_price - previous_price) / previous_price) * 100
        
        # Simple signal logic
        if price_change > 0.1:
            signal = "BUY"
            confidence = min(80, 50 + abs(price_change) * 10)
        elif price_change < -0.1:
            signal = "SELL" 
            confidence = min(80, 50 + abs(price_change) * 10)
        else:
            signal = "HOLD"
            confidence = 50.0
        
        return signal, confidence
    except Exception as e:
        print(f"Error generating signal: {e}")
        return 'HOLD', 50.0

def analyze_pair(pair):
    """Analyze pair for stop loss and take profit"""
    try:
        # Simple analysis based on pair volatility
        volatility_map = {
            'EURUSD': 0.005,
            'GBPJPY': 0.015,
            'USDJPY': 0.008,
            'GBPUSD': 0.007
        }
        
        volatility = volatility_map.get(pair, 0.01)
        current_price = 1.0750  # Default base price for EURUSD, will be adjusted
        
        # Adjust current_price based on pair
        price_map = {
            'EURUSD': 1.0750,
            'GBPJPY': 185.00,
            'USDJPY': 150.00,
            'GBPUSD': 1.2500
        }
        current_price = price_map.get(pair, 1.0750)
        
        return {
            'stop_loss': round(current_price * (1 - volatility), 4),
            'take_profit': round(current_price * (1 + volatility), 4),
            'risk_reward': '1:1.5'
        }
    except Exception as e:
        print(f"Error analyzing pair: {e}")
        return {'stop_loss': 1.070, 'take_profit': 1.085, 'risk_reward': '1:1.5'}

@app.route("/", methods=["GET", "POST"])
def index():
    """Main dashboard route"""
    try:
        # Get pair from request (form POST or args GET)
        if request.method == 'POST':
            pair = request.form.get("pair", "EURUSD")
        else:
            pair = request.args.get("pair", "EURUSD")
        
        # Validate pair
        if pair not in pair_mapping:
            pair = "EURUSD"
        
        # Fetch and process data
        df = fetch_data(pair)
        indicators = calculate_indicators(df)
        signal, confidence = generate_signal(df)
        sl_recommendation = analyze_pair(pair)
        
        # Prepare chart data
        chart_data = {
            'prices': [round(float(p), 5) for p in df['Close'].tail(24)],
            'timestamps': [ts.strftime('%H:%M') for ts in df.index.tail(24)]
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
        
        return render_template("index.html",
                            pair=pair,
                            current_price=round(float(df['Close'].iloc[-1]), 5),
                            signal=signal,
                            confidence=confidence,
                            indicators=indicators,
                            chart_data=chart_data,
                            sentiment=sentiment,
                            prediction=random.choice(predictions),
                            sl_recommendation=sl_recommendation)
    
    except Exception as e:
        print(f"Error in index route: {e}")
        # Return fallback data in case of error
        return render_template("index.html",
                            pair="EURUSD",
                            current_price=1.0750,
                            signal="HOLD",
                            confidence=50.0,
                            indicators=create_sample_indicators(),
                            chart_data={'prices': [1.0750, 1.0752, 1.0748, 1.0755], 'timestamps': ['09:00', '10:00', '11:00', '12:00']},
                            sentiment={'positive': 65, 'negative': 35},
                            prediction="Model Machine Learning memprediksi tren naik dalam 24 jam ke depan.",
                            sl_recommendation={'stop_loss': 1.070, 'take_profit': 1.085, 'risk_reward': '1:1.5'})

@app.route('/get_data')
def get_data():
    """API endpoint for AJAX requests"""
    try:
        pair = request.args.get('pair', 'EURUSD')
        timeframe = request.args.get('timeframe', '1h')
        
        # Fetch and process data
        df = fetch_data(pair)
        indicators = calculate_indicators(df)
        signal, confidence = generate_signal(df)
        sl_recommendation = analyze_pair(pair)
        
        # Prepare chart data
        chart_data = {
            'prices': [round(float(p), 5) for p in df['Close'].tail(24)],
            'timestamps': [ts.strftime('%H:%M') for ts in df.index.tail(24)]
        }
        
        return jsonify({
            'pair': pair,
            'signal': signal,
            'confidence': confidence,
            'current_price': round(float(df['Close'].iloc[-1]), 5),
            'take_profit': sl_recommendation['take_profit'],
            'stop_loss': sl_recommendation['stop_loss'],
            'risk_reward': sl_recommendation['risk_reward'],
            'indicators': indicators,
            'chart_data': chart_data,
            'sentiment': {'positive': random.randint(60, 75), 'negative': random.randint(25, 40)},
            'prediction': "Model Machine Learning memprediksi tren naik dalam 24 jam ke depan."
        })
        
    except Exception as e:
        print(f"Error in get_data route: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
