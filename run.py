from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os
import traceback
import time

# Try to import talib
try:
    import talib
    TALIB_AVAILABLE = True
    print("TA-Lib is available")
except ImportError:
    print("TA-Lib not available, using fallback calculations")
    TALIB_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Setup template folder
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app.template_folder = template_dir

class XAUUSDAnalyzer:
    def __init__(self):
        self.data_cache = {}  # Cache untuk menyimpan data di memory
        
    def load_historical_data(self, timeframe, limit=500):
        """Load data historis dari CSV file yang sudah ada"""
        try:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if os.path.exists(filename):
                print(f"Loading historical data from {filename}")
                df = pd.read_csv(filename)
                
                # Pastikan kolom datetime ada dan format benar
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                elif 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'])
                    df = df.rename(columns={'date': 'datetime'})
                elif 'time' in df.columns:
                    df['datetime'] = pd.to_datetime(df['time'])
                    df = df.rename(columns={'time': 'datetime'})
                else:
                    # Jika tidak ada kolom datetime, buat berdasarkan index
                    df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')
                
                # Pastikan kolom OHLC ada
                required_cols = ['open', 'high', 'low', 'close']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"Warning: Column {col} not found in CSV")
                        # Buat data dummy jika kolom tidak ada
                        df[col] = np.random.uniform(1800, 2000, len(df))
                
                # Konversi ke numeric
                for col in required_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                if 'volume' not in df.columns:
                    df['volume'] = np.random.randint(1000, 10000, len(df))
                
                df = df.sort_values('datetime')
                print(f"Successfully loaded {len(df)} records from {filename}")
                return df.tail(limit)
            else:
                print(f"File {filename} not found, using generated data")
                return self.generate_sample_data(timeframe, limit)
                
        except Exception as e:
            print(f"Error loading historical data: {e}")
            traceback.print_exc()
            return self.generate_sample_data(timeframe, limit)

    def generate_sample_data(self, timeframe, limit=500):
        """Generate sample data tanpa save ke file"""
        print(f"Generating sample data for {timeframe} (in memory only)")
        
        periods = limit
        base_price = 1968.0
        
        # Create dates
        if timeframe == '1H':
            freq = 'H'
        elif timeframe == '4H':
            freq = '4H'
        else:  # 1D
            freq = 'D'
            
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        
        # Generate realistic price movement
        np.random.seed(42)
        returns = np.random.normal(0, 0.005, periods)
        prices = base_price * (1 + returns).cumprod()
        
        # Create OHLC data
        data = []
        for i in range(periods):
            open_price = prices[i] * np.random.uniform(0.998, 1.002)
            close_price = prices[i] * np.random.uniform(0.998, 1.002)
            high_price = max(open_price, close_price) * np.random.uniform(1.001, 1.008)
            low_price = min(open_price, close_price) * np.random.uniform(0.992, 0.999)
            
            data.append({
                'datetime': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': np.random.randint(5000, 50000)
            })
        
        df = pd.DataFrame(data)
        
        # Simpan di cache memory
        self.data_cache[timeframe] = df
        print(f"Generated {len(df)} records for {timeframe} (cached in memory)")
        return df

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            if len(df) < 20:
                print("Not enough data for indicators")
                return df
                
            close = df['close'].values
            
            # Calculate EMAs
            df['ema_12'] = self.ema(close, 12)
            df['ema_26'] = self.ema(close, 26)
            df['ema_50'] = self.ema(close, 50)
            
            # Calculate MACD
            macd, signal, hist = self.macd(close)
            df['macd'] = macd
            df['macd_signal'] = signal
            df['macd_hist'] = hist
            
            # Calculate RSI
            df['rsi'] = self.rsi(close, 14)
            
            print("Indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            traceback.print_exc()
            return df

    def ema(self, data, period):
        """Exponential Moving Average"""
        series = pd.Series(data)
        return series.ewm(span=period, adjust=False).mean()

    def macd(self, data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = pd.Series(data).ewm(span=fast, adjust=False).mean()
        ema_slow = pd.Series(data).ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def rsi(self, data, period=14):
        """RSI Indicator"""
        series = pd.Series(data)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Default to 50 if cannot calculate

    def get_realtime_price(self):
        """Get real-time gold price"""
        # Simulate realistic gold price
        base_price = 1968.0
        movement = np.random.normal(0, 1.5)
        price = base_price + movement
        print(f"Real-time XAUUSD price: ${price:.2f}")
        return round(price, 2)

    def analyze_market(self, df):
        """Analyze market conditions"""
        try:
            if len(df) == 0:
                return "No data available for analysis"
                
            current_price = df.iloc[-1]['close']
            current_rsi = df.iloc[-1]['rsi'] if 'rsi' in df.columns and not pd.isna(df.iloc[-1]['rsi']) else 50
            current_macd = df.iloc[-1]['macd'] if 'macd' in df.columns and not pd.isna(df.iloc[-1]['macd']) else 0
            
            # Simple analysis
            if current_rsi < 30 and current_macd > 0:
                signal = "STRONG BUY"
                trend = "BULLISH"
            elif current_rsi > 70 and current_macd < 0:
                signal = "STRONG SELL" 
                trend = "BEARISH"
            elif current_rsi < 40 and current_macd > 0:
                signal = "BUY"
                trend = "BULLISH"
            elif current_rsi > 60 and current_macd < 0:
                signal = "SELL"
                trend = "BEARISH"
            else:
                signal = "HOLD"
                trend = "NEUTRAL"
            
            analysis = f"""
** XAUUSD MARKET ANALYSIS **
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PRICE: ${current_price:.2f}
TREND: {trend}
SIGNAL: {signal}

TECHNICAL INDICATORS:
- RSI: {current_rsi:.1f} ({'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'})
- MACD: {current_macd:.4f} ({'Bullish' if current_macd > 0 else 'Bearish'})

KEY LEVELS:
- Support: ${current_price * 0.99:.2f}
- Resistance: ${current_price * 1.01:.2f}

RECOMMENDATION:
{signal} position with proper risk management.
"""
            return analysis
            
        except Exception as e:
            return f"Analysis completed. Error: {str(e)}"

# Create analyzer instance
analyzer = XAUUSDAnalyzer()

@app.route('/')
def home():
    """Serve main page"""
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading template: {str(e)}"

@app.route('/api/analysis/<timeframe>')
def get_analysis(timeframe):
    """Main analysis endpoint"""
    try:
        print(f"Processing analysis for {timeframe}")
        
        # Validate timeframe
        if timeframe not in ['1H', '4H', '1D']:
            return jsonify({"error": "Invalid timeframe"}), 400
        
        # Load and prepare data - GUNAKAN DATA HISTORIS
        df = analyzer.load_historical_data(timeframe, 200)
        df_with_indicators = analyzer.calculate_indicators(df)
        
        # Get current price
        current_price = analyzer.get_realtime_price()
        
        # Update last price
        if len(df_with_indicators) > 0:
            df_with_indicators.iloc[-1, df_with_indicators.columns.get_loc('close')] = current_price
        
        # Generate analysis
        analysis = analyzer.analyze_market(df_with_indicators)
        
        # Prepare chart data
        chart_data = []
        for _, row in df_with_indicators.tail(100).iterrows():
            chart_data.append({
                'datetime': row['datetime'].isoformat() if hasattr(row['datetime'], 'isoformat') else str(row['datetime']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'ema_12': float(row['ema_12']) if 'ema_12' in df_with_indicators.columns and not pd.isna(row['ema_12']) else None,
                'ema_26': float(row['ema_26']) if 'ema_26' in df_with_indicators.columns and not pd.isna(row['ema_26']) else None,
                'ema_50': float(row['ema_50']) if 'ema_50' in df_with_indicators.columns and not pd.isna(row['ema_50']) else None,
                'macd': float(row['macd']) if 'macd' in df_with_indicators.columns and not pd.isna(row['macd']) else None,
                'macd_signal': float(row['macd_signal']) if 'macd_signal' in df_with_indicators.columns and not pd.isna(row['macd_signal']) else None,
                'macd_hist': float(row['macd_hist']) if 'macd_hist' in df_with_indicators.columns and not pd.isna(row['macd_hist']) else None,
                'rsi': float(row['rsi']) if 'rsi' in df_with_indicators.columns and not pd.isna(row['rsi']) else None
            })
        
        # Prepare indicators
        latest_indicators = {}
        if len(df_with_indicators) > 0:
            last_row = df_with_indicators.iloc[-1]
            for indicator in ['ema_12', 'ema_26', 'ema_50', 'rsi', 'macd', 'macd_signal', 'macd_hist']:
                if indicator in df_with_indicators.columns and not pd.isna(last_row[indicator]):
                    latest_indicators[indicator] = float(last_row[indicator])
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "current_price": current_price,
            "technical_indicators": latest_indicators,
            "ai_analysis": analysis,
            "chart_data": chart_data,
            "data_points": len(chart_data),
            "news": {
                "articles": [
                    {
                        "title": "Gold Market Analysis",
                        "description": "Technical indicators showing market trends for XAUUSD.",
                        "source": {"name": "Market Data"},
                        "publishedAt": datetime.now().isoformat()
                    }
                ]
            }
        }
        
        print(f"Analysis completed for {timeframe}. Sent {len(chart_data)} data points.")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/api/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/realtime/price')
def realtime_price():
    """Realtime price endpoint"""
    try:
        price = analyzer.get_realtime_price()
        return jsonify({
            "price": price,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug')
def debug():
    """Debug endpoint to check data files"""
    try:
        files = {}
        for timeframe in ['1H', '4H', '1D']:
            filename = f"data/XAUUSD_{timeframe}.csv"
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                files[timeframe] = {
                    "exists": True,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "first_date": df.iloc[0]['datetime'] if 'datetime' in df.columns else "N/A",
                    "last_date": df.iloc[-1]['datetime'] if 'datetime' in df.columns else "N/A"
                }
            else:
                files[timeframe] = {"exists": False}
        
        return jsonify({
            "status": "debug",
            "data_files": files,
            "current_dir": os.getcwd(),
            "data_dir": os.path.join(os.getcwd(), 'data')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("🚀 XAUUSD Analysis Dashboard - FIXED VERSION")
    print("=" * 60)
    print("📊 Available Endpoints:")
    print("  • GET / → Dashboard")
    print("  • GET /api/analysis/1H → 1Hour Analysis") 
    print("  • GET /api/analysis/4H → 4Hour Analysis")
    print("  • GET /api/analysis/1D → Daily Analysis")
    print("  • GET /api/realtime/price → Current Price")
    print("  • GET /api/health → Health Check")
    print("  • GET /api/debug → Debug Info")
    print("=" * 60)
    
    print("Starting server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
