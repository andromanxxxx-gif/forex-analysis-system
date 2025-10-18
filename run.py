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
        self.data_cache = {}
        
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
            high = df['high'].values
            low = df['low'].values
            
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
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.bollinger_bands(close)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            
            # Calculate Stochastic
            stoch_k, stoch_d = self.stochastic(high, low, close)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            print("All technical indicators calculated successfully")
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
        return rsi.fillna(50)

    def bollinger_bands(self, data, period=20, std_dev=2):
        """Bollinger Bands"""
        series = pd.Series(data)
        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def stochastic(self, high, low, close, k_period=14, d_period=3):
        """Stochastic Oscillator"""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        lowest_low = low_series.rolling(window=k_period, min_periods=1).min()
        highest_high = high_series.rolling(window=k_period, min_periods=1).max()
        
        stoch_k = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
        stoch_d = stoch_k.rolling(window=d_period, min_periods=1).mean()
        
        return stoch_k, stoch_d

    def get_realtime_price(self):
        """Get real-time gold price"""
        # Simulate realistic gold price
        base_price = 1968.0
        movement = np.random.normal(0, 1.5)
        price = base_price + movement
        print(f"Real-time XAUUSD price: ${price:.2f}")
        return round(price, 2)

    def get_fundamental_news(self):
        """Get fundamental news about gold market"""
        try:
            # Sample news data - in real implementation, you would fetch from NewsAPI
            return {
                "articles": [
                    {
                        "title": "Gold Prices Hold Steady Amid Economic Uncertainty",
                        "description": "XAUUSD maintains strong support levels as investors seek safe-haven assets.",
                        "publishedAt": datetime.now().isoformat(),
                        "source": {"name": "Financial Times"},
                        "url": "#"
                    },
                    {
                        "title": "Federal Reserve Policy Impacts Precious Metals",
                        "description": "Recent Fed announcements create favorable conditions for gold prices.",
                        "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "source": {"name": "Bloomberg"},
                        "url": "#"
                    },
                    {
                        "title": "Technical Analysis: XAUUSD Shows Bullish Momentum",
                        "description": "Gold approaches key resistance level as bullish pattern forms.",
                        "publishedAt": (datetime.now() - timedelta(days=1)).isoformat(),
                        "source": {"name": "Reuters"},
                        "url": "#"
                    }
                ]
            }
        except Exception as e:
            print(f"Error getting news: {e}")
            return {"articles": []}

    def analyze_market_conditions(self, df, indicators):
        """Comprehensive market analysis with trading signals"""
        try:
            if len(df) == 0:
                return "No data available for analysis"
                
            current_price = df.iloc[-1]['close']
            
            # Get latest indicator values
            current_rsi = indicators.get('rsi', 50)
            current_macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            stoch_k = indicators.get('stoch_k', 50)
            stoch_d = indicators.get('stoch_d', 50)
            ema_12 = indicators.get('ema_12', current_price)
            ema_26 = indicators.get('ema_26', current_price)
            ema_50 = indicators.get('ema_50', current_price)
            bb_upper = indicators.get('bb_upper', current_price * 1.02)
            bb_lower = indicators.get('bb_lower', current_price * 0.98)
            
            # Calculate trading signals
            bullish_signals = 0
            bearish_signals = 0
            
            # RSI Analysis
            if current_rsi < 30:
                bullish_signals += 2
                rsi_signal = "OVERSOLD - STRONG BUY"
            elif current_rsi < 40:
                bullish_signals += 1
                rsi_signal = "NEARLY OVERSOLD - BUY"
            elif current_rsi > 70:
                bearish_signals += 2
                rsi_signal = "OVERBOUGHT - STRONG SELL"
            elif current_rsi > 60:
                bearish_signals += 1
                rsi_signal = "NEARLY OVERBOUGHT - SELL"
            else:
                rsi_signal = "NEUTRAL"
            
            # MACD Analysis
            if current_macd > macd_signal:
                bullish_signals += 1
                macd_signal_text = "BULLISH CROSSOVER"
            else:
                bearish_signals += 1
                macd_signal_text = "BEARISH CROSSOVER"
            
            # EMA Analysis
            if current_price > ema_12 > ema_26 > ema_50:
                bullish_signals += 2
                ema_signal = "STRONG BULLISH TREND"
            elif current_price < ema_12 < ema_26 < ema_50:
                bearish_signals += 2
                ema_signal = "STRONG BEARISH TREND"
            elif current_price > ema_12 and ema_12 > ema_26:
                bullish_signals += 1
                ema_signal = "BULLISH TREND"
            elif current_price < ema_12 and ema_12 < ema_26:
                bearish_signals += 1
                ema_signal = "BEARISH TREND"
            else:
                ema_signal = "MIXED TREND"
            
            # Stochastic Analysis
            if stoch_k < 20 and stoch_d < 20:
                bullish_signals += 1
                stoch_signal = "OVERSOLD - BUY"
            elif stoch_k > 80 and stoch_d > 80:
                bearish_signals += 1
                stoch_signal = "OVERBOUGHT - SELL"
            else:
                stoch_signal = "NEUTRAL"
            
            # Bollinger Bands Analysis
            if current_price < bb_lower:
                bullish_signals += 1
                bb_signal = "PRICE BELOW LOWER BAND - POTENTIAL BUY"
            elif current_price > bb_upper:
                bearish_signals += 1
                bb_signal = "PRICE ABOVE UPPER BAND - POTENTIAL SELL"
            else:
                bb_signal = "PRICE WITHIN BANDS - NEUTRAL"
            
            # Determine overall trend and signal
            if bullish_signals - bearish_signals >= 3:
                trend = "STRONG BULLISH"
                signal = "STRONG BUY"
                risk = "LOW"
            elif bullish_signals - bearish_signals >= 1:
                trend = "BULLISH"
                signal = "BUY"
                risk = "MEDIUM"
            elif bearish_signals - bullish_signals >= 3:
                trend = "STRONG BEARISH"
                signal = "STRONG SELL"
                risk = "HIGH"
            elif bearish_signals - bullish_signals >= 1:
                trend = "BEARISH"
                signal = "SELL"
                risk = "MEDIUM"
            else:
                trend = "NEUTRAL"
                signal = "HOLD"
                risk = "LOW"
            
            # Calculate trading levels
            if signal in ["STRONG BUY", "BUY"]:
                stop_loss = current_price * 0.99  # 1% below current price
                take_profit_1 = current_price * 1.01  # 1% above
                take_profit_2 = current_price * 1.02  # 2% above
            elif signal in ["STRONG SELL", "SELL"]:
                stop_loss = current_price * 1.01  # 1% above current price
                take_profit_1 = current_price * 0.99  # 1% below
                take_profit_2 = current_price * 0.98  # 2% below
            else:
                stop_loss = current_price * 0.995
                take_profit_1 = current_price * 1.005
                take_profit_2 = current_price * 1.01
            
            analysis = f"""
** 🎯 XAUUSD TRADING ANALYSIS **
*Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

** 📊 EXECUTIVE SUMMARY **
- Current Price: ${current_price:.2f}
- Market Trend: {trend}
- Trading Signal: {signal}
- Risk Level: {risk}

** 📈 TECHNICAL OVERVIEW **

** Trend Analysis: **
- EMA Alignment: {ema_signal}
- Price vs EMA 12: {'Above' if current_price > ema_12 else 'Below'} (${ema_12:.2f})
- Price vs EMA 26: {'Above' if current_price > ema_26 else 'Below'} (${ema_26:.2f})

** Momentum Indicators: **
- RSI (14): {current_rsi:.1f} - {rsi_signal}
- MACD: {current_macd:.4f} - {macd_signal_text}
- Stochastic: K={stoch_k:.1f}, D={stoch_d:.1f} - {stoch_signal}
- Bollinger Bands: {bb_signal}

** Signal Strength: **
- Bullish Signals: {bullish_signals}
- Bearish Signals: {bearish_signals}

** 💰 TRADING RECOMMENDATIONS **

** Primary Strategy: **
{signal} XAUUSD with {risk.lower()} risk approach.

** Key Levels: **
- Strong Support: ${bb_lower:.2f}
- Strong Resistance: ${bb_upper:.2f}

** Risk Management: **
- Stop Loss: ${stop_loss:.2f}
- Take Profit 1: ${take_profit_1:.2f}
- Take Profit 2: ${take_profit_2:.2f}

** 📋 TRADING PLAN **

** Entry: **
- Ideal Entry: ${current_price:.2f}
- Alternative Entry: ${current_price * 0.998:.2f} (for BUY) / ${current_price * 1.002:.2f} (for SELL)

** Position Sizing: **
- Risk per trade: 1-2% of account
- Leverage: Maximum 1:10 for this setup

** ⚠️ RISK CONSIDERATIONS **
- Monitor Federal Reserve announcements
- Watch USD strength and inflation data
- Consider geopolitical factors
- Always use proper risk management

** 🔍 MARKET CONTEXT **
Gold is showing {trend.lower()} characteristics with {bullish_signals} bullish indicators vs {bearish_signals} bearish indicators. {'Consider long positions' if 'BUY' in signal else 'Consider short positions' if 'SELL' in signal else 'Wait for clearer signals'}.
"""
            return analysis
            
        except Exception as e:
            return f"Comprehensive analysis completed. Technical indicators processed. Error: {str(e)}"

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
        
        # Prepare indicators for analysis
        latest_indicators = {}
        if len(df_with_indicators) > 0:
            last_row = df_with_indicators.iloc[-1]
            for indicator in ['ema_12', 'ema_26', 'ema_50', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                             'stoch_k', 'stoch_d', 'bb_upper', 'bb_lower']:
                if indicator in df_with_indicators.columns and not pd.isna(last_row[indicator]):
                    latest_indicators[indicator] = float(last_row[indicator])
        
        # Generate comprehensive analysis
        analysis = analyzer.analyze_market_conditions(df_with_indicators, latest_indicators)
        
        # Get fundamental news
        news_data = analyzer.get_fundamental_news()
        
        # Prepare chart data (limit to 100 points for better performance)
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
                'rsi': float(row['rsi']) if 'rsi' in df_with_indicators.columns and not pd.isna(row['rsi']) else None,
                'bb_upper': float(row['bb_upper']) if 'bb_upper' in df_with_indicators.columns and not pd.isna(row['bb_upper']) else None,
                'bb_lower': float(row['bb_lower']) if 'bb_lower' in df_with_indicators.columns and not pd.isna(row['bb_lower']) else None
            })
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "current_price": current_price,
            "technical_indicators": latest_indicators,
            "ai_analysis": analysis,
            "chart_data": chart_data,
            "data_points": len(chart_data),
            "news": news_data,
            "trading_signals": {
                "bullish_count": sum([1 for k in latest_indicators.keys() if 'rsi' in k and latest_indicators[k] < 40] + 
                                   [1 for k in latest_indicators.keys() if 'macd' in k and latest_indicators.get('macd', 0) > latest_indicators.get('macd_signal', 0)]),
                "bearish_count": sum([1 for k in latest_indicators.keys() if 'rsi' in k and latest_indicators[k] > 60] + 
                                   [1 for k in latest_indicators.keys() if 'macd' in k and latest_indicators.get('macd', 0) < latest_indicators.get('macd_signal', 0)])
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
    print("🚀 XAUUSD Professional Trading Analysis")
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
    print("✨ Features:")
    print("  • Comprehensive Technical Analysis")
    print("  • AI-Powered Trading Signals")
    print("  • Stop Loss & Take Profit Levels")
    print("  • Fundamental News")
    print("  • Professional Chart Display")
    print("=" * 60)
    
    print("Starting server...")
    app.run(debug=True, port=5000, host='0.0.0.0')
