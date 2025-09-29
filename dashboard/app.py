from flask import Flask, request, jsonify, send_from_directory, render_template
import pandas as pd
import numpy as np
import requests
import os
import json
import sqlite3
import traceback
from datetime import datetime, timedelta
import random
import logging
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
class Config:
    DB_PATH = 'forex_analysis.db'
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 2
    CACHE_DURATION = 300
    
    # Trading parameters
    DEFAULT_TIMEFRAME = "4H"
    SUPPORTED_PAIRS = ["USDJPY", "GBPJPY", "EURJPY", "CHFJPY"]
    SUPPORTED_TIMEFRAMES = ["1H", "4H", "1D", "1W"]
    
    # Data periods
    DATA_PERIODS = {
        "1H": 30 * 24,   # 30 hari * 24 jam
        "4H": 30 * 6,    # 30 hari * 6 interval 4 jam
        "1D": 120,       # 120 hari
        "1W": 52         # 52 minggu
    }
    
    # Risk management
    DEFAULT_STOP_LOSS_PCT = 0.01
    DEFAULT_TAKE_PROFIT_PCT = 0.02

# API Keys from environment variables
TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY", "1a5a4b69dae6419c951a4fb62e4ad7b2")
ALPHA_API_KEY = os.environ.get("ALPHA_API_KEY", "G8588U1ISMGM8GZB")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "b90862d072ce41e4b0505cbd7b710b66")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# API URLs
TWELVE_API_URL = "https://api.twelvedata.com"
ALPHA_API_URL = "https://www.alphavantage.co/query"
NEWS_API_URL = "https://newsapi.org/v2/everything"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Global variables
HISTORICAL = {}
PAIR_MAP = {
    "USDJPY": "USD/JPY",
    "GBPJPY": "GBP/JPY",
    "EURJPY": "EUR/JPY",
    "CHFJPY": "CHF/JPY",
}

# ---------------- DATABASE FUNCTIONS ----------------
def init_db():
    """Initialize database with enhanced tables"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        
        # Main analysis results table
        c.execute('''CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            current_price REAL,
            technical_indicators TEXT,
            ai_analysis TEXT,
            fundamental_news TEXT,
            chart_data TEXT,
            data_source TEXT,
            confidence_score REAL,
            ai_provider TEXT
        )''')
        
        # Performance tracking table
        c.execute('''CREATE TABLE IF NOT EXISTS signal_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            signal_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            signal_type TEXT,
            entry_price REAL,
            stop_loss REAL,
            take_profit_1 REAL,
            take_profit_2 REAL,
            outcome TEXT,
            pnl REAL,
            duration_hours INTEGER,
            confidence_level INTEGER
        )''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
    finally:
        conn.close()

def save_analysis_result(data: Dict):
    """Save analysis result to database"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        
        c.execute('''INSERT INTO analysis_results 
                    (pair, timeframe, current_price, technical_indicators, 
                     ai_analysis, fundamental_news, chart_data, data_source, 
                     confidence_score, ai_provider)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (data['pair'], data['timeframe'], data['current_price'],
                  json.dumps(data['technical_indicators']), json.dumps(data['ai_analysis']),
                  data.get('fundamental_news', ''), json.dumps(data.get('chart_data', {})),
                  data.get('data_source', ''), data['ai_analysis'].get('CONFIDENCE_LEVEL', 50),
                  "DeepSeek" if DEEPSEEK_API_KEY else "Fallback"))
        
        conn.commit()
        logger.info(f"Analysis saved for {data['pair']}-{data['timeframe']}")
    except Exception as e:
        logger.error(f"Error saving analysis: {e}")
    finally:
        conn.close()

# ---------------- DATA LOADING ----------------
def load_csv_data():
    """Load historical CSV data with enhanced error handling and support for longer periods"""
    search_dirs = [".", "data", "historical_data"]
    loaded_count = 0
    
    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
            
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                try:
                    df = pd.read_csv(file_path)
                    df.columns = [col.lower().strip() for col in df.columns]
                    
                    # Handle different column naming conventions
                    if 'close' not in df.columns:
                        if 'price' in df.columns:
                            df['close'] = df['price']
                        elif 'last' in df.columns:
                            df['close'] = df['last']
                        else:
                            logger.warning(f"No price column found in {filename}")
                            continue
                    
                    # Parse date column
                    date_column = None
                    for col in ['date', 'time', 'datetime', 'timestamp']:
                        if col in df.columns:
                            date_column = col
                            break
                    
                    if date_column:
                        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                        df = df.dropna(subset=[date_column])
                        # Sort by date ascending
                        df = df.sort_values(date_column)
                    else:
                        logger.warning(f"No date column found in {filename}")
                        continue
                    
                    # Extract pair and timeframe from filename
                    base_name = os.path.basename(filename).replace(".csv", "")
                    parts = base_name.split("_")
                    pair = parts[0].upper() if parts else "UNKNOWN"
                    timeframe = parts[1].upper() if len(parts) > 1 else "1D"
                    
                    if pair not in Config.SUPPORTED_PAIRS:
                        logger.warning(f"Unsupported pair {pair} in file {filename}")
                        continue
                    
                    # Initialize nested dictionaries
                    if pair not in HISTORICAL:
                        HISTORICAL[pair] = {}
                    
                    HISTORICAL[pair][timeframe] = df
                    loaded_count += 1
                    logger.info(f"✅ Loaded {pair}-{timeframe} from {file_path}, {len(df)} rows")
                    
                except Exception as e:
                    logger.error(f"⚠️ Error loading {file_path}: {e}")
    
    # Log summary of loaded data
    for pair in HISTORICAL:
        for timeframe in HISTORICAL[pair]:
            data_points = len(HISTORICAL[pair][timeframe])
            logger.info(f"📊 {pair}-{timeframe}: {data_points} data points available")
    
    logger.info(f"Total loaded datasets: {loaded_count}")

# ---------------- TECHNICAL INDICATORS CALCULATION ----------------
def calculate_ema_series(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA for entire series"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd_series(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD components for entire series"""
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def calc_indicators(series: List[float], volumes: Optional[List[float]] = None) -> Dict:
    """Calculate technical indicators with enhanced features"""
    if not series:
        return {"error": "No price data available"}
    
    close = pd.Series(series)
    
    # Basic statistics
    cp = close.iloc[-1]
    price_change = close.pct_change().iloc[-1] * 100 if len(close) > 1 else 0
    
    # RSI
    delta = close.diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan)).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    
    # Moving averages
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    ema200 = close.ewm(span=200).mean()
    
    # MACD
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()
    macd_histogram = macd_line - macd_signal
    
    # OBV
    obv = None
    if volumes is not None and len(volumes) == len(series):
        vol = pd.Series(volumes)
        direction = np.sign(delta.fillna(0))
        obv_series = (direction * vol).cumsum()
        obv = obv_series.iloc[-1]
    
    # Bollinger Bands
    bb_upper = sma20 + (close.rolling(20).std() * 2)
    bb_lower = sma20 - (close.rolling(20).std() * 2)
    
    # Support and Resistance (simplified)
    recent_high = close.tail(20).max()
    recent_low = close.tail(20).min()
    
    return {
        "current_price": round(cp, 4),
        "price_change_pct": round(price_change, 2),
        "RSI": round(rsi.iloc[-1], 2),
        "RSI_14": round(rsi.iloc[-1], 2),
        "SMA20": round(sma20.iloc[-1], 4) if not pd.isna(sma20.iloc[-1]) else cp,
        "SMA50": round(sma50.iloc[-1], 4) if not pd.isna(sma50.iloc[-1]) else cp,
        "EMA200": round(ema200.iloc[-1], 4) if not pd.isna(ema200.iloc[-1]) else cp,
        "MACD": round(macd_line.iloc[-1], 4),
        "MACD_Signal": round(macd_signal.iloc[-1], 4),
        "MACD_Histogram": round(macd_histogram.iloc[-1], 4),
        "OBV": round(obv, 2) if obv is not None else "N/A",
        "Bollinger_Upper": round(bb_upper.iloc[-1], 4) if not pd.isna(bb_upper.iloc[-1]) else cp,
        "Bollinger_Lower": round(bb_lower.iloc[-1], 4) if not pd.isna(bb_lower.iloc[-1]) else cp,
        "Resistance": round(recent_high, 4),
        "Support": round(recent_low, 4),
        "Volatility": round(close.pct_change().std() * 100, 2) if len(close) > 1 else 0,
    }

# ---------------- DATA PROVIDERS ----------------
def get_price_twelvedata(pair: str) -> Optional[float]:
    """Get real-time price from Twelve Data API"""
    try:
        symbol = f"{pair[:3]}/{pair[3:]}"
        url = f"{TWELVE_API_URL}/exchange_rate?symbol={symbol}&apikey={TWELVE_API_KEY}"
        
        response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        if "rate" in data:
            price = float(data["rate"])
            logger.info(f"TwelveData price for {pair}: {price}")
            return price
        else:
            logger.warning(f"TwelveData API error for {pair}: {data.get('message', 'Unknown error')}")
            return None
            
    except requests.exceptions.Timeout:
        logger.error(f"TwelveData timeout for {pair}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"TwelveData request error for {pair}: {e}")
        return None
    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"TwelveData data parsing error for {pair}: {e}")
        return None

def get_fundamental_news(pair: str = "USDJPY") -> str:
    """Get fundamental news with sentiment analysis"""
    news_sources = []
    
    # Alpha Vantage News
    try:
        ticker = pair[3:]  # JPY for JPY pairs
        url = f"{ALPHA_API_URL}?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHA_API_KEY}&limit=2"
        response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
        data = response.json()
        
        if "feed" in data and data["feed"]:
            for item in data["feed"][:2]:
                title = item.get('title', '')
                source = item.get('source', 'Unknown')
                sentiment = item.get('overall_sentiment_label', 'Neutral')
                news_sources.append(f"{title} [{source}, {sentiment}]")
    except Exception as e:
        logger.error(f"AlphaVantage news error: {e}")
    
    # NewsAPI Fallback
    if not news_sources:
        try:
            query = f"{pair[:3]} {pair[3:]} forex economy"
            url = f"{NEWS_API_URL}?q={query}&language=en&pageSize=2&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
            response = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
            data = response.json()
            
            if "articles" in data and data["articles"]:
                for article in data["articles"][:2]:
                    title = article.get('title', '')
                    source = article.get('source', {}).get('name', 'Unknown')
                    news_sources.append(f"{title} [{source}]")
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
    
    # Final fallback
    if not news_sources:
        news_text = f"Tidak ada berita terbaru untuk {pair}. Pantau kalender ekonomi untuk update."
    else:
        news_text = " | ".join(news_sources)
    
    return news_text

# ---------------- AI ANALYSIS ----------------
def ai_fallback(tech: Dict, news_summary: str = "") -> Dict:
    """Enhanced fallback AI analysis in Bahasa Indonesia"""
    cp = tech["current_price"]
    rsi = tech["RSI"]
    macd = tech["MACD"]
    macd_signal = tech["MACD_Signal"]
    
    # Multi-factor signal determination
    signal_score = 0
    
    # RSI factor
    if rsi < 30:
        signal_score += 2
    elif rsi > 70:
        signal_score -= 2
    elif 40 < rsi < 60:
        signal_score += 0.5
    
    # MACD factor
    if macd > macd_signal:
        signal_score += 1
    else:
        signal_score -= 1
    
    # Price position factor
    if cp > tech['SMA20']:
        signal_score += 0.5
    else:
        signal_score -= 0.5
    
    # Determine signal
    if signal_score >= 2:
        signal = "BELI KUAT"
        confidence = 80
        sl = cp * 0.995
        tp1 = cp * 1.01
        tp2 = cp * 1.02
        advice = "RSI oversold, momentum bullish kuat. Entry dengan risk-reward menarik."
    elif signal_score >= 1:
        signal = "BELI"
        confidence = 65
        sl = cp * 0.997
        tp1 = cp * 1.008
        tp2 = cp * 1.015
        advice = "Kondisi teknis mendukung bullish. Tunggu konfirmasi breakout resistance."
    elif signal_score <= -2:
        signal = "JUAL KUAT"
        confidence = 80
        sl = cp * 1.005
        tp1 = cp * 0.99
        tp2 = cp * 0.98
        advice = "RSI overbought, momentum bearish kuat. Pertimbangkan short position."
    elif signal_score <= -1:
        signal = "JUAL"
        confidence = 65
        sl = cp * 1.003
        tp1 = cp * 0.992
        tp2 = cp * 0.985
        advice = "Tekanan jual meningkat. Wait for breakdown below support."
    else:
        signal = "TUNGGU"
        confidence = 50
        sl = cp * 0.998
        tp1 = cp * 1.005
        tp2 = cp * 1.01
        advice = "Market dalam konsolidasi. Tunggu breakout yang jelas."
    
    return {
        "SIGNAL": signal,
        "ENTRY_PRICE": round(cp, 4),
        "STOP_LOSS": round(sl, 4),
        "TAKE_PROFIT_1": round(tp1, 4),
        "TAKE_PROFIT_2": round(tp2, 4),
        "CONFIDENCE_LEVEL": confidence,
        "TRADING_ADVICE": f"Analisis teknikal: {advice} Berita: {news_summary[:100]}...",
        "RISK_LEVEL": "RENDAH" if confidence < 60 else "SEDANG" if confidence < 75 else "TINGGI",
        "EXPECTED_MOVEMENT": f"{abs(round((tp1-cp)/cp*100, 2))}%",
        "AI_PROVIDER": "Fallback System"
    }

def ai_deepseek_analysis(pair: str, tech: Dict, fundamentals: str) -> Dict:
    """AI analysis using DeepSeek API in Bahasa Indonesia"""
    if not DEEPSEEK_API_KEY:
        logger.info("No DeepSeek API key, using fallback analysis")
        return ai_fallback(tech, fundamentals)
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
Sebagai analis trading forex profesional, berikan rekomendasi trading dalam format JSON dengan struktur berikut:

{{
    "SIGNAL": "BELI KUAT/BELI/JUAL KUAT/JUAL/TUNGGU",
    "ENTRY_PRICE": number,
    "STOP_LOSS": number, 
    "TAKE_PROFIT_1": number,
    "TAKE_PROFIT_2": number,
    "CONFIDENCE_LEVEL": number (0-100),
    "TRADING_ADVICE": "string dengan analisis detail dalam Bahasa Indonesia",
    "RISK_LEVEL": "RENDAH/SEDANG/TINGGI",
    "EXPECTED_MOVEMENT": "string dengan pergerakan yang diharapkan",
    "AI_PROVIDER": "DeepSeek AI"
}}

DATA TRADING:
Pair: {pair} ({PAIR_MAP.get(pair, pair)})
Timeframe: {request.args.get('timeframe', '4H')}

INDIKATOR TEKNIKAL:
- Harga Saat Ini: {tech['current_price']}
- RSI (14): {tech['RSI']} {"(Oversold)" if tech['RSI'] < 30 else "(Overbought)" if tech['RSI'] > 70 else "(Neutral)"}
- MACD: {tech['MACD']}, Signal: {tech['MACD_Signal']}
- SMA20: {tech['SMA20']}, SMA50: {tech['SMA50']}, EMA200: {tech['EMA200']}
- Support: {tech['Support']}, Resistance: {tech['Resistance']}
- Bollinger Bands: Upper {tech['Bollinger_Upper']}, Lower {tech['Bollinger_Lower']}
- Volatilitas: {tech['Volatility']}%

KONTEKS FUNDAMENTAL:
{fundamentals}

Berikan analisis yang profesional dan realistis dalam Bahasa Indonesia. Pertimbangkan faktor teknikal dan fundamental.
"""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "choices" not in data or not data["choices"]:
            logger.error("DeepSeek API returned no choices")
            return ai_fallback(tech, fundamentals)
        
        ai_response = data["choices"][0]["message"]["content"]
        logger.info(f"DeepSeek raw response: {ai_response[:200]}...")
        
        # Clean the response and parse JSON
        ai_response = ai_response.strip()
        if ai_response.startswith("```json"):
            ai_response = ai_response[7:]
        if ai_response.endswith("```"):
            ai_response = ai_response[:-3]
        
        analysis_result = json.loads(ai_response)
        
        # Validate required fields
        required_fields = ["SIGNAL", "ENTRY_PRICE", "STOP_LOSS", "TAKE_PROFIT_1", "TAKE_PROFIT_2", "CONFIDENCE_LEVEL"]
        if all(field in analysis_result for field in required_fields):
            logger.info(f"DeepSeek analysis completed for {pair}")
            analysis_result["AI_PROVIDER"] = "DeepSeek AI"
            return analysis_result
        else:
            logger.warning("DeepSeek response missing required fields, using fallback")
            return ai_fallback(tech, fundamentals)
            
    except json.JSONDecodeError as e:
        logger.error(f"DeepSeek JSON parsing error: {e}")
        return ai_fallback(tech, fundamentals)
    except requests.exceptions.RequestException as e:
        logger.error(f"DeepSeek API request error: {e}")
        return ai_fallback(tech, fundamentals)
    except Exception as e:
        logger.error(f"DeepSeek unexpected error: {e}")
        return ai_fallback(tech, fundamentals)

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html', 
                         pairs=Config.SUPPORTED_PAIRS,
                         timeframes=Config.SUPPORTED_TIMEFRAMES,
                         ai_available=bool(DEEPSEEK_API_KEY))

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/get_analysis')
def get_analysis():
    """Main analysis endpoint with extended data periods"""
    start_time = datetime.now()
    
    try:
        pair = request.args.get("pair", "USDJPY").upper()
        timeframe = request.args.get("timeframe", "4H").upper()
        use_history = request.args.get("use_history", "0") == "1"
        
        logger.info(f"Analysis request: {pair}-{timeframe}, use_history: {use_history}")
        
        # Validate inputs
        if pair not in Config.SUPPORTED_PAIRS:
            return jsonify({"error": f"Unsupported pair: {pair}"}), 400
        
        if timeframe not in Config.SUPPORTED_TIMEFRAMES:
            return jsonify({"error": f"Unsupported timeframe: {timeframe}"}), 400
        
        # Get price data
        current_price = get_price_twelvedata(pair)
        data_source = "Twelve Data"
        
        if current_price is None:
            # Fallback to historical data
            if pair in HISTORICAL and timeframe in HISTORICAL[pair]:
                current_price = float(HISTORICAL[pair][timeframe].tail(1)["close"].iloc[0])
                data_source = "Historical CSV"
            else:
                # Final fallback - synthetic data
                base_prices = {"USDJPY": 148.50, "GBPJPY": 187.25, "EURJPY": 159.80, "CHFJPY": 169.45}
                current_price = base_prices.get(pair, 150.0) + random.uniform(-0.5, 0.5)
                data_source = "Synthetic"
                logger.info(f"Using synthetic price for {pair}: {current_price}")
        
        # Determine required bars based on timeframe
        required_bars = Config.DATA_PERIODS.get(timeframe, 100)
        
        # Get historical data for indicators
        if use_history and pair in HISTORICAL and timeframe in HISTORICAL[pair]:
            df = HISTORICAL[pair][timeframe].tail(required_bars)
            
            # Jika data tidak cukup, ambil sebanyak yang tersedia
            if len(df) < required_bars:
                logger.warning(f"Insufficient data for {pair}-{timeframe}: {len(df)} bars, needed {required_bars}")
                df = HISTORICAL[pair][timeframe]  # Ambil semua data yang tersedia
            
            closes = df["close"].tolist()
            volumes = df["vol."].fillna(0).tolist() if "vol." in df.columns else None
            
            # Format dates berdasarkan timeframe
            if timeframe == '1D':
                dates = df["date"].dt.strftime("%d/%m/%Y").tolist()
            elif timeframe == '1W':
                dates = df["date"].dt.strftime("%d/%m/%y").tolist()
            else:
                dates = df["date"].dt.strftime("%d/%m %H:%M").tolist()
            
            # Calculate EMA200 and MACD series for chart
            close_series = pd.Series(closes)
            ema200_series = calculate_ema_series(close_series, 200)
            ema200_values = [round(x, 4) for x in ema200_series.tolist()]
            
            macd_line, macd_signal, macd_histogram = calculate_macd_series(close_series)
            macd_line_values = [round(x, 4) for x in macd_line.tolist()]
            macd_signal_values = [round(x, 4) for x in macd_signal.tolist()]
            macd_histogram_values = [round(x, 4) for x in macd_histogram.tolist()]
            
        else:
            # Generate synthetic data dengan jumlah yang sesuai
            num_points = required_bars
            
            # Generate realistic price data dengan trend
            base_price = current_price
            trend_direction = random.choice([-1, 1])
            volatility = 0.5 if timeframe == '1H' else 1.0 if timeframe == '4H' else 2.0 if timeframe == '1D' else 5.0
            
            closes = []
            for i in range(num_points):
                # Add trend and random movement
                trend = trend_direction * volatility * 0.1 * (i / num_points)
                random_move = random.uniform(-volatility, volatility) * 0.01
                price = base_price + trend + random_move
                closes.append(price)
            
            volumes = [random.randint(1000, 10000) for _ in range(num_points)]
            
            # Generate dates yang sesuai
            base_date = datetime.now()
            if timeframe == '1D':
                dates = [(base_date - timedelta(days=i)).strftime("%d/%m/%Y") 
                        for i in range(num_points-1, -1, -1)]
            elif timeframe == '4H':
                dates = [(base_date - timedelta(hours=4*i)).strftime("%d/%m %H:%M") 
                        for i in range(num_points-1, -1, -1)]
            elif timeframe == '1W':
                dates = [(base_date - timedelta(weeks=i)).strftime("%d/%m/%y") 
                        for i in range(num_points-1, -1, -1)]
            else:  # 1H
                dates = [(base_date - timedelta(hours=i)).strftime("%d/%m %H:%M") 
                        for i in range(num_points-1, -1, -1)]
            
            # Calculate EMA200 and MACD untuk synthetic data
            close_series = pd.Series(closes)
            ema200_series = calculate_ema_series(close_series, 200)
            ema200_values = [round(x, 4) for x in ema200_series.tolist()]
            
            macd_line, macd_signal, macd_histogram = calculate_macd_series(close_series)
            macd_line_values = [round(x, 4) for x in macd_line.tolist()]
            macd_signal_values = [round(x, 4) for x in macd_signal.tolist()]
            macd_histogram_values = [round(x, 4) for x in macd_histogram.tolist()]
        
        # Calculate technical indicators
        technical_indicators = calc_indicators(closes, volumes)
        
        # Get fundamental analysis
        fundamental_news = get_fundamental_news(pair)
        
        # AI Analysis
        ai_analysis = ai_deepseek_analysis(pair, technical_indicators, fundamental_news)
        
        # Prepare response
        response_data = {
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "current_price": current_price,
            "technical_indicators": technical_indicators,
            "ai_analysis": ai_analysis,
            "fundamental_news": fundamental_news,
            "chart_data": {
                "dates": dates,
                "close": closes,
                "volume": volumes if volumes else [0] * len(closes),
                "ema200": ema200_values,
                "macd_line": macd_line_values,
                "macd_signal": macd_signal_values,
                "macd_histogram": macd_histogram_values
            },
            "data_source": data_source,
            "processing_time": round((datetime.now() - start_time).total_seconds(), 2),
            "ai_provider": ai_analysis.get("AI_PROVIDER", "Fallback System"),
            "data_points": len(dates)
        }
        
        # Save to database
        save_analysis_result(response_data)
        
        logger.info(f"Analysis completed for {pair}-{timeframe} in {response_data['processing_time']}s, {len(dates)} data points")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/quick_overview')
def quick_overview():
    """Quick overview of all supported pairs"""
    overview = {}
    
    for pair in Config.SUPPORTED_PAIRS:
        try:
            price = get_price_twelvedata(pair)
            source = "Twelve Data"
            
            if price is None and pair in HISTORICAL and "1D" in HISTORICAL[pair]:
                price = float(HISTORICAL[pair]["1D"].tail(1)["close"].iloc[0])
                source = "Historical"
            elif price is None:
                base_prices = {"USDJPY": 148.50, "GBPJPY": 187.25, "EURJPY": 159.80, "CHFJPY": 169.45}
                price = base_prices.get(pair, 150.0) + random.uniform(-0.1, 0.1)
                source = "Synthetic"
            
            if pair in HISTORICAL and "1D" in HISTORICAL[pair]:
                df = HISTORICAL[pair]["1D"]
                if len(df) > 1:
                    prev_close = df.iloc[-2]['close']
                    change_pct = ((price - prev_close) / prev_close) * 100
                else:
                    change_pct = 0
            else:
                change_pct = random.uniform(-0.3, 0.3)
            
            overview[pair] = {
                "price": round(price, 4),
                "change": round(change_pct, 2),
                "source": source,
                "trend": "up" if change_pct > 0 else "down" if change_pct < 0 else "flat"
            }
            
        except Exception as e:
            logger.error(f"Quick overview error for {pair}: {e}")
            overview[pair] = {
                "price": None,
                "change": 0,
                "source": "error",
                "trend": "flat"
            }
    
    return jsonify(overview)

@app.route('/ai_status')
def ai_status():
    """Check AI DeepSeek status"""
    status = {
        "ai_available": bool(DEEPSEEK_API_KEY),
        "ai_provider": "DeepSeek",
        "fallback_ready": True,
        "message": "AI DeepSeek siap digunakan" if DEEPSEEK_API_KEY else "AI DeepSeek tidak tersedia, menggunakan sistem fallback"
    }
    return jsonify(status)

@app.route('/performance')
def performance_metrics():
    """Get performance metrics from database"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        
        # Get recent analyses count
        c = conn.cursor()
        c.execute('''SELECT pair, COUNT(*) as count, 
                    AVG(confidence_score) as avg_confidence,
                    MAX(timestamp) as last_analysis,
                    ai_provider
                    FROM analysis_results 
                    WHERE timestamp > datetime('now', '-1 day')
                    GROUP BY pair, ai_provider''')
        
        results = c.fetchall()
        performance_data = []
        
        for row in results:
            performance_data.append({
                "pair": row[0],
                "analysis_count": row[1],
                "avg_confidence": round(row[2] or 0, 1),
                "last_analysis": row[3],
                "ai_provider": row[4]
            })
        
        conn.close()
        return jsonify(performance_data)
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------- INITIALIZATION ----------------
if __name__ == "__main__":
    logger.info("Starting Forex Analysis Application...")
    
    # Initialize components
    init_db()
    load_csv_data()
    
    # Log AI status
    if DEEPSEEK_API_KEY:
        logger.info("✅ DeepSeek AI integration ENABLED")
    else:
        logger.info("⚠️ DeepSeek AI integration DISABLED - using fallback system")
    
    # Log data periods
    logger.info("📅 Data periods configured:")
    for tf, bars in Config.DATA_PERIODS.items():
        logger.info(f"   {tf}: {bars} bars")
    
    # Start Flask application
    logger.info("Application initialized successfully")
    app.run(debug=True, host='0.0.0.0', port=5000)
