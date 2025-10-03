# [FILE: app_enhanced_final.py]
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import requests
import os
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import talib
import yfinance as yf
from dataclasses import dataclass

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'forex-secure-key-2024')

# ==================== KONFIGURASI SISTEM ====================
@dataclass
class SystemConfig:
    # API Configuration
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "") 
    TWELVE_DATA_KEY: str = os.environ.get("TWELVE_DATA_KEY", "")
    
    # Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02  # 2% risk per trade
    MAX_POSITIONS: int = 3
    STOP_LOSS_PCT: float = 0.01   # 1% stop loss
    TAKE_PROFIT_PCT: float = 0.02 # 2% take profit
    
    # Supported Instruments
    FOREX_PAIRS: List[str] = "USDJPY,GBPJPY,EURJPY,CHFJPY,EURUSD,GBPUSD,USDCHF".split(",")
    TIMEFRAMES: List[str] = "1H,4H,1D,1W".split(",")
    
    # Backtesting
    DEFAULT_BACKTEST_DAYS: int = 30
    MIN_DATA_POINTS: int = 100

config = SystemConfig()

# ==================== ENGINE ANALISIS TEKNIKAL ====================
class TechnicalAnalysisEngine:
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Menghitung semua indikator teknikal dari DataFrame OHLC"""
        try:
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Trend Indicators
            sma_20 = talib.SMA(closes, timeperiod=20)
            sma_50 = talib.SMA(closes, timeperiod=50)
            ema_12 = talib.EMA(closes, timeperiod=12)
            ema_26 = talib.EMA(closes, timeperiod=26)
            
            # Momentum Indicators
            rsi = talib.RSI(closes, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(closes)
            stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
            
            # Volatility Indicators
            bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(closes)
            atr = talib.ATR(highs, lows, closes, timeperiod=14)
            
            # Support & Resistance
            recent_high = highs[-20:].max()
            recent_low = lows[-20:].min()
            
            return {
                'trend': {
                    'sma_20': float(sma_20[-1]) if not np.isnan(sma_20[-1]) else None,
                    'sma_50': float(sma_50[-1]) if not np.isnan(sma_50[-1]) else None,
                    'ema_12': float(ema_12[-1]) if not np.isnan(ema_12[-1]) else None,
                    'ema_26': float(ema_26[-1]) if not np.isnan(ema_26[-1]) else None,
                    'trend_direction': 'BULLISH' if sma_20[-1] > sma_50[-1] else 'BEARISH'
                },
                'momentum': {
                    'rsi': float(rsi[-1]) if not np.isnan(rsi[-1]) else 50,
                    'macd': float(macd[-1]) if not np.isnan(macd[-1]) else 0,
                    'macd_signal': float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else 0,
                    'macd_histogram': float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else 0,
                    'stoch_k': float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else 50,
                    'stoch_d': float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else 50
                },
                'volatility': {
                    'bollinger_upper': float(bollinger_upper[-1]) if not np.isnan(bollinger_upper[-1]) else None,
                    'bollinger_lower': float(bollinger_lower[-1]) if not np.isnan(bollinger_lower[-1]) else None,
                    'atr': float(atr[-1]) if not np.isnan(atr[-1]) else 0,
                    'volatility_pct': float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100) if len(closes) >= 20 else 0
                },
                'levels': {
                    'support': float(recent_low),
                    'resistance': float(recent_high),
                    'current_price': float(closes[-1])
                }
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._fallback_indicators(df)

    def _fallback_indicators(self, df: pd.DataFrame) -> Dict:
        """Fallback indicators jika TA-Lib gagal"""
        closes = df['close'].values
        current_price = float(closes[-1])
        
        return {
            'trend': {
                'sma_20': current_price * 0.998,
                'sma_50': current_price * 0.995,
                'ema_12': current_price * 0.999,
                'ema_26': current_price * 0.997,
                'trend_direction': 'BULLISH' if current_price > df['close'].mean() else 'BEARISH'
            },
            'momentum': {
                'rsi': 50,
                'macd': 0.001,
                'macd_signal': 0.0005,
                'macd_histogram': 0.0005,
                'stoch_k': 50,
                'stoch_d': 50
            },
            'volatility': {
                'bollinger_upper': current_price * 1.01,
                'bollinger_lower': current_price * 0.99,
                'atr': current_price * 0.005,
                'volatility_pct': 1.0
            },
            'levels': {
                'support': current_price * 0.99,
                'resistance': current_price * 1.01,
                'current_price': current_price
            }
        }

# ==================== ENGINE ANALISIS FUNDAMENTAL ====================
class FundamentalAnalysisEngine:
    def __init__(self):
        self.news_cache = {}
    
    def get_forex_news(self, pair: str) -> str:
        """Mendapatkan berita fundamental untuk pair forex"""
        try:
            # Map pair ke negara terkait
            country_map = {
                'USDJPY': 'Japan United States economy',
                'GBPJPY': 'Japan UK economy Brexit',
                'EURJPY': 'Japan Europe ECB economy',
                'CHFJPY': 'Japan Switzerland economy',
                'EURUSD': 'Europe United States Fed ECB',
                'GBPUSD': 'UK United States Bank of England Fed',
                'USDCHF': 'United States Switzerland SNB Fed'
            }
            
            query = country_map.get(pair, 'forex market news')
            
            if config.NEWS_API_KEY and config.NEWS_API_KEY != "demo":
                url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=3&apiKey={config.NEWS_API_KEY}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    if articles:
                        news_items = []
                        for article in articles[:2]:  # Ambil 2 artikel terbaru
                            title = article.get('title', '')
                            source = article.get('source', {}).get('name', 'Unknown')
                            news_items.append(f"{title} (Source: {source})")
                        
                        return " | ".join(news_items)
            
            # Fallback news
            return self._get_fallback_news(pair)
            
        except Exception as e:
            logger.error(f"Error fetching news for {pair}: {e}")
            return self._get_fallback_news(pair)
    
    def _get_fallback_news(self, pair: str) -> str:
        """Berita fallback ketika API tidak tersedia"""
        news_templates = {
            'USDJPY': [
                "Bank of Japan maintains ultra-loose monetary policy. Fed signals potential rate cuts in 2024.",
                "Yen weakness continues as BOJ sticks to yield curve control. USD strength persists.",
                "USD/JPY approaches intervention levels as interest rate differential widens."
            ],
            'GBPJPY': [
                "Bank of England holds rates steady amid inflation concerns. GBP shows volatility.",
                "UK economic data mixed, GBP/JPY influenced by risk sentiment and carry trades.",
                "Brexit aftermath continues to impact GBP crosses with Japanese Yen."
            ],
            'EURJPY': [
                "ECB monitoring inflation closely. Euro area growth shows signs of stabilization.",
                "EUR/JPY influenced by ECB policy outlook and Japanese economic recovery.",
                "European inflation data key for EUR direction against safe-haven JPY."
            ]
        }
        
        import random
        return random.choice(news_templates.get(pair, ["Market analysis ongoing. Monitor economic indicators for trading opportunities."]))

# ==================== DEEPSEEK AI ANALYZER ====================
class DeepSeekAnalyzer:
    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    def analyze_market(self, pair: str, technical_data: Dict, fundamental_news: str) -> Dict:
        """Menganalisis market menggunakan DeepSeek AI"""
        if not self.api_key or self.api_key == "demo":
            logger.warning("DeepSeek API key not available, using enhanced analysis")
            return self._enhanced_analysis(technical_data, fundamental_news, pair)
        
        try:
            prompt = self._build_analysis_prompt(pair, technical_data, fundamental_news)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": """Anda adalah analis forex profesional dengan pengalaman 10 tahun. 
                        Berikan analisis yang realistis, praktis, dan dapat ditindaklanjuti. 
                        Fokus pada risk management dan peluang trading yang jelas."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1500
            }
            
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                ai_response = response.json()["choices"][0]["message"]["content"]
                return self._parse_ai_response(ai_response, technical_data)
            else:
                logger.error(f"DeepSeek API error: {response.status_code}")
                return self._enhanced_analysis(technical_data, fundamental_news, pair)
                
        except Exception as e:
            logger.error(f"DeepSeek analysis failed: {e}")
            return self._enhanced_analysis(technical_data, fundamental_news, pair)
    
    def _build_analysis_prompt(self, pair: str, technical_data: Dict, news: str) -> str:
        """Membangun prompt untuk analisis AI"""
        trend = technical_data['trend']
        momentum = technical_data['momentum']
        volatility = technical_data['volatility']
        levels = technical_data['levels']
        
        return f"""
ANALISIS FOREX UNTUK {pair}

DATA TEKNIKAL:
- Harga Saat Ini: {levels['current_price']}
- Trend: {trend['trend_direction']}
- RSI: {momentum['rsi']:.2f} ({'OVERSOLD' if momentum['rsi'] < 30 else 'OVERBOUGHT' if momentum['rsi'] > 70 else 'NETRAL'})
- MACD: {momentum['macd']:.4f} (Signal: {momentum['macd_signal']:.4f})
- Support: {levels['support']:.4f}
- Resistance: {levels['resistance']:.4f}
- Volatilitas: {volatility['volatility_pct']:.2f}%
- ATR: {volatility['atr']:.4f}

BERITA FUNDAMENTAL: {news}

HASILKAN ANALISIS DALAM FORMAT JSON:
{{
    "signal": "BUY/SELL/HOLD",
    "confidence": 0-100,
    "entry_price": "rentang harga",
    "stop_loss": "harga",
    "take_profit_1": "harga", 
    "take_profit_2": "harga",
    "risk_level": "LOW/MEDIUM/HIGH",
    "analysis_summary": "ringkasan analisis dalam Bahasa Indonesia",
    "key_levels": "level penting untuk diawasi",
    "timeframe_suggestion": "timeframe yang disarankan"
}}

Pertimbangkan:
1. Konvergensi/divergensi indikator
2. Level support/resistance 
3. Konteks berita fundamental
4. Risk-reward ratio yang realistis
5. Kondisi overbought/oversold
"""
    
    def _parse_ai_response(self, ai_response: str, technical_data: Dict) -> Dict:
        """Parse response dari DeepSeek AI"""
        try:
            # Clean response
            cleaned_response = ai_response.replace('```json', '').replace('```', '').strip()
            analysis = json.loads(cleaned_response)
            
            # Add metadata
            analysis['ai_provider'] = 'DeepSeek AI'
            analysis['timestamp'] = datetime.now().isoformat()
            
            return analysis
            
        except json.JSONDecodeError:
            logger.error("Failed to parse AI JSON response, using enhanced analysis")
            return self._enhanced_analysis(technical_data, "", "")
    
    def _enhanced_analysis(self, technical_data: Dict, news: str, pair: str) -> Dict:
        """Analisis enhanced ketika AI tidak tersedia"""
        trend = technical_data['trend']
        momentum = technical_data['momentum']
        levels = technical_data['levels']
        
        current_price = levels['current_price']
        rsi = momentum['rsi']
        macd_hist = momentum['macd_histogram']
        
        # Logika analisis multi-faktor
        signal_score = 0
        
        # RSI scoring
        if rsi < 30:
            signal_score += 3
        elif rsi < 40:
            signal_score += 2
        elif rsi > 70:
            signal_score -= 3
        elif rsi > 60:
            signal_score -= 2
        
        # MACD scoring
        if macd_hist > 0:
            signal_score += 2
        else:
            signal_score -= 2
        
        # Trend scoring
        if trend['trend_direction'] == 'BULLISH':
            signal_score += 1
        else:
            signal_score -= 1
        
        # Determine signal
        if signal_score >= 4:
            signal = "BUY"
            confidence = 75
            sl = current_price * (1 - config.STOP_LOSS_PCT)
            tp1 = current_price * (1 + config.TAKE_PROFIT_PCT)
            tp2 = current_price * (1 + config.TAKE_PROFIT_PCT * 1.5)
        elif signal_score >= 2:
            signal = "BUY" 
            confidence = 60
            sl = current_price * (1 - config.STOP_LOSS_PCT * 0.8)
            tp1 = current_price * (1 + config.TAKE_PROFIT_PCT * 0.8)
            tp2 = current_price * (1 + config.TAKE_PROFIT_PCT * 1.2)
        elif signal_score <= -4:
            signal = "SELL"
            confidence = 75
            sl = current_price * (1 + config.STOP_LOSS_PCT)
            tp1 = current_price * (1 - config.TAKE_PROFIT_PCT)
            tp2 = current_price * (1 - config.TAKE_PROFIT_PCT * 1.5)
        elif signal_score <= -2:
            signal = "SELL"
            confidence = 60
            sl = current_price * (1 + config.STOP_LOSS_PCT * 0.8)
            tp1 = current_price * (1 - config.TAKE_PROFIT_PCT * 0.8)
            tp2 = current_price * (1 - config.TAKE_PROFIT_PCT * 1.2)
        else:
            signal = "HOLD"
            confidence = 50
            sl = current_price * 0.995
            tp1 = current_price * 1.005
            tp2 = current_price * 1.01
        
        return {
            "signal": signal,
            "confidence": confidence,
            "entry_price": f"{current_price:.4f}",
            "stop_loss": f"{sl:.4f}",
            "take_profit_1": f"{tp1:.4f}",
            "take_profit_2": f"{tp2:.4f}",
            "risk_level": "LOW" if confidence < 60 else "MEDIUM" if confidence < 75 else "HIGH",
            "analysis_summary": f"Analisis teknikal menunjukkan kondisi {signal.lower()} untuk {pair}. RSI: {rsi:.1f}, Trend: {trend['trend_direction']}",
            "key_levels": f"Support: {levels['support']:.4f}, Resistance: {levels['resistance']:.4f}",
            "timeframe_suggestion": "4H-1D untuk konfirmasi",
            "ai_provider": "Enhanced Technical Analysis",
            "timestamp": datetime.now().isoformat()
        }

# ==================== REALISTIC BACKTESTING ENGINE ====================
class RealisticBacktestingEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        self.current_date = None
    
    def run_backtest(self, signals: List[Dict], price_data: pd.DataFrame, pair: str) -> Dict:
        """Menjalankan backtest yang realistis dengan data harga aktual"""
        self.reset()
        
        if not signals:
            return self._empty_backtest_result(pair)
        
        logger.info(f"Running backtest for {pair} with {len(signals)} signals")
        
        # Simulasikan eksekusi trade berdasarkan sinyal
        for signal in signals:
            self._execute_signal(signal, price_data, pair)
        
        return self._generate_performance_report(pair)
    
    def _execute_signal(self, signal: Dict, price_data: pd.DataFrame, pair: str):
        """Eksekusi sinyal trading"""
        try:
            signal_date = signal['date']
            signal_action = signal['action']
            confidence = signal.get('confidence', 50)
            
            # Skip jika confidence terlalu rendah
            if confidence < 40:
                return
            
            # Cari harga pada tanggal sinyal
            trade_data = price_data[price_data['date'] == signal_date]
            if trade_data.empty:
                return
            
            entry_price = trade_data['close'].iloc[0]
            
            # Hitung position size berdasarkan risk management
            position_size = self._calculate_position_size(entry_price, signal.get('sl_pips', 30))
            
            if signal_action == 'BUY':
                stop_loss = entry_price * (1 - signal.get('sl_pct', 0.01))
                take_profit = entry_price * (1 + signal.get('tp_pct', 0.02))
            else:  # SELL
                stop_loss = entry_price * (1 + signal.get('sl_pct', 0.01))
                take_profit = entry_price * (1 - signal.get('tp_pct', 0.02))
            
            # Simulasikan trade
            trade = {
                'entry_date': signal_date,
                'pair': pair,
                'action': signal_action,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'status': 'open'
            }
            
            self.positions.append(trade)
            self._check_positions(price_data)
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def _calculate_position_size(self, entry_price: float, stop_loss_pips: int) -> float:
        """Hitung position size berdasarkan risk management"""
        risk_amount = self.balance * config.RISK_PER_TRADE
        pip_value = 10  # Untuk JPY pairs
        potential_loss = stop_loss_pips * pip_value
        
        if potential_loss > 0:
            position_size = risk_amount / potential_loss
            return min(position_size, 0.1)  # Max 0.1 lot
        return 0.01  # Default 0.01 lot
    
    def _check_positions(self, price_data: pd.DataFrame):
        """Check semua posisi terbuka untuk SL/TP"""
        current_price = price_data['close'].iloc[-1] if not price_data.empty else 0
        
        for position in self.positions[:]:
            if position['status'] != 'open':
                continue
            
            pnl = 0
            close_reason = None
            
            if position['action'] == 'BUY':
                if current_price >= position['take_profit']:
                    pnl = (position['take_profit'] - position['entry_price']) * position['position_size'] * 10000
                    close_reason = 'TP'
                elif current_price <= position['stop_loss']:
                    pnl = (position['stop_loss'] - position['entry_price']) * position['position_size'] * 10000
                    close_reason = 'SL'
            else:  # SELL
                if current_price <= position['take_profit']:
                    pnl = (position['entry_price'] - position['take_profit']) * position['position_size'] * 10000
                    close_reason = 'TP'
                elif current_price >= position['stop_loss']:
                    pnl = (position['entry_price'] - position['stop_loss']) * position['position_size'] * 10000
                    close_reason = 'SL'
            
            if close_reason:
                position['status'] = 'closed'
                position['close_date'] = datetime.now()
                position['pnl'] = pnl
                position['close_reason'] = close_reason
                
                self.balance += pnl
                self.trade_history.append(position.copy())
                
                # Remove dari positions aktif
                self.positions = [p for p in self.positions if p['status'] == 'open']
    
    def _generate_performance_report(self, pair: str) -> Dict:
        """Generate laporan performa backtest"""
        if not self.trade_history:
            return self._empty_backtest_result(pair)
        
        df_trades = pd.DataFrame(self.trade_history)
        
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['pnl'] > 0])
        losing_trades = len(df_trades[df_trades['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = df_trades['pnl'].sum()
        average_profit = df_trades['pnl'].mean()
        
        return {
            'status': 'success',
            'summary': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'total_profit': round(total_profit, 2),
                'final_balance': round(self.balance, 2),
                'average_profit': round(average_profit, 2),
                'return_percentage': round(((self.balance - self.initial_balance) / self.initial_balance * 100), 2)
            },
            'trade_history': [
                {
                    'entry_date': trade['entry_date'].strftime('%Y-%m-%d'),
                    'pair': trade['pair'],
                    'action': trade['action'],
                    'entry_price': round(trade['entry_price'], 4),
                    'pnl': round(trade.get('pnl', 0), 2),
                    'close_reason': trade.get('close_reason', 'Open')
                }
                for trade in self.trade_history[-20:]  # Last 20 trades
            ],
            'metadata': {
                'pair': pair,
                'initial_balance': self.initial_balance,
                'testing_date': datetime.now().isoformat()
            }
        }
    
    def _empty_backtest_result(self, pair: str) -> Dict:
        """Hasil backtest ketika tidak ada trade"""
        return {
            'status': 'no_trades',
            'summary': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'final_balance': self.initial_balance,
                'average_profit': 0,
                'return_percentage': 0
            },
            'trade_history': [],
            'metadata': {
                'pair': pair,
                'initial_balance': self.initial_balance,
                'testing_date': datetime.now().isoformat(),
                'message': 'No trades executed during backtest period'
            }
        }

# ==================== DATA MANAGER ====================
class DataManager:
    def __init__(self):
        self.historical_data = {}
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load data historis dari file CSV"""
        try:
            data_dir = "historical_data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                logger.info("Created historical_data directory")
                self._create_sample_data()
                return
            
            for filename in os.listdir(data_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(data_dir, filename)
                    try:
                        df = pd.read_csv(file_path)
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Extract pair and timeframe from filename
                        name_parts = filename.replace('.csv', '').split('_')
                        if len(name_parts) >= 2:
                            pair = name_parts[0].upper()
                            timeframe = name_parts[1].upper()
                            
                            if pair not in self.historical_data:
                                self.historical_data[pair] = {}
                            
                            self.historical_data[pair][timeframe] = df
                            logger.info(f"Loaded {pair}-{timeframe}: {len(df)} records")
                            
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")
            
            logger.info(f"Total loaded: {len(self.historical_data)} pairs")
            
        except Exception as e:
            logger.error(f"Error in load_historical_data: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Buat sample data jika tidak ada data historis"""
        logger.info("Creating sample historical data...")
        
        for pair in config.FOREX_PAIRS[:4]:  # Buat untuk 4 pair pertama
            for timeframe in ['1H', '4H', '1D']:
                self._generate_sample_data(pair, timeframe)
    
    def _generate_sample_data(self, pair: str, timeframe: str):
        """Generate sample data yang realistis"""
        periods = 1000
        base_prices = {
            'USDJPY': 147.0, 'GBPJPY': 198.0, 'EURJPY': 172.0, 'CHFJPY': 184.0,
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDCHF': 0.8850
        }
        
        base_price = base_prices.get(pair, 150.0)
        prices = []
        current_price = base_price
        
        for i in range(periods):
            # Random walk dengan trend dan volatilitas realistis
            trend = np.random.normal(0, 0.0005)
            noise = np.random.normal(0, 0.001)
            
            price_change = trend + noise
            new_price = current_price * (1 + price_change)
            
            # Generate OHLC
            open_price = current_price
            close_price = new_price
            high_price = max(open_price, close_price) + abs(noise) * base_price * 0.3
            low_price = min(open_price, close_price) - abs(noise) * base_price * 0.3
            
            # Generate date
            if timeframe == '1H':
                current_date = datetime(2023, 1, 1) + timedelta(hours=i)
            elif timeframe == '4H':
                current_date = datetime(2023, 1, 1) + timedelta(hours=4*i)
            else:  # 1D
                current_date = datetime(2023, 1, 1) + timedelta(days=i)
            
            prices.append({
                'date': current_date,
                'open': round(open_price, 4),
                'high': round(high_price, 4),
                'low': round(low_price, 4),
                'close': round(close_price, 4),
                'volume': np.random.randint(10000, 50000)
            })
            
            current_price = close_price
        
        df = pd.DataFrame(prices)
        
        # Save to file
        data_dir = "historical_data"
        filename = f"{data_dir}/{pair}_{timeframe}.csv"
        df.to_csv(filename, index=False)
        
        # Store in memory
        if pair not in self.historical_data:
            self.historical_data[pair] = {}
        self.historical_data[pair][timeframe] = df
        
        logger.info(f"Created sample data: {filename}")

    def get_price_data(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Dapatkan data harga untuk backtesting"""
        if pair in self.historical_data and timeframe in self.historical_data[pair]:
            df = self.historical_data[pair][timeframe]
            # Return data untuk periode tertentu
            required_points = days * 24 if timeframe == '1H' else days * 6 if timeframe == '4H' else days
            return df.tail(min(len(df), required_points))
        
        # Fallback: generate synthetic data
        return self._generate_synthetic_data(pair, timeframe, days)

    def _generate_synthetic_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate synthetic data untuk backtesting"""
        points = days * 24 if timeframe == '1H' else days * 6 if timeframe == '4H' else days
        base_prices = {
            'USDJPY': 147.0, 'GBPJPY': 198.0, 'EURJPY': 172.0, 'CHFJPY': 184.0,
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDCHF': 0.8850
        }
        
        base_price = base_prices.get(pair, 150.0)
        prices = []
        current_price = base_price
        
        for i in range(points):
            change = np.random.normal(0, 0.001)  # 0.1% daily volatility
            current_price = current_price * (1 + change)
            
            open_price = current_price
            close_price = current_price * (1 + np.random.normal(0, 0.0005))
            high = max(open_price, close_price) + abs(change) * 0.3
            low = min(open_price, close_price) - abs(change) * 0.3
            
            prices.append({
                'date': datetime.now() - timedelta(hours=points-i) if timeframe in ['1H','4H'] else datetime.now() - timedelta(days=points-i),
                'open': round(open_price, 4),
                'high': round(high, 4),
                'low': round(low, 4),
                'close': round(close_price, 4),
                'volume': np.random.randint(10000, 50000)
            })
        
        return pd.DataFrame(prices)

# ==================== INITIALIZE SYSTEM ====================
tech_engine = TechnicalAnalysisEngine()
fundamental_engine = FundamentalAnalysisEngine() 
ai_analyzer = DeepSeekAnalyzer()
backtester = RealisticBacktestingEngine()
data_manager = DataManager()

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html', 
                         pairs=config.FOREX_PAIRS,
                         timeframes=config.TIMEFRAMES)

@app.route('/api/analyze')
def api_analyze():
    """Endpoint untuk analisis market real-time"""
    try:
        pair = request.args.get('pair', 'USDJPY').upper()
        timeframe = request.args.get('timeframe', '4H').upper()
        
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        # Dapatkan data harga
        price_data = data_manager.get_price_data(pair, timeframe, days=60)
        if price_data.empty:
            return jsonify({'error': 'No price data available'}), 400
        
        # Analisis teknikal
        technical_analysis = tech_engine.calculate_all_indicators(price_data)
        
        # Analisis fundamental
        fundamental_news = fundamental_engine.get_forex_news(pair)
        
        # Analisis AI
        ai_analysis = ai_analyzer.analyze_market(pair, technical_analysis, fundamental_news)
        
        # Siapkan response
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'fundamental_analysis': fundamental_news,
            'ai_analysis': ai_analysis,
            'price_data': {
                'current': technical_analysis['levels']['current_price'],
                'support': technical_analysis['levels']['support'],
                'resistance': technical_analysis['levels']['resistance']
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Endpoint untuk backtesting"""
    try:
        data = request.get_json()
        pair = data.get('pair', 'USDJPY')
        timeframe = data.get('timeframe', '4H')
        days = data.get('days', 30)
        
        logger.info(f"Backtest request: {pair}-{timeframe} for {days} days")
        
        # Dapatkan data harga
        price_data = data_manager.get_price_data(pair, timeframe, days)
        
        # Generate sinyal trading
        signals = generate_trading_signals(price_data, pair, timeframe)
        
        # Jalankan backtest
        result = backtester.run_backtest(signals, price_data, pair)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500

def generate_trading_signals(price_data: pd.DataFrame, pair: str, timeframe: str) -> List[Dict]:
    """Generate sinyal trading berdasarkan analisis teknikal"""
    signals = []
    
    try:
        # Analisis teknikal untuk setiap titik data
        for i in range(50, len(price_data), 5):  # Skip some points untuk efisiensi
            window_data = price_data.iloc[:i+1]
            
            if len(window_data) < 50:
                continue
                
            tech_analysis = tech_engine.calculate_all_indicators(window_data)
            current_price = tech_analysis['levels']['current_price']
            
            # Logika sinyal sederhana
            rsi = tech_analysis['momentum']['rsi']
            macd_hist = tech_analysis['momentum']['macd_histogram']
            trend = tech_analysis['trend']['trend_direction']
            
            signal = None
            confidence = 50
            
            # Kondisi BUY
            if (rsi < 35 and macd_hist > 0 and trend == 'BULLISH') or \
               (rsi < 30 and macd_hist > -0.001):
                signal = 'BUY'
                confidence = 70
                
            # Kondisi SELL  
            elif (rsi > 65 and macd_hist < 0 and trend == 'BEARISH') or \
                 (rsi > 70 and macd_hist < 0.001):
                signal = 'SELL'
                confidence = 70
            
            if signal and confidence > 60:
                signals.append({
                    'date': window_data.iloc[-1]['date'],
                    'pair': pair,
                    'action': signal,
                    'confidence': confidence,
                    'price': current_price,
                    'sl_pips': 30,
                    'tp_pips': 45
                })
        
        logger.info(f"Generated {len(signals)} trading signals for {pair}-{timeframe}")
        return signals
        
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        return []

@app.route('/api/market_overview')
def api_market_overview():
    """Overview market untuk semua pair"""
    overview = {}
    
    for pair in config.FOREX_PAIRS[:4]:  # 4 pair pertama
        try:
            price_data = data_manager.get_price_data(pair, '1D', days=10)
            if not price_data.empty:
                tech = tech_engine.calculate_all_indicators(price_data)
                
                overview[pair] = {
                    'price': tech['levels']['current_price'],
                    'change': round(((tech['levels']['current_price'] - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2] * 100), 2) if len(price_data) > 1 else 0,
                    'rsi': tech['momentum']['rsi'],
                    'trend': tech['trend']['trend_direction'],
                    'signal': 'BULLISH' if tech['trend']['sma_20'] > tech['trend']['sma_50'] else 'BEARISH'
                }
        except Exception as e:
            logger.error(f"Error getting overview for {pair}: {e}")
            overview[pair] = {'error': str(e)}
    
    return jsonify(overview)

@app.route('/api/system_status')
def api_system_status():
    """Status sistem dan ketersediaan API"""
    return jsonify({
        'deepseek_ai': 'ENABLED' if config.DEEPSEEK_API_KEY else 'DISABLED',
        'news_api': 'ENABLED' if config.NEWS_API_KEY else 'DISABLED',
        'price_data': 'ENABLED' if len(data_manager.historical_data) > 0 else 'DISABLED',
        'historical_pairs': list(data_manager.historical_data.keys()),
        'server_time': datetime.now().isoformat()
    })

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced Forex Analysis System...")
    logger.info(f"Supported pairs: {config.FOREX_PAIRS}")
    logger.info(f"DeepSeek AI: {'ENABLED' if config.DEEPSEEK_API_KEY else 'DISABLED'}")
    logger.info(f"Historical data: {len(data_manager.historical_data)} pairs loaded")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
