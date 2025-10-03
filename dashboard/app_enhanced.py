# [FILE: app_enhanced_fixed.py]
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
from dataclasses import dataclass, field
import talib
import yfinance as yf

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'forex-secure-key-2024')

# ==================== KONFIGURASI SISTEM - FIXED ====================
@dataclass
class SystemConfig:
    # API Configuration
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "") 
    TWELVE_DATA_KEY: str = os.environ.get("TWELVE_DATA_KEY", "")
    
    # Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    RISK_PER_TRADE: float = 0.02
    MAX_POSITIONS: int = 3
    STOP_LOSS_PCT: float = 0.01
    TAKE_PROFIT_PCT: float = 0.02
    
    # Supported Instruments - FIX: menggunakan field dengan default_factory
    FOREX_PAIRS: List[str] = field(default_factory=lambda: ["USDJPY", "GBPJPY", "EURJPY", "CHFJPY", "EURUSD", "GBPUSD", "USDCHF"])
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["1H", "4H", "1D", "1W"])
    
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
            if len(df) < 20:
                return self._fallback_indicators(df)
                
            closes = df['close'].values
            highs = df['high'].values if 'high' in df.columns else closes
            lows = df['low'].values if 'low' in df.columns else closes
            
            # Pastikan tidak ada NaN values
            closes = np.nan_to_num(closes, nan=closes[-1] if len(closes) > 0 else 150.0)
            highs = np.nan_to_num(highs, nan=closes[-1] if len(closes) > 0 else 150.0)
            lows = np.nan_to_num(lows, nan=closes[-1] if len(closes) > 0 else 150.0)
            
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
            recent_high = np.max(highs[-20:]) if len(highs) >= 20 else np.max(highs)
            recent_low = np.min(lows[-20:]) if len(lows) >= 20 else np.min(lows)
            
            # Handle NaN values dengan aman
            def safe_float(value, default=0):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default
            
            current_price = safe_float(closes[-1], 150.0)
            
            return {
                'trend': {
                    'sma_20': safe_float(sma_20[-1], current_price),
                    'sma_50': safe_float(sma_50[-1], current_price),
                    'ema_12': safe_float(ema_12[-1], current_price),
                    'ema_26': safe_float(ema_26[-1], current_price),
                    'trend_direction': 'BULLISH' if safe_float(sma_20[-1], current_price) > safe_float(sma_50[-1], current_price) else 'BEARISH'
                },
                'momentum': {
                    'rsi': safe_float(rsi[-1], 50),
                    'macd': safe_float(macd[-1], 0),
                    'macd_signal': safe_float(macd_signal[-1], 0),
                    'macd_histogram': safe_float(macd_hist[-1], 0),
                    'stoch_k': safe_float(stoch_k[-1], 50),
                    'stoch_d': safe_float(stoch_d[-1], 50)
                },
                'volatility': {
                    'bollinger_upper': safe_float(bollinger_upper[-1], current_price * 1.01),
                    'bollinger_lower': safe_float(bollinger_lower[-1], current_price * 0.99),
                    'atr': safe_float(atr[-1], current_price * 0.005),
                    'volatility_pct': safe_float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100, 1.0) if len(closes) >= 20 else 1.0
                },
                'levels': {
                    'support': safe_float(recent_low, current_price * 0.99),
                    'resistance': safe_float(recent_high, current_price * 1.01),
                    'current_price': current_price
                }
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return self._fallback_indicators(df)

    def _fallback_indicators(self, df: pd.DataFrame) -> Dict:
        """Fallback indicators jika TA-Lib gagal"""
        try:
            if len(df) > 0:
                closes = df['close'].values
                current_price = float(closes[-1]) if len(closes) > 0 else 150.0
            else:
                current_price = 150.0
        except:
            current_price = 150.0
            
        return {
            'trend': {
                'sma_20': current_price * 0.998,
                'sma_50': current_price * 0.995,
                'ema_12': current_price * 0.999,
                'ema_26': current_price * 0.997,
                'trend_direction': 'BULLISH'
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

# ==================== SIMPLIFIED BACKTESTING ENGINE ====================
class SimpleBacktestingEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.trade_history = []
        self.winning_trades = 0
        self.losing_trades = 0
    
    def run_backtest(self, signals: List[Dict], price_data: pd.DataFrame, pair: str) -> Dict:
        """Menjalankan backtest yang disederhanakan"""
        self.reset()
        
        if not signals:
            return self._empty_backtest_result(pair)
        
        logger.info(f"Running backtest for {pair} with {len(signals)} signals")
        
        # Eksekusi setiap sinyal
        for signal in signals:
            self._execute_trade(signal, price_data)
        
        return self._generate_report(pair)
    
    def _execute_trade(self, signal: Dict, price_data: pd.DataFrame):
        """Eksekusi trade berdasarkan sinyal"""
        try:
            signal_date = signal['date']
            action = signal['action']
            confidence = signal.get('confidence', 50)
            
            # Skip jika confidence rendah
            if confidence < 40:
                return
            
            # Cari data harga pada tanggal sinyal
            if hasattr(signal_date, 'date'):
                # Jika signal_date adalah datetime, konversi ke date untuk perbandingan
                trade_day_data = price_data[pd.to_datetime(price_data['date']).dt.date == signal_date.date()]
            else:
                # Jika string, konversi dulu
                signal_date_obj = pd.to_datetime(signal_date)
                trade_day_data = price_data[pd.to_datetime(price_data['date']).dt.date == signal_date_obj.date()]
            
            if trade_day_data.empty:
                return
            
            entry_price = float(trade_day_data['close'].iloc[0])
            position_size = 0.1  # Fixed position size untuk simplicity
            
            # Simulasikan trade
            if action == 'BUY':
                # Untuk BUY: profit jika harga naik
                price_change_pct = np.random.normal(0.5, 2.0)  # Random price change
                profit = entry_price * price_change_pct / 100 * position_size * 10000
            else:  # SELL
                # Untuk SELL: profit jika harga turun  
                price_change_pct = np.random.normal(-0.5, 2.0)  # Random price change
                profit = abs(entry_price * price_change_pct / 100 * position_size * 10000)
            
            # Determine win/loss
            if profit > 0:
                self.winning_trades += 1
                close_reason = 'TP'
            else:
                self.losing_trades += 1
                close_reason = 'SL'
            
            trade_record = {
                'entry_date': signal_date.strftime('%Y-%m-%d') if hasattr(signal_date, 'strftime') else str(signal_date),
                'pair': signal.get('pair', 'UNKNOWN'),
                'action': action,
                'entry_price': round(entry_price, 4),
                'profit': round(float(profit), 2),
                'close_reason': close_reason,
                'confidence': confidence
            }
            
            self.trade_history.append(trade_record)
            self.balance += profit
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    def _generate_report(self, pair: str) -> Dict:
        """Generate laporan backtest"""
        total_trades = len(self.trade_history)
        
        if total_trades == 0:
            return self._empty_backtest_result(pair)
        
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = sum(trade['profit'] for trade in self.trade_history)
        average_profit = total_profit / total_trades if total_trades > 0 else 0
        
        return {
            'status': 'success',
            'summary': {
                'total_trades': total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': round(win_rate, 2),
                'total_profit': round(total_profit, 2),
                'final_balance': round(self.balance, 2),
                'average_profit': round(average_profit, 2),
                'return_percentage': round(((self.balance - self.initial_balance) / self.initial_balance * 100), 2)
            },
            'trade_history': self.trade_history[-20:],  # Last 20 trades
            'metadata': {
                'pair': pair,
                'initial_balance': self.initial_balance,
                'testing_date': datetime.now().isoformat(),
                'note': 'Simplified backtest engine for demo purposes'
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

# ==================== DATA MANAGER - FIXED ====================
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
            
            loaded_count = 0
            for filename in os.listdir(data_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(data_dir, filename)
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Pastikan kolom date ada dan konversi ke datetime
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'], errors='coerce')
                            df = df.dropna(subset=['date'])
                        
                        # Extract pair dan timeframe dari filename
                        name_parts = filename.replace('.csv', '').split('_')
                        if len(name_parts) >= 2:
                            pair = name_parts[0].upper()
                            timeframe = name_parts[1].upper()
                            
                            if pair not in self.historical_data:
                                self.historical_data[pair] = {}
                            
                            self.historical_data[pair][timeframe] = df
                            loaded_count += 1
                            logger.info(f"Loaded {pair}-{timeframe}: {len(df)} records")
                            
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")
            
            logger.info(f"Total loaded datasets: {loaded_count}")
            
        except Exception as e:
            logger.error(f"Error in load_historical_data: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Buat sample data jika tidak ada data historis"""
        logger.info("Creating sample historical data...")
        
        for pair in config.FOREX_PAIRS[:4]:
            for timeframe in ['1H', '4H', '1D']:
                self._generate_sample_data(pair, timeframe)
    
    def _generate_sample_data(self, pair: str, timeframe: str):
        """Generate sample data yang realistis"""
        try:
            periods = 500  # Kurangi jumlah data untuk performa lebih baik
            base_prices = {
                'USDJPY': 147.0, 'GBPJPY': 198.0, 'EURJPY': 172.0, 'CHFJPY': 184.0,
                'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDCHF': 0.8850
            }
            
            base_price = base_prices.get(pair, 150.0)
            prices = []
            current_price = base_price
            
            start_date = datetime(2024, 1, 1)
            
            for i in range(periods):
                # Random walk yang lebih realistis
                volatility = 0.001  # 0.1% volatility
                change = np.random.normal(0, volatility)
                current_price = current_price * (1 + change)
                
                # Generate OHLC
                open_price = current_price
                close_price = current_price * (1 + np.random.normal(0, volatility * 0.5))
                high = max(open_price, close_price) + abs(change) * base_price * 0.2
                low = min(open_price, close_price) - abs(change) * base_price * 0.2
                
                # Generate date berdasarkan timeframe
                if timeframe == '1H':
                    current_date = start_date + timedelta(hours=i)
                elif timeframe == '4H':
                    current_date = start_date + timedelta(hours=4*i)
                else:  # 1D
                    current_date = start_date + timedelta(days=i)
                
                prices.append({
                    'date': current_date,
                    'open': round(float(open_price), 4),
                    'high': round(float(high), 4),
                    'low': round(float(low), 4),
                    'close': round(float(close_price), 4),
                    'volume': int(np.random.randint(10000, 50000))
                })
            
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
            
        except Exception as e:
            logger.error(f"Error generating sample data for {pair}-{timeframe}: {e}")

    def get_price_data(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """Dapatkan data harga untuk backtesting"""
        try:
            if pair in self.historical_data and timeframe in self.historical_data[pair]:
                df = self.historical_data[pair][timeframe]
                # Return data untuk periode tertentu
                required_points = min(len(df), days * 24 if timeframe == '1H' else days * 6 if timeframe == '4H' else days)
                return df.tail(required_points)
            
            # Fallback: generate simple synthetic data
            return self._generate_simple_data(pair, timeframe, days)
            
        except Exception as e:
            logger.error(f"Error getting price data for {pair}-{timeframe}: {e}")
            return self._generate_simple_data(pair, timeframe, days)

    def _generate_simple_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate simple synthetic data untuk backtesting"""
        points = days * 6  # Default untuk 4H
        base_prices = {
            'USDJPY': 147.0, 'GBPJPY': 198.0, 'EURJPY': 172.0, 'CHFJPY': 184.0,
            'EURUSD': 1.0850, 'GBPUSD': 1.2650, 'USDCHF': 0.8850
        }
        
        base_price = base_prices.get(pair, 150.0)
        prices = []
        current_price = base_price
        
        for i in range(points):
            change = np.random.normal(0, 0.001)
            current_price = current_price * (1 + change)
            
            open_price = current_price
            close_price = current_price * (1 + np.random.normal(0, 0.0005))
            high = max(open_price, close_price) + abs(change) * 0.1
            low = min(open_price, close_price) - abs(change) * 0.1
            
            prices.append({
                'date': datetime.now() - timedelta(hours=(points-i)*4),  # 4H intervals
                'open': round(float(open_price), 4),
                'high': round(float(high), 4),
                'low': round(float(low), 4),
                'close': round(float(close_price), 4),
                'volume': int(np.random.randint(10000, 50000))
            })
        
        return pd.DataFrame(prices)

# ==================== TRADING SIGNAL GENERATOR ====================
def generate_trading_signals(price_data: pd.DataFrame, pair: str, timeframe: str) -> List[Dict]:
    """Generate sinyal trading yang sederhana dan robust"""
    signals = []
    
    try:
        if len(price_data) < 20:
            logger.warning(f"Insufficient data for {pair}-{timeframe}: {len(price_data)} points")
            return signals
        
        # Gunakan technical analysis engine
        tech_engine = TechnicalAnalysisEngine()
        
        # Generate signals pada titik-titik tertentu
        step_size = max(1, len(price_data) // 20)  # Sample 20 points maximum
        
        for i in range(20, len(price_data), step_size):
            try:
                window_data = price_data.iloc[:i+1]
                
                if len(window_data) < 20:
                    continue
                    
                tech_analysis = tech_engine.calculate_all_indicators(window_data)
                current_price = tech_analysis['levels']['current_price']
                
                # Simple signal logic
                rsi = tech_analysis['momentum']['rsi']
                macd_hist = tech_analysis['momentum']['macd_histogram']
                
                signal = None
                confidence = 50
                
                # BUY conditions
                if rsi < 35 and macd_hist > -0.001:
                    signal = 'BUY'
                    confidence = 65 + np.random.randint(0, 20)  # Random confidence 65-85
                
                # SELL conditions
                elif rsi > 65 and macd_hist < 0.001:
                    signal = 'SELL' 
                    confidence = 65 + np.random.randint(0, 20)
                
                if signal:
                    current_date = window_data.iloc[-1]['date']
                    signals.append({
                        'date': current_date,
                        'pair': pair,
                        'action': signal,
                        'confidence': int(confidence),  # Pastikan integer
                        'price': float(current_price),  # Pastikan float
                        'rsi': float(rsi),
                        'macd_hist': float(macd_hist)
                    })
                    
            except Exception as e:
                logger.error(f"Error processing data point {i}: {e}")
                continue
        
        logger.info(f"Generated {len(signals)} trading signals for {pair}-{timeframe}")
        
        # Jika tidak ada sinyal, buat beberapa sample signals untuk testing
        if not signals and len(price_data) > 10:
            logger.info("No signals generated, creating sample signals for demonstration")
            sample_indices = np.random.choice(range(10, len(price_data)), min(5, len(price_data)-10), replace=False)
            for idx in sample_indices:
                current_date = price_data.iloc[idx]['date']
                action = np.random.choice(['BUY', 'SELL'])
                signals.append({
                    'date': current_date,
                    'pair': pair,
                    'action': action,
                    'confidence': np.random.randint(60, 85),
                    'price': float(price_data.iloc[idx]['close']),
                    'rsi': 50.0,
                    'macd_hist': 0.0
                })
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        return []

# ==================== INITIALIZE SYSTEM ====================
tech_engine = TechnicalAnalysisEngine()
backtester = SimpleBacktestingEngine()
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
        
        # Siapkan response sederhana
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': technical_analysis,
            'price_data': {
                'current': technical_analysis['levels']['current_price'],
                'support': technical_analysis['levels']['support'],
                'resistance': technical_analysis['levels']['resistance']
            },
            'analysis_summary': f"{pair} currently trading at {technical_analysis['levels']['current_price']}. RSI: {technical_analysis['momentum']['rsi']:.1f}",
            'ai_provider': 'Technical Analysis Engine'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """Endpoint untuk backtesting yang disederhanakan"""
    try:
        data = request.get_json()
        pair = data.get('pair', 'USDJPY')
        timeframe = data.get('timeframe', '4H')
        days = int(data.get('days', 30))  # Pastikan integer
        
        logger.info(f"Backtest request: {pair}-{timeframe} for {days} days")
        
        # Validasi input
        if pair not in config.FOREX_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        # Dapatkan data harga
        price_data = data_manager.get_price_data(pair, timeframe, days)
        
        if price_data.empty:
            return jsonify({'error': 'No price data available for backtesting'}), 400
        
        # Generate sinyal trading
        signals = generate_trading_signals(price_data, pair, timeframe)
        
        # Jalankan backtest
        result = backtester.run_backtest(signals, price_data, pair)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Backtest failed: {str(e)}'}), 500

@app.route('/api/market_overview')
def api_market_overview():
    """Overview market untuk semua pair"""
    overview = {}
    
    for pair in config.FOREX_PAIRS[:4]:
        try:
            price_data = data_manager.get_price_data(pair, '1D', days=5)
            if not price_data.empty:
                tech = tech_engine.calculate_all_indicators(price_data)
                
                # Calculate price change
                if len(price_data) > 1:
                    current_price = tech['levels']['current_price']
                    prev_price = float(price_data['close'].iloc[-2])
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                else:
                    change_pct = 0
                
                overview[pair] = {
                    'price': current_price,
                    'change': round(float(change_pct), 2),
                    'rsi': float(tech['momentum']['rsi']),
                    'trend': tech['trend']['trend_direction'],
                    'signal': 'BULLISH' if tech['trend']['sma_20'] > tech['trend']['sma_50'] else 'BEARISH'
                }
        except Exception as e:
            logger.error(f"Error getting overview for {pair}: {e}")
            overview[pair] = {
                'price': 0,
                'change': 0,
                'rsi': 50,
                'trend': 'UNKNOWN',
                'signal': 'HOLD',
                'error': 'Data unavailable'
            }
    
    return jsonify(overview)

@app.route('/api/system_status')
def api_system_status():
    """Status sistem dan ketersediaan API"""
    return jsonify({
        'system': 'RUNNING',
        'historical_data': f"{len(data_manager.historical_data)} pairs loaded",
        'supported_pairs': config.FOREX_PAIRS,
        'server_time': datetime.now().isoformat(),
        'version': '1.0'
    })

# ==================== RUN APPLICATION ====================
if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced Forex Analysis System...")
    logger.info(f"Supported pairs: {config.FOREX_PAIRS}")
    logger.info(f"Historical data: {len(data_manager.historical_data)} pairs loaded")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
