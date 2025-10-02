[file name]: app_enhanced.py
[file content begin]
from flask import Flask, request, jsonify, send_from_directory, render_template, session
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
app.secret_key = os.environ.get('SECRET_KEY', 'forex-analysis-backtest-secret-key-2024')

# ENHANCED Configuration dengan Pair-Specific Parameters
class Config:
    DB_PATH = 'forex_analysis_enhanced.db'
    REQUEST_TIMEOUT = 15
    MAX_RETRIES = 2
    CACHE_DURATION = 300
    
    # Trading parameters
    DEFAULT_TIMEFRAME = "4H"
    SUPPORTED_PAIRS = ["USDJPY", "GBPJPY", "EURJPY", "CHFJPY"]
    SUPPORTED_TIMEFRAMES = ["1H", "4H", "1D", "1W"]
    
    # Data periods
    DATA_PERIODS = {
        "1H": 30 * 24,   # 30 days * 24 hours
        "4H": 30 * 6,    # 30 days * 6 four-hour intervals
        "1D": 120,       # 120 days
        "1W": 52         # 52 weeks
    }
    
    # Enhanced Risk management
    DEFAULT_STOP_LOSS_PCT = 0.01
    DEFAULT_TAKE_PROFIT_PCT = 0.02
    MAX_DRAWDOWN_PCT = 0.05  # Max 5% drawdown
    DAILY_LOSS_LIMIT = 0.03  # Max 3% loss per day

    # Backtesting
    INITIAL_BALANCE = 10000
    DEFAULT_LOT_SIZE = 0.1
    
    # Enhanced Trading Parameters - lebih selektif
    PAIR_PRIORITY = {
        'GBPJPY': 1,  
        'USDJPY': 2,
        'EURJPY': 3,
        'CHFJPY': 4
    }
    
    # Pair-specific parameters untuk optimal performance
    PAIR_SPECIFIC_PARAMS = {
        'USDJPY': {
            'optimal_hours': [1, 2, 3, 9, 10, 11, 12, 13, 14],  # Tokyo-London overlap
            'min_volatility': 0.4,
            'max_volatility': 8.0,
            'preferred_direction': 'BOTH',
            'news_sensitivity': 'HIGH',
            'volatility_multiplier': 1.2,
            'confidence_boost': 5
        },
        'GBPJPY': {
            'optimal_hours': [8, 9, 10, 11, 12, 13, 14, 15],
            'min_volatility': 0.5,
            'max_volatility': 10.0,
            'preferred_direction': 'BOTH',
            'news_sensitivity': 'HIGH',
            'volatility_multiplier': 1.3,
            'confidence_boost': 0
        },
        'EURJPY': {
            'optimal_hours': [7, 8, 9, 10, 11, 12, 13, 14],
            'min_volatility': 0.3,
            'max_volatility': 7.0,
            'preferred_direction': 'BOTH',
            'news_sensitivity': 'MEDIUM',
            'volatility_multiplier': 1.1,
            'confidence_boost': 0
        },
        'CHFJPY': {
            'optimal_hours': [8, 9, 10, 11, 12, 13, 14],
            'min_volatility': 0.4,
            'max_volatility': 6.0,
            'preferred_direction': 'BOTH',
            'news_sensitivity': 'MEDIUM',
            'volatility_multiplier': 1.0,
            'confidence_boost': 0
        }
    }
    
    # Enhanced Timeframe-specific parameters - lebih selektif
    TIMEFRAME_PARAMS = {
        "1H": {
            "min_volatility": 0.3,
            "max_volatility": 8.0,
            "min_trend_strength": 0.01,
            "optimal_hours": list(range(0, 24)),
            "required_bars": 30 * 24,
            "confidence_threshold": 40,
            "start_index": 20,
            "max_daily_trades": 5
        },
        "4H": {
            "min_volatility": 0.4,
            "max_volatility": 8.0,
            "min_trend_strength": 0.02,
            "optimal_hours": list(range(0, 24)),
            "required_bars": 30 * 6,
            "confidence_threshold": 45,
            "start_index": 10,
            "max_daily_trades": 3
        },
        "1D": {
            "min_volatility": 0.2,
            "max_volatility": 12.0,
            "min_trend_strength": 0.01,
            "optimal_hours": list(range(0, 24)),
            "required_bars": 60,
            "confidence_threshold": 35,
            "start_index": 20,
            "max_daily_trades": 2
        },
        "1W": {
            "min_volatility": 0.1,
            "max_volatility": 15.0,
            "min_trend_strength": 0.005,
            "optimal_hours": list(range(0, 24)),
            "required_bars": 30,
            "confidence_threshold": 30,
            "start_index": 10,
            "max_daily_trades": 1
        }
    }

# API Keys from environment variables
TWELVE_API_KEY = os.environ.get("TWELVE_API_KEY", "")
ALPHA_API_KEY = os.environ.get("ALPHA_API_KEY", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "b")
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

# ---------------- ENHANCED UTILITY FUNCTIONS ----------------
def is_optimal_session(pair: str, hour: int) -> bool:
    """Filter trading berdasarkan session optimal untuk setiap pair"""
    pair_params = Config.PAIR_SPECIFIC_PARAMS.get(pair, {})
    optimal_hours = pair_params.get('optimal_hours', list(range(0, 24)))
    return hour in optimal_hours

def calculate_dynamic_position_size(confidence: float, consecutive_losses: int, base_lot_size: float = 0.1) -> float:
    """Adjust lot size berdasarkan confidence dan loss streak"""
    # Kurangi position size setelah consecutive losses
    if consecutive_losses >= 3:
        return base_lot_size * 0.3
    elif consecutive_losses >= 2:
        return base_lot_size * 0.5
    elif consecutive_losses >= 1:
        return base_lot_size * 0.7
    
    # Adjust berdasarkan confidence
    if confidence >= 80:
        return min(base_lot_size * 1.5, 0.2)  # Max 0.2 lot
    elif confidence >= 70:
        return base_lot_size * 1.2
    elif confidence >= 60:
        return base_lot_size
    elif confidence >= 50:
        return base_lot_size * 0.8
    else:
        return base_lot_size * 0.5  # Lot size kecil untuk confidence rendah

def get_pair_specific_params(pair: str, param_type: str):
    """Get pair-specific parameters"""
    pair_params = Config.PAIR_SPECIFIC_PARAMS.get(pair, {})
    return pair_params.get(param_type, None)

# ---------------- ENHANCED BACKTESTING MODULE ----------------
class EnhancedForexBacktester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        self.data = {}
        self.pairs = []
        self.daily_loss_tracker = {}
        self.consecutive_losses = 0
        self.daily_trade_count = {}
        
    def load_historical_data(self, historical_data):
        self.data = historical_data
        self.pairs = list(historical_data.keys())
        return self.data
    
    def calculate_pip_value(self, pair, lot_size=0.1):
        jpy_pairs = ['GBPJPY', 'USDJPY', 'EURJPY', 'CHFJPY']
        if any(p in pair for p in jpy_pairs):
            return lot_size * 1000
        else:
            return lot_size * 10
    
    def execute_trade(self, signal, current_price):
        # Dynamic position sizing
        adjusted_lot_size = calculate_dynamic_position_size(
            signal.get('confidence', 50), 
            self.consecutive_losses,
            signal.get('lot_size', 0.1)
        )
        
        pip_value = self.calculate_pip_value(signal['pair'], adjusted_lot_size)
        
        # Risk management
        risk_per_trade = self.balance * 0.02
        pip_risk = signal['sl']
        potential_loss = pip_risk * pip_value
        
        if potential_loss > risk_per_trade and pip_risk > 0:
            adjusted_lot_size = (risk_per_trade / (pip_risk * pip_value)) * adjusted_lot_size
            adjusted_lot_size = max(0.01, min(adjusted_lot_size, 0.2))  # Max 0.2 lot
            pip_value = self.calculate_pip_value(signal['pair'], adjusted_lot_size)
        
        # Check daily loss limit
        today = datetime.now().date()
        daily_loss = self.daily_loss_tracker.get(today, 0)
        if daily_loss <= -self.balance * Config.DAILY_LOSS_LIMIT:
            logger.warning(f"Daily loss limit reached for {today}, skipping trade")
            return False
        
        # Check daily trade limit
        timeframe_params = Config.TIMEFRAME_PARAMS.get(signal.get('timeframe', '4H'), {})
        max_daily_trades = timeframe_params.get('max_daily_trades', 3)
        
        if today not in self.daily_trade_count:
            self.daily_trade_count[today] = 0
            
        if self.daily_trade_count[today] >= max_daily_trades:
            logger.warning(f"Daily trade limit reached for {today}, skipping trade")
            return False
        
        if signal['action'].upper() == 'BUY':
            entry_price = current_price
            stop_loss = entry_price - signal['sl'] * 0.01
            take_profit = entry_price + signal['tp'] * 0.01
            direction = 1
        else:  # SELL
            entry_price = current_price
            stop_loss = entry_price + signal['sl'] * 0.01
            take_profit = entry_price - signal['tp'] * 0.01
            direction = -1
            
        position = {
            'entry_date': signal['date'],
            'pair': signal['pair'],
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'lot_size': adjusted_lot_size,
            'pip_value': pip_value,
            'status': 'open',
            'signal_confidence': signal.get('confidence', 50),
            'timeframe': signal.get('timeframe', '4H')
        }
        
        # Limit maximum open positions
        if len(self.positions) < 3:
            self.positions.append(position)
            self.daily_trade_count[today] += 1
            logger.info(f"Executed {signal['action']} on {signal['pair']} at {current_price}, Lot: {adjusted_lot_size}")
            return True
        else:
            logger.warning(f"Max positions reached, skipping trade on {signal['pair']}")
            return False
    
    def check_positions(self, current_prices):
        today = datetime.now().date()
        daily_profit = 0
        
        for position in self.positions[:]:
            if position['status'] == 'open':
                pair = position['pair']
                current_price = current_prices.get(pair)
                
                if current_price is None:
                    continue
                    
                if position['direction'] == 1:  # Buy
                    if current_price >= position['take_profit']:
                        pips = (position['take_profit'] - position['entry_price']) / 0.01
                        profit = pips * position['pip_value']
                        self.close_position(position, profit, 'TP')
                        daily_profit += profit
                    elif current_price <= position['stop_loss']:
                        pips = (position['stop_loss'] - position['entry_price']) / 0.01
                        profit = pips * position['pip_value']
                        self.close_position(position, profit, 'SL')
                        daily_profit += profit
                else:  # Sell
                    if current_price <= position['take_profit']:
                        pips = (position['entry_price'] - position['take_profit']) / 0.01
                        profit = pips * position['pip_value']
                        self.close_position(position, profit, 'TP')
                        daily_profit += profit
                    elif current_price >= position['stop_loss']:
                        pips = (position['entry_price'] - position['stop_loss']) / 0.01
                        profit = pips * position['pip_value']
                        self.close_position(position, profit, 'SL')
                        daily_profit += profit
        
        # Update daily loss tracker
        if today not in self.daily_loss_tracker:
            self.daily_loss_tracker[today] = 0
        self.daily_loss_tracker[today] += daily_profit
    
    def close_position(self, position, profit, close_reason):
        position['status'] = 'closed'
        position['close_reason'] = close_reason
        position['profit'] = profit
        position['close_date'] = datetime.now()
        
        self.balance += profit
        self.trade_history.append(position.copy())
        
        # Update consecutive losses tracker
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Remove from open positions
        self.positions = [p for p in self.positions if p['status'] == 'open']
        
        logger.info(f"Closed {position['pair']} position: {close_reason}, P&L: ${profit:.2f}")
    
    def run_backtest(self, signals, timeframe="4H"):
        logger.info("Starting enhanced backtesting...")
        
        # Reset state
        self.balance = self.initial_balance
        self.positions = []
        self.trade_history = []
        self.equity_curve = []
        self.daily_loss_tracker = {}
        self.consecutive_losses = 0
        self.daily_trade_count = {}
        
        # Sort signals by date
        signals.sort(key=lambda x: x['date'])
        
        if not signals:
            return {"error": "No signals provided for backtesting"}
            
        start_date = min([s['date'] for s in signals])
        end_date = max([s['date'] for s in signals])
        
        current_date = start_date
        trades_executed = 0
        
        while current_date <= end_date:
            # Check for maximum drawdown
            current_equity = self.balance + sum(
                p['pip_value'] * ((current_prices.get(p['pair'], p['entry_price']) - p['entry_price']) / 0.01 * p['direction']) 
                for p in self.positions if p['status'] == 'open'
            )
            
            drawdown = (current_equity - self.initial_balance) / self.initial_balance * 100
            if drawdown < -Config.MAX_DRAWDOWN_PCT * 100:
                logger.warning(f"Max drawdown reached at {current_date}, stopping backtest")
                break
            
            # Execute signals for current date
            daily_signals = [s for s in signals if s['date'].date() == current_date.date()]
            
            for signal in daily_signals:
                # Skip if too many consecutive losses
                if self.consecutive_losses >= 3:
                    logger.warning(f"3 consecutive losses reached, skipping signals for {current_date}")
                    break
                    
                pair = signal['pair']
                if pair in self.data and timeframe in self.data[pair]:
                    df_pair = self.data[pair][timeframe]
                    date_data = df_pair[df_pair['date'] == current_date]
                    if not date_data.empty:
                        current_price = date_data['open'].values[0]
                        if self.execute_trade(signal, current_price):
                            trades_executed += 1
            
            # Check open positions
            current_prices = {}
            for pair in self.pairs:
                if pair in self.data and timeframe in self.data[pair]:
                    df_pair = self.data[pair][timeframe]
                    date_data = df_pair[df_pair['date'] == current_date]
                    if not date_data.empty:
                        current_prices[pair] = date_data['close'].values[0]
            
            self.check_positions(current_prices)
            
            # Record equity
            self.equity_curve.append({
                'date': current_date,
                'balance': self.balance,
                'open_positions': len(self.positions),
                'drawdown': drawdown
            })
            
            current_date += timedelta(days=1)
        
        logger.info(f"Backtesting completed: {trades_executed} trades executed, final balance: ${self.balance:.2f}")
        return self.generate_enhanced_report()
    
    def generate_enhanced_report(self):
        if not self.trade_history:
            return {
                'status': 'error',
                'message': 'No trades executed during backtesting period'
            }
        
        try:
            df_trades = pd.DataFrame(self.trade_history)
            df_equity = pd.DataFrame(self.equity_curve)
            
            # Calculate performance metrics dengan error handling
            total_trades = len(df_trades)
            winning_trades = len(df_trades[df_trades['profit'] > 0]) if 'profit' in df_trades.columns else 0
            losing_trades = len(df_trades[df_trades['profit'] < 0]) if 'profit' in df_trades.columns else 0
            win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
            
            total_profit = df_trades['profit'].sum() if 'profit' in df_trades.columns else 0
            average_profit = df_trades['profit'].mean() if 'profit' in df_trades.columns and total_trades > 0 else 0
            average_win = df_trades[df_trades['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
            average_loss = df_trades[df_trades['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
            
            # Risk Reward Ratio dengan safe division
            avg_risk_reward = abs(average_win / average_loss) if average_loss != 0 else 0
            
            # Maximum drawdown
            if not df_equity.empty and 'balance' in df_equity.columns:
                df_equity['peak'] = df_equity['balance'].expanding().max()
                df_equity['drawdown'] = (df_equity['balance'] - df_equity['peak']) / df_equity['peak'] * 100
                max_drawdown = df_equity['drawdown'].min() if 'drawdown' in df_equity.columns else 0
            else:
                max_drawdown = 0
            
            # Profit Factor
            gross_profit = df_trades[df_trades['profit'] > 0]['profit'].sum() if winning_trades > 0 else 0
            gross_loss = abs(df_trades[df_trades['profit'] < 0]['profit'].sum()) if losing_trades > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            
            # Expectancy
            expectancy = (win_rate/100 * average_win) - ((100-win_rate)/100 * abs(average_loss))
            
            # Compile report
            report = {
                'status': 'success',
                'summary': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': round(win_rate, 2),
                    'total_profit': round(total_profit, 2),
                    'return_percentage': round(((self.balance - self.initial_balance) / self.initial_balance * 100), 2),
                    'average_profit': round(average_profit, 2),
                    'average_win': round(average_win, 2),
                    'average_loss': round(average_loss, 2),
                    'risk_reward_ratio': round(avg_risk_reward, 2),
                    'profit_factor': round(profit_factor, 2),
                    'expectancy': round(expectancy, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'final_balance': round(self.balance, 2),
                    'initial_balance': self.initial_balance,
                    'consecutive_losses': self.consecutive_losses
                },
                'performance_by_pair': {},
                'performance_by_timeframe': {},
                'trade_history': [
                    {
                        'entry_date': trade['entry_date'].strftime('%Y-%m-%d') if hasattr(trade['entry_date'], 'strftime') else str(trade['entry_date']),
                        'pair': trade['pair'],
                        'direction': 'BUY' if trade['direction'] == 1 else 'SELL',
                        'entry_price': round(trade['entry_price'], 4),
                        'profit': round(trade.get('profit', 0), 2),
                        'close_reason': trade.get('close_reason', 'Open'),
                        'confidence': trade.get('signal_confidence', 50),
                        'timeframe': trade.get('timeframe', '4H'),
                        'lot_size': trade.get('lot_size', 0.1)
                    }
                    for trade in self.trade_history[-20:]
                ]
            }
            
            # Enhanced performance by pair analysis
            for pair in self.pairs:
                pair_trades = df_trades[df_trades['pair'] == pair]
                if len(pair_trades) > 0:
                    pair_profit = pair_trades['profit'].sum() if 'profit' in pair_trades.columns else 0
                    pair_win_rate = len(pair_trades[pair_trades['profit'] > 0]) / len(pair_trades) * 100 if 'profit' in pair_trades.columns else 0
                    avg_confidence = pair_trades['signal_confidence'].mean() if 'signal_confidence' in pair_trades.columns else 50
                    
                    report['performance_by_pair'][pair] = {
                        'trades': len(pair_trades),
                        'profit': round(pair_profit, 2),
                        'win_rate': round(pair_win_rate, 2),
                        'avg_confidence': round(avg_confidence, 1),
                        'performance': 'EXCELLENT' if pair_win_rate > 60 and pair_profit > 0 else 'GOOD' if pair_win_rate > 50 and pair_profit > 0 else 'POOR'
                    }
            
            # Performance by timeframe analysis
            if 'timeframe' in df_trades.columns:
                for timeframe in df_trades['timeframe'].unique():
                    tf_trades = df_trades[df_trades['timeframe'] == timeframe]
                    if len(tf_trades) > 0:
                        tf_profit = tf_trades['profit'].sum() if 'profit' in tf_trades.columns else 0
                        tf_win_rate = len(tf_trades[tf_trades['profit'] > 0]) / len(tf_trades) * 100 if 'profit' in tf_trades.columns else 0
                        
                        report['performance_by_timeframe'][timeframe] = {
                            'trades': len(tf_trades),
                            'profit': round(tf_profit, 2),
                            'win_rate': round(tf_win_rate, 2),
                            'performance': 'GOOD' if tf_win_rate > 50 and tf_profit > 0 else 'POOR'
                        }
            
            # Add enhanced recommendations
            report['recommendations'] = self.generate_enhanced_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating enhanced report: {e}")
            traceback.print_exc()
            return {
                'status': 'error',
                'message': f'Error generating report: {str(e)}'
            }
    
    def generate_enhanced_recommendations(self, report):
        recommendations = []
        
        if report['status'] == 'error':
            return ["⚠️ Error in report generation - check logs for details"]
        
        summary = report.get('summary', {})
        win_rate = summary.get('win_rate', 0)
        profit_factor = summary.get('profit_factor', 0)
        max_drawdown = summary.get('max_drawdown', 0)
        consecutive_losses = summary.get('consecutive_losses', 0)
        
        # Win rate recommendations
        if win_rate < 40:
            recommendations.append("🎯 CRITICAL: Increase signal quality filters drastically - win rate too low")
        elif win_rate < 50:
            recommendations.append("⚠️ Increase signal quality filters - current win rate below 50%")
        elif win_rate > 65:
            recommendations.append("✅ EXCELLENT: Maintain current strategy - win rate optimal")
        elif win_rate > 55:
            recommendations.append("👍 GOOD: Current win rate acceptable")
        
        # Profit factor recommendations
        if profit_factor < 0.8:
            recommendations.append("⚡ CRITICAL: Improve risk-reward ratio immediately - heavy losses")
        elif profit_factor < 1.0:
            recommendations.append("⚠️ Improve risk-reward ratio - strategy not profitable")
        elif profit_factor > 1.5:
            recommendations.append("💰 EXCELLENT: Profit factor optimal - strategy profitable")
        elif profit_factor > 1.2:
            recommendations.append("👍 GOOD: Profit factor acceptable")
        
        # Drawdown recommendations
        if max_drawdown < -15:
            recommendations.append("🛡️ CRITICAL: Strengthen risk management immediately - dangerous drawdown")
        elif max_drawdown < -10:
            recommendations.append("⚠️ Strengthen risk management - drawdown too high")
        elif max_drawdown > -5:
            recommendations.append("📊 HEALTHY: Drawdown level acceptable")
        
        # Consecutive losses recommendations
        if consecutive_losses >= 5:
            recommendations.append("🚨 CRITICAL: Stop trading - review strategy completely")
        elif consecutive_losses >= 3:
            recommendations.append("⚠️ High consecutive losses - reduce position size significantly")
        
        # Pair-specific recommendations
        for pair, perf in report.get('performance_by_pair', {}).items():
            if perf['performance'] == 'POOR':
                recommendations.append(f"🔍 Review {pair} strategy completely - underperforming (Win Rate: {perf['win_rate']}%)")
            elif perf['performance'] == 'GOOD':
                recommendations.append(f"✅ {pair} performing well (Win Rate: {perf['win_rate']}%)")
            elif perf['performance'] == 'EXCELLENT':
                recommendations.append(f"🎯 {pair} performing excellently - consider increasing allocation")
        
        # Timeframe-specific recommendations
        for timeframe, perf in report.get('performance_by_timeframe', {}).items():
            if perf['performance'] == 'POOR':
                recommendations.append(f"⏰ Avoid {timeframe} timeframe - poor performance (Win Rate: {perf['win_rate']}%)")
            else:
                recommendations.append(f"✅ {timeframe} timeframe performing well")
        
        if not recommendations:
            recommendations.append("📈 Strategy performing well - continue with current parameters")
        
        return recommendations

# Initialize enhanced backtester
backtester = EnhancedForexBacktester(initial_balance=Config.INITIAL_BALANCE)

# ---------------- DATABASE FUNCTIONS ----------------
def init_db():
    """Initialize database with enhanced tables"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        
        # Drop and recreate tables to ensure clean schema
        c.execute('DROP TABLE IF EXISTS analysis_results')
        c.execute('DROP TABLE IF EXISTS backtesting_results')
        
        # Main analysis results table
        c.execute('''CREATE TABLE analysis_results (
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

        # Enhanced backtesting results table
        c.execute('''CREATE TABLE backtesting_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            pair TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            period_days INTEGER,
            total_trades INTEGER,
            winning_trades INTEGER,
            win_rate REAL,
            total_profit REAL,
            final_balance REAL,
            max_drawdown REAL,
            profit_factor REAL,
            risk_reward_ratio REAL,
            expectancy REAL,
            report_data TEXT,
            recommendations TEXT,
            strategy_version TEXT DEFAULT 'enhanced_v2'
        )''')
        
        conn.commit()
        logger.info("Enhanced database initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        traceback.print_exc()
    finally:
        conn.close()

def save_analysis_result(data: Dict):
    """Save analysis result to database"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        
        # Get confidence score safely
        confidence = data['ai_analysis'].get('CONFIDENCE_LEVEL', 50) if data.get('ai_analysis') else 50
        ai_provider = data.get('ai_provider', "Fallback")
        
        c.execute('''INSERT INTO analysis_results 
                    (pair, timeframe, current_price, technical_indicators, 
                     ai_analysis, fundamental_news, chart_data, data_source, 
                     confidence_score, ai_provider)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (data['pair'], data['timeframe'], data['current_price'],
                  json.dumps(data['technical_indicators']), 
                  json.dumps(data.get('ai_analysis', {})),
                  data.get('fundamental_news', ''),
                  json.dumps(data.get('chart_data', {})),
                  data.get('data_source', ''),
                  confidence,
                  ai_provider))
        
        conn.commit()
        logger.info(f"Analysis saved for {data['pair']}-{data['timeframe']}")
    except Exception as e:
        logger.error(f"Error saving analysis: {e}")
        traceback.print_exc()
    finally:
        conn.close()

def save_backtest_result(report_data: Dict):
    """Save enhanced backtesting result to database"""
    try:
        # Check if report has summary section
        if 'summary' not in report_data:
            logger.warning("No summary found in report data, skipping save")
            return
            
        summary = report_data['summary']
        
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        
        recommendations = json.dumps(report_data.get('recommendations', []))
        
        c.execute('''INSERT INTO backtesting_results 
                    (pair, timeframe, period_days, total_trades, winning_trades, 
                     win_rate, total_profit, final_balance, max_drawdown, profit_factor, 
                     risk_reward_ratio, expectancy, report_data, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (report_data.get('pair', 'MULTI'),
                  report_data.get('timeframe', '4H'),
                  report_data.get('period_days', 30),
                  summary.get('total_trades', 0),
                  summary.get('winning_trades', 0),
                  summary.get('win_rate', 0),
                  summary.get('total_profit', 0),
                  summary.get('final_balance', Config.INITIAL_BALANCE),
                  summary.get('max_drawdown', 0),
                  summary.get('profit_factor', 0),
                  summary.get('risk_reward_ratio', 0),
                  summary.get('expectancy', 0),
                  json.dumps(report_data),
                  recommendations))
        
        conn.commit()
        logger.info("Enhanced backtesting result saved to database")
    except Exception as e:
        logger.error(f"Error saving backtest result: {e}")
        traceback.print_exc()
    finally:
        conn.close()

# ---------------- ENHANCED DATA LOADING ----------------
def load_csv_data():
    """Load historical CSV data with enhanced error handling"""
    search_dirs = [".", "data", "historical_data"]
    loaded_count = 0
    
    for directory in search_dirs:
        if not os.path.exists(directory):
            logger.info(f"Directory {directory} does not exist, skipping")
            continue
            
        logger.info(f"Searching for CSV files in {directory}")
        for filename in os.listdir(directory):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                try:
                    logger.info(f"Loading CSV file: {file_path}")
                    
                    # Try different delimiters and encodings
                    df = None
                    delimiters = [',', '\t', ';', '|']
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                    
                    for encoding in encodings:
                        for delimiter in delimiters:
                            try:
                                df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
                                if df.shape[1] > 1:  # If we have more than 1 column, it's probably correct
                                    logger.info(f"Successfully loaded with delimiter '{delimiter}' and encoding '{encoding}'")
                                    break
                            except Exception as e:
                                continue
                        if df is not None and df.shape[1] > 1:
                            break
                    
                    # If still no success, try reading with no header and infer structure
                    if df is None or df.shape[1] <= 1:
                        logger.warning(f"Standard loading failed for {filename}, trying alternative methods")
                        try:
                            # Read the first few lines to understand structure
                            with open(file_path, 'r', encoding='utf-8') as f:
                                first_lines = [f.readline().strip() for _ in range(5)]
                            
                            # Check if it's tab-separated with combined columns
                            if any('\t' in line for line in first_lines):
                                # Try reading with tab separator and no header
                                df = pd.read_csv(file_path, delimiter='\t', header=None, encoding='utf-8')
                                logger.info("Loaded as tab-separated without header")
                                
                                # Assign column names based on number of columns
                                if df.shape[1] >= 5:
                                    df.columns = ['datetime', 'open', 'high', 'low', 'close'] + [f'extra_{i}' for i in range(5, df.shape[1])]
                                elif df.shape[1] >= 4:
                                    df.columns = ['datetime', 'open', 'high', 'low'] + [f'extra_{i}' for i in range(4, df.shape[1])]
                                    df['close'] = df['low']  # Use low as close if close is missing
                                else:
                                    logger.warning(f"Not enough columns in {filename}: {df.shape[1]}")
                                    continue
                            else:
                                logger.error(f"Cannot determine file structure for {filename}")
                                continue
                                
                        except Exception as e:
                            logger.error(f"Alternative loading also failed for {filename}: {e}")
                            continue
                    
                    # Standardize column names
                    df.columns = [str(col).lower().strip().replace(' ', '_').replace('.', '') for col in df.columns]
                    logger.info(f"Columns found: {list(df.columns)}")
                    
                    # Handle different column naming conventions
                    if 'close' not in df.columns:
                        price_found = False
                        for col in df.columns:
                            if col in ['close', 'price', 'last', 'value']:
                                df['close'] = df[col]
                                price_found = True
                                break
                        
                        if not price_found:
                            # Use the last numeric column as close price
                            for col in df.columns:
                                if pd.api.types.is_numeric_dtype(df[col]):
                                    df['close'] = df[col]
                                    price_found = True
                                    break
                        
                        if not price_found:
                            logger.warning(f"No suitable price column found in {filename}")
                            continue
                    
                    # Ensure we have open, high, low columns
                    if 'open' not in df.columns:
                        if any(col in df.columns for col in ['open', 'o']):
                            for col in ['open', 'o']:
                                if col in df.columns:
                                    df['open'] = df[col]
                                    break
                        else:
                            df['open'] = df['close']
                    
                    if 'high' not in df.columns:
                        if any(col in df.columns for col in ['high', 'h']):
                            for col in ['high', 'h']:
                                if col in df.columns:
                                    df['high'] = df[col]
                                    break
                        else:
                            df['high'] = df['close']
                    
                    if 'low' not in df.columns:
                        if any(col in df.columns for col in ['low', 'l']):
                            for col in ['low', 'l']:
                                if col in df.columns:
                                    df['low'] = df[col]
                                    break
                        else:
                            df['low'] = df['close']
                    
                    if 'volume' not in df.columns:
                        if 'vol' in df.columns:
                            df['volume'] = df['vol']
                        else:
                            df['volume'] = 10000
                    
                    # Parse date column - handle various date formats
                    date_column = None
                    for col in ['date', 'time', 'datetime', 'timestamp']:
                        if col in df.columns:
                            date_column = col
                            break
                    
                    if date_column:
                        # Clean the date column - remove any non-date characters
                        df[date_column] = df[date_column].astype(str).str.strip()
                        
                        # Try different date formats
                        date_formats = [
                            '%Y-%m-%d %H:%M:%S',
                            '%Y-%m-%d %H:%M',
                            '%Y-%m-%d',
                            '%d/%m/%Y %H:%M:%S', 
                            '%d/%m/%Y %H:%M',
                            '%d/%m/%Y',
                            '%m/%d/%Y %H:%M:%S',
                            '%m/%d/%Y %H:%M',
                            '%m/%d/%Y',
                            '%Y.%m.%d %H:%M:%S',
                            '%Y.%m.%d %H:%M', 
                            '%Y.%m.%d',
                            '%d-%m-%Y %H:%M:%S',
                            '%d-%m-%Y %H:%M',
                            '%d-%m-%Y',
                            '%Y%m%d %H:%M:%S',
                            '%Y%m%d %H:%M',
                            '%Y%m%d'
                        ]
                        
                        # Special handling for formats like "2009-09-15_08:00"
                        if df[date_column].str.contains('_').any():
                            df[date_column] = df[date_column].str.replace('_', ' ', regex=False)
                            date_formats.insert(0, '%Y-%m-%d %H:%M')
                        
                        parsed_dates = pd.Series([pd.NaT] * len(df))
                        
                        for fmt in date_formats:
                            try:
                                temp_dates = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
                                success_rate = temp_dates.notna().mean()
                                if success_rate > 0.8:  # If format works for most rows
                                    parsed_dates = temp_dates
                                    logger.info(f"Successfully parsed dates with format: {fmt}")
                                    break
                            except:
                                continue
                        
                        if parsed_dates.isna().all():
                            # Final fallback - let pandas infer
                            logger.warning(f"Using pandas to infer date format for {filename}")
                            parsed_dates = pd.to_datetime(df[date_column], errors='coerce')
                        
                        df[date_column] = parsed_dates
                        df = df.dropna(subset=[date_column])
                        
                        if len(df) == 0:
                            logger.warning(f"All dates failed to parse in {filename}")
                            continue
                            
                        # Sort by date ascending
                        df = df.sort_values(date_column)
                        df = df.reset_index(drop=True)
                    else:
                        # Create date index if no date column
                        logger.warning(f"No date column found in {filename}, creating date index")
                        start_date = datetime(2024, 1, 1)
                        if len(df) > 1000:
                            dates = pd.date_range(start=start_date, periods=len(df), freq='1H')
                        else:
                            dates = pd.date_range(start=start_date, periods=len(df), freq='1D')
                        df['date'] = dates
                    
                    # Extract pair and timeframe from filename
                    base_name = os.path.basename(filename).replace(".csv", "").upper()
                    parts = base_name.split("_")
                    pair = parts[0] if parts else "UNKNOWN"
                    
                    # Auto-detect timeframe based on data characteristics
                    if len(parts) > 1:
                        timeframe = parts[1]
                    else:
                        # Detect timeframe from data
                        if len(df) > 0 and 'date' in df.columns:
                            try:
                                time_diff = df['date'].diff().mean()
                                if time_diff <= pd.Timedelta(hours=1):
                                    timeframe = "1H"
                                elif time_diff <= pd.Timedelta(hours=4):
                                    timeframe = "4H"
                                elif time_diff <= pd.Timedelta(days=1):
                                    timeframe = "1D"
                                else:
                                    timeframe = "1W"
                            except:
                                timeframe = "1D"
                        else:
                            timeframe = "1D"
                    
                    if pair not in Config.SUPPORTED_PAIRS:
                        logger.warning(f"Unsupported pair {pair} in file {filename}")
                        continue
                    
                    # Clean data - remove any rows with invalid prices
                    numeric_cols = ['open', 'high', 'low', 'close']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    df = df.dropna(subset=['close'])
                    
                    if len(df) == 0:
                        logger.warning(f"No valid data rows in {filename} after cleaning")
                        continue
                    
                    # Initialize nested dictionaries
                    if pair not in HISTORICAL:
                        HISTORICAL[pair] = {}
                    
                    HISTORICAL[pair][timeframe] = df
                    loaded_count += 1
                    logger.info(f"✅ Loaded {pair}-{timeframe} from {file_path}, {len(df)} rows")
                    
                except Exception as e:
                    logger.error(f"⚠️ Error loading {file_path}: {e}")
                    traceback.print_exc()
    
    # Log summary of loaded data
    if HISTORICAL:
        for pair in HISTORICAL:
            for timeframe in HISTORICAL[pair]:
                data_points = len(HISTORICAL[pair][timeframe])
                logger.info(f"📊 {pair}-{timeframe}: {data_points} data points available")
    else:
        logger.warning("❌ No historical data loaded!")
    
    logger.info(f"Total loaded datasets: {loaded_count}")

    # Load data to backtester
    if HISTORICAL:
        backtester.load_historical_data(HISTORICAL)
        logger.info("✅ Historical data loaded to backtester")
    else:
        logger.error("No historical data available for backtester!")

# ---------------- SAMPLE DATA CREATION ----------------
def create_sample_data():
    """Create sample historical data if none exists"""
    try:
        # Create historical_data directory if not exists
        if not os.path.exists('historical_data'):
            os.makedirs('historical_data')
            logger.info("Created historical_data directory")
        
        # Generate sample data for all pairs and timeframes
        pairs = ['USDJPY', 'GBPJPY', 'EURJPY', 'CHFJPY']
        timeframes = ['1H', '4H', '1D']
        
        base_prices = {
            'USDJPY': 147.0,
            'GBPJPY': 198.0, 
            'EURJPY': 172.0,
            'CHFJPY': 184.0
        }
        
        volatility_multipliers = {
            '1H': 1.5,
            '4H': 2.0,  
            '1D': 3.0
        }
        
        data_points = {
            '1H': 24 * 120,  # 120 days
            '4H': 6 * 120,   # 120 days
            '1D': 365        # 1 year
        }
        
        files_created = 0
        
        for pair in pairs:
            for timeframe in timeframes:
                filename = f"historical_data/{pair}_{timeframe}.csv"
                
                # Skip if file already exists
                if os.path.exists(filename):
                    logger.info(f"File {filename} already exists, skipping")
                    continue
                
                logger.info(f"Creating sample data for {pair}-{timeframe}")
                
                # Generate dates based on timeframe
                start_date = datetime(2023, 1, 1)
                periods = data_points[timeframe]
                
                if timeframe == '1H':
                    date_range = [start_date + timedelta(hours=i) for i in range(periods)]
                elif timeframe == '4H':
                    date_range = [start_date + timedelta(hours=4*i) for i in range(periods)]
                else:  # 1D
                    date_range = [start_date + timedelta(days=i) for i in range(periods)]
                
                # Generate price data dengan realistic trends
                base_price = base_prices[pair]
                prices = []
                current_price = base_price
                volatility = volatility_multipliers[timeframe]
                
                # Add some realistic trends and patterns
                trend_periods = [50, 100, 200]  # Various trend lengths
                current_trend = 0
                trend_duration = 0
                trend_direction = 1
                
                for i in range(periods):
                    # Change trend occasionally
                    if trend_duration <= 0:
                        trend_direction = random.choice([-1, 1])
                        trend_duration = random.choice(trend_periods)
                        current_trend = trend_direction * random.uniform(0.1, 0.5) * volatility
                    else:
                        trend_duration -= 1
                    
                    open_price = current_price
                    trend_move = current_trend * (random.random() * 0.1)
                    random_move = (random.random() - 0.5) * 2 * volatility * 0.01
                    
                    close = open_price + trend_move + random_move
                    
                    # Ensure realistic high/low values
                    high = max(open_price, close) + abs(random_move) * 0.3
                    low = min(open_price, close) - abs(random_move) * 0.3
                    
                    prices.append({
                        'date': date_range[i],
                        'open': round(open_price, 4),
                        'high': round(high, 4),
                        'low': round(low, 4),
                        'close': round(close, 4),
                        'volume': int(10000 + random.random() * 10000)
                    })
                    
                    current_price = close
                
                # Create DataFrame and save
                df = pd.DataFrame(prices)
                df.to_csv(filename, index=False)
                files_created += 1
                logger.info(f"✅ Created {filename} with {len(df)} rows")
                
        logger.info(f"Sample data creation completed: {files_created} files created")
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        traceback.print_exc()

# ---------------- ENHANCED TECHNICAL INDICATORS CALCULATION ----------------
def calculate_ema_series(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd_series(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def calc_indicators(series: List[float], volumes: Optional[List[float]] = None) -> Dict:
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
    
    # Bollinger Bands
    bb_upper = sma20 + (close.rolling(20).std() * 2)
    bb_lower = sma20 - (close.rolling(20).std() * 2)
    bb_width = (bb_upper - bb_lower) / sma20 * 100
    
    # Support and Resistance
    recent_high = close.tail(20).max()
    recent_low = close.tail(20).min()
    
    # Trend strength
    trend_strength = abs(sma20.iloc[-1] - sma50.iloc[-1]) / cp * 100 if not pd.isna(sma20.iloc[-1]) and not pd.isna(sma50.iloc[-1]) else 0
    
    # ATR
    high = pd.Series([c * 1.001 for c in close])
    low = pd.Series([c * 0.999 for c in close])
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr = tr.rolling(14).mean()
    
    # Volume analysis (if available)
    volume_trend = 0
    if volumes is not None and len(volumes) > 0:
        volume_series = pd.Series(volumes)
        volume_ma = volume_series.rolling(20).mean()
        current_volume = volumes[-1] if volumes else 0
        avg_volume = volume_ma.iloc[-1] if not pd.isna(volume_ma.iloc[-1]) else current_volume
        volume_trend = (current_volume - avg_volume) / avg_volume * 100 if avg_volume > 0 else 0
    
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
        "Bollinger_Upper": round(bb_upper.iloc[-1], 4) if not pd.isna(bb_upper.iloc[-1]) else cp,
        "Bollinger_Lower": round(bb_lower.iloc[-1], 4) if not pd.isna(bb_lower.iloc[-1]) else cp,
        "Bollinger_Width": round(bb_width.iloc[-1], 4) if not pd.isna(bb_width.iloc[-1]) else 0,
        "Resistance": round(recent_high, 4),
        "Support": round(recent_low, 4),
        "Volatility": round(close.pct_change().std() * 100, 2) if len(close) > 1 else 0,
        "Trend_Strength": round(trend_strength, 2),
        "ATR": round(atr.iloc[-1], 4) if not pd.isna(atr.iloc[-1]) else 0,
        "Volume_Trend": round(volume_trend, 2) if volumes else 0,
        "Volume": volumes[-1] if volumes else 0
    }

# ---------------- ENHANCED SIGNAL GENERATION ----------------
def calculate_signal_confidence(signal: Dict, tech: Dict) -> float:
    confidence = 50
    
    # RSI confidence
    rsi = tech['RSI']
    if 30 <= rsi <= 40 and signal['action'] == 'BUY':
        confidence += 20
    elif 60 <= rsi <= 70 and signal['action'] == 'SELL':
        confidence += 20
    elif 25 <= rsi <= 75:
        confidence += 10
    
    # MACD confidence
    macd_strength = abs(tech['MACD_Histogram'])
    if macd_strength > 0.1:
        confidence += 15
    elif macd_strength > 0.05:
        confidence += 10
    elif macd_strength > 0.02:
        confidence += 5
    
    # Trend alignment confidence - LEBIH KETAT
    if (signal['action'] == 'BUY' and tech['SMA20'] > tech['SMA50'] and tech['SMA50'] > tech['EMA200']) or \
       (signal['action'] == 'SELL' and tech['SMA20'] < tech['SMA50'] and tech['SMA50'] < tech['EMA200']):
        confidence += 25  # Strong trend alignment
    elif (signal['action'] == 'BUY' and tech['SMA20'] > tech['SMA50']) or \
         (signal['action'] == 'SELL' and tech['SMA20'] < tech['SMA50']):
        confidence += 10  # Medium trend alignment
    
    # Risk-Reward Ratio scoring - LEBIH SELEKTIF
    rr_ratio = signal['tp'] / signal['sl'] if signal['sl'] > 0 else 1
    if rr_ratio >= 2.0:
        confidence += 15
    elif rr_ratio >= 1.5:
        confidence += 10
    elif rr_ratio < 1.0:
        confidence -= 10  # Penalize poor risk-reward
    
    # Volatility confidence
    volatility = tech.get('Volatility', 0)
    if 0.5 <= volatility <= 5.0:
        confidence += 10
    elif volatility > 8.0:
        confidence -= 10  # Too volatile
    
    # Volume confirmation (jika tersedia)
    volume_trend = tech.get('Volume_Trend', 0)
    if volume_trend > 20:  # Volume above average
        confidence += 5
    
    # Pair-specific confidence boost
    pair_boost = get_pair_specific_params(signal.get('pair', ''), 'confidence_boost') or 0
    confidence += pair_boost
    
    return min(95, max(15, confidence))  # Minimum confidence 15%

def generate_enhanced_signal(tech: Dict, current_price: float, pair: str, timeframe: str = "4H") -> Dict:
    rsi = tech['RSI']
    macd = tech['MACD']
    macd_signal = tech['MACD_Signal']
    macd_histogram = tech['MACD_Histogram']
    sma20 = tech['SMA20']
    sma50 = tech['SMA50']
    ema200 = tech['EMA200']
    volatility = tech.get('Volatility', 1.0)
    trend_strength = tech.get('Trend_Strength', 0)
    bb_width = tech.get('Bollinger_Width', 0)
    volume = tech.get('Volume', 0)
    
    # Get timeframe-specific parameters
    timeframe_params = Config.TIMEFRAME_PARAMS.get(timeframe, Config.TIMEFRAME_PARAMS["4H"])
    min_volatility = timeframe_params["min_volatility"]
    max_volatility = timeframe_params["max_volatility"]
    min_trend_strength = timeframe_params["min_trend_strength"]
    
    # Get pair-specific parameters
    pair_params = Config.PAIR_SPECIFIC_PARAMS.get(pair, {})
    pair_min_volatility = pair_params.get('min_volatility', min_volatility)
    pair_max_volatility = pair_params.get('max_volatility', max_volatility)
    
    trend_bullish = sma20 > sma50 and sma50 > ema200 and current_price > sma20
    trend_bearish = sma20 < sma50 and sma50 < ema200 and current_price < sma20
    
    # Dynamic SL/TP based on volatility and ATR - lebih konservatif
    atr = tech.get('ATR', 0.5)
    
    # Adjust SL/TP berdasarkan timeframe dan pair
    volatility_multiplier = pair_params.get('volatility_multiplier', 1.0)
    
    if timeframe == "1D":
        base_sl = max(20, min(60, atr * 100 * 1.5 * volatility_multiplier))
        base_tp = base_sl * 2.0  # Risk-reward 1:2
    elif timeframe == "1W":
        base_sl = max(30, min(80, atr * 100 * 2.0 * volatility_multiplier))
        base_tp = base_sl * 2.5
    else:
        base_sl = max(15, min(40, atr * 100 * 1.2 * volatility_multiplier))
        base_tp = base_sl * 1.8
    
    # Adjust for volatility - lebih moderat
    volatility_multiplier = max(0.5, min(2.0, volatility / 2.0))
    
    sl_pips = int(base_sl * volatility_multiplier)
    tp_pips = int(base_tp * volatility_multiplier)
    
    # FILTER KRITIS: Hindari trading saat kondisi tidak optimal
    # Filter 1: Market terlalu datar
    if volatility < pair_min_volatility:
        return {'action': 'HOLD', 'tp': 0, 'sl': 0, 'strength': 'WEAK', 'reason': 'Low volatility'}
    
    # Filter 2: Market terlalu volatile
    if volatility > pair_max_volatility:
        return {'action': 'HOLD', 'tp': 0, 'sl': 0, 'strength': 'WEAK', 'reason': 'High volatility'}
    
    # Filter 3: Trend terlalu lemah
    if trend_strength < min_trend_strength:
        return {'action': 'HOLD', 'tp': 0, 'sl': 0, 'strength': 'WEAK', 'reason': 'Weak trend'}
    
    # Filter 4: Bollinger Band terlalu sempit (market compressed)
    if bb_width < 0.5:
        return {'action': 'HOLD', 'tp': 0, 'sl': 0, 'strength': 'WEAK', 'reason': 'Compressed market'}
    
    # Filter 5: Volume terlalu rendah (jika data tersedia)
    if volume > 0 and volume < 5000:
        return {'action': 'HOLD', 'tp': 0, 'sl': 0, 'strength': 'WEAK', 'reason': 'Low volume'}
    
    # STRONG BUY CONDITIONS (lebih selektif)
    strong_buy_conditions = (
        rsi < 35 and
        macd > macd_signal and 
        macd_histogram > 0.02 and
        trend_bullish and
        current_price > tech.get('Bollinger_Lower', current_price * 0.99) and
        current_price < sma20 and  # Price pullback to SMA20
        trend_strength > min_trend_strength * 2 and
        pair_min_volatility <= volatility <= pair_max_volatility
    )
    
    # STRONG SELL CONDITIONS (lebih selektif)
    strong_sell_conditions = (
        rsi > 65 and
        macd < macd_signal and 
        macd_histogram < -0.02 and
        trend_bearish and
        current_price < tech.get('Bollinger_Upper', current_price * 1.01) and
        current_price > sma20 and  # Price pullback to SMA20
        trend_strength > min_trend_strength * 2 and
        pair_min_volatility <= volatility <= pair_max_volatility
    )
    
    # MODERATE BUY CONDITIONS
    moderate_buy_conditions = (
        rsi < 45 and
        macd > macd_signal and
        current_price > sma20 and
        trend_strength > min_trend_strength
    )
    
    # MODERATE SELL CONDITIONS
    moderate_sell_conditions = (
        rsi > 55 and
        macd < macd_signal and
        current_price < sma20 and
        trend_strength > min_trend_strength
    )
    
    if strong_buy_conditions:
        return {
            'action': 'BUY',
            'tp': tp_pips,
            'sl': sl_pips,
            'strength': 'STRONG',
            'reason': 'Strong bullish momentum with RSI oversold'
        }
    elif strong_sell_conditions:
        return {
            'action': 'SELL', 
            'tp': tp_pips,
            'sl': sl_pips,
            'strength': 'STRONG',
            'reason': 'Strong bearish momentum with RSI overbought'
        }
    elif moderate_buy_conditions:
        return {
            'action': 'BUY',
            'tp': int(tp_pips * 0.8),
            'sl': int(sl_pips * 1.2),
            'strength': 'MODERATE',
            'reason': 'Moderate bullish conditions'
        }
    elif moderate_sell_conditions:
        return {
            'action': 'SELL',
            'tp': int(tp_pips * 0.8),
            'sl': int(sl_pips * 1.2),
            'strength': 'MODERATE',
            'reason': 'Moderate bearish conditions'
        }
    else:
        return {
            'action': 'HOLD',
            'tp': 0,
            'sl': 0,
            'strength': 'WEAK',
            'reason': 'No clear trading signal'
        }

def generate_backtest_signals_from_analysis(pair: str, timeframe: str, days: int = 30) -> List[Dict]:
    signals = []
    
    try:
        # Get timeframe-specific parameters
        timeframe_params = Config.TIMEFRAME_PARAMS.get(timeframe, Config.TIMEFRAME_PARAMS["4H"])
        min_volatility = timeframe_params["min_volatility"]
        max_volatility = timeframe_params["max_volatility"]
        min_trend_strength = timeframe_params["min_trend_strength"]
        confidence_threshold = timeframe_params["confidence_threshold"]
        start_index = timeframe_params["start_index"]
        max_daily_trades = timeframe_params["max_daily_trades"]
        
        # Get pair-specific parameters
        pair_params = Config.PAIR_SPECIFIC_PARAMS.get(pair, {})
        pair_optimal_hours = pair_params.get('optimal_hours', list(range(0, 24)))
        
        if pair not in HISTORICAL or timeframe not in HISTORICAL[pair]:
            logger.error(f"No historical data found for {pair}-{timeframe}")
            return signals
        
        # Adjust data points based on timeframe
        data_multiplier = 2 if timeframe in ["1H", "4H"] else 1
        df = HISTORICAL[pair][timeframe].tail(days * data_multiplier)
        
        # Minimum data points
        min_data_points = timeframe_params["required_bars"]
        if len(df) < min_data_points:
            logger.warning(f"Insufficient data for {pair}-{timeframe}: {len(df)} points, needed {min_data_points}")
            df = HISTORICAL[pair][timeframe]
            if len(df) < 50:  # Absolute minimum
                logger.error(f"Absolutely insufficient data for {pair}-{timeframe}: {len(df)} points")
                return signals
        
        signal_count = 0
        skip_count = 0
        daily_signal_count = {}
        
        logger.info(f"Enhanced signal generation for {pair}-{timeframe}:")
        logger.info(f"  - Volatility range: {min_volatility}-{max_volatility}%")
        logger.info(f"  - Min trend strength: {min_trend_strength}%")
        logger.info(f"  - Confidence threshold: {confidence_threshold}%")
        logger.info(f"  - Max daily trades: {max_daily_trades}")
        logger.info(f"  - Optimal hours: {pair_optimal_hours}")
        
        for i in range(start_index, len(df)):
            try:
                current_data = df.iloc[:i+1]
                current_date = current_data.iloc[-1]['date']
                current_price = current_data.iloc[-1]['close']
                
                # Session time filtering - gunakan pair-specific optimal hours
                hour = current_date.hour if hasattr(current_date, 'hour') else 12
                if not is_optimal_session(pair, hour):
                    skip_count += 1
                    continue
                
                # Daily trade limit check
                date_key = current_date.date()
                if date_key not in daily_signal_count:
                    daily_signal_count[date_key] = 0
                
                if daily_signal_count[date_key] >= max_daily_trades:
                    skip_count += 1
                    continue
                    
                # Calculate technical indicators
                closes = current_data['close'].tolist()
                volumes = current_data['volume'].fillna(0).tolist() if 'volume' in current_data.columns else None
                
                tech_indicators = calc_indicators(closes, volumes)
                
                # Filter berdasarkan volatility
                volatility = tech_indicators.get('Volatility', 0)
                if volatility < min_volatility or volatility > max_volatility:
                    skip_count += 1
                    continue
                    
                # Filter berdasarkan trend strength
                trend_strength = tech_indicators.get('Trend_Strength', 0)
                if trend_strength < min_trend_strength:
                    skip_count += 1
                    continue
                
                signal = generate_enhanced_signal(tech_indicators, current_price, pair, timeframe)
                
                if signal['action'] != 'HOLD':
                    confidence = calculate_signal_confidence(signal, tech_indicators)
                    
                    if confidence >= confidence_threshold:
                        signals.append({
                            'date': current_date,
                            'pair': pair,
                            'action': signal['action'],
                            'tp': signal['tp'],
                            'sl': signal['sl'],
                            'lot_size': Config.DEFAULT_LOT_SIZE,
                            'confidence': confidence,
                            'strength': signal['strength'],
                            'volatility': volatility,
                            'trend_strength': trend_strength,
                            'timeframe': timeframe,
                            'signal_reason': signal.get('reason', 'N/A')
                        })
                        signal_count += 1
                        daily_signal_count[date_key] += 1
                        logger.info(f"✅ Signal: {signal['action']} {pair} at {current_price}, "
                                   f"Confidence: {confidence}%, TP/SL: {signal['tp']}/{signal['sl']}, "
                                   f"Reason: {signal.get('reason', 'N/A')}")
                    else:
                        logger.debug(f"❌ Low confidence: {signal['action']} {pair}, Confidence: {confidence}%")
            except Exception as e:
                logger.error(f"Error processing data point {i} for {pair}-{timeframe}: {e}")
                continue
        
        logger.info(f"Generated {signal_count} quality signals for {pair}-{timeframe} ({skip_count} points skipped)")
        
        # Jika tidak ada sinyal, buat sinyal sample untuk testing (hanya di development)
        if signal_count == 0 and os.environ.get('DEBUG', 'False') == 'True':
            logger.warning("No signals generated, creating sample signals for testing")
            # Create minimal sample signals for testing
            sample_dates = df['date'].tail(5).tolist()
            for sample_date in sample_dates:
                action = random.choice(['BUY', 'SELL'])
                signals.append({
                    'date': sample_date,
                    'pair': pair,
                    'action': action,
                    'tp': random.randint(25, 50),
                    'sl': random.randint(15, 25),
                    'lot_size': Config.DEFAULT_LOT_SIZE * 0.5,  # Smaller lot for samples
                    'confidence': random.randint(40, 60),
                    'strength': 'MODERATE',
                    'volatility': 1.5,
                    'trend_strength': 0.5,
                    'timeframe': timeframe,
                    'signal_reason': 'Sample signal for testing'
                })
        
        return signals
        
    except Exception as e:
        logger.error(f"Error generating backtest signals: {e}")
        traceback.print_exc()
        return signals

# ---------------- DATA PROVIDERS & AI ANALYSIS ----------------
def get_price_twelvedata(pair: str) -> Optional[float]:
    """Get real-time price from Twelve Data API"""
    try:
        if TWELVE_API_KEY == "demo":
            logger.info(f"Using demo API key, skipping TwelveData API call for {pair}")
            return None
            
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
            logger.warning(f"TwelveData API error for {pair}: {data.get('message', 'Using fallback')}")
            return None
            
    except Exception as e:
        logger.warning(f"TwelveData unavailable for {pair}, using fallback: {e}")
        return None

def get_fundamental_news(pair: str = "USDJPY") -> str:
    """Get fundamental news dengan pair-specific content"""
    try:
        # Pair-specific news items
        news_templates = {
            "USDJPY": [
                f"USD/JPY: Bank of Japan policy meeting in focus. Current yield differential {random.randint(130, 160)}bps.",
                f"USD/JPY: US Treasury yields and BOJ yield curve control under scrutiny.",
                f"USD/JPY: Monitoring Fed vs BOJ policy divergence for direction clues.",
                f"USD/JPY: Technical patterns suggest {random.choice(['bullish', 'bearish'])} bias in near term."
            ],
            "GBPJPY": [
                f"GBP/JPY: Bank of England interest rate expectations driving volatility.",
                f"GBP/JPY: Risk sentiment and carry trade flows impacting price action.",
                f"GBP/JPY: UK economic data and BOJ policy key drivers for cross.",
                f"GBP/JPY: Technical analysis shows {random.choice(['consolidation', 'breakout'])} pattern forming."
            ],
            "EURJPY": [
                f"EUR/JPY: ECB policy outlook and Japanese inflation data in focus.",
                f"EUR/JPY: European economic indicators and risk appetite driving moves.",
                f"EUR/JPY: Monitoring Eurozone GDP and Japan trade balance for direction.",
                f"EUR/JPY: Market positioning suggests {random.choice(['long', 'short'])} bias building."
            ],
            "CHFJPY": [
                f"CHF/JPY: Swiss National Bank interventions and safe-haven flows key factors.",
                f"CHF/JPY: Risk-off sentiment typically benefits CHF against JPY.",
                f"CHF/JPY: Monitoring SNB currency reserves and Japan current account.",
                f"CHF/JPY: Technical setup indicates {random.choice(['range-bound', 'trending'])} conditions."
            ]
        }
        
        pair_news = news_templates.get(pair, news_templates["USDJPY"])
        return random.choice(pair_news)
    except Exception as e:
        logger.error(f"News error: {e}")
        return f"Market analysis ongoing for {pair}. Monitor economic calendar for updates."

def ai_fallback(tech: Dict, news_summary: str = "", pair: str = "USDJPY") -> Dict:
    """Enhanced fallback AI analysis dengan pair-specific logic"""
    cp = tech["current_price"]
    rsi = tech["RSI"]
    macd = tech["MACD"]
    macd_signal = tech["MACD_Signal"]
    volatility = tech.get('Volatility', 1.0)
    
    # Pair-specific analysis
    pair_analysis = {
        "USDJPY": "USD/JPY sensitive to US-Japan yield differential and risk sentiment",
        "GBPJPY": "GBP/JPY influenced by BOE policy and carry trade flows", 
        "EURJPY": "EUR/JPY driven by ECB policy and Eurozone economic data",
        "CHFJPY": "CHF/JPY affected by SNB policy and safe-haven flows"
    }
    
    # Multi-factor signal determination
    signal_score = 0
    
    # RSI factor
    if rsi < 30:
        signal_score += 3
    elif rsi < 40:
        signal_score += 2
    elif rsi > 70:
        signal_score -= 3
    elif rsi > 60:
        signal_score -= 2
    elif 40 < rsi < 60:
        signal_score += 1
    
    # MACD factor
    if macd > macd_signal:
        signal_score += 2
    else:
        signal_score -= 2
    
    # Price position factor
    if cp > tech['SMA20'] and tech['SMA20'] > tech['SMA50']:
        signal_score += 2
    elif cp < tech['SMA20'] and tech['SMA20'] < tech['SMA50']:
        signal_score -= 2
    
    # Volatility factor
    if 0.5 <= volatility <= 3.0:
        signal_score += 1
    elif volatility > 5.0:
        signal_score -= 1
    
    # Determine signal dengan threshold yang lebih tinggi
    if signal_score >= 4:
        signal = "BELI KUAT"
        confidence = 75
        sl = cp * 0.99
        tp1 = cp * 1.015
        tp2 = cp * 1.03
        advice = f"Konfirmasi teknis kuat untuk {pair}. RSI oversold, momentum bullish jelas."
    elif signal_score >= 2:
        signal = "BELI"
        confidence = 60
        sl = cp * 0.992
        tp1 = cp * 1.01
        tp2 = cp * 1.02
        advice = f"Kondisi teknis mendukung {pair}. Tunggu konfirmasi breakout resistance."
    elif signal_score <= -4:
        signal = "JUAL KUAT"
        confidence = 75
        sl = cp * 1.01
        tp1 = cp * 0.985
        tp2 = cp * 0.97
        advice = f"Tekanan jual kuat pada {pair}. RSI overbought, momentum bearish dominan."
    elif signal_score <= -2:
        signal = "JUAL"
        confidence = 60
        sl = cp * 1.008
        tp1 = cp * 0.99
        tp2 = cp * 0.98
        advice = f"Tekanan jual meningkat untuk {pair}. Wait for breakdown below support."
    else:
        signal = "TUNGGU"
        confidence = 50
        sl = cp * 0.995
        tp1 = cp * 1.005
        tp2 = cp * 1.01
        advice = f"{pair} dalam konsolidasi. {pair_analysis.get(pair, 'Tunggu breakout yang jelas.')}"
    
    return {
        "SIGNAL": signal,
        "ENTRY_PRICE": round(cp, 4),
        "STOP_LOSS": round(sl, 4),
        "TAKE_PROFIT_1": round(tp1, 4),
        "TAKE_PROFIT_2": round(tp2, 4),
        "CONFIDENCE_LEVEL": confidence,
        "TRADING_ADVICE": advice,
        "RISK_LEVEL": "RENDAH" if confidence < 60 else "SEDANG" if confidence < 75 else "TINGGI",
        "EXPECTED_MOVEMENT": f"{abs(round((tp1-cp)/cp*100, 2))}%",
        "AI_PROVIDER": "Enhanced Fallback System",
        "PAIR_ANALYSIS": pair_analysis.get(pair, "")
    }

def ai_deepseek_analysis(pair: str, tech: Dict, fundamentals: str) -> Dict:
    if not DEEPSEEK_API_KEY:
        return ai_fallback(tech, fundamentals, pair)
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
Analyze {pair} Forex trading opportunity with enhanced technical data:

Technical Indicators:
- Current Price: {tech['current_price']}
- RSI: {tech['RSI']} 
- MACD: {tech['MACD']} (Signal: {tech['MACD_Signal']})
- Trend Strength: {tech['Trend_Strength']}%
- Volatility: {tech['Volatility']}%
- Support: {tech['Support']}, Resistance: {tech['Resistance']}
- SMA20: {tech['SMA20']}, SMA50: {tech['SMA50']}

Fundamental Context: {fundamentals}

Provide comprehensive JSON analysis with:
- SIGNAL (BUY/SELL/HOLD)
- ENTRY_PRICE
- STOP_LOSS  
- TAKE_PROFIT_1
- TAKE_PROFIT_2
- CONFIDENCE_LEVEL (0-100)
- TRADING_ADVICE (detailed reasoning)
- RISK_LEVEL (LOW/MEDIUM/HIGH)
- EXPECTED_MOVEMENT (percentage)
- KEY_LEVELS_TO_WATCH

Focus on risk management and realistic profit targets.
"""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "choices" not in data or not data["choices"]:
            return ai_fallback(tech, fundamentals, pair)
        
        ai_response = data["choices"][0]["message"]["content"]
        
        # Clean and parse response
        ai_response = ai_response.strip()
        if ai_response.startswith("```json"):
            ai_response = ai_response[7:]
        if ai_response.endswith("```"):
            ai_response = ai_response[:-3]
        
        analysis_result = json.loads(ai_response)
        analysis_result["AI_PROVIDER"] = "DeepSeek AI"
        return analysis_result
            
    except Exception as e:
        logger.error(f"DeepSeek error: {e}")
        return ai_fallback(tech, fundamentals, pair)

# ---------------- ENHANCED ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html', 
                         pairs=Config.SUPPORTED_PAIRS,
                         timeframes=Config.SUPPORTED_TIMEFRAMES,
                         ai_available=bool(DEEPSEEK_API_KEY))

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/get_analysis')
def get_analysis():
    start_time = datetime.now()
    
    try:
        pair = request.args.get("pair", "USDJPY").upper()
        timeframe = request.args.get("timeframe", "4H").upper()
        use_history = request.args.get("use_history", "0") == "1"
        
        logger.info(f"Enhanced analysis request: {pair}-{timeframe}")
        
        # Validate inputs
        if pair not in Config.SUPPORTED_PAIRS:
            return jsonify({"error": f"Unsupported pair: {pair}"}), 400
        
        if timeframe not in Config.SUPPORTED_TIMEFRAMES:
            return jsonify({"error": f"Unsupported timeframe: {timeframe}"}), 400
        
        # Get price data with enhanced fallback
        current_price = get_price_twelvedata(pair)
        data_source = "Twelve Data"
        
        if current_price is None:
            # Fallback to historical data
            if pair in HISTORICAL and timeframe in HISTORICAL[pair]:
                current_price = float(HISTORICAL[pair][timeframe].tail(1)["close"].iloc[0])
                data_source = "Historical CSV"
                logger.info(f"Using historical price for {pair}: {current_price}")
            else:
                # Enhanced final fallback - menggunakan base price yang realistic
                base_prices = {"USDJPY": 147.13, "GBPJPY": 198.29, "EURJPY": 172.56, "CHFJPY": 184.41}
                base_price = base_prices.get(pair, 150.0)
                current_price = base_price + random.uniform(-0.5, 0.5)
                data_source = "Synthetic"
                logger.info(f"Using synthetic price for {pair}: {current_price}")
        
        # Determine required bars based on timeframe
        required_bars = Config.DATA_PERIODS.get(timeframe, 100)
        
        # Get historical data for indicators
        if use_history and pair in HISTORICAL and timeframe in HISTORICAL[pair]:
            df = HISTORICAL[pair][timeframe].tail(required_bars)
            
            if len(df) < required_bars:
                logger.warning(f"Insufficient data for {pair}-{timeframe}: {len(df)} bars, needed {required_bars}")
                df = HISTORICAL[pair][timeframe]
            
            closes = df["close"].tolist()
            volumes = df["volume"].fillna(0).tolist() if "volume" in df.columns else None
            
            # Format dates
            if timeframe == '1D':
                dates = df["date"].dt.strftime("%d/%m/%Y").tolist()
            elif timeframe == '1W':
                dates = df["date"].dt.strftime("%d/%m/%y").tolist()
            else:
                dates = df["date"].dt.strftime("%d/%m %H:%M").tolist()
            
            # Calculate technical series
            close_series = pd.Series(closes)
            ema200_series = calculate_ema_series(close_series, 200)
            ema200_values = [round(x, 4) for x in ema200_series.tolist()]
            
            macd_line, macd_signal, macd_histogram = calculate_macd_series(close_series)
            macd_line_values = [round(x, 4) for x in macd_line.tolist()]
            macd_signal_values = [round(x, 4) for x in macd_signal.tolist()]
            macd_histogram_values = [round(x, 4) for x in macd_histogram.tolist()]
            
        else:
            # Generate synthetic data dengan lebih realistic patterns
            num_points = required_bars
            base_price = current_price
            
            closes = []
            for i in range(num_points):
                # More realistic price movement dengan mean reversion
                if i == 0:
                    closes.append(base_price)
                else:
                    prev_price = closes[-1]
                    # Random walk dengan sedikit mean reversion
                    change = random.uniform(-0.002, 0.002) * prev_price
                    new_price = prev_price + change
                    closes.append(new_price)
            
            volumes = [random.randint(8000, 15000) for _ in range(num_points)]
            
            # Generate dates
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
            else:
                dates = [(base_date - timedelta(hours=i)).strftime("%d/%m %H:%M") 
                        for i in range(num_points-1, -1, -1)]
            
            # Calculate technical series for synthetic data
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
        
        # Prepare enhanced response
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
            "ai_provider": ai_analysis.get("AI_PROVIDER", "Enhanced Fallback System"),
            "data_points": len(dates),
            "pair_priority": Config.PAIR_PRIORITY.get(pair, 5),
            "optimal_hours": Config.PAIR_SPECIFIC_PARAMS.get(pair, {}).get('optimal_hours', [])
        }
        
        # Save to database
        save_analysis_result(response_data)
        
        logger.info(f"Enhanced analysis completed for {pair}-{timeframe} in {response_data['processing_time']}s")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/run_backtest', methods=['POST'])
def api_run_backtest():
    try:
        data = request.get_json()
        pair = data.get('pair', 'USDJPY')
        timeframe = data.get('timeframe', '4H')
        days = data.get('days', 30)
        
        logger.info(f"Enhanced backtest request: {pair}-{timeframe} for {days} days")
        
        # Validate inputs
        if pair not in Config.SUPPORTED_PAIRS:
            return jsonify({'error': f'Unsupported pair: {pair}'}), 400
        
        if timeframe not in Config.SUPPORTED_TIMEFRAMES:
            return jsonify({'error': f'Unsupported timeframe: {timeframe}'}), 400
        
        # Check if historical data exists
        if pair not in HISTORICAL or timeframe not in HISTORICAL[pair]:
            available_data = []
            for p in HISTORICAL:
                for tf in HISTORICAL[p]:
                    available_data.append(f"{p}-{tf}")
            
            return jsonify({
                'error': f'No historical data found for {pair}-{timeframe}',
                'available_data': available_data,
                'suggestion': 'Please ensure CSV files exist in historical_data folder'
            }), 400
        
        # Generate enhanced signals untuk backtesting
        signals = generate_backtest_signals_from_analysis(pair, timeframe, days)
        
        if not signals:
            return jsonify({
                'error': 'No signals generated for backtesting',
                'reason': 'Insufficient data or no trading conditions met',
                'data_points': len(HISTORICAL[pair][timeframe]) if pair in HISTORICAL and timeframe in HISTORICAL[pair] else 0,
                'suggestion': 'Try different timeframe or adjust signal parameters',
                'debug_info': {
                    'timeframe_params': Config.TIMEFRAME_PARAMS.get(timeframe, {}),
                    'pair_params': Config.PAIR_SPECIFIC_PARAMS.get(pair, {})
                }
            }), 400
        
        # Run enhanced backtest
        report = backtester.run_backtest(signals, timeframe)
        
        # Add enhanced metadata to report
        report['metadata'] = {
            'pair': pair,
            'timeframe': timeframe,
            'period_days': days,
            'signals_generated': len(signals),
            'initial_balance': Config.INITIAL_BALANCE,
            'strategy': 'Enhanced Multi-Filter Strategy v2',
            'strategy_features': [
                'Pair-specific optimal session filtering',
                'Dynamic position sizing based on confidence',
                'Enhanced volatility and trend filters',
                'Daily trade limits',
                'Consecutive loss protection'
            ],
            'parameters_used': {
                'timeframe': Config.TIMEFRAME_PARAMS.get(timeframe, {}),
                'pair_specific': Config.PAIR_SPECIFIC_PARAMS.get(pair, {})
            }
        }
        
        # Save to database
        save_backtest_result(report)
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"Enhanced backtesting error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Backtesting failed: {str(e)}'}), 500

@app.route('/api/backtest_status')
def api_backtest_status():
    status = {
        'historical_data_loaded': len(HISTORICAL) > 0,
        'available_pairs': list(HISTORICAL.keys()),
        'initial_balance': Config.INITIAL_BALANCE,
        'supported_timeframes': Config.SUPPORTED_TIMEFRAMES,
        'enhanced_features': {
            'pair_priority': Config.PAIR_PRIORITY,
            'pair_specific_params': Config.PAIR_SPECIFIC_PARAMS,
            'timeframe_specific_params': Config.TIMEFRAME_PARAMS,
            'max_positions': 3,
            'max_drawdown': f"{Config.MAX_DRAWDOWN_PCT * 100}%",
            'daily_loss_limit': f"{Config.DAILY_LOSS_LIMIT * 100}%",
            'dynamic_position_sizing': True,
            'session_filtering': True
        }
    }
    
    # Add data points info
    for pair in HISTORICAL:
        status[f'{pair}_data'] = {}
        for tf in HISTORICAL[pair]:
            status[f'{pair}_data'][tf] = len(HISTORICAL[pair][tf])
    
    return jsonify(status)

@app.route('/api/backtest_history')
def api_backtest_history():
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        c = conn.cursor()
        
        c.execute('''SELECT pair, timeframe, period_days, total_trades, winning_trades, 
                    win_rate, total_profit, final_balance, max_drawdown, profit_factor, timestamp
                    FROM backtesting_results 
                    ORDER BY timestamp DESC LIMIT 10''')
        
        results = c.fetchall()
        history = []
        
        for row in results:
            history.append({
                'pair': row[0],
                'timeframe': row[1],
                'period_days': row[2],
                'total_trades': row[3],
                'winning_trades': row[4],
                'win_rate': row[5],
                'total_profit': row[6],
                'final_balance': row[7],
                'max_drawdown': row[8],
                'profit_factor': row[9],
                'timestamp': row[10]
            })
        
        conn.close()
        return jsonify(history)
        
    except Exception as e:
        logger.error(f"Error fetching backtest history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/quick_overview')
def quick_overview():
    overview = {}
    
    for pair in Config.SUPPORTED_PAIRS:
        try:
            price = get_price_twelvedata(pair)
            source = "Twelve Data"
            
            if price is None and pair in HISTORICAL and "1D" in HISTORICAL[pair]:
                price = float(HISTORICAL[pair]["1D"].tail(1)["close"].iloc[0])
                source = "Historical"
            elif price is None:
                base_prices = {"USDJPY": 147.13, "GBPJPY": 198.29, "EURJPY": 172.56, "CHFJPY": 184.41}
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
            
            # Get pair-specific info
            pair_params = Config.PAIR_SPECIFIC_PARAMS.get(pair, {})
            
            overview[pair] = {
                "price": round(price, 4),
                "change": round(change_pct, 2),
                "source": source,
                "trend": "up" if change_pct > 0 else "down" if change_pct < 0 else "flat",
                "priority": Config.PAIR_PRIORITY.get(pair, 5),
                "optimal_hours": pair_params.get('optimal_hours', []),
                "volatility_range": f"{pair_params.get('min_volatility', 0.3)}-{pair_params.get('max_volatility', 8.0)}%"
            }
            
        except Exception as e:
            logger.error(f"Quick overview error for {pair}: {e}")
            overview[pair] = {
                "price": None,
                "change": 0,
                "source": "error",
                "trend": "flat",
                "priority": 5,
                "optimal_hours": [],
                "volatility_range": "N/A"
            }
    
    return jsonify(overview)

@app.route('/ai_status')
def ai_status():
    status = {
        "ai_available": bool(DEEPSEEK_API_KEY),
        "ai_provider": "DeepSeek",
        "fallback_ready": True,
        "enhanced_fallback": True,
        "pair_specific_analysis": True,
        "message": "AI DeepSeek siap digunakan" if DEEPSEEK_API_KEY else "AI DeepSeek tidak tersedia, menggunakan enhanced fallback system"
    }
    return jsonify(status)

@app.route('/performance')
def performance_metrics():
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

# ---------------- ENHANCED INITIALIZATION ----------------
if __name__ == "__main__":
    logger.info("🚀 Starting ENHANCED Forex Analysis Application with Advanced Backtesting...")
    
    # Delete old database to ensure clean schema
    if os.path.exists(Config.DB_PATH):
        os.remove(Config.DB_PATH)
        logger.info("Removed old database for clean schema")
    
    # Initialize components
    init_db()
    
    # Always create sample data for demonstration
    logger.info("Creating enhanced sample data for demonstration...")
    create_sample_data()
    
    # Load historical data
    load_csv_data()
    
    # If still no data, create emergency sample data
    if not HISTORICAL:
        logger.error("No historical data loaded after initial attempt, creating emergency data...")
        create_sample_data()
        load_csv_data()
    
    # Log enhanced features status
    logger.info("🎯 ENHANCED FEATURES ENABLED:")
    logger.info(f"   Pair Priority System: {Config.PAIR_PRIORITY}")
    logger.info(f"   Pair-Specific Parameters: Activated")
    for pair, params in Config.PAIR_SPECIFIC_PARAMS.items():
        logger.info(f"     {pair}: Hours {params['optimal_hours']}, Vol {params['min_volatility']}-{params['max_volatility']}%")
    
    logger.info(f"   Timeframe-Specific Parameters:")
    for tf, params in Config.TIMEFRAME_PARAMS.items():
        logger.info(f"     {tf}: Confidence {params['confidence_threshold']}%, Max Trades {params.get('max_daily_trades', 3)}/day")
    
    logger.info(f"   Risk Management: Max Drawdown {Config.MAX_DRAWDOWN_PCT * 100}%, Daily Loss Limit {Config.DAILY_LOSS_LIMIT * 100}%")
    logger.info(f"   Dynamic Position Sizing: ENABLED")
    logger.info(f"   Session Filtering: ENABLED")
    
    # Log AI status
    if DEEPSEEK_API_KEY:
        logger.info("✅ DeepSeek AI integration ENABLED")
    else:
        logger.info("⚠️ DeepSeek AI integration DISABLED - using enhanced fallback system")
    
    # Log backtesting status
    logger.info(f"✅ Enhanced Backtesting module INITIALIZED with initial balance: ${Config.INITIAL_BALANCE}")
    logger.info(f"📊 Historical data loaded for {len(HISTORICAL)} pairs")
    
    for pair in HISTORICAL:
        for tf in HISTORICAL[pair]:
            logger.info(f"   {pair}-{tf}: {len(HISTORICAL[pair][tf])} data points")
    
    # Start Flask application
    logger.info("🎯 ENHANCED Application initialized successfully - Ready for improved backtesting!")
    app.run(debug=True, host='0.0.0.0', port=5000)
[file content end]
