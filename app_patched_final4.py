# app_complete.py
# Backend lengkap untuk Forex Analysis System
# Dilengkapi logging, data manager, endpoints API, dan perbaikan get_price_data

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import requests
import os
import json
import sqlite3
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import talib
import random

# ==================== LOGGING ====================
def setup_logging():
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler('forex_trading.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()

# ==================== FLASK APP ====================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'forex-secure-key-2024')

# ==================== CONFIG ====================
@dataclass
class SystemConfig:
    DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "demo")
    NEWS_API_KEY: str = os.environ.get("NEWS_API_KEY", "demo")
    TWELVE_DATA_KEY: str = os.environ.get("TWELVE_DATA_KEY", "demo")

    FOREX_PAIRS: List[str] = field(default_factory=lambda: [
        "USDJPY", "GBPJPY", "EURJPY", "CHFJPY",
        "EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"
    ])
    TIMEFRAMES: List[str] = field(default_factory=lambda: ["M30", "1H", "4H", "1D", "1W"])

    INITIAL_BALANCE: float = 10000.0
    MIN_DATA_POINTS: int = 50

    BACKTEST_DAILY_TRADE_LIMIT: int = 100
    BACKTEST_MIN_CONFIDENCE: int = 40
    BACKTEST_RISK_SCORE_THRESHOLD: int = 8

config = SystemConfig()

# ==================== DATA MANAGER ====================
class DataManager:
    def __init__(self, data_dir: str = "historical_data"):
        self.historical_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.load_historical_data()

    def load_historical_data(self):
        try:
            for filename in os.listdir(self.data_dir):
                if not filename.endswith(".csv"):
                    continue
                path = os.path.join(self.data_dir, filename)
                try:
                    df = pd.read_csv(path)
                    df.columns = [c.lower() for c in df.columns]
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    parts = filename.replace(".csv", "").split("_")
                    if len(parts) >= 2:
                        pair = parts[0].upper()
                        tf = parts[1].upper()
                        if pair not in self.historical_data:
                            self.historical_data[pair] = {}
                        self.historical_data[pair][tf] = df
                        logger.info(f"Loaded {pair}-{tf}: {len(df)} records")
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {e}")
        except Exception as e:
            logger.error(f"Error in load_historical_data: {e}")

    def _generate_simple_data(self, pair: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate synthetic sample data (fallback)"""
        try:
            # Create simple date range depending on timeframe
            if timeframe == "M30":
                periods = max(720, days * 48)
                freq = "30T"
            elif timeframe == "1H":
                periods = max(720, days * 24)
                freq = "H"
            elif timeframe == "4H":
                periods = max(720, days * 6)
                freq = "4H"
            else:  # 1D or 1W fallback
                periods = max(365, days)
                freq = "D"

            end = datetime.utcnow()
            idx = pd.date_range(end=end, periods=periods, freq=freq)
            prices = []
            base = random.uniform(1.0, 150.0)
            for i in range(len(idx)):
                o = base + random.uniform(-0.5, 0.5)
                h = o + random.uniform(0, 0.5)
                l = o - random.uniform(0, 0.5)
                c = o + random.uniform(-0.3, 0.3)
                prices.append({"date": idx[i], "open": o, "high": h, "low": l, "close": c})
                base = c
            df = pd.DataFrame(prices)
            return df
        except Exception as e:
            logger.error(f"Error generating simple data: {e}")
            return pd.DataFrame()

    def get_price_data(self, pair: str, timeframe: str, days: int = 30) -> pd.DataFrame:
        """
        Ambil data harga dengan perbaikan:
         - normalisasi timeframe
         - jika 1D default ambil hingga 500 hari
         - selalu definisikan required_points
        """
        try:
            timeframe = (timeframe or "1D").upper()
            if timeframe == "1D":
                days = 500  # tampilkan hingga 500 hari untuk chart 1D

            if pair in self.historical_data and timeframe in self.historical_data[pair]:
                df = self.historical_data[pair][timeframe]
                if df is None or df.empty:
                    return self._generate_simple_data(pair, timeframe, days)

                # tentukan required_points dengan fallback yang aman
                if timeframe == "M30":
                    required_points = min(len(df), max(1, days * 48))
                elif timeframe == "1H":
                    required_points = min(len(df), max(1, days * 24))
                elif timeframe == "4H":
                    required_points = min(len(df), max(1, days * 6))
                elif timeframe in ("1D", "1W"):
                    required_points = min(len(df), max(1, days))
                else:
                    required_points = min(len(df), max(1, days))

                result_df = df.tail(required_points).copy()

                # pastikan kolom penting ada
                required_cols = ["date", "open", "high", "low", "close"]
                missing = [c for c in required_cols if c not in result_df.columns]
                if missing:
                    logger.warning(f"Missing columns {missing} for {pair}-{timeframe}, using synthetic data")
                    return self._generate_simple_data(pair, timeframe, days)

                logger.info(f"Sending {len(result_df)} records for {pair}-{timeframe}")
                return result_df

            # fallback: generate synthetic
            return self._generate_simple_data(pair, timeframe, days)
        except Exception as e:
            logger.error(f"Error getting price data for {pair}-{timeframe}: {e}")
            return self._generate_simple_data(pair, timeframe, days)

data_manager = DataManager()

# ==================== STUB CLIENTS / ENGINES (simple implementations) ====================
class TwelveDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.demo_mode = (api_key == "demo")

    def get_real_time_price(self, pair: str) -> float:
        # Simplified stub: return last close from historical or synthetic
        df = data_manager.get_price_data(pair, "1D", days=5)
        if not df.empty and "close" in df.columns:
            try:
                return float(df['close'].iloc[-1])
            except:
                pass
        # fallback synthetic
        return round(random.uniform(0.5, 200.0), 5)

twelve_data_client = TwelveDataClient(config.TWELVE_DATA_KEY)

class TechnicalAnalysisEngine:
    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        # Basic indicators: rsi, atr, adx approximations (safe fallbacks)
        try:
            closes = df['close'].astype(float)
            # RSI fallback using TA-Lib if available
            rsi = 50.0
            try:
                r = talib.RSI(closes, timeperiod=14)
                rsi = float(r.iloc[-1]) if not np.all(np.isnan(r)) else 50.0
            except Exception:
                rsi = float(50.0)
            # Basic trend detection
            if len(closes) >= 20:
                ma_short = closes.rolling(window=10).mean().iloc[-1]
                ma_long = closes.rolling(window=50).mean().iloc[-1] if len(closes) >= 50 else closes.mean()
                trend = "BULLISH" if ma_short >= ma_long else "BEARISH"
                trend_strength = abs((ma_short - ma_long) / (ma_long if ma_long != 0 else 1))
            else:
                trend = "NEUTRAL"
                trend_strength = 0.0
            volatility_pct = float(closes.pct_change().std() * 100) if len(closes) > 1 else 0.0
            atr = 0.0
            try:
                highs = df['high'].astype(float)
                lows = df['low'].astype(float)
                tr = pd.concat([highs - lows, (highs - closes.shift()).abs(), (lows - closes.shift()).abs()], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.mean())
            except Exception:
                atr = 0.0
            return {
                "momentum": {"rsi": rsi},
                "trend": {"trend_direction": trend, "trend_strength": trend_strength, "adx": 20.0},
                "volatility": {"volatility_pct": volatility_pct, "atr": atr},
                "levels": {"support": None, "resistance": None}
            }
        except Exception as e:
            logger.error(f"TechnicalAnalysis error: {e}")
            return {
                "momentum": {"rsi": 50.0},
                "trend": {"trend_direction": "UNKNOWN", "trend_strength": 0.0, "adx": 0.0},
                "volatility": {"volatility_pct": 0.0, "atr": 0.0},
                "levels": {"support": None, "resistance": None}
            }

tech_engine = TechnicalAnalysisEngine()

class RiskManager:
    def get_risk_report(self) -> Dict:
        # simplified stub
        return {"status": "OK", "active_positions": 0, "daily_loss": 0.0}

risk_manager = RiskManager()

class AdvancedBacktestingEngine:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance

    def run_comprehensive_backtest(self, signals: List[Dict], price_data: pd.DataFrame, pair: str, timeframe: str) -> Dict:
        # Simplified backtest summary
        summary = {
            "initial_balance": self.initial_balance,
            "final_balance": self.initial_balance + random.uniform(-1000, 1000),
            "total_trades": max(0, int(len(signals) * 0.5)),
            "winning_trades": max(0, int(len(signals) * 0.3)),
            "losing_trades": max(0, int(len(signals) * 0.2)),
        }
        return {"summary": summary, "detailed": [], "equity_curve": []}

advanced_backtester = AdvancedBacktestingEngine(config.INITIAL_BALANCE)

# ==================== HELPERS ====================
def generate_trading_signals(price_data: pd.DataFrame, pair: str, timeframe: str) -> List[Dict]:
    # Very simple signal generator for demo: look at last two candles
    signals = []
    try:
        if price_data is None or price_data.empty:
            return signals
        if len(price_data) < 2:
            return signals
        last = price_data.iloc[-1]
        prev = price_data.iloc[-2]
        if last['close'] > prev['close']:
            signals.append({"pair": pair, "timeframe": timeframe, "signal": "BUY", "confidence": 60})
        elif last['close'] < prev['close']:
            signals.append({"pair": pair, "timeframe": timeframe, "signal": "SELL", "confidence": 60})
        else:
            signals.append({"pair": pair, "timeframe": timeframe, "signal": "HOLD", "confidence": 30})
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
    return signals

# ==================== ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html", pairs=config.FOREX_PAIRS, timeframes=config.TIMEFRAMES)

@app.route("/api/chart_data")
def api_chart_data():
    try:
        pair = request.args.get("pair", "USDJPY").upper()
        timeframe = request.args.get("timeframe", "1D").upper()
        days = int(request.args.get("days", 500 if timeframe == "1D" else 30))
        df = data_manager.get_price_data(pair, timeframe, days=days)
        if df is None or df.empty:
            return jsonify({"error": "No data available"}), 404
        # Convert dates to isoformat for JSON
        df2 = df.copy()
        if "date" in df2.columns:
            df2['date'] = df2['date'].apply(lambda x: pd.to_datetime(x).isoformat())
        data = df2.to_dict(orient="records")
        return jsonify({"pair": pair, "timeframe": timeframe, "data": data})
    except Exception as e:
        logger.error(f"/api/chart_data error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/analyze")
def api_analyze():
    try:
        pair = request.args.get("pair", "USDJPY").upper()
        timeframe = request.args.get("timeframe", "1D").upper()
        df = data_manager.get_price_data(pair, timeframe, days=90)
        if df is None or df.empty:
            return jsonify({"error": "No data for analysis"}), 404
        tech = tech_engine.calculate_all_indicators(df)
        # simple AI/stub analysis
        recommendation = "HOLD"
        confidence = 50
        if tech['momentum']['rsi'] < 35 and tech['trend']['trend_direction'] == 'BULLISH':
            recommendation = "BUY"; confidence = 75
        elif tech['momentum']['rsi'] > 65 and tech['trend']['trend_direction'] == 'BEARISH':
            recommendation = "SELL"; confidence = 75
        return jsonify({
            "pair": pair,
            "timeframe": timeframe,
            "recommendation": recommendation,
            "confidence": confidence,
            "indicators": tech
        })
    except Exception as e:
        logger.error(f"/api/analyze error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    try:
        data = request.get_json() or {}
        pair = data.get("pair", "USDJPY").upper()
        timeframe = data.get("timeframe", "4H").upper()
        days = int(data.get("days", 30))
        initial_balance = float(data.get("initial_balance", config.INITIAL_BALANCE))
        df = data_manager.get_price_data(pair, timeframe, days=days)
        if df is None or df.empty:
            return jsonify({"error": "No price data for backtest"}), 400
        signals = generate_trading_signals(df, pair, timeframe)
        backtester = AdvancedBacktestingEngine(initial_balance)
        result = backtester.run_comprehensive_backtest(signals, df, pair, timeframe)
        result["backtest_parameters"] = {
            "pair": pair, "timeframe": timeframe, "days": days, "initial_balance": initial_balance
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"/api/backtest error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/advanced_backtest", methods=["POST"])
def api_advanced_backtest():
    try:
        data = request.get_json() or {}
        pair = data.get("pair", "USDJPY").upper()
        timeframe = data.get("timeframe", "4H").upper()
        days = int(data.get("days", 30))
        initial_balance = float(data.get("initial_balance", config.INITIAL_BALANCE))
        df = data_manager.get_price_data(pair, timeframe, days=days)
        if df is None or df.empty:
            return jsonify({"error": "No price data for advanced backtest"}), 400
        signals = generate_trading_signals(df, pair, timeframe)
        advanced_backtester.initial_balance = initial_balance
        result = advanced_backtester.run_comprehensive_backtest(signals, df, pair, timeframe)
        result["backtest_parameters"] = {
            "pair": pair, "timeframe": timeframe, "days": days, "initial_balance": initial_balance
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f"/api/advanced_backtest error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/market_overview")
def api_market_overview():
    try:
        overview = {}
        for pair in config.FOREX_PAIRS[:6]:
            try:
                rt = twelve_data_client.get_real_time_price(pair)
                df = data_manager.get_price_data(pair, "1D", days=5)
                if df is None or df.empty:
                    overview[pair] = {"price": rt, "error": "no history"}
                    continue
                tech = tech_engine.calculate_all_indicators(df)
                prev = float(df['close'].iloc[-2]) if len(df) > 1 else float(df['close'].iloc[-1])
                change_pct = ((rt - prev) / prev) * 100 if prev != 0 else 0
                overview[pair] = {
                    "price": rt,
                    "change": round(change_pct, 3),
                    "rsi": float(tech['momentum']['rsi']),
                    "trend": tech['trend']['trend_direction'],
                    "volatility": round(tech['volatility']['volatility_pct'], 3)
                }
            except Exception as e:
                logger.error(f"market overview error for {pair}: {e}")
                overview[pair] = {"price": 0, "error": str(e)}
        return jsonify(overview)
    except Exception as e:
        logger.error(f"/api/market_overview error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/risk_dashboard")
def api_risk_dashboard():
    try:
        risk = risk_manager.get_risk_report()
        market = {}
        for pair in config.FOREX_PAIRS[:4]:
            try:
                df = data_manager.get_price_data(pair, "4H", days=10)
                tech = tech_engine.calculate_all_indicators(df) if not df.empty else {}
                market[pair] = {
                    "trend": tech.get("trend", {}).get("trend_direction", "UNKNOWN"),
                    "rsi": tech.get("momentum", {}).get("rsi", 50),
                    "volatility": tech.get("volatility", {}).get("volatility_pct", 0)
                }
            except Exception as e:
                market[pair] = {"error": str(e)}
        return jsonify({"risk": risk, "market": market})
    except Exception as e:
        logger.error(f"/api/risk_dashboard error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/system_status")
def api_system_status():
    try:
        return jsonify({
            "system": "RUNNING",
            "historical_data_pairs": len(data_manager.historical_data),
            "supported_pairs": config.FOREX_PAIRS,
            "twelve_data_mode": "DEMO" if twelve_data_client.demo_mode else "LIVE",
            "server_time": datetime.utcnow().isoformat(),
            "version": "1.0"
        })
    except Exception as e:
        logger.error(f"/api/system_status error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# ==================== START APP ====================
if __name__ == "__main__":
    logger.info("Starting Forex Analysis System...")
    try:
        logger.info(f"Supported pairs: {config.FOREX_PAIRS}")
        logger.info(f"Historical data: {len(data_manager.historical_data)} pairs loaded")
    except Exception:
        logger.info("Data manager not ready yet")
    app.run(host="0.0.0.0", port=5000, debug=False)
