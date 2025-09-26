import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """
    Analisis teknikal untuk forex data
    """
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Hitung semua indikator teknikal
        """
        if data.empty or 'Close' not in data.columns:
            raise ValueError("Data must contain 'Close' column")
        
        df = data.copy()
        
        # Trend Indicators
        df = self._add_trend_indicators(df)
        
        # Momentum Indicators
        df = self._add_momentum_indicators(df)
        
        # Volatility Indicators
        df = self._add_volatility_indicators(df)
        
        # Volume Indicators
        df = self._add_volume_indicators(df)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tambahkan indikator trend"""
        # Moving Averages
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tambahkan indikator momentum"""
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Williams %R
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # CCI
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tambahkan indikator volatilitas"""
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        
        # ATR (Average True Range)
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tambahkan indikator volume"""
        if 'Volume' not in df.columns:
            logger.warning("Volume data not available, skipping volume indicators")
            return df
        
        # On Balance Volume
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        
        # Volume SMA
        df['Volume_SMA_20'] = ta.volume.sma_indicator(df['Volume'], window=20)
        
        # Chaikin Money Flow
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> Dict:
        """
        Generate sinyal trading berdasarkan indikator
        """
        if df.empty:
            return {}
        
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        signals = {}
        
        # Trend Signals
        signals['trend_ema'] = self._get_ema_signal(current)
        signals['trend_sma'] = self._get_sma_signal(current)
        signals['macd_signal'] = self._get_macd_signal(current, previous)
        
        # Momentum Signals
        signals['rsi_signal'] = self._get_rsi_signal(current)
        signals['stoch_signal'] = self._get_stoch_signal(current)
        
        # Volatility Signals
        signals['bb_signal'] = self._get_bb_signal(current)
        
        # Volume Signals
        signals['volume_signal'] = self._get_volume_signal(current, previous)
        
        # Combined Signal
        signals['overall_signal'] = self._get_overall_signal(signals)
        
        return signals
    
    def _get_ema_signal(self, current) -> str:
        """Sinyal berdasarkan EMA"""
        if current['Close'] > current['EMA_200']:
            return 'BULLISH'
        else:
            return 'BEARISH'
    
    def _get_sma_signal(self, current) -> str:
        """Sinyal berdasarkan SMA crossover"""
        if current['SMA_20'] > current['SMA_50']:
            return 'BULLISH'
        else:
            return 'BEARISH'
    
    def _get_macd_signal(self, current, previous) -> str:
        """Sinyal berdasarkan MACD"""
        if current['MACD'] > current['MACD_Signal'] and previous['MACD'] <= previous['MACD_Signal']:
            return 'BULLISH_CROSS'
        elif current['MACD'] < current['MACD_Signal'] and previous['MACD'] >= previous['MACD_Signal']:
            return 'BEARISH_CROSS'
        elif current['MACD'] > current['MACD_Signal']:
            return 'BULLISH'
        else:
            return 'BEARISH'
    
    def _get_rsi_signal(self, current) -> str:
        """Sinyal berdasarkan RSI"""
        if current['RSI'] < 30:
            return 'OVERSOLD'
        elif current['RSI'] > 70:
            return 'OVERBOUGHT'
        else:
            return 'NEUTRAL'
    
    def _get_stoch_signal(self, current) -> str:
        """Sinyal berdasarkan Stochastic"""
        if current['Stoch_K'] < 20 and current['Stoch_D'] < 20:
            return 'OVERSOLD'
        elif current['Stoch_K'] > 80 and current['Stoch_D'] > 80:
            return 'OVERBOUGHT'
        else:
            return 'NEUTRAL'
    
    def _get_bb_signal(self, current) -> str:
        """Sinyal berdasarkan Bollinger Bands"""
        if current['Close'] < current['BB_Lower']:
            return 'OVERSOLD'
        elif current['Close'] > current['BB_Upper']:
            return 'OVERBOUGHT'
        else:
            return 'NEUTRAL'
    
    def _get_volume_signal(self, current, previous) -> str:
        """Sinyal berdasarkan Volume"""
        if 'OBV' in current and 'OBV' in previous:
            if current['OBV'] > previous['OBV']:
                return 'BULLISH'
            else:
                return 'BEARISH'
        return 'NEUTRAL'
    
    def _get_overall_signal(self, signals: Dict) -> Dict:
        """Kombinasikan semua sinyal menjadi sinyal overall"""
        bull_signals = 0
        bear_signals = 0
        total_signals = 0
        
        for key, signal in signals.items():
            if key == 'overall_signal':
                continue
                
            if signal in ['BULLISH', 'BULLISH_CROSS', 'OVERSOLD']:
                bull_signals += 1
            elif signal in ['BEARISH', 'BEARISH_CROSS', 'OVERBOUGHT']:
                bear_signals += 1
            total_signals += 1
        
        if total_signals == 0:
            return {'action': 'HOLD', 'confidence': 0.5}
        
        bull_ratio = bull_signals / total_signals
        bear_ratio = bear_signals / total_signals
        
        if bull_ratio > 0.6:
            return {'action': 'BUY', 'confidence': bull_ratio}
        elif bear_ratio > 0.6:
            return {'action': 'SELL', 'confidence': bear_ratio}
        else:
            return {'action': 'HOLD', 'confidence': max(bull_ratio, bear_ratio)}
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Dict:
        """
        Calculate support and resistance levels
        """
        if len(df) < window:
            return {}
        
        recent_data = df.tail(window)
        
        resistance = recent_data['High'].max()
        support = recent_data['Low'].min()
        pivot = (resistance + support + recent_data['Close'].iloc[-1]) / 3
        
        return {
            'resistance': resistance,
            'support': support,
            'pivot_point': pivot,
            'r1': 2 * pivot - support,
            'r2': pivot + (resistance - support),
            's1': 2 * pivot - resistance,
            's2': pivot - (resistance - support)
        }

# Global instance
TECHNICAL_ANALYZER = TechnicalAnalyzer()
