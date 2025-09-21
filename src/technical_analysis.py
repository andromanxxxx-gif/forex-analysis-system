import pandas as pd
import numpy as np
from src.utils import calculate_atr

class TechnicalAnalyzer:
    def __init__(self, ema_period=200, macd_fast=12, macd_slow=26, macd_signal=9):
        self.ema_period = ema_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
    
    def calculate_ema(self, data, period):
        """Menghitung Exponential Moving Average (EMA)"""
        return data['Close'].ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, data):
        """Menghitung MACD"""
        exp1 = data['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = data['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    def analyze(self, data):
        """Melakukan analisis teknikal lengkap"""
        if data is None or data.empty:
            return None
        
        # Buat salinan data
        result = data.copy()
        
        # Hitung EMA200
        result['EMA200'] = self.calculate_ema(result, self.ema_period)
        
        # Hitung MACD
        macd, signal, histogram = self.calculate_macd(result)
        result['MACD'] = macd
        result['MACD_Signal'] = signal
        result['MACD_Histogram'] = histogram
        
        # Hitung ATR
        result['ATR'] = calculate_atr(result)
        
        return result
