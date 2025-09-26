import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class ForexDataFetcher:
    def __init__(self):
        self.pair_mapping = {
            'GBPJPY': 'GBPJPY=X',
            'USDJPY': 'USDJPY=X',
            'CHFJPY': 'CHFJPY=X',
            'EURJPY': 'EURJPY=X',
            'EURNZD': 'EURNZD=X'
        }
    
    def get_data(self, pair, period='60d'):
        """Mendapatkan data dari Yahoo Finance"""
        try:
            yf_symbol = self.pair_mapping.get(pair, pair)
            data = yf.download(yf_symbol, period=period, interval='1h')
            return data
        except Exception as e:
            print(f"Error fetching data for {pair}: {e}")
            return None
    
    def resample_data(self, data, timeframe):
        """Resample data berdasarkan timeframe"""
        if timeframe == '2H':
            return data.resample('2H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif timeframe == '4H':
            return data.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif timeframe == '1D':
            return data.resample('1D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            return data
