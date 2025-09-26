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
        """Mendapatkan data forex"""
        try:
            yf_pair = self.pair_mapping.get(pair, pair)
            data = yf.download(yf_pair, period=period, interval='1h')
            return data
        except Exception as e:
            print(f"Error fetching data for {pair}: {e}")
            return None
    
    def get_multiple_pairs(self, pairs, period='30d'):
        """Mendapatkan data untuk multiple pairs"""
        data_dict = {}
        for pair in pairs:
            data = self.get_data(pair, period)
            if data is not None:
                data_dict[pair] = data
        return data_dict
