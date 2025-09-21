import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import time
import os

def get_historical_data(pair, period, interval):
    """Mengambil data historis dari Yahoo Finance"""
    try:
        data = yf.download(pair, period=period, interval=interval, progress=False)
        if data.empty:
            print(f"Tidak ada data untuk {pair}")
            return None
        return data
    except Exception as e:
        print(f"Error mengambil data untuk {pair}: {e}")
        return None

def calculate_atr(data, period=14):
    """Menghitung Average True Range (ATR)"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = true_range.max(axis=1)
    
    atr = true_range.rolling(period).mean()
    return atr

def save_to_csv(data, filename):
    """Menyimpan data ke CSV"""
    os.makedirs('data/results', exist_ok=True)
    filepath = f"data/results/{filename}"
    data.to_csv(filepath, index=False)
    print(f"Data disimpan ke {filepath}")

def load_pairs_list(filepath='data/pairs_list.txt'):
    """Memuat daftar pasangan dari file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            pairs = [line.strip() for line in f if line.strip()]
        return pairs
    else:
        # Jika file tidak ada, kembalikan daftar default
        from config import settings
        return settings.FOREX_PAIRS
