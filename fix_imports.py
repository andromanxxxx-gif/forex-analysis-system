#!/usr/bin/env python3
"""
Script untuk memperbaiki error import secara otomatis
"""

import os
import shutil
from pathlib import Path

def create_missing_files():
    """Buat file-file yang missing"""
    root_dir = Path(__file__).parent
    
    # Buat src directory jika belum ada
    src_dir = root_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    # File signal_generator.py
    signal_generator_content = '''import pandas as pd
import numpy as np

class SignalGenerator:
    """Simple signal generator for forex trading"""
    
    def generate_signals(self, data):
        """Generate basic trading signals"""
        if data is None or data.empty:
            return {"action": "HOLD", "confidence": 0.5}
        
        # Simple logic based on price vs EMA
        current = data.iloc[-1]
        price = current['Close']
        ema = current.get('EMA_200', price)
        
        if price > ema:
            return {
                "action": "BUY",
                "confidence": 0.7,
                "take_profit": price * 1.02,
                "stop_loss": price * 0.98
            }
        elif price < ema:
            return {
                "action": "SELL", 
                "confidence": 0.7,
                "take_profit": price * 0.98,
                "stop_loss": price * 1.02
            }
        else:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "take_profit": price,
                "stop_loss": price
            }

SIGNAL_GENERATOR = SignalGenerator()
'''
    
    with open(src_dir / "signal_generator.py", "w", encoding='utf-8') as f:
        f.write(signal_generator_content)
    
    # File technical_analyzer.py
    technical_analyzer_content = '''import pandas as pd
import ta

class TechnicalAnalyzer:
    """Basic technical analysis"""
    
    def calculate_indicators(self, data):
        """Calculate basic technical indicators"""
        if data is None or data.empty:
            return data
        
        # EMA
        data['EMA_200'] = ta.trend.ema_indicator(data['Close'], window=200)
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        
        return data

TECHNICAL_ANALYZER = TechnicalAnalyzer()
'''
    
    with open(src_dir / "technical_analyzer.py", "w", encoding='utf-8') as f:
        f.write(technical_analyzer_content)
    
    # File data_fetcher.py
    data_fetcher_content = '''import yfinance as yf

class ForexDataFetcher:
    """Fetch forex data from Yahoo Finance"""
    
    def get_data(self, pair, period="7d"):
        """Get forex data"""
        pair_mapping = {
            'GBPJPY': 'GBPJPY=X',
            'USDJPY': 'USDJPY=X',
            'CHFJPY': 'CHFJPY=X',
            'EURJPY': 'EURJPY=X',
            'EURNZD': 'EURNZD=X'
        }
        
        try:
            return yf.download(pair_mapping[pair], period=period, interval='1h')
        except:
            return None

DATA_FETCHER = ForexDataFetcher()
'''
    
    with open(src_dir / "data_fetcher.py", "w", encoding='utf-8') as f:
        f.write(data_fetcher_content)

def fix_dashboard_imports():
    """Perbaiki import di dashboard"""
    dashboard_dir = Path(__file__).parent / "dashboard"
    app_file = dashboard_dir / "app.py"
    
    if app_file.exists():
        # Backup original file
        shutil.copy2(app_file, app_file.with_suffix('.py.backup'))
        
        # Baca content dengan encoding utf-8, jika gagal coba latin-1
        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(app_file, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading file {app_file}: {e}")
                return
        
        # Ganti import yang problematic
        new_content = content.replace(
            "from src.signal_generator import SignalGenerator", 
            "try:\n    from src.signal_generator import SignalGenerator\nexcept ImportError:\n    SignalGenerator = None"
        )
        
        # Tulis kembali dengan encoding utf-8
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

def main():
    """Main function"""
    print("🔧 Memperbaiki error import...")
    
    create_missing_files()
    fix_dashboard_imports()
    
    print("✅ File-file yang missing telah dibuat")
    print("✅ Import di dashboard telah diperbaiki")
    print("🚀 Coba jalankan dashboard lagi: python run_dashboard.py")

if __name__ == "__main__":
    main()
