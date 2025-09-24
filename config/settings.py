# Daftar semua pasangan forex utama
FOREX_PAIRS = [
    # Major Pairs
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 
    'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X',
    
    # JPY Crosses
    'EURJPY=X', 'GBPJPY=X', 'CHFJPY=X', 'AUDJPY=X', 
    'CADJPY=X', 'NZDJPY=X',
    
    # EUR Crosses
    'EURGBP=X', 'EURCHF=X', 'EURAUD=X', 'EURCAD=X', 
    'EURNZD=X',
    
    # GBP Crosses
    'GBPCHF=X', 'GBPAUD=X', 'GBPCAD=X', 'GBPNZD=X',
    
    # AUD Crosses
    'AUDCHF=X', 'AUDCAD=X', 'AUDNZD=X',
    
    # CAD Crosses
    'CADCHF=X',
    
    # CHF Crosses
    'CHFCAD=X',
    
    # Exotic Pairs
    'USDHKD=X', 'USDSGD=X', 'USDTRY=X', 'USDZAR=X',
    'USDSEK=X', 'USDDKK=X', 'USDNOK=X', 'USDMXN=X'
]

# Timeframe untuk analisis
TIMEFRAME = '4h'
PERIOD = '60d'  # 60 hari data historis

# Parameter indikator
EMA_PERIOD = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Parameter risk management
RISK_TO_REWARD = 2.0
ATR_MULTIPLIER = 2.0

# Sumber berita
NEWS_SOURCES = [
    'https://www.forexfactory.com/',
    'https://www.investing.com/news/forex-news',
    'https://www.fxstreet.com/news'
]

# Warna untuk output
COLORS = {
    'buy': '\033[92m',  # Hijau
    'sell': '\033[91m', # Merah
    'neutral': '\033[93m', # Kuning
    'end': '\033[0m'    # Akhir warna
}
# config/settings.py
import os
from pathlib import Path

# Google Drive Auth Configuration
GOOGLE_AUTH_PATH = Path("C:/hp/Json/google-auth.json")

# Fallback jika path utama tidak ada
if not GOOGLE_AUTH_PATH.exists():
    GOOGLE_AUTH_PATH = Path.home() / "hp" / "Json" / "google-auth.json"

# Other existing settings...
FOREX_PAIRS = ['GBPJPY=X', 'CHFJPY=X', 'USDJPY=X', 'EURJPY=X']
# ... rest of your settings
