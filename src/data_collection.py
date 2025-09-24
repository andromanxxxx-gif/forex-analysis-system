import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time
import sys
from pathlib import Path

# Tambahkan path untuk import modul lain
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.google_drive_auth import drive_auth
    HAS_DRIVE_ACCESS = True
except ImportError:
    HAS_DRIVE_ACCESS = False

class DataCollector:
    def __init__(self):
        self.pairs = ['GBPJPY=X', 'CHFJPY=X', 'USDJPY=X', 'EURJPY=X']
        self.period = '60d'
        self.timeframe = '4h'
    
    def get_historical_data(self, pair, period, interval):
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
    
    def fetch_all_data(self, pairs=None):
        """Mengambil data untuk semua pasangan forex"""
        if pairs is None:
            pairs = self.pairs
            
        all_data = {}
        for pair in pairs:
            print(f"Mengambil data untuk {pair}...")
            data = self.get_historical_data(pair, self.period, self.timeframe)
            if data is not None:
                all_data[pair] = data
            time.sleep(1)  # Delay untuk menghindari rate limiting
        return all_data
    
    def scrape_news(self, query="forex news"):
        """Scraping berita forex"""
        news_items = []
        
        try:
            # Contoh sederhana - bisa diganti dengan sumber berita yang lebih reliable
            url = f"https://news.google.com/search?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = soup.find_all('article')[:10]
            
            for article in articles:
                try:
                    title_element = article.find('h3')
                    if title_element:
                        title = title_element.text
                        link_element = article.find('a')
                        if link_element and link_element.get('href'):
                            link = "https://news.google.com" + link_element['href'].replace('./', '/')
                            
                            news_items.append({
                                'title': title,
                                'link': link,
                                'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'source': 'Google News'
                            })
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping news: {e}")
        
        return news_items
    
    def save_to_drive(self, data, filename):
        """Save data to Google Drive (jika diperlukan)"""
        if not HAS_DRIVE_ACCESS:
            print("Google Drive access not available")
            return False
        
        try:
            # Implementasi save to Google Drive akan ditambahkan later
            print(f"Simulating save to Google Drive: {filename}")
            return True
        except Exception as e:
            print(f"Error saving to Google Drive: {e}")
            return False
    
    def calculate_atr(self, data, period=14):
        """Menghitung Average True Range (ATR)"""
        try:
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = true_range.max(axis=1)
            
            atr = true_range.rolling(period).mean()
            return atr
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return None

# Contoh penggunaan
if __name__ == "__main__":
    collector = DataCollector()
    data = collector.fetch_all_data()
    print(f"Data collected for {len(data)} pairs")
