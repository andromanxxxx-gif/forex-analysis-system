import pandas as pd
import yfinance as yf
from src.utils import get_historical_data
from config import settings
import requests
from bs4 import BeautifulSoup
import re
import time
import sys
from pathlib import Path

class DataCollector:
    def __init__(self):
        self.pairs = settings.FOREX_PAIRS
        self.period = settings.PERIOD
        self.timeframe = settings.TIMEFRAME
    
    def fetch_all_data(self, pairs=None):
        """Mengambil data untuk semua pasangan forex"""
        if pairs is None:
            pairs = self.pairs
            
        all_data = {}
        for pair in pairs:
            print(f"Mengambil data untuk {pair}...")
            data = get_historical_data(pair, self.period, self.timeframe)
            if data is not None:
                all_data[pair] = data
            time.sleep(0.5)  # Delay untuk menghindari rate limiting
        return all_data
    
    def scrape_news(self, query="forex news"):
        """Scraping berita forex"""
        news_items = []
        
        try:
            # Scraping dari Google News
            url = f"https://news.google.com/search?q={query}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = soup.find_all('article')[:15]  # Ambil 15 berita terbaru
            
            for article in articles:
                try:
                    title = article.find('h3').text
                    link = "https://news.google.com" + article.find('a')['href'].replace('./', '/')
                    time_element = article.find('time')
                    time_str = time_element['datetime'] if time_element else "Unknown"
                    
                    news_items.append({
                        'title': title,
                        'link': link,
                        'time': time_str,
                        'source': 'Google News'
                    })
                except:
                    continue
                    
        except Exception as e:
            print(f"Error scraping news: {e}")
        # Akses google drive
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.google_drive_auth import drive_auth
    HAS_DRIVE_ACCESS = True
except ImportError:
    HAS_DRIVE_ACCESS = False

class DataCollector:
    def __init__(self):
        # ... kode existing
        
    def save_to_drive(self, data, filename):
        """Save data to Google Drive (jika diperlukan)"""
        if not HAS_DRIVE_ACCESS:
            print("Google Drive access not available")
            return False
        
        try:
            drive_service = drive_auth.get_service()
            # ... implementasi save to drive
            return True
        except Exception as e:
            print(f"Error saving to Google Drive: {e}")
            return False
        return news_items
