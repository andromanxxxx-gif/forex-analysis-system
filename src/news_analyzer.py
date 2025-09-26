import requests
from bs4 import BeautifulSoup
import pandas as pd

class NewsAnalyzer:
    def __init__(self):
        self.sources = [
            {
                'name': 'ForexFactory',
                'url': 'https://www.forexfactory.com/',
                'parser': self.parse_forexfactory
            },
            # Tambahkan sumber berita lainnya
        ]
    
    def get_news(self):
        """Mendapatkan berita dari berbagai sumber"""
        all_news = []
        
        for source in self.sources:
            try:
                response = requests.get(source['url'])
                if response.status_code == 200:
                    news_items = source['parser'](response.text)
                    all_news.extend(news_items)
            except Exception as e:
                print(f"Error scraping {source['name']}: {e}")
        
        # Jika tidak ada berita, kembalikan berita default
        if not all_news:
            all_news = [
                {
                    'title': 'Bank of Japan Maintains Monetary Policy',
                    'source': 'ForexFactory',
                    'impact': 'High'
                }
            ]
        
        return all_news
    
    def parse_forexfactory(self, html):
        """Parse berita dari ForexFactory"""
        # Implementasi parsing ForexFactory
        # Karena kompleksitas, kita kembalikan data dummy untuk sementara
        return [
            {
                'title': 'BOJ Maintains Current Monetary Policy',
                'source': 'ForexFactory',
                'impact': 'High'
            }
        ]
