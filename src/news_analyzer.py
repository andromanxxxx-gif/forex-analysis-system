import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    """
    Analisis berita forex dari berbagai sumber
    """
    
    def __init__(self):
        self.sources = [
            {
                'name': 'ForexFactory',
                'url': 'https://www.forexfactory.com/',
                'parser': self._parse_forexfactory
            },
            {
                'name': 'Investing.com',
                'url': 'https://www.investing.com/news/forex-news',
                'parser': self._parse_investing
            },
            {
                'name': 'DailyFX',
                'url': 'https://www.dailyfx.com/latest-news',
                'parser': self._parse_dailyfx
            }
        ]
        
        self.currency_keywords = {
            'USD': ['fed', 'usd', 'dollar', 'federal reserve', 'jerome powell'],
            'EUR': ['ecb', 'euro', 'eur', 'european central bank', 'lagarde'],
            'JPY': ['boj', 'jpy', 'yen', 'bank of japan', 'kuroda'],
            'GBP': ['boe', 'gbp', 'pound', 'bank of england', 'bailey'],
            'CHF': ['snb', 'chf', 'franc', 'swiss national bank'],
            'NZD': ['rbnz', 'nzd', 'kiwi', 'reserve bank of new zealand'],
            'AUD': ['rba', 'aud', 'aussie', 'reserve bank of australia'],
            'CAD': ['boc', 'cad', 'loonie', 'bank of canada']
        }
    
    def get_news(self, max_articles: int = 10) -> List[Dict]:
        """
        Dapatkan berita terkini dari semua sumber
        """
        all_news = []
        
        for source in self.sources:
            try:
                logger.info(f"Fetching news from {source['name']}")
                articles = source['parser'](source['url'])
                all_news.extend(articles[:max_articles])
                
            except Exception as e:
                logger.error(f"Error fetching from {source['name']}: {e}")
                # Fallback to sample data
                all_news.extend(self._get_sample_news(source['name']))
        
        # Sort by date and limit
        all_news.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
        return all_news[:max_articles]
    
    def _parse_forexfactory(self, url: str) -> List[Dict]:
        """Parse news dari ForexFactory (simulated)"""
        # Simulated data - in practice you'd use BeautifulSoup
        return [
            {
                'title': 'FOMC Maintains Rates, Signals Future Hikes',
                'summary': 'Federal Reserve keeps interest rates unchanged but indicates potential hikes later this year.',
                'source': 'ForexFactory',
                'timestamp': datetime.now() - timedelta(hours=2),
                'impact': 'High',
                'currencies': ['USD'],
                'url': 'https://www.forexfactory.com/news/fomc-rates'
            },
            {
                'title': 'ECB President Speech on Inflation',
                'summary': 'European Central Bank President discusses inflation outlook and monetary policy.',
                'source': 'ForexFactory', 
                'timestamp': datetime.now() - timedelta(hours=5),
                'impact': 'Medium',
                'currencies': ['EUR'],
                'url': 'https://www.forexfactory.com/news/ecb-speech'
            }
        ]
    
    def _parse_investing(self, url: str) -> List[Dict]:
        """Parse news dari Investing.com (simulated)"""
        return [
            {
                'title': 'Bank of Japan Unexpected Policy Shift',
                'summary': 'BOJ surprises markets with adjustment to yield curve control policy.',
                'source': 'Investing.com',
                'timestamp': datetime.now() - timedelta(hours=1),
                'impact': 'High',
                'currencies': ['JPY'],
                'url': 'https://www.investing.com/news/boj-policy'
            }
        ]
    
    def _parse_dailyfx(self, url: str) -> List[Dict]:
        """Parse news dari DailyFX (simulated)"""
        return [
            {
                'title': 'UK GDP Beats Expectations',
                'summary': 'UK economic growth exceeds forecasts, supporting GBP strength.',
                'source': 'DailyFX',
                'timestamp': datetime.now() - timedelta(hours=3),
                'impact': 'Medium',
                'currencies': ['GBP'],
                'url': 'https://www.dailyfx.com/news/uk-gdp'
            }
        ]
    
    def _get_sample_news(self, source: str) -> List[Dict]:
        """Sample news data sebagai fallback"""
        return [
            {
                'title': f'Market Analysis - {source}',
                'summary': 'Technical analysis suggests potential breakout in major currency pairs.',
                'source': source,
                'timestamp': datetime.now(),
                'impact': 'Low',
                'currencies': ['USD', 'EUR', 'JPY'],
                'url': f'https://{source.lower().replace(" ", "")}.com/sample'
            }
        ]
    
    def analyze_sentiment(self, news_articles: List[Dict]) -> Dict:
        """
        Analisis sentimen dari berita
        """
        positive_keywords = ['bullish', 'strong', 'growth', 'positive', 'hike', 'hawkish', 'beat']
        negative_keywords = ['bearish', 'weak', 'decline', 'negative', 'cut', 'dovish', 'miss']
        
        currency_sentiment = {}
        
        for article in news_articles:
            title = article.get('title', '').lower()
            summary = article.get('summary', '').lower()
            impact = article.get('impact', 'Low')
            currencies = article.get('currencies', [])
            
            # Calculate sentiment score
            positive_score = sum(1 for word in positive_keywords if word in title + summary)
            negative_score = sum(1 for word in negative_keywords if word in title + summary)
            
            sentiment_score = positive_score - negative_score
            impact_weight = {'High': 3, 'Medium': 2, 'Low': 1}.get(impact, 1)
            weighted_score = sentiment_score * impact_weight
            
            for currency in currencies:
                if currency not in currency_sentiment:
                    currency_sentiment[currency] = 0
                currency_sentiment[currency] += weighted_score
        
        # Normalize scores
        for currency in currency_sentiment:
            if currency_sentiment[currency] > 5:
                currency_sentiment[currency] = 5
            elif currency_sentiment[currency] < -5:
                currency_sentiment[currency] = -5
        
        return currency_sentiment
    
    def get_currency_news(self, currency: str, news_articles: List[Dict]) -> List[Dict]:
        """
        Dapatkan berita spesifik untuk currency tertentu
        """
        currency_news = []
        
        for article in news_articles:
            if currency in article.get('currencies', []):
                currency_news.append(article)
            else:
                # Check if currency is mentioned in title/summary
                title = article.get('title', '').lower()
                summary = article.get('summary', '').lower()
                currency_lower = currency.lower()
                
                if (currency_lower in title or 
                    currency_lower in summary or
                    any(keyword in title + summary for keyword in self.currency_keywords.get(currency, []))):
                    currency_news.append(article)
        
        return currency_news
    
    def generate_news_summary(self, news_articles: List[Dict]) -> str:
        """
        Generate summary text dari berita untuk AI analysis
        """
        if not news_articles:
            return "No significant news events today."
        
        high_impact = [article for article in news_articles if article.get('impact') == 'High']
        medium_impact = [article for article in news_articles if article.get('impact') == 'Medium']
        
        summary_parts = []
        
        if high_impact:
            summary_parts.append("HIGH IMPACT NEWS:")
            for article in high_impact[:3]:  # Max 3 high impact articles
                summary_parts.append(f"- {article['title']} ({', '.join(article.get('currencies', []))})")
        
        if medium_impact:
            summary_parts.append("MEDIUM IMPACT NEWS:")
            for article in medium_impact[:2]:  # Max 2 medium impact articles
                summary_parts.append(f"- {article['title']}")
        
        return "\n".join(summary_parts)

# Global instance
NEWS_ANALYZER = NewsAnalyzer()
