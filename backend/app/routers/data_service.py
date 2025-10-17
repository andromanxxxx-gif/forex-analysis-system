import pandas as pd
import numpy as np
import aiohttp
import asyncio
from datetime import datetime, timedelta
import os
import json
import logging

# Direct import
try:
    from config import settings
except ImportError:
    # Fallback settings
    class FallbackSettings:
        DATA_PATH = "data/"
        TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
        TWELVEDATA_API_KEY = "demo"
        NEWS_API_BASE_URL = "https://newsapi.org/v2"
        NEWS_API_KEY = "demo"
        CACHE_TTL = 300
    
    settings = FallbackSettings()

logger = logging.getLogger(__name__)

class EnhancedDataService:
    def __init__(self):
        self.data_path = settings.DATA_PATH
        self.cache = {}
        self.cache_expiry = {}
        
    async def load_historical_data(self, timeframe: str, limit: int = 600) -> pd.DataFrame:
        """Load historical data from CSV files with caching"""
        cache_key = f"historical_{timeframe}_{limit}"
        
        # Check cache
        if await self._is_cache_valid(cache_key):
            return self.cache[cache_key]
            
        filename = f"XAUUSD_{timeframe}.csv"
        filepath = os.path.join(self.data_path, filename)
        
        try:
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').tail(limit)
            
            # Cache the result
            await self._set_cache(cache_key, df)
            
            return df
        except Exception as e:
            raise Exception(f"Error loading historical data: {str(e)}")
    
    async def get_realtime_price(self) -> float:
        """Get real-time price from TwelveData API with fallback"""
        cache_key = "realtime_price"
        
        # Check cache (1 minute cache for price)
        if await self._is_cache_valid(cache_key, ttl=60):
            return self.cache[cache_key]
            
        # Try TwelveData API first
        try:
            price = await self._fetch_twelvedata_price()
            await self._set_cache(cache_key, price, ttl=60)
            return price
        except Exception as e:
            logger.warning(f"TwelveData API failed: {e}. Trying fallback...")
            
        # Fallback: Calculate from latest historical data
        try:
            price = await self._get_price_from_historical()
            await self._set_cache(cache_key, price, ttl=60)
            return price
        except Exception as e:
            raise Exception(f"All price sources failed: {e}")
    
    async def _fetch_twelvedata_price(self) -> float:
        """Fetch price from TwelveData API"""
        url = f"{settings.TWELVEDATA_BASE_URL}/price"
        params = {
            'symbol': 'XAU/USD',
            'apikey': settings.TWELVEDATA_API_KEY
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['price'])
                else:
                    raise Exception(f"API returned status {response.status}")
    
    async def _get_price_from_historical(self) -> float:
        """Get approximate price from latest historical data"""
        try:
            # Try 1H data first as it's most recent
            df = await self.load_historical_data('1H', limit=1)
            if not df.empty:
                return float(df.iloc[-1]['close'])
                
            # Fallback to other timeframes
            for tf in ['4H', '1D']:
                df = await self.load_historical_data(tf, limit=1)
                if not df.empty:
                    return float(df.iloc[-1]['close'])
                    
            raise Exception("No historical data available")
        except Exception as e:
            raise Exception(f"Failed to get price from historical data: {e}")
    
    async def get_fundamental_news(self, limit: int = 10) -> list:
        """Get fundamental news with caching"""
        cache_key = f"news_{limit}"
        
        if await self._is_cache_valid(cache_key, ttl=300):  # 5 minutes cache
            return self.cache[cache_key]
            
        url = f"{settings.NEWS_API_BASE_URL}/everything"
        params = {
            'q': 'gold OR XAUUSD OR Federal Reserve OR inflation OR USD',
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': settings.NEWS_API_KEY,
            'pageSize': limit
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    news = data.get('articles', [])
                    
                    # Process news items
                    processed_news = []
                    for article in news:
                        processed_news.append({
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'sentiment': await self._analyze_sentiment(article)
                        })
                    
                    await self._set_cache(cache_key, processed_news, ttl=300)
                    return processed_news
                else:
                    logger.warning(f"News API failed: {response.status}")
                    return []
    
    async def _analyze_sentiment(self, article: dict) -> str:
        """Simple sentiment analysis based on keywords"""
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        
        text = f"{title} {description}"
        
        positive_words = ['bullish', 'rise', 'gain', 'up', 'high', 'strong', 'positive', 'buy']
        negative_words = ['bearish', 'fall', 'drop', 'down', 'low', 'weak', 'negative', 'sell']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'POSITIVE'
        elif negative_count > positive_count:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    
    def update_realtime_candle(self, df: pd.DataFrame, realtime_price: float) -> pd.DataFrame:
        """Update the last candle with real-time price"""
        if len(df) == 0:
            return df
        
        # Create a copy to avoid modifying original
        df_updated = df.copy()
        last_index = df_updated.index[-1]
        
        # Update the last row with real-time data
        df_updated.at[last_index, 'close'] = realtime_price
        df_updated.at[last_index, 'high'] = max(df_updated.at[last_index, 'high'], realtime_price)
        df_updated.at[last_index, 'low'] = min(df_updated.at[last_index, 'low'], realtime_price)
        
        return df_updated
    
    async def _set_cache(self, key: str, value, ttl: int = settings.CACHE_TTL):
        """Set cache value with TTL"""
        self.cache[key] = value
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=ttl)
    
    async def _is_cache_valid(self, key: str, ttl: int = None) -> bool:
        """Check if cache is still valid"""
        if key not in self.cache or key not in self.cache_expiry:
            return False
            
        if ttl:
            # Use provided TTL
            return datetime.now() < self.cache_expiry[key]
        else:
            # Use default TTL
            return datetime.now() < self.cache_expiry[key]
    
    async def get_multiple_timeframe_data(self) -> dict:
        """Get data for all timeframes at once"""
        timeframes = ['1D', '4H', '1H']
        result = {}
        
        for tf in timeframes:
            try:
                data = await self.load_historical_data(tf, 100)  # Last 100 points
                result[tf] = data
            except Exception as e:
                logger.error(f"Failed to load data for {tf}: {e}")
                result[tf] = None
                
        return result

# Update the global instance
data_service = EnhancedDataService()
