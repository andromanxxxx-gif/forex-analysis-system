import pandas as pd
import numpy as np
import aiohttp
import asyncio
from datetime import datetime, timedelta
import os
import json
import logging
from typing import List, Dict, Any, Optional

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
        """Load historical data from CSV files with enhanced error handling"""
        cache_key = f"historical_{timeframe}_{limit}"
        
        # Check cache
        if await self._is_cache_valid(cache_key):
            return self.cache[cache_key]
            
        filename = f"XAUUSD_{timeframe}.csv"
        filepath = os.path.join(self.data_path, filename)
        
        try:
            logger.info(f"Loading historical data from: {filepath}")
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}. Using sample data.")
                return self._create_sample_data(limit)
            
            # Load CSV with flexible column handling
            df = pd.read_csv(filepath)
            logger.info(f"CSV columns: {df.columns.tolist()}")
            logger.info(f"CSV shape: {df.shape}")
            
            # Handle different timestamp column names
            timestamp_col = None
            for col in ['timestamp', 'time', 'date', 'datetime', 'Timestamp', 'Time', 'Date']:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                df['timestamp'] = pd.to_datetime(df[timestamp_col])
                logger.info(f"Using timestamp column: {timestamp_col}")
            else:
                # If no timestamp column, create one
                logger.warning("No timestamp column found. Creating default timestamps.")
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close']
            for col in required_cols:
                if col not in df.columns:
                    # Try capitalized versions
                    col_cap = col.capitalize()
                    if col_cap in df.columns:
                        df[col] = df[col_cap]
                    else:
                        # Create sample data if columns missing
                        logger.warning(f"Missing column {col}. Using sample data.")
                        return self._create_sample_data(limit)
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = 1000 + np.random.randint(-100, 100, len(df))
            
            df = df.sort_values('timestamp').tail(limit)
            
            # Cache the result
            await self._set_cache(cache_key, df)
            
            logger.info(f"Successfully loaded {len(df)} records for timeframe {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data from {filepath}: {str(e)}")
            logger.info("Using sample data as fallback")
            return self._create_sample_data(limit)
    
    def _create_sample_data(self, limit: int = 600) -> pd.DataFrame:
        """Create sample price data for testing"""
        logger.info(f"Creating sample data for {limit} periods")
        
        # Create realistic sample data for XAUUSD
        base_price = 1950.0
        dates = pd.date_range(start='2024-01-01', periods=limit, freq='1H')
        
        # Generate realistic price movement
        prices = [base_price]
        for i in range(1, limit):
            change = np.random.normal(0, 5)  # Small random changes
            new_price = prices[-1] + change
            # Keep price in realistic range
            new_price = max(1800, min(2200, new_price))
            prices.append(new_price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [price + abs(np.random.normal(2, 1)) for price in prices],
            'low': [price - abs(np.random.normal(2, 1)) for price in prices],
            'close': prices,
            'volume': [1000 + np.random.randint(-100, 100) for _ in prices]
        })
        
        # Add some trend
        df['close'] = df['close'] + np.arange(len(df)) * 0.01
        
        return df
    
    async def get_realtime_price(self) -> float:
        """Get real-time price with fallback to sample data"""
        cache_key = "realtime_price"
        
        # Check cache (1 minute cache for price)
        if await self._is_cache_valid(cache_key, ttl=60):
            return self.cache[cache_key]
            
        try:
            # Try to get price from historical data first
            historical_data = await self.load_historical_data('1H', 1)
            if not historical_data.empty:
                price = float(historical_data.iloc[-1]['close'])
                await self._set_cache(cache_key, price, ttl=60)
                return price
        except Exception as e:
            logger.warning(f"Failed to get price from historical data: {e}")
        
        # Fallback: Generate realistic price
        base_price = 1950.0
        variation = np.random.normal(0, 10)  # +/- $10 variation
        price = max(1800, min(2200, base_price + variation))
        
        await self._set_cache(cache_key, price, ttl=60)
        logger.info(f"Using sample real-time price: ${price:.2f}")
        return price
    
    async def get_fundamental_news(self, limit: int = 5) -> list:
        """Get sample fundamental news"""
        cache_key = f"news_{limit}"
        
        if await self._is_cache_valid(cache_key, ttl=300):
            return self.cache[cache_key]
            
        # Sample news data
        sample_news = [
            {
                'title': 'Federal Reserve Maintains Interest Rates',
                'description': 'The Fed keeps rates steady, impacting gold prices.',
                'published_at': datetime.now().isoformat(),
                'source': 'Financial News',
                'sentiment': 'NEUTRAL'
            },
            {
                'title': 'Gold Prices Show Strength Amid Economic Uncertainty',
                'description': 'Investors flock to safe-haven assets like gold.',
                'published_at': (datetime.now() - timedelta(hours=1)).isoformat(),
                'source': 'Market Watch',
                'sentiment': 'POSITIVE'
            },
            {
                'title': 'US Dollar Strengthens, Pressuring Gold',
                'description': 'Strong USD makes gold more expensive for foreign buyers.',
                'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': 'Economic Times',
                'sentiment': 'NEGATIVE'
            }
        ]
        
        await self._set_cache(cache_key, sample_news[:limit], ttl=300)
        return sample_news[:limit]
    
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
    
    async def _set_cache(self, key: str, value, ttl: int = 300):
        """Set cache value with TTL"""
        self.cache[key] = value
        self.cache_expiry[key] = datetime.now() + timedelta(seconds=ttl)
    
    async def _is_cache_valid(self, key: str, ttl: int = None) -> bool:
        """Check if cache is still valid"""
        if key not in self.cache or key not in self.cache_expiry:
            return False
        return datetime.now() < self.cache_expiry[key]
    
    async def get_multiple_timeframe_data(self) -> dict:
        """Get data for all timeframes at once"""
        timeframes = ['1D', '4H', '1H']
        result = {}
        
        for tf in timeframes:
            try:
                data = await self.load_historical_data(tf, 100)
                result[tf] = data
            except Exception as e:
                logger.error(f"Failed to load data for {tf}: {e}")
                result[tf] = self._create_sample_data(100)
                
        return result

# Global instance
data_service = EnhancedDataService()
