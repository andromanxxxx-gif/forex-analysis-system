import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from app.config import settings
from app.services.websocket_service import websocket_manager
from app.services.data_service import data_service

logger = logging.getLogger(__name__)

class RealTimeService:
    def __init__(self):
        self.is_running = False
        self.previous_price: Optional[float] = None
        self.price_history: list = []
        self.max_history_size = 100
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_update_time: Optional[datetime] = None
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def connect(self):
        """Create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def start_price_updates(self):
        """Start real-time price updates with proper initialization"""
        if self.is_running:
            logger.warning("Real-time service already running")
            return
            
        await self.connect()
        self.is_running = True
        logger.info("🚀 Starting real-time price updates")
        
        # Initial price fetch
        try:
            initial_price = await data_service.get_realtime_price()
            if initial_price is not None:
                self.previous_price = initial_price
                logger.info(f"✅ Initial price set: ${initial_price:.2f}")
        except Exception as e:
            logger.error(f"Failed to get initial price: {e}")
        
        # Start update loop
        while self.is_running:
            try:
                await self._fetch_and_broadcast_price()
                await asyncio.sleep(settings.PRICE_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in real-time service: {e}")
                await asyncio.sleep(30)  # Wait longer on error
                
    async def stop_price_updates(self):
        """Stop real-time price updates and cleanup"""
        self.is_running = False
        logger.info("🛑 Stopping real-time price updates")
        await self.close()
        
    async def _fetch_and_broadcast_price(self):
        """Fetch current price and broadcast via WebSocket with comprehensive error handling"""
        try:
            # Get real-time price with timeout
            current_price = await self._get_price_with_fallback()
            
            # Validate current_price
            if current_price is None:
                logger.warning("Current price is None, using fallback price")
                current_price = await self._get_fallback_price()
                
            if current_price is None:
                logger.error("All price sources failed, skipping update")
                return
            
            # Ensure price is float
            try:
                current_price = float(current_price)
            except (ValueError, TypeError):
                logger.error(f"Invalid price format: {current_price}")
                return
            
            # Validate price range (reasonable gold price range)
            if not (1000 <= current_price <= 3000):
                logger.warning(f"Price outside expected range: ${current_price:.2f}")
                # Continue anyway, might be valid in extreme market conditions
            
            # Calculate change percentage (handle None previous_price)
            change_percent = await self._calculate_change_percent(current_price)
            
            # Update price history
            await self._update_price_history(current_price)
            
            # Broadcast price update
            await self._broadcast_price_update(current_price, change_percent)
            
            # Update previous price
            self.previous_price = current_price
            self.last_update_time = datetime.now()
            
            # Log the update
            await self._log_price_update(current_price, change_percent)
            
        except Exception as e:
            logger.error(f"Failed to fetch/broadcast price: {e}")
            # Don't update previous price on error
    
    async def _get_price_with_fallback(self) -> Optional[float]:
        """Get price with multiple fallback strategies"""
        try:
            # Try primary data service first
            price = await data_service.get_realtime_price()
            if price is not None:
                return price
        except Exception as e:
            logger.warning(f"Primary price source failed: {e}")
        
        # Try historical data fallback
        try:
            price = await self._get_price_from_historical()
            if price is not None:
                logger.info("Using historical data as price fallback")
                return price
        except Exception as e:
            logger.warning(f"Historical price fallback failed: {e}")
        
        # Try cache fallback
        try:
            price = await self._get_price_from_cache()
            if price is not None:
                logger.info("Using cached price as fallback")
                return price
        except Exception as e:
            logger.warning(f"Cache price fallback failed: {e}")
            
        return None
    
    async def _get_fallback_price(self) -> Optional[float]:
        """Get fallback price when all sources fail"""
        # Use previous price if available and recent
        if (self.previous_price is not None and 
            self.last_update_time and 
            (datetime.now() - self.last_update_time).total_seconds() < 300):  # 5 minutes
            logger.info("Using previous price as fallback")
            return self.previous_price
        
        # Use last known price from history
        if self.price_history:
            logger.info("Using last known price from history as fallback")
            return self.price_history[-1]['price']
        
        # Ultimate fallback - reasonable default
        logger.warning("Using default fallback price")
        return 1950.0  # Reasonable default gold price
    
    async def _get_price_from_historical(self) -> Optional[float]:
        """Get approximate price from latest historical data"""
        try:
            # Try different timeframes in order of recency
            for timeframe in ['1H', '4H', '1D']:
                try:
                    df = await data_service.load_historical_data(timeframe, limit=1)
                    if not df.empty and 'close' in df.columns:
                        price = float(df.iloc[-1]['close'])
                        if price > 0:
                            return price
                except Exception as e:
                    logger.debug(f"Failed to get price from {timeframe}: {e}")
                    continue
                    
            return None
        except Exception as e:
            logger.error(f"Historical price extraction failed: {e}")
            return None
    
    async def _get_price_from_cache(self) -> Optional[float]:
        """Get price from cache if available"""
        try:
            cache_key = "realtime_price"
            if hasattr(data_service, 'cache') and cache_key in data_service.cache:
                cached_price = data_service.cache[cache_key]
                if cached_price is not None:
                    return cached_price
            return None
        except Exception as e:
            logger.debug(f"Cache access failed: {e}")
            return None
    
    async def _calculate_change_percent(self, current_price: float) -> Optional[float]:
        """Calculate price change percentage safely"""
        try:
            if (self.previous_price is not None and 
                self.previous_price > 0 and 
                current_price > 0):
                
                change_percent = ((current_price - self.previous_price) / self.previous_price) * 100
                
                # Validate change is reasonable (less than 10% change in one update)
                if abs(change_percent) < 10:
                    return round(change_percent, 4)  # Round to 4 decimal places
                else:
                    logger.warning(f"Unusual price change detected: {change_percent:.2f}%")
                    return None
                    
            return None
        except (ZeroDivisionError, TypeError, ValueError) as e:
            logger.debug(f"Change calculation error: {e}")
            return None
    
    async def _update_price_history(self, current_price: float):
        """Update price history for tracking"""
        price_point = {
            'price': current_price,
            'timestamp': datetime.now(),
            'volume': None  # Would be actual volume if available
        }
        
        self.price_history.append(price_point)
        
        # Limit history size
        if len(self.price_history) > self.max_history_size:
            self.price_history = self.price_history[-self.max_history_size:]
    
    async def _broadcast_price_update(self, current_price: float, change_percent: Optional[float]):
        """Broadcast price update via WebSocket"""
        try:
            await websocket_manager.handle_price_update(
                symbol="XAUUSD",
                price=current_price,
                change=change_percent
            )
        except Exception as e:
            logger.error(f"WebSocket broadcast failed: {e}")
    
    async def _log_price_update(self, current_price: float, change_percent: Optional[float]):
        """Log price update appropriately"""
        if change_percent is not None:
            change_direction = "↑" if change_percent > 0 else "↓" if change_percent < 0 else "→"
            logger.debug(f"Price update: ${current_price:.2f} {change_direction} ({change_percent:+.2f}%)")
        else:
            logger.debug(f"Price update: ${current_price:.2f} (No change data)")
        
        # Log significant changes
        if change_percent is not None and abs(change_percent) > 1.0:
            logger.info(f"Significant price movement: ${current_price:.2f} ({change_percent:+.2f}%)")
    
    async def get_current_status(self) -> Dict[str, Any]:
        """Get current real-time service status"""
        return {
            "is_running": self.is_running,
            "previous_price": self.previous_price,
            "last_update_time": self.last_update_time.isoformat() if self.last_update_time else None,
            "history_size": len(self.price_history),
            "update_interval": settings.PRICE_UPDATE_INTERVAL
        }
    
    async def get_price_history_data(self, limit: int = 50) -> list:
        """Get recent price history for charting"""
        try:
            history = self.price_history[-limit:] if self.price_history else []
            
            # Format for frontend
            formatted_history = []
            for point in history:
                formatted_history.append({
                    'timestamp': point['timestamp'].isoformat(),
                    'price': point['price'],
                    'volume': point['volume']
                })
            
            return formatted_history
        except Exception as e:
            logger.error(f"Failed to get price history: {e}")
            return []
    
    async def get_price_statistics(self) -> Dict[str, Any]:
        """Get price statistics from recent history"""
        try:
            if not self.price_history:
                return {}
            
            prices = [point['price'] for point in self.price_history if point['price'] is not None]
            
            if not prices:
                return {}
            
            current_price = prices[-1]
            min_price = min(prices)
            max_price = max(prices)
            avg_price = sum(prices) / len(prices)
            
            # Calculate volatility (standard deviation)
            if len(prices) > 1:
                price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
                volatility = sum(price_changes) / len(price_changes)
            else:
                volatility = 0
            
            return {
                "current": current_price,
                "min": min_price,
                "max": max_price,
                "average": round(avg_price, 2),
                "volatility": round(volatility, 4),
                "data_points": len(prices),
                "time_period": f"Last {len(prices)} updates"
            }
        except Exception as e:
            logger.error(f"Failed to calculate price statistics: {e}")
            return {}
    
    async def force_price_update(self):
        """Force an immediate price update"""
        logger.info("Forcing immediate price update")
        await self._fetch_and_broadcast_price()
    
    async def reset_service(self):
        """Reset the service state"""
        logger.info("Resetting real-time service")
        self.previous_price = None
        self.price_history = []
        self.last_update_time = None
        logger.info("Real-time service reset complete")

# Global real-time service instance
realtime_service = RealTimeService()

# Utility functions for external use
async def start_realtime_service():
    """Start the real-time service (for use in main.py)"""
    await realtime_service.start_price_updates()

async def stop_realtime_service():
    """Stop the real-time service (for use in main.py)"""
    await realtime_service.stop_price_updates()

async def get_realtime_status():
    """Get real-time service status (for API endpoints)"""
    return await realtime_service.get_current_status()

async def get_realtime_history(limit: int = 50):
    """Get real-time price history (for API endpoints)"""
    return await realtime_service.get_price_history_data(limit)

async def get_realtime_statistics():
    """Get real-time price statistics (for API endpoints)"""
    return await realtime_service.get_price_statistics()
