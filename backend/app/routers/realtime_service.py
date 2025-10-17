import asyncio
import aiohttp
import logging
from datetime import datetime
from app.config import settings
from app.services.websocket_service import websocket_manager
from app.services.data_service import data_service

logger = logging.getLogger(__name__)

class RealTimeService:
    def __init__(self):
        self.is_running = False
        self.previous_price = None
        
    async def start_price_updates(self):
        """Start real-time price updates"""
        if self.is_running:
            logger.warning("Real-time service already running")
            return
            
        self.is_running = True
        logger.info("🚀 Starting real-time price updates")
        
        while self.is_running:
            try:
                await self._fetch_and_broadcast_price()
                await asyncio.sleep(settings.PRICE_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in real-time service: {e}")
                await asyncio.sleep(30)  # Wait longer on error
                
    async def stop_price_updates(self):
        """Stop real-time price updates"""
        self.is_running = False
        logger.info("🛑 Stopping real-time price updates")
        
    async def _fetch_and_broadcast_price(self):
        """Fetch current price and broadcast via WebSocket"""
        try:
            # Get real-time price
            current_price = await data_service.get_realtime_price()
            
            # Calculate change percentage
            change_percent = None
            if self.previous_price and self.previous_price > 0:
                change_percent = ((current_price - self.previous_price) / self.previous_price) * 100
                
            # Broadcast price update
            await websocket_manager.handle_price_update(
                symbol="XAUUSD",
                price=current_price,
                change=change_percent
            )
            
            # Update previous price
            self.previous_price = current_price
            
            logger.debug(f"Price update broadcast: ${current_price:.2f} ({change_percent:+.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to fetch/broadcast price: {e}")
            # Don't update previous price on error
            
    async def get_price_history(self, hours: int = 24) -> list:
        """Get recent price history for charting"""
        try:
            # This would typically fetch from database or cache
            # For now, return mock data or empty list
            return []
        except Exception as e:
            logger.error(f"Failed to get price history: {e}")
            return []

# Global real-time service instance
realtime_service = RealTimeService()
