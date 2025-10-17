import asyncio
import json
import logging
from typing import List
from fastapi import WebSocket
import redis.asyncio as redis
from app.config import settings

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self.redis_client = None
        
    async def connect_redis(self):
        """Connect to Redis for pub/sub"""
        try:
            self.redis_client = await redis.from_url(settings.REDIS_URL)
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.connections:
            return
            
        message_json = json.dumps(message)
        disconnected = []
        
        for connection in self.connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected.append(connection)
                
        # Remove disconnected clients
        for connection in disconnected:
            self.connections.remove(connection)
            
    async def connect(self, websocket: WebSocket):
        """Register a new WebSocket connection"""
        await websocket.accept()
        self.connections.append(websocket)
        logger.info(f"✅ New WebSocket connection. Total: {len(self.connections)}")
        
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "message": "Connected to XAUUSD Real-time Server",
            "timestamp": self._get_timestamp()
        }
        await websocket.send_text(json.dumps(welcome_msg))
        
    def disconnect(self, websocket: WebSocket):
        """Unregister a WebSocket connection"""
        if websocket in self.connections:
            self.connections.remove(websocket)
            logger.info(f"🔌 WebSocket disconnected. Total: {len(self.connections)}")
            
    async def handle_price_update(self, symbol: str, price: float, change: float = None):
        """Handle price update and broadcast to clients"""
        message = {
            "type": "price_update",
            "symbol": symbol,
            "price": price,
            "change_percent": change,
            "timestamp": self._get_timestamp()
        }
        await self.broadcast(message)
        
    async def handle_analysis_update(self, analysis_data: dict):
        """Handle analysis update and broadcast to clients"""
        message = {
            "type": "analysis_update",
            "data": analysis_data,
            "timestamp": self._get_timestamp()
        }
        await self.broadcast(message)
        
    def _get_timestamp(self):
        import datetime
        return datetime.datetime.now().isoformat()

# Global WebSocket manager instance
websocket_manager = WebSocketManager()

# Hapus fungsi-fungsi yang tidak digunakan lagi: websocket_handler, start_websocket_server, handle_client_message
