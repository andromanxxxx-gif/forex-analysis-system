import asyncio
import json
import logging
from typing import Dict, List
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis
from ..config import settings

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.connections: List[WebSocket] = []  # Changed to FastAPI WebSocket
        self.redis_client = None
        
    async def connect_redis(self):
        """Connect to Redis for pub/sub"""
        try:
            self.redis_client = await redis.from_url(settings.REDIS_URL)
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            
    async def connect(self, websocket: WebSocket):
        """Register a new WebSocket connection (FastAPI version)"""
        await websocket.accept()
        self.connections.append(websocket)
        logger.info(f"✅ New WebSocket connection. Total: {len(self.connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Unregister a WebSocket connection"""
        if websocket in self.connections:
            self.connections.remove(websocket)
            logger.info(f"🔌 WebSocket disconnected. Total: {len(self.connections)}")
            
    async def _send_message(self, websocket: WebSocket, message: dict):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send message to client: {e}")
            self.disconnect(websocket)
            
    async def safe_broadcast(self, message: dict):
        """Safe broadcast that handles disconnected clients"""
        if not self.connections:
            return
            
        message_json = json.dumps(message)
        disconnected = []
        
        for connection in self.connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"Client disconnected, removing from connections: {e}")
                disconnected.append(connection)
                
        # Remove disconnected clients
        for connection in disconnected:
            self.connections.remove(connection)
        
        if disconnected:
            logger.info(f"Removed {len(disconnected)} disconnected clients")
            
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        await self.safe_broadcast(message)
            
    async def handle_client_message(self, websocket: WebSocket, data: dict):
        """Handle messages from clients"""
        message_type = data.get("type")
        
        if message_type == "subscribe":
            # Handle subscription requests
            symbols = data.get("symbols", ["XAUUSD"])
            logger.info(f"Client subscribed to: {symbols}")
            
            response = {
                "type": "subscription_confirmed",
                "symbols": symbols,
                "timestamp": self._get_timestamp()
            }
            await self._send_message(websocket, response)
            
        elif message_type == "unsubscribe":
            # Handle unsubscription requests
            symbols = data.get("symbols", [])
            logger.info(f"Client unsubscribed from: {symbols}")
            
        elif message_type == "ping":
            # Handle ping/pong
            response = {
                "type": "pong",
                "timestamp": self._get_timestamp()
            }
            await self._send_message(websocket, response)
        
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
