import asyncio
import json
import logging
from typing import Dict, List
from websockets import serve, WebSocketServerProtocol
import redis.asyncio as redis
from app.config import settings

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.connections: List[WebSocketServerProtocol] = []
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
                await connection.send(message_json)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                disconnected.append(connection)
                
        # Remove disconnected clients
        for connection in disconnected:
            self.connections.remove(connection)
            
    async def register(self, websocket: WebSocketServerProtocol):
        """Register a new WebSocket connection"""
        self.connections.append(websocket)
        logger.info(f"✅ New WebSocket connection. Total: {len(self.connections)}")
        
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "message": "Connected to XAUUSD Real-time Server",
            "timestamp": self._get_timestamp()
        }
        await websocket.send(json.dumps(welcome_msg))
        
    async def unregister(self, websocket: WebSocketServerProtocol):
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

async def websocket_handler(websocket: WebSocketServerProtocol, path: str):
    """Main WebSocket handler"""
    await websocket_manager.register(websocket)
    try:
        async for message in websocket:
            # Handle incoming messages from clients
            try:
                data = json.loads(message)
                await handle_client_message(data, websocket)
            except json.JSONDecodeError:
                logger.error("Invalid JSON received from client")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket_manager.unregister(websocket)

async def handle_client_message(data: dict, websocket: WebSocketServerProtocol):
    """Handle messages from clients"""
    message_type = data.get("type")
    
    if message_type == "subscribe":
        # Handle subscription requests
        symbols = data.get("symbols", ["XAUUSD"])
        logger.info(f"Client subscribed to: {symbols}")
        
        response = {
            "type": "subscription_confirmed",
            "symbols": symbols,
            "timestamp": websocket_manager._get_timestamp()
        }
        await websocket.send(json.dumps(response))
        
    elif message_type == "unsubscribe":
        # Handle unsubscription requests
        symbols = data.get("symbols", [])
        logger.info(f"Client unsubscribed from: {symbols}")

async def start_websocket_server():
    """Start the WebSocket server"""
    await websocket_manager.connect_redis()
    server = await serve(
        websocket_handler,
        settings.WS_HOST,
        settings.WS_PORT,
        ping_interval=20,
        ping_timeout=10
    )
    logger.info(f"🚀 WebSocket server started on ws://{settings.WS_HOST}:{settings.WS_PORT}")
    return server
