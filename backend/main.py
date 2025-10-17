import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import asyncio
import logging
import json
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="XAUUSD AI Analysis API",
    description="Backend untuk analisis teknikal dan AI XAUUSD dengan DeepSeek",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
try:
    from routers.analysis import router as analysis_router
    app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])
    logger.info("✅ Analysis router loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to load analysis router: {e}")

# Basic WebSocket endpoint (without services for now)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "message": "Connected to XAUUSD AI Analysis",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_json(welcome_msg)
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    json_data = json.loads(data)
                    # Handle basic messages
                    if json_data.get('type') == 'subscribe':
                        response = {
                            "type": "subscription_confirmed",
                            "symbols": json_data.get('symbols', ['XAUUSD']),
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket.send_json(response)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# Basic startup without external services for now
@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Starting XAUUSD AI Analyzer Backend")
    logger.info("✅ Basic services started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("🛑 Shutting down services...")

@app.get("/")
async def root():
    return {
        "message": "XAUUSD AI Analysis API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "websocket": "active",
            "api": "active"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "websocket": "active"
        },
        "environment": "development"
    }

@app.get("/api/ws-info")
async def websocket_info():
    """WebSocket connection information"""
    return {
        "websocket_url": "ws://localhost:8000/ws",
        "supported_messages": [
            "connection_established",
            "subscription_confirmed"
        ]
    }

# Simple test endpoints
@app.get("/api/test")
async def test_endpoint():
    return {"message": "Test endpoint working!", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/analysis/simple")
async def simple_analysis():
    """Simple analysis endpoint for testing"""
    return {
        "symbol": "XAUUSD",
        "current_price": 1950.0,
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "trend": "BULLISH",
            "confidence": 0.75,
            "recommendation": "BUY"
        }
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting server...")
        uvicorn.run(
            app,  # Use app object directly
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for stability
            log_level="info"
        )
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        input("Press Enter to exit...")
