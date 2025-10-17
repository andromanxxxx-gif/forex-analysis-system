from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime
import asyncio
import logging

from app.routers import analysis
from app.config import settings
from app.services.websocket_service import websocket_manager
from app.services.realtime_service import realtime_service

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

# CORS middleware - updated for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",  # Tambahkan ini
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

# WebSocket endpoint langsung di FastAPI
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Starting XAUUSD AI Analyzer Backend")
    
    # HAPUS: start_websocket_server() - sekarang sudah terintegrasi di FastAPI
    
    # Start real-time price updates
    asyncio.create_task(realtime_service.start_price_updates())
    
    logger.info("✅ All services started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("🛑 Shutting down services...")
    await realtime_service.stop_price_updates()
    
    # Disconnect all WebSocket clients
    await websocket_manager.broadcast({"type": "server_shutdown", "message": "Server is shutting down"})
    for connection in websocket_manager.connections:
        await connection.close()
    
    logger.info("✅ Services shut down successfully")

@app.get("/")
async def root():
    return {
        "message": "XAUUSD AI Analysis API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "websocket": "active",
            "realtime_updates": "active",
            "ai_analysis": "active"
        }
    }

@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Test data service
        from app.services.data_service import data_service
        price = await data_service.get_realtime_price()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "data_service": "operational",
                "technical_analysis": "operational",
                "ai_analysis": "operational",
                "websocket": "active",
                "realtime_updates": "active"
            },
            "current_price": price,
            "environment": "development" if settings.DEBUG else "production"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "services": {
                "data_service": "degraded",
                "technical_analysis": "unknown",
                "ai_analysis": "unknown",
                "websocket": "active",
                "realtime_updates": "active"
            }
        }

@app.get("/api/ws-info")
async def websocket_info():
    """WebSocket connection information"""
    return {
        "websocket_url": "ws://localhost:8000/ws",  # Update URL
        "connected_clients": len(websocket_manager.connections),
        "supported_messages": [
            "price_update",
            "analysis_update",
            "subscription_confirmed"
        ]
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
    uvicorn.run(
        "main:app",  # Update ini
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
