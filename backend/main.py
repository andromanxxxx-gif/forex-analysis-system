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

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        from services.websocket_service import websocket_manager
        await websocket_manager.connect(websocket)
        
        # Send immediate welcome message
        welcome_msg = {
            "type": "connection_established",
            "message": "Connected to XAUUSD AI Analysis",
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_json(welcome_msg)
        
        # Keep connection alive and listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                try:
                    json_data = json.loads(data)
                    await websocket_manager.handle_client_message(websocket, json_data)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {data}")
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket receive error: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if 'websocket_manager' in locals():
            websocket_manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Starting XAUUSD AI Analyzer Backend")
    
    try:
        from services.websocket_service import websocket_manager
        from services.realtime_service import realtime_service
        
        # Connect Redis for WebSocket manager
        await websocket_manager.connect_redis()
        
        # Start real-time price updates
        asyncio.create_task(realtime_service.start_price_updates())
        
        logger.info("✅ All services started successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to start services: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("🛑 Shutting down services...")
    
    try:
        from services.realtime_service import realtime_service
        from services.websocket_service import websocket_manager
        
        await realtime_service.stop_price_updates()
        
        # Disconnect all WebSocket clients
        try:
            await websocket_manager.broadcast({
                "type": "server_shutdown", 
                "message": "Server is shutting down"
            })
            # Close all connections
            for connection in websocket_manager.connections[:]:
                try:
                    await connection.close()
                except:
                    pass
            websocket_manager.connections.clear()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    except ImportError as e:
        logger.error(f"❌ Failed to shutdown services: {e}")
    
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
        from services.data_service import data_service
        from config import settings
        
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
    try:
        from services.websocket_service import websocket_manager
        return {
            "websocket_url": "ws://localhost:8000/ws",
            "connected_clients": len(websocket_manager.connections),
            "supported_messages": [
                "price_update",
                "analysis_update",
                "subscription_confirmed",
                "connection_established"
            ]
        }
    except ImportError:
        return {
            "websocket_url": "ws://localhost:8000/ws",
            "connected_clients": 0,
            "supported_messages": [],
            "error": "WebSocket service not available"
        }

# New endpoints for chart data
@app.get("/api/v1/chart/{timeframe}")
async def get_chart_data(
    timeframe: str,
    limit: int = 100,
    include_indicators: bool = True
):
    """Get historical chart data with technical indicators"""
    try:
        from services.data_service import data_service
        from services.technical_analysis import technical_analyzer
        from config import settings
        
        # Load historical data
        historical_data = await data_service.load_historical_data(timeframe, limit)
        
        if historical_data.empty:
            raise HTTPException(status_code=404, detail="No data available for the specified timeframe")
        
        # Calculate technical indicators
        if include_indicators:
            df_with_indicators = technical_analyzer.calculate_indicators(historical_data)
        else:
            df_with_indicators = historical_data
        
        # Convert to list for JSON response
        chart_data = []
        for _, row in df_with_indicators.iterrows():
            candle = {
                'timestamp': row['timestamp'].isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row.get('volume', 0))
            }
            
            # Add technical indicators if available
            if include_indicators:
                indicators = {}
                for indicator in settings.TECHNICAL_INDICATORS:
                    if indicator in row and pd.notna(row[indicator]):
                        indicators[indicator] = float(row[indicator])
                candle['indicators'] = indicators
            
            chart_data.append(candle)
        
        return {
            'symbol': 'XAUUSD',
            'timeframe': timeframe,
            'data_points': len(chart_data),
            'data': chart_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chart data: {str(e)}")

@app.get("/api/v1/analysis/multi-timeframe")
async def get_multi_timeframe_analysis():
    """Get analysis for all timeframes"""
    try:
        from services.data_service import data_service
        from services.technical_analysis import technical_analyzer
        
        timeframes = ["1D", "4H", "1H"]
        results = {}
        
        for tf in timeframes:
            try:
                historical_data = await data_service.load_historical_data(tf, 100)
                realtime_price = await data_service.get_realtime_price()
                updated_data = data_service.update_realtime_candle(historical_data, realtime_price)
                
                # Technical analysis
                analysis = technical_analyzer.analyze(updated_data)
                
                results[tf] = {
                    "current_price": realtime_price,
                    "trend": analysis.summary.value,
                    "confidence": analysis.confidence,
                    "support_levels": analysis.support_levels[-3:],
                    "resistance_levels": analysis.resistance_levels[:3],
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                results[tf] = {"error": str(e)}
        
        return {
            "symbol": "XAUUSD",
            "timestamp": datetime.now().isoformat(),
            "timeframes": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-timeframe analysis failed: {str(e)}")

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
        app,  # Use the app object directly
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
