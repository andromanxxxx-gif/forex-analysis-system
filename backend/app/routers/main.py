from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime
import asyncio
import logging

from app.routers import analysis
from app.config import settings
from app.services.websocket_service import start_websocket_server, websocket_manager
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
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis.router, prefix="/api/v1", tags=["analysis"])

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("🚀 Starting XAUUSD AI Analyzer Backend")
    
    # Start WebSocket server in background
    asyncio.create_task(start_websocket_server())
    
    # Start real-time price updates
    asyncio.create_task(realtime_service.start_price_updates())
    
    logger.info("✅ All services started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("🛑 Shutting down services...")
    await realtime_service.stop_price_updates()
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
        "websocket_url": f"ws://{settings.WS_HOST}:{settings.WS_PORT}",
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
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
# Tambahkan di main.py setelah endpoint yang sudah ada
@app.get("/api/v1/chart/{timeframe}")
async def get_chart_data(
    timeframe: str,
    limit: int = Query(100, ge=10, le=1000),
    include_indicators: bool = Query(True)
):
    """Get historical chart data with technical indicators"""
    try:
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
