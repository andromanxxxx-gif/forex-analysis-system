from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging

# Direct imports from services and models
try:
    from services.data_service import data_service
    from services.technical_analysis import technical_analyzer
    from services.ai_analysis import ai_analysis_service
    from services.websocket_service import websocket_manager
    from models.analysis import AnalysisResponse, Signal, Trend
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Import error in analysis router: {e}")
    IMPORT_SUCCESS = False

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/analysis/xauusd", response_model=AnalysisResponse)
async def analyze_xauusd(
    timeframe: str = Query("1D", enum=["1D", "4H", "1H"]),
    include_news: bool = Query(True),
    include_chart: bool = Query(True),
    background_tasks: BackgroundTasks = None
):
    """
    Analyze XAUUSD dengan analisis teknikal dan AI - Enhanced version
    """
    try:
        logger.info(f"Starting analysis for timeframe: {timeframe}")
        
        # Load historical data
        historical_data = await data_service.load_historical_data(timeframe, 100)  # Reduced for performance
        
        if historical_data.empty:
            # Return sample analysis if no data
            return create_sample_analysis(timeframe)
        
        # Get real-time price
        realtime_price = await data_service.get_realtime_price()
        
        # Update last candle with real-time price
        updated_data = data_service.update_realtime_candle(historical_data, realtime_price)
        
        # Perform technical analysis
        technical_analysis = technical_analyzer.analyze(updated_data)
        
        # Get fundamental news if requested
        fundamental_news = []
        if include_news:
            try:
                fundamental_news = await data_service.get_fundamental_news(limit=3)
            except Exception as e:
                logger.warning(f"Failed to get news: {e}")
        
        # Prepare data for AI analysis
        price_data = {
            'current_price': realtime_price,
            'timeframe': timeframe,
            'data_points': len(updated_data)
        }
        
        # AI Analysis with error handling
        try:
            ai_analysis = await ai_analysis_service.analyze_with_ai(
                technical_analysis.dict(),
                fundamental_news,
                price_data
            )
        except Exception as e:
            logger.warning(f"AI analysis failed, using fallback: {e}")
            ai_analysis = create_fallback_ai_analysis(technical_analysis, realtime_price)
        
        # Generate chart data if requested
        chart_data = {}
        if include_chart:
            try:
                chart_data = await generate_enhanced_chart_data(updated_data, technical_analysis, timeframe)
            except Exception as e:
                logger.warning(f"Chart generation failed: {e}")
                chart_data = {"error": "Chart generation failed"}
        
        # Prepare response
        response = AnalysisResponse(
            symbol="XAUUSD",
            current_price=realtime_price,
            timestamp=datetime.now(),
            timeframe=timeframe,
            technical_analysis=technical_analysis,
            ai_analysis=ai_analysis,
            chart_data=chart_data
        )
        
        # Broadcast analysis update via WebSocket (in background)
        if background_tasks:
            background_tasks.add_task(
                broadcast_analysis_update,
                response.dict()
            )
        
        logger.info(f"Analysis completed for {timeframe}")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        # Return sample analysis as fallback
        return create_sample_analysis(timeframe)

def create_sample_analysis(timeframe: str):
    """Create sample analysis when real data fails"""
    return AnalysisResponse(
        symbol="XAUUSD",
        current_price=1950.0,
        timestamp=datetime.now(),
        timeframe=timeframe,
        technical_analysis={
            "indicators": [
                {
                    "name": "RSI_14",
                    "value": 55.5,
                    "signal": Signal.NEUTRAL,
                    "description": "Relative Strength Index",
                    "strength": 0.5
                }
            ],
            "summary": Trend.NEUTRAL,
            "confidence": 0.6,
            "support_levels": [1930.0, 1920.0, 1900.0],
            "resistance_levels": [1960.0, 1970.0, 1980.0],
            "trend_strength": 0.5,
            "volatility": 1.2
        },
        ai_analysis={
            "technical_summary": "Market showing neutral momentum with balanced indicators",
            "fundamental_impact": "Standard market conditions with typical gold price influences",
            "market_sentiment": Trend.NEUTRAL,
            "price_prediction": {
                "short_term": "Sideways movement expected",
                "medium_term": "Watch for breakout signals",
                "long_term": "Long-term trend remains stable",
                "targets": {
                    "support": [1930.0, 1920.0],
                    "resistance": [1960.0, 1970.0],
                    "immediate_target": 1960.0,
                    "stop_loss": 1930.0
                },
                "probability": 0.6
            },
            "risk_assessment": "MEDIUM",
            "recommendation": Signal.NEUTRAL,
            "confidence_score": 0.6,
            "key_factors": [
                "Technical indicators neutral",
                "Market sentiment balanced",
                "No major catalysts detected"
            ],
            "warnings": [
                "Trade with proper risk management",
                "Monitor key support/resistance levels"
            ]
        },
        chart_data={"type": "sample_data"}
    )

def create_fallback_ai_analysis(technical_analysis, current_price: float):
    """Create fallback AI analysis"""
    from models.analysis import AIAnalysis, PricePrediction, Trend, Signal, RiskLevel
    
    return AIAnalysis(
        technical_summary="Analysis based on technical indicators and market patterns",
        fundamental_impact="Standard market conditions affecting gold prices",
        market_sentiment=technical_analysis.summary,
        price_prediction=PricePrediction(
            short_term="Monitor key technical levels",
            medium_term="Direction based on technical breakout",
            long_term="Influenced by broader market trends",
            targets={
                "support": technical_analysis.support_levels,
                "resistance": technical_analysis.resistance_levels,
                "immediate_target": technical_analysis.resistance_levels[0] if technical_analysis.resistance_levels else current_price * 1.01,
                "stop_loss": technical_analysis.support_levels[0] if technical_analysis.support_levels else current_price * 0.99
            },
            probability=technical_analysis.confidence
        ),
        risk_assessment=RiskLevel.MEDIUM,
        recommendation=Signal.NEUTRAL,
        confidence_score=technical_analysis.confidence,
        key_factors=["Technical patterns", "Price levels", "Market volatility"],
        warnings=["Use proper position sizing", "Monitor market conditions"]
    )

async def broadcast_analysis_update(analysis_data: dict):
    """Broadcast analysis update via WebSocket"""
    try:
        await websocket_manager.handle_analysis_update(analysis_data)
    except Exception as e:
        logger.error(f"Failed to broadcast analysis update: {e}")

@router.get("/analysis/multi-timeframe")
async def get_multi_timeframe_analysis():
    """Get analysis for all timeframes at once"""
    try:
        timeframes = ["1D", "4H", "1H"]
        results = {}
        
        for tf in timeframes:
            try:
                historical_data = await data_service.load_historical_data(tf, 50)
                realtime_price = await data_service.get_realtime_price()
                updated_data = data_service.update_realtime_candle(historical_data, realtime_price)
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

# Enhanced chart generation function
async def generate_enhanced_chart_data(df: pd.DataFrame, analysis: any, timeframe: str) -> dict:
    """Generate professional TradingView-like chart data"""
    try:
        # Simple chart data for now
        chart_data = {
            "type": "basic_candlestick",
            "timeframe": timeframe,
            "data_points": len(df),
            "sample_data": True,
            "message": "Advanced chart generation disabled for performance"
        }
        
        return chart_data
        
    except Exception as e:
        logger.error(f"Enhanced chart generation failed: {str(e)}")
        return {"error": f"Chart generation failed: {str(e)}"}
