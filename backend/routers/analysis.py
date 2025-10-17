from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging

# Import services directly
try:
    from services.data_service import data_service
    from services.technical_analysis import technical_analyzer
    from services.ai_analysis import ai_analysis_service
    from services.websocket_service import websocket_manager
    from models.analysis import AnalysisResponse
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"❌ Import error in analysis router: {e}")
    IMPORT_SUCCESS = False

logger = logging.getLogger(__name__)

router = APIRouter()

if IMPORT_SUCCESS:
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
            historical_data = await data_service.load_historical_data(timeframe, 600)
            
            if historical_data.empty:
                raise HTTPException(status_code=404, detail="Historical data not found")
            
            # Get real-time price
            realtime_price = await data_service.get_realtime_price()
            
            # Update last candle with real-time price
            updated_data = data_service.update_realtime_candle(historical_data, realtime_price)
            
            # Perform technical analysis
            technical_analysis = technical_analyzer.analyze(updated_data)
            
            # Get fundamental news if requested
            fundamental_news = []
            if include_news:
                fundamental_news = await data_service.get_fundamental_news()
            
            # Prepare data for AI analysis
            price_data = {
                'current_price': realtime_price,
                'timeframe': timeframe,
                'data_points': len(updated_data)
            }
            
            # AI Analysis
            ai_analysis = await ai_analysis_service.analyze_with_ai(
                technical_analysis.dict(),
                fundamental_news,
                price_data
            )
            
            # Generate chart data if requested
            chart_data = {}
            if include_chart:
                chart_data = await generate_enhanced_chart_data(updated_data, technical_analysis, timeframe)
            
            # Prepare response
            response = AnalysisResponse(
                symbol="XAUUSD",
                current_price=realtime_price,
                timestamp=datetime.now(),
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
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
                    # Use a simplified version for multi-timeframe to avoid overloading
                    historical_data = await data_service.load_historical_data(tf, 100)
                    realtime_price = await data_service.get_realtime_price()
                    updated_data = data_service.update_realtime_candle(historical_data, realtime_price)
                    analysis = technical_analyzer.analyze(updated_data)
                    
                    results[tf] = {
                        "current_price": realtime_price,
                        "trend": analysis.summary,
                        "confidence": analysis.confidence,
                        "support_levels": analysis.support_levels[-3:],
                        "resistance_levels": analysis.resistance_levels[-3:],
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
        """Generate professional TradingView-like chart data with enhanced features"""
        try:
            # Create subplots with more features
            fig = make_subplots(
                rows=3, cols=1,
                shared_x=True,
                vertical_spacing=0.08,
                subplot_titles=('Price Chart', 'Technical Indicators', 'Volume'),
                row_width=[0.5, 0.25, 0.25]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='XAUUSD'
                ),
                row=1, col=1
            )
            
            # Add technical indicators if available
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['sma_20'],
                        line=dict(color='orange', width=1.5),
                        name='SMA 20'
                    ),
                    row=1, col=1
                )
            
            if 'sma_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['sma_50'],
                        line=dict(color='blue', width=1.5),
                        name='SMA 50'
                    ),
                    row=1, col=1
                )
            
            # RSI in second subplot
            if 'rsi_14' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['rsi_14'],
                        line=dict(color='purple', width=1.5),
                        name='RSI 14'
                    ),
                    row=2, col=1
                )
                
                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Volume in third subplot
            if 'volume' in df.columns:
                colors = ['red' if row['close'] < row['open'] else 'green' 
                         for _, row in df.iterrows()]
                
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['volume'],
                        marker_color=colors,
                        name='Volume'
                    ),
                    row=3, col=1
                )
            
            # Update layout for better appearance
            fig.update_layout(
                title=f"XAUUSD {timeframe} Chart - Comprehensive Analysis",
                xaxis_rangeslider_visible=False,
                height=700,
                showlegend=True,
                template="plotly_white",
                hovermode='x unified'
            )
            
            # Update axis labels
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
            fig.update_yaxes(title_text="Volume", row=3, col=1)
            
            # Convert to JSON
            chart_json = fig.to_json()
            
            return {
                "chart_type": "advanced_candlestick",
                "timeframe": timeframe,
                "data_points": len(df),
                "plotly_config": json.loads(chart_json),
                "technical_indicators": {
                    "moving_averages": ["sma_20", "sma_50"] if all(col in df.columns for col in ['sma_20', 'sma_50']) else [],
                    "oscillators": ["rsi_14"] if 'rsi_14' in df.columns else [],
                    "volume": "volume" in df.columns,
                    "bollinger_bands": "bollinger_upper" in df.columns and "bollinger_lower" in df.columns
                },
                "support_levels": analysis.support_levels,
                "resistance_levels": analysis.resistance_levels
            }
            
        except Exception as e:
            logger.error(f"Enhanced chart generation failed: {str(e)}")
            return {"error": f"Chart generation failed: {str(e)}"}

else:
    @router.get("/analysis/xauusd")
    async def analyze_xauusd_fallback():
        """Fallback analysis endpoint when imports fail"""
        return {
            "symbol": "XAUUSD",
            "current_price": 1950.0,
            "timestamp": datetime.now().isoformat(),
            "error": "Analysis services not available - check imports",
            "technical_analysis": {
                "summary": "NEUTRAL",
                "confidence": 0.5,
                "support_levels": [],
                "resistance_levels": []
            },
            "ai_analysis": {
                "recommendation": "HOLD",
                "confidence_score": 0.5,
                "risk_assessment": "MEDIUM"
            }
        }
