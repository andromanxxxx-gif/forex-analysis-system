from fastapi import APIRouter, HTTPException, Query
from app.models.analysis import AnalysisResponse  # Import models
from app.services.data_service import data_service
from app.services.technical_analysis import technical_analyzer
from app.services.ai_analysis import ai_analysis_service

router = APIRouter()

@router.get("/analysis/xauusd", response_model=AnalysisResponse)
async def analyze_xauusd(timeframe: str = Query("1D")):
    """
    API Endpoint: Business logic untuk analisis XAUUSD
    """
    try:
        # 1. Load data
        historical_data = await data_service.load_historical_data(timeframe, 600)
        
        # 2. Get real-time price
        realtime_price = await data_service.get_realtime_price()
        
        # 3. Update dengan real-time data
        updated_data = data_service.update_realtime_candle(historical_data, realtime_price)
        
        # 4. Technical analysis
        technical_analysis = technical_analyzer.analyze(updated_data)
        
        # 5. AI analysis
        ai_analysis = await ai_analysis_service.analyze_with_ai(...)
        
        # 6. Generate chart
        chart_data = generate_chart_data(updated_data, technical_analysis, timeframe)
        
        # 7. Return response (menggunakan model dari models/analysis.py)
        return AnalysisResponse(
            symbol="XAUUSD",
            current_price=realtime_price,
            timestamp=datetime.now(),
            technical_analysis=technical_analysis,
            ai_analysis=ai_analysis,
            chart_data=chart_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
