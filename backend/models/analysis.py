from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class Timeframe(str, Enum):
    DAILY = "1D"
    FOUR_HOURS = "4H"
    ONE_HOUR = "1H"

class Period(str, Enum):
    SIX_MONTHS = "6m"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"

class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class Trend(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class PriceData(BaseModel):
    timestamp: datetime
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: Optional[float] = Field(None, description="Volume")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class TechnicalIndicator(BaseModel):
    name: str = Field(..., description="Indicator name")
    value: float = Field(..., description="Current value")
    signal: Signal = Field(..., description="Trading signal")
    description: Optional[str] = Field(None, description="Indicator description")
    previous_value: Optional[float] = Field(None, description="Previous value")
    strength: Optional[float] = Field(None, description="Signal strength 0-1")

    class Config:
        use_enum_values = True

class TechnicalAnalysis(BaseModel):
    indicators: List[TechnicalIndicator] = Field(..., description="List of technical indicators")
    summary: Trend = Field(..., description="Overall trend summary")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence 0-1")
    support_levels: List[float] = Field(..., description="Support price levels")
    resistance_levels: List[float] = Field(..., description="Resistance price levels")
    trend_strength: Optional[float] = Field(None, ge=0, le=1, description="Trend strength 0-1")
    volatility: Optional[float] = Field(None, description="Market volatility")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class FundamentalNews(BaseModel):
    title: str = Field(..., description="News title")
    description: str = Field(..., description="News description")
    url: Optional[str] = Field(None, description="News URL")
    published_at: datetime = Field(..., description="Publication time")
    source: str = Field(..., description="News source")
    sentiment: str = Field(..., description="Sentiment analysis")
    impact_score: Optional[float] = Field(None, ge=0, le=1, description="Impact score 0-1")
    category: Optional[str] = Field(None, description="News category")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PricePrediction(BaseModel):
    short_term: str = Field(..., description="Short-term prediction")
    medium_term: Optional[str] = Field(None, description="Medium-term prediction")
    long_term: Optional[str] = Field(None, description="Long-term prediction")
    targets: Dict[str, Any] = Field(..., description="Price targets")
    time_horizon: Optional[str] = Field(None, description="Prediction time horizon")
    probability: Optional[float] = Field(None, ge=0, le=1, description="Prediction probability")

class AIAnalysis(BaseModel):
    technical_summary: str = Field(..., description="Technical analysis summary")
    fundamental_impact: str = Field(..., description="Fundamental analysis impact")
    market_sentiment: Trend = Field(..., description="Overall market sentiment")
    price_prediction: PricePrediction = Field(..., description="Price predictions")
    risk_assessment: RiskLevel = Field(..., description="Risk assessment")
    recommendation: Signal = Field(..., description="Trading recommendation")
    confidence_score: float = Field(..., ge=0, le=1, description="AI confidence score")
    key_factors: List[str] = Field(default_factory=list, description="Key influencing factors")
    warnings: List[str] = Field(default_factory=list, description="Risk warnings")
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AnalysisResponse(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    current_price: float = Field(..., description="Current market price")
    price_change: Optional[float] = Field(None, description="Price change amount")
    price_change_percent: Optional[float] = Field(None, description="Price change percentage")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    timeframe: Timeframe = Field(..., description="Analysis timeframe")
    technical_analysis: TechnicalAnalysis = Field(..., description="Technical analysis")
    ai_analysis: AIAnalysis = Field(..., description="AI analysis")
    chart_data: Dict[str, Any] = Field(..., description="Chart configuration data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HistoricalAnalysis(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    timeframe: Timeframe = Field(..., description="Analysis timeframe")
    period: Period = Field(..., description="Historical period")
    data_points: int = Field(..., description="Number of data points")
    chart_data: Dict[str, Any] = Field(..., description="Chart data")
    technical_summary: str = Field(..., description="Technical summary")
    confidence: float = Field(..., ge=0, le=1, description="Analysis confidence")
    start_date: datetime = Field(..., description="Analysis start date")
    end_date: datetime = Field(..., description="Analysis end date")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Statistical data")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MarketStatusResponse(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Status timestamp")
    timeframes: Dict[str, Dict[str, Any]] = Field(..., description="Status per timeframe")
    overall_trend: Optional[Trend] = Field(None, description="Overall market trend")
    market_condition: Optional[str] = Field(None, description="Market condition description")

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# WebSocket Message Models
class WebSocketMessage(BaseModel):
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = Field(None, description="Message data")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PriceUpdateMessage(WebSocketMessage):
    type: str = "price_update"
    symbol: str = Field(..., description="Trading symbol")
    price: float = Field(..., description="Current price")
    change_percent: Optional[float] = Field(None, description="Price change percentage")
    volume: Optional[float] = Field(None, description="Trading volume")

class AnalysisUpdateMessage(WebSocketMessage):
    type: str = "analysis_update"
    symbol: str = Field(..., description="Trading symbol")
    timeframe: Timeframe = Field(..., description="Analysis timeframe")
    analysis: Dict[str, Any] = Field(..., description="Analysis data")

class SubscriptionMessage(WebSocketMessage):
    type: str = "subscription"
    action: str = Field(..., description="subscribe or unsubscribe")
    symbols: List[str] = Field(..., description="Symbols to subscribe/unsubscribe")
    timeframes: Optional[List[Timeframe]] = Field(None, description="Timeframes to subscribe")

class ErrorMessage(WebSocketMessage):
    type: str = "error"
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error description")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")

# Request Models
class AnalysisRequest(BaseModel):
    timeframe: Timeframe = Field(default=Timeframe.DAILY, description="Analysis timeframe")
    include_news: bool = Field(default=True, description="Include fundamental news")
    include_chart: bool = Field(default=True, description="Include chart data")
    indicators: Optional[List[str]] = Field(None, description="Specific indicators to include")

class HistoricalRequest(BaseModel):
    timeframe: Timeframe = Field(..., description="Analysis timeframe")
    period: Period = Field(..., description="Historical period")
    include_statistics: bool = Field(default=False, description="Include statistical data")

# Response Models for API
class APIResponse(BaseModel):
    success: bool = Field(..., description="Request success status")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error: Optional[str] = Field(None, description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Request identifier")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    services: Dict[str, str] = Field(..., description="Service statuses")
    version: str = Field(..., description="API version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")
    environment: str = Field(..., description="Deployment environment")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Statistics Models
class PriceStatistics(BaseModel):
    current: float = Field(..., description="Current price")
    high_24h: Optional[float] = Field(None, description="24h high")
    low_24h: Optional[float] = Field(None, description="24h low")
    change_24h: Optional[float] = Field(None, description="24h change")
    change_percent_24h: Optional[float] = Field(None, description="24h change percentage")
    volume_24h: Optional[float] = Field(None, description="24h volume")
    average_volume_24h: Optional[float] = Field(None, description="24h average volume")

class MarketStatistics(BaseModel):
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Statistics timestamp")
    price: PriceStatistics = Field(..., description="Price statistics")
    volatility: Optional[float] = Field(None, description="Market volatility")
    trend_strength: Optional[float] = Field(None, description="Trend strength")
    support_count: int = Field(..., description="Number of support levels")
    resistance_count: int = Field(..., description="Number of resistance levels")
    rsi: Optional[float] = Field(None, description="Current RSI")
    macd: Optional[float] = Field(None, description="Current MACD")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
