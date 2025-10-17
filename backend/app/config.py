import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    # Database paths
    DATA_PATH = "data/"
    
    # API URLs
    TWELVEDATA_BASE_URL = "https://api.twelvedata.com"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    NEWS_API_BASE_URL = "https://newsapi.org/v2"
    
    # WebSocket settings
    WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
    WS_PORT = int(os.getenv("WS_PORT", "8001"))
    
    # Redis for caching and WebSocket
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Analysis settings
    HISTORICAL_DATA_POINTS = 600
    TECHNICAL_INDICATORS = [
        "sma_20", "sma_50", "sma_200",
        "ema_12", "ema_26",
        "rsi_14", 
        "macd", "macd_signal", "macd_histogram",
        "bollinger_upper", "bollinger_lower",
        "stoch_k", "stoch_d",
        "atr_14",
        "volume_sma_20"
    ]
    
    # Real-time settings
    PRICE_UPDATE_INTERVAL = 10  # seconds
    CACHE_TTL = 300  # 5 minutes

settings = Settings()
