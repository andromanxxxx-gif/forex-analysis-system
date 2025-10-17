import aiohttp
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.config import settings
from app.models.analysis import (
    AIAnalysis, 
    PricePrediction, 
    Trend, 
    Signal, 
    RiskLevel,
    FundamentalNews
)

logger = logging.getLogger(__name__)

class EnhancedAIAnalysisService:
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def connect(self):
        """Create aiohttp session"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
    async def close(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def analyze_with_ai(self, 
                            technical_data: Dict[str, Any],
                            fundamental_news: List[Dict],
                            price_data: Dict[str, Any]) -> AIAnalysis:
        """
        Enhanced AI analysis with better context and reasoning
        """
        logger.info("Starting AI analysis with DeepSeek")
        
        prompt = self._create_enhanced_analysis_prompt(
            technical_data, 
            fundamental_news, 
            price_data
        )
        
        try:
            await self.connect()
            
            # Try DeepSeek API first if available
            if self._is_deepseek_available():
                analysis_text = await self._call_deepseek_api(prompt)
                analysis = self._parse_enhanced_ai_response(
                    analysis_text, 
                    technical_data, 
                    fundamental_news
                )
                
                # Validate analysis results
                if analysis.confidence_score > 0.3:
                    logger.info(f"AI analysis completed with confidence: {analysis.confidence_score:.1%}")
                    return analysis
                else:
                    logger.warning("AI analysis returned low confidence, using enhanced rule-based fallback")
            
            # Fallback to enhanced rule-based analysis
            analysis = self._enhanced_rule_based_analysis(
                technical_data, 
                fundamental_news, 
                price_data
            )
            logger.info("Using enhanced rule-based analysis")
            return analysis
            
        except Exception as e:
            logger.error(f"AI analysis failed: {str(e)}")
            # Return comprehensive fallback analysis
            return self._create_enhanced_fallback_analysis(
                technical_data, 
                fundamental_news
            )
    
    def _is_deepseek_available(self) -> bool:
        """Check if DeepSeek API is available"""
        return (self.api_key and 
                self.api_key != "your_deepseek_api_key_here" and 
                self.base_url)
    
    def _create_enhanced_analysis_prompt(self, 
                                       technical_data: Dict[str, Any],
                                       fundamental_news: List[Dict],
                                       price_data: Dict[str, Any]) -> str:
        """
        Create enhanced prompt for AI analysis with comprehensive context
        """
        current_price = price_data.get('current_price', 0)
        timeframe = price_data.get('timeframe', '1D')
        
        # Format technical analysis details
        technical_indicators = []
        for indicator in technical_data.get('indicators', []):
            signal_emoji = "🟢" if indicator.get('signal') == 'BUY' else "🔴" if indicator.get('signal') == 'SELL' else "⚪"
            strength = indicator.get('strength', 0.5)
            technical_indicators.append(
                f"{signal_emoji} {indicator.get('name')}: {indicator.get('value', 0):.2f} "
                f"({indicator.get('signal')}, Strength: {strength:.1%})"
            )
        
        # Format fundamental news with sentiment analysis
        news_analysis = self._analyze_news_sentiment_detailed(fundamental_news)
        
        # Format support and resistance levels
        support_levels = technical_data.get('support_levels', [])
        resistance_levels = technical_data.get('resistance_levels', [])
        
        prompt = f"""
# XAUUSD (GOLD) TRADING ANALYSIS REQUEST

## MARKET CONTEXT
- **Symbol**: XAU/USD (Gold vs US Dollar)
- **Current Price**: ${current_price:.2f}
- **Timeframe**: {timeframe}
- **Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## TECHNICAL ANALYSIS SUMMARY
- **Primary Trend**: {technical_data.get('summary', 'NEUTRAL')}
- **Confidence Level**: {technical_data.get('confidence', 0) * 100:.1f}%
- **Trend Strength**: {technical_data.get('trend_strength', 0) * 100:.1f}%
- **Market Volatility**: {technical_data.get('volatility', 0):.2f}%

## KEY TECHNICAL LEVELS
**Support Levels**: {', '.join([f'${level:.2f}' for level in support_levels]) if support_levels else 'None identified'}
**Resistance Levels**: {', '.join([f'${level:.2f}' for level in resistance_levels]) if resistance_levels else 'None identified'}

## TECHNICAL INDICATORS DETAILS
{chr(10).join(technical_indicators) if technical_indicators else 'No indicator data available'}

## FUNDAMENTAL CONTEXT
{news_analysis}

## ANALYSIS REQUEST

Please provide a comprehensive trading analysis for XAUUSD covering:

### 1. TECHNICAL ASSESSMENT
- Evaluate the strength and reliability of technical signals
- Assess trend continuity and potential reversal points
- Identify key chart patterns and their implications

### 2. FUNDAMENTAL IMPACT
- Analyze how current news and economic factors affect gold prices
- Consider USD strength, inflation data, and geopolitical factors
- Assess central bank policies and their impact on gold

### 3. MARKET SENTIMENT
- Determine overall market sentiment (Bullish/Bearish/Neutral)
- Evaluate sentiment strength and potential changes
- Consider positioning and market psychology

### 4. PRICE PREDICTION
- Provide short-term (1-3 days) price outlook
- Provide medium-term (1-2 weeks) projection  
- Provide long-term (1 month) perspective
- Identify specific price targets and key levels

### 5. RISK ASSESSMENT
- Evaluate overall trading risk (Low/Medium/High)
- Identify specific risk factors and warnings
- Suggest risk management strategies

### 6. TRADING RECOMMENDATION
- Clear trading recommendation (BUY/SELL/HOLD)
- Entry, target, and stop-loss levels if applicable
- Position sizing suggestions

## RESPONSE FORMAT

Please respond with the following JSON structure:

```json
{{
    "technical_summary": "Detailed technical analysis summary (2-3 paragraphs)",
    "fundamental_impact": "Comprehensive fundamental analysis (2-3 paragraphs)",
    "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
    "price_prediction": {{
        "short_term": "Specific short-term price outlook",
        "medium_term": "Medium-term price projection", 
        "long_term": "Long-term price perspective",
        "targets": {{
            "support": [list of support levels],
            "resistance": [list of resistance levels],
            "immediate_target": target_price,
            "stop_loss": stop_loss_price
        }},
        "probability": 0.85
    }},
    "risk_assessment": "LOW/MEDIUM/HIGH",
    "recommendation": "BUY/SELL/HOLD",
    "confidence_score": 0.85,
    "key_factors": [
        "Factor 1: Technical pattern development",
        "Factor 2: Fundamental driver impact",
        "Factor 3: Market sentiment shift",
        "Factor 4: Economic data influence",
        "Factor 5: Geopolitical considerations"
    ],
    "warnings": [
        "Warning 1: Specific risk factor",
        "Warning 2: Market condition to monitor",
        "Warning 3: Potential unexpected events"
    ]
}}
