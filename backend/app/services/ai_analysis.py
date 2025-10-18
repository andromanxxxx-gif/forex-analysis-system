import aiohttp
import json
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional

# Direct imports
try:
    from config import settings
    from models.analysis import AIAnalysis, PricePrediction, Trend, Signal, RiskLevel, FundamentalNews
except ImportError:
    # Fallback classes
    class Signal:
        BUY = "BUY"
        SELL = "SELL"
        HOLD = "HOLD"
        NEUTRAL = "NEUTRAL"

    class Trend:
        BULLISH = "BULLISH"
        BEARISH = "BEARISH"
        NEUTRAL = "NEUTRAL"

    class RiskLevel:
        LOW = "LOW"
        MEDIUM = "MEDIUM"
        HIGH = "HIGH"

    class PricePrediction:
        def __init__(self, short_term, medium_term=None, long_term=None, targets=None, probability=None):
            self.short_term = short_term
            self.medium_term = medium_term
            self.long_term = long_term
            self.targets = targets or {}
            self.probability = probability

    class AIAnalysis:
        def __init__(self, technical_summary, fundamental_impact, market_sentiment, price_prediction, risk_assessment, recommendation, confidence_score, key_factors=None, warnings=None):
            self.technical_summary = technical_summary
            self.fundamental_impact = fundamental_impact
            self.market_sentiment = market_sentiment
            self.price_prediction = price_prediction
            self.risk_assessment = risk_assessment
            self.recommendation = recommendation
            self.confidence_score = confidence_score
            self.key_factors = key_factors or []
            self.warnings = warnings or []

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
        """Create enhanced prompt for AI analysis"""
        # ... (kode prompt creation tetap sama)
        return prompt
    
    def _enhanced_rule_based_analysis(self, 
                                    technical_data: Dict[str, Any],
                                    fundamental_news: List[Dict],
                                    price_data: Dict[str, Any]) -> AIAnalysis:
        """
        Enhanced rule-based analysis with sophisticated decision logic
        """
        logger.info("Executing enhanced rule-based analysis")
        
        current_price = price_data.get('current_price', 1950.0)
        timeframe = price_data.get('timeframe', '1D')
        
        # Extract technical metrics
        technical_summary = technical_data.get('summary', 'NEUTRAL')
        confidence = technical_data.get('confidence', 0.5)
        trend_strength = technical_data.get('trend_strength', 0.5)
        volatility = technical_data.get('volatility', 0.0)
        support_levels = technical_data.get('support_levels', [])
        resistance_levels = technical_data.get('resistance_levels', [])
        
        # Analyze indicators
        indicators = technical_data.get('indicators', [])
        buy_signals = len([i for i in indicators if i.get('signal') == 'BUY'])
        sell_signals = len([i for i in indicators if i.get('signal') == 'SELL'])
        signal_strengths = [i.get('strength', 0.5) for i in indicators if i.get('strength') is not None]
        avg_signal_strength = sum(signal_strengths) / len(signal_strengths) if signal_strengths else 0.5
        
        # Multi-factor decision matrix
        factors = {
            'technical_trend': 1.0 if technical_summary == 'BULLISH' else -1.0 if technical_summary == 'BEARISH' else 0.0,
            'signal_balance': (buy_signals - sell_signals) / max(len(indicators), 1),
            'trend_strength': trend_strength if technical_summary == 'BULLISH' else -trend_strength if technical_summary == 'BEARISH' else 0.0,
            'volatility_impact': -0.5 if volatility > 3.0 else 0.0
        }
        
        # Calculate composite score
        weights = {
            'technical_trend': 0.35,
            'signal_balance': 0.25,
            'trend_strength': 0.15,
            'volatility_impact': 0.05
        }
        
        composite_score = sum(factors[key] * weights[key] for key in factors)
        
        # Determine recommendation based on composite score
        if composite_score > 0.3:
            recommendation = Signal.BUY
            market_sentiment = Trend.BULLISH
            confidence_score = min(0.85, confidence + 0.15)
        elif composite_score < -0.3:
            recommendation = Signal.SELL
            market_sentiment = Trend.BEARISH
            confidence_score = min(0.85, confidence + 0.15)
        else:
            recommendation = Signal.NEUTRAL  # ✅ PERBAIKAN: Ganti HOLD dengan NEUTRAL
            market_sentiment = Trend.NEUTRAL
            confidence_score = max(0.4, confidence)
        
        # Adjust for volatility
        if volatility > 5.0:
            risk_level = RiskLevel.HIGH
            confidence_score *= 0.8
        elif volatility > 2.0:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Generate price targets
        price_targets = self._generate_price_targets(
            current_price, 
            support_levels, 
            resistance_levels, 
            recommendation
        )
        
        # Generate analysis text
        technical_summary_text = self._generate_technical_summary(
            technical_summary, confidence, trend_strength, volatility, indicators
        )
        
        fundamental_impact_text = "Market fundamentals suggest typical gold price drivers are in play. Monitor USD strength and economic indicators."
        
        # Identify key factors and warnings
        key_factors = self._identify_key_factors(factors, weights)
        warnings = self._generate_warnings(volatility, recommendation, len(fundamental_news))
        
        return AIAnalysis(
            technical_summary=technical_summary_text,
            fundamental_impact=fundamental_impact_text,
            market_sentiment=market_sentiment,
            price_prediction=PricePrediction(
                short_term=price_targets['short_term'],
                medium_term=price_targets['medium_term'],
                long_term=price_targets['long_term'],
                targets=price_targets['targets'],
                probability=confidence_score,
                time_horizon=f"{timeframe} perspective"
            ),
            risk_assessment=risk_level,
            recommendation=recommendation,
            confidence_score=confidence_score,
            key_factors=key_factors,
            warnings=warnings
        )
    
    def _create_enhanced_fallback_analysis(self, 
                                         technical_data: Dict[str, Any],
                                         fundamental_news: List[Dict]) -> AIAnalysis:
        """
        Create comprehensive fallback analysis when AI fails
        """
        logger.info("Creating enhanced fallback analysis")
        
        current_price = 1950.0
        support_levels = technical_data.get('support_levels', [])
        resistance_levels = technical_data.get('resistance_levels', [])
        
        return AIAnalysis(
            technical_summary="Comprehensive technical analysis based on multiple indicators and price action patterns.",
            fundamental_impact="Market fundamentals suggest typical gold price drivers are in play.",
            market_sentiment=technical_data.get('summary', 'NEUTRAL'),
            price_prediction=PricePrediction(
                short_term="Monitor key technical levels for breakout confirmation",
                medium_term="Direction dependent on fundamental catalyst development",
                long_term="Long-term trend remains influenced by macroeconomic factors",
                targets={
                    "support": support_levels,
                    "resistance": resistance_levels,
                    "immediate_target": resistance_levels[0] if resistance_levels else current_price * 1.02,
                    "stop_loss": support_levels[0] if support_levels else current_price * 0.98
                },
                probability=technical_data.get('confidence', 0.5)
            ),
            risk_assessment=RiskLevel.MEDIUM,
            recommendation=Signal.NEUTRAL,  # ✅ PERBAIKAN: Ganti HOLD dengan NEUTRAL
            confidence_score=technical_data.get('confidence', 0.5),
            key_factors=[
                "Technical pattern confirmation required",
                "Market sentiment and positioning analysis",
                "Key support/resistance level behavior"
            ],
            warnings=[
                "Exercise caution in current market conditions",
                "Ensure proper risk management with position sizing"
            ]
        )
    
    def _parse_enhanced_ai_response(self, response_text: str, 
                                  technical_data: Dict[str, Any],
                                  fundamental_news: List[Dict]) -> AIAnalysis:
        """
        Parse enhanced AI response with robust error handling
        """
        try:
            logger.info("Parsing AI response")
            
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON found in AI response, using fallback")
                return self._create_enhanced_fallback_analysis(technical_data, fundamental_news)
            
            json_str = response_text[start_idx:end_idx]
            
            # Clean the JSON string
            json_str = self._clean_json_string(json_str)
            
            analysis_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['technical_summary', 'fundamental_impact', 'market_sentiment', 
                             'price_prediction', 'risk_assessment', 'recommendation', 'confidence_score']
            
            for field in required_fields:
                if field not in analysis_data:
                    logger.warning(f"Missing required field in AI response: {field}")
                    return self._create_enhanced_fallback_analysis(technical_data, fundamental_news)
            
            # Handle recommendation - convert to valid Signal enum
            recommendation_str = analysis_data.get('recommendation', 'NEUTRAL')
            if recommendation_str == 'HOLD':
                recommendation_str = 'NEUTRAL'  # ✅ Convert HOLD to NEUTRAL
            
            try:
                recommendation = Signal(recommendation_str)
            except ValueError:
                logger.warning(f"Invalid recommendation: {recommendation_str}, using NEUTRAL")
                recommendation = Signal.NEUTRAL
            
            # Create PricePrediction object
            price_prediction_data = analysis_data.get('price_prediction', {})
            price_prediction = PricePrediction(
                short_term=price_prediction_data.get('short_term', ''),
                medium_term=price_prediction_data.get('medium_term', ''),
                long_term=price_prediction_data.get('long_term', ''),
                targets=price_prediction_data.get('targets', {}),
                probability=price_prediction_data.get('probability', 0.5),
                time_horizon=price_prediction_data.get('time_horizon', 'Short-term')
            )
            
            # Create AIAnalysis object
            analysis = AIAnalysis(
                technical_summary=analysis_data.get('technical_summary', ''),
                fundamental_impact=analysis_data.get('fundamental_impact', ''),
                market_sentiment=analysis_data.get('market_sentiment', 'NEUTRAL'),
                price_prediction=price_prediction,
                risk_assessment=analysis_data.get('risk_assessment', 'MEDIUM'),
                recommendation=recommendation,
                confidence_score=min(1.0, max(0.0, analysis_data.get('confidence_score', 0.5))),
                key_factors=analysis_data.get('key_factors', []),
                warnings=analysis_data.get('warnings', [])
            )
            
            logger.info("AI response parsed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Unexpected error parsing AI response: {str(e)}")
            return self._create_enhanced_fallback_analysis(technical_data, fundamental_news)
    
    # ... (method lainnya tetap sama)

# Global instance with context manager support
ai_analysis_service = EnhancedAIAnalysisService()
