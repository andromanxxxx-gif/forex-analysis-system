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
        
        # Build the prompt in parts to avoid the triple-quote issue
        prompt_parts = [
            "# XAUUSD (GOLD) TRADING ANALYSIS REQUEST",
            "",
            "## MARKET CONTEXT",
            f"- **Symbol**: XAU/USD (Gold vs US Dollar)",
            f"- **Current Price**: ${current_price:.2f}",
            f"- **Timeframe**: {timeframe}",
            f"- **Analysis Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## TECHNICAL ANALYSIS SUMMARY",
            f"- **Primary Trend**: {technical_data.get('summary', 'NEUTRAL')}",
            f"- **Confidence Level**: {technical_data.get('confidence', 0) * 100:.1f}%",
            f"- **Trend Strength**: {technical_data.get('trend_strength', 0) * 100:.1f}%",
            f"- **Market Volatility**: {technical_data.get('volatility', 0):.2f}%",
            "",
            "## KEY TECHNICAL LEVELS",
            f"**Support Levels**: {', '.join([f'${level:.2f}' for level in support_levels]) if support_levels else 'None identified'}",
            f"**Resistance Levels**: {', '.join([f'${level:.2f}' for level in resistance_levels]) if resistance_levels else 'None identified'}",
            "",
            "## TECHNICAL INDICATORS DETAILS",
            chr(10).join(technical_indicators) if technical_indicators else 'No indicator data available',
            "",
            "## FUNDAMENTAL CONTEXT",
            news_analysis,
            "",
            "## ANALYSIS REQUEST",
            "",
            "Please provide a comprehensive trading analysis for XAUUSD covering:",
            "",
            "### 1. TECHNICAL ASSESSMENT",
            "- Evaluate the strength and reliability of technical signals",
            "- Assess trend continuity and potential reversal points",
            "- Identify key chart patterns and their implications",
            "",
            "### 2. FUNDAMENTAL IMPACT", 
            "- Analyze how current news and economic factors affect gold prices",
            "- Consider USD strength, inflation data, and geopolitical factors",
            "- Assess central bank policies and their impact on gold",
            "",
            "### 3. MARKET SENTIMENT",
            "- Determine overall market sentiment (Bullish/Bearish/Neutral)",
            "- Evaluate sentiment strength and potential changes",
            "- Consider positioning and market psychology",
            "",
            "### 4. PRICE PREDICTION",
            "- Provide short-term (1-3 days) price outlook",
            "- Provide medium-term (1-2 weeks) projection",  
            "- Provide long-term (1 month) perspective",
            "- Identify specific price targets and key levels",
            "",
            "### 5. RISK ASSESSMENT",
            "- Evaluate overall trading risk (Low/Medium/High)",
            "- Identify specific risk factors and warnings",
            "- Suggest risk management strategies",
            "",
            "### 6. TRADING RECOMMENDATION",
            "- Clear trading recommendation (BUY/SELL/HOLD)",
            "- Entry, target, and stop-loss levels if applicable",
            "- Position sizing suggestions",
            "",
            "## RESPONSE FORMAT",
            "",
            "Please respond with the following JSON structure:",
            "",
            "```json",
            "{",
            '    "technical_summary": "Detailed technical analysis summary (2-3 paragraphs)",',
            '    "fundamental_impact": "Comprehensive fundamental analysis (2-3 paragraphs)",',
            '    "market_sentiment": "BULLISH/BEARISH/NEUTRAL",',
            '    "price_prediction": {',
            '        "short_term": "Specific short-term price outlook",',
            '        "medium_term": "Medium-term price projection",', 
            '        "long_term": "Long-term price perspective",',
            '        "targets": {',
            '            "support": [list of support levels],',
            '            "resistance": [list of resistance levels],',
            '            "immediate_target": target_price,',
            '            "stop_loss": stop_loss_price',
            '        },',
            '        "probability": 0.85',
            '    },',
            '    "risk_assessment": "LOW/MEDIUM/HIGH",',
            '    "recommendation": "BUY/SELL/HOLD",',
            '    "confidence_score": 0.85,',
            '    "key_factors": [',
            '        "Factor 1: Technical pattern development",',
            '        "Factor 2: Fundamental driver impact",',
            '        "Factor 3: Market sentiment shift",',
            '        "Factor 4: Economic data influence",',
            '        "Factor 5: Geopolitical considerations"',
            '    ],',
            '    "warnings": [',
            '        "Warning 1: Specific risk factor",',
            '        "Warning 2: Market condition to monitor",',
            '        "Warning 3: Potential unexpected events"',
            '    ]',
            "}",
            "```",
            "",
            "Provide objective, data-driven analysis suitable for professional traders."
        ]
        
        return "\n".join(prompt_parts)
    
    def _analyze_news_sentiment_detailed(self, news: List[Dict]) -> str:
        """Provide detailed news sentiment analysis"""
        if not news:
            return "No significant fundamental news available. Monitor standard economic indicators and USD trends."
        
        positive_news = []
        negative_news = []
        neutral_news = []
        
        for item in news:
            sentiment = item.get('sentiment', 'NEUTRAL')
            title = item.get('title', '')
            
            if sentiment == 'POSITIVE':
                positive_news.append(f"• {title}")
            elif sentiment == 'NEGATIVE':
                negative_news.append(f"• {title}")
            else:
                neutral_news.append(f"• {title}")
        
        analysis_parts = []
        
        if positive_news:
            analysis_parts.append("🟢 POSITIVE FACTORS:")
            analysis_parts.extend(positive_news[:3])  # Top 3 positive
        
        if negative_news:
            analysis_parts.append("🔴 NEGATIVE FACTORS:")
            analysis_parts.extend(negative_news[:3])  # Top 3 negative
        
        if neutral_news:
            analysis_parts.append("⚪ NEUTRAL DEVELOPMENTS:")
            analysis_parts.extend(neutral_news[:2])  # Top 2 neutral
        
        # Calculate overall sentiment
        total_news = len(news)
        positive_count = len(positive_news)
        negative_count = len(negative_news)
        
        if total_news > 0:
            sentiment_ratio = (positive_count - negative_count) / total_news
            if sentiment_ratio > 0.2:
                overall_sentiment = "BULLISH"
            elif sentiment_ratio < -0.2:
                overall_sentiment = "BEARISH"
            else:
                overall_sentiment = "NEUTRAL"
            
            analysis_parts.append(f"\nOverall News Sentiment: {overall_sentiment}")
        
        return "\n".join(analysis_parts) if analysis_parts else "Limited news impact detected."
    
    async def _call_deepseek_api(self, prompt: str) -> str:
        """
        Call DeepSeek API with enhanced error handling and retry logic
        """
        if not self.session:
            await self.connect()
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "XAUUSD-Analyzer/2.0.0"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a senior financial market analyst specializing in XAUUSD (Gold) trading. Provide professional, data-driven analysis with clear recommendations. Focus on risk management and practical trading insights. Be objective and avoid emotional language. Always consider both technical and fundamental factors."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
        
        max_retries = 2
        retry_delay = 1
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"Calling DeepSeek API (attempt {attempt + 1})")
                
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        logger.info("DeepSeek API call successful")
                        return content
                    
                    elif response.status == 429:
                        logger.warning("DeepSeek API rate limit exceeded")
                        if attempt < max_retries:
                            await asyncio.sleep(retry_delay * (attempt + 1))
                            continue
                        else:
                            raise Exception("API rate limit exceeded after retries")
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error {response.status}: {error_text}")
                        raise Exception(f"API returned status {response.status}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"DeepSeek API timeout (attempt {attempt + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    raise Exception("API timeout after retries")
                    
            except Exception as e:
                logger.error(f"DeepSeek API call failed: {str(e)}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    raise
        
        raise Exception("All DeepSeek API attempts failed")
    
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
        
        # Analyze news sentiment
        news_sentiment_score = self._calculate_news_sentiment_score(fundamental_news)
        news_impact = self._assess_news_impact(fundamental_news)
        
        # Multi-factor decision matrix
        factors = {
            'technical_trend': 1.0 if technical_summary == 'BULLISH' else -1.0 if technical_summary == 'BEARISH' else 0.0,
            'signal_balance': (buy_signals - sell_signals) / max(len(indicators), 1),
            'news_sentiment': news_sentiment_score,
            'trend_strength': trend_strength if technical_summary == 'BULLISH' else -trend_strength if technical_summary == 'BEARISH' else 0.0,
            'volatility_impact': -0.5 if volatility > 3.0 else 0.0  # High volatility reduces confidence
        }
        
        # Calculate composite score
        weights = {
            'technical_trend': 0.35,
            'signal_balance': 0.25,
            'news_sentiment': 0.20,
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
            recommendation = Signal.HOLD
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
        
        fundamental_impact_text = self._generate_fundamental_impact(news_impact, fundamental_news)
        
        # Identify key factors and warnings
        key_factors = self._identify_key_factors(factors, weights, news_impact)
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
    
    def _calculate_news_sentiment_score(self, news: List[Dict]) -> float:
        """Calculate numerical news sentiment score (-1 to 1)"""
        if not news:
            return 0.0
        
        sentiment_values = {
            'POSITIVE': 1.0,
            'NEUTRAL': 0.0,
            'NEGATIVE': -1.0
        }
        
        scores = [sentiment_values.get(item.get('sentiment', 'NEUTRAL'), 0.0) for item in news]
        return sum(scores) / len(scores)
    
    def _assess_news_impact(self, news: List[Dict]) -> Dict[str, Any]:
        """Assess the potential impact of news on gold prices"""
        high_impact_keywords = ['fed', 'rate', 'inflation', 'jobs report', 'cpi', 'interest']
        medium_impact_keywords = ['dollar', 'treasury', 'employment', 'manufacturing']
        
        high_impact_count = 0
        medium_impact_count = 0
        
        for item in news:
            title = item.get('title', '').lower()
            description = item.get('description', '').lower()
            text = f"{title} {description}"
            
            if any(keyword in text for keyword in high_impact_keywords):
                high_impact_count += 1
            elif any(keyword in text for keyword in medium_impact_keywords):
                medium_impact_count += 1
        
        total_impact = high_impact_count * 2 + medium_impact_count
        
        if total_impact >= 3:
            return {"level": "HIGH", "description": "Significant fundamental developments"}
        elif total_impact >= 1:
            return {"level": "MEDIUM", "description": "Moderate fundamental influences"}
        else:
            return {"level": "LOW", "description": "Limited fundamental impact"}
    
    def _generate_price_targets(self, current_price: float, 
                              support_levels: List[float], 
                              resistance_levels: List[float],
                              recommendation: Signal) -> Dict[str, Any]:
        """Generate realistic price targets based on technical levels"""
        
        # Calculate immediate target and stop loss
        if recommendation == Signal.BUY:
            immediate_target = resistance_levels[0] if resistance_levels else current_price * 1.015
            stop_loss = support_levels[0] if support_levels else current_price * 0.985
            short_term = f"Targeting ${immediate_target:.2f} resistance"
            medium_term = f"Potential move to ${resistance_levels[1] if len(resistance_levels) > 1 else current_price * 1.03:.2f}" if resistance_levels else "Monitor resistance breaks"
            long_term = "Dependent on broader trend confirmation"
        elif recommendation == Signal.SELL:
            immediate_target = support_levels[0] if support_levels else current_price * 0.985
            stop_loss = resistance_levels[0] if resistance_levels else current_price * 1.015
            short_term = f"Targeting ${immediate_target:.2f} support"
            medium_term = f"Potential move to ${support_levels[1] if len(support_levels) > 1 else current_price * 0.97:.2f}" if support_levels else "Monitor support breaks"
            long_term = "Dependent on broader trend confirmation"
        else:
            immediate_target = current_price
            stop_loss = current_price * 0.98
            short_term = "Sideways movement expected"
            medium_term = "Await clearer directional signals"
            long_term = "Monitor key level breaks for direction"
        
        return {
            'short_term': short_term,
            'medium_term': medium_term,
            'long_term': long_term,
            'targets': {
                'support': support_levels,
                'resistance': resistance_levels,
                'immediate_target': round(immediate_target, 2),
                'stop_loss': round(stop_loss, 2)
            }
        }
    
    def _generate_technical_summary(self, trend: str, confidence: float, 
                                  trend_strength: float, volatility: float,
                                  indicators: List[Dict]) -> str:
        """Generate comprehensive technical summary"""
        
        base_summary = f"Technical analysis indicates a {trend} trend with {confidence:.1%} confidence. "
        
        strength_assessment = (
            f"Trend strength is {'strong' if trend_strength > 0.7 else 'moderate' if trend_strength > 0.4 else 'weak'}. "
        )
        
        volatility_assessment = (
            f"Market volatility is {'high' if volatility > 3.0 else 'moderate' if volatility > 1.5 else 'low'}. "
        )
        
        signal_count = len(indicators)
        buy_signals = len([i for i in indicators if i.get('signal') == 'BUY'])
        sell_signals = len([i for i in indicators if i.get('signal') == 'SELL'])
        
        signal_assessment = (
            f"Among {signal_count} technical indicators, {buy_signals} suggest BUY, "
            f"{sell_signals} suggest SELL, and {signal_count - buy_signals - sell_signals} are neutral. "
        )
        
        return base_summary + strength_assessment + volatility_assessment + signal_assessment
    
    def _generate_fundamental_impact(self, news_impact: Dict[str, Any], news: List[Dict]) -> str:
        """Generate fundamental impact analysis"""
        
        if not news:
            return "Limited fundamental news available. Focus on technical analysis and monitor upcoming economic data releases."
        
        impact_level = news_impact['level']
        
        if impact_level == "HIGH":
            return f"High impact fundamental developments detected. {news_impact['description']}. Monitor these factors closely as they may drive significant price movements."
        elif impact_level == "MEDIUM":
            return f"Moderate fundamental influences present. {news_impact['description']}. These factors should be considered in trading decisions."
        else:
            return f"Limited fundamental impact. {news_impact['description']}. Technical factors likely to dominate short-term price action."
    
    def _identify_key_factors(self, factors: Dict[str, float], 
                            weights: Dict[str, float],
                            news_impact: Dict[str, Any]) -> List[str]:
        """Identify the most influential factors in the analysis"""
        
        key_factors = []
        
        # Technical factors
        if abs(factors['technical_trend']) > 0.5:
            direction = "bullish" if factors['technical_trend'] > 0 else "bearish"
            key_factors.append(f"Strong {direction} technical trend pattern")
        
        if abs(factors['signal_balance']) > 0.3:
            direction = "buying" if factors['signal_balance'] > 0 else "selling"
            key_factors.append(f"Indicator consensus favors {direction} pressure")
        
        # Fundamental factors
        if news_impact['level'] == "HIGH":
            key_factors.append("High-impact fundamental developments")
        elif news_impact['level'] == "MEDIUM":
            key_factors.append("Moderate fundamental influences")
        
        # Market condition factors
        if factors['volatility_impact'] < -0.2:
            key_factors.append("Elevated market volatility conditions")
        
        if factors['trend_strength'] > 0.6:
            key_factors.append("Strong trend momentum confirmation")
        elif factors['trend_strength'] < 0.3:
            key_factors.append("Weak trend momentum requiring confirmation")
        
        # Always include these standard factors
        key_factors.extend([
            "USD strength and Federal Reserve policy outlook",
            "Global economic uncertainty and safe-haven demand",
            "Inflation expectations and real yield movements"
        ])
        
        return key_factors[:5]  # Return top 5 factors
    
    def _generate_warnings(self, volatility: float, 
                         recommendation: Signal, 
                         news_count: int) -> List[str]:
        """Generate appropriate risk warnings"""
        
        warnings = [
            "All trading involves substantial risk of loss",
            "Use proper position sizing and risk management",
            "Monitor positions closely and adjust stops accordingly"
        ]
        
        if volatility > 4.0:
            warnings.append("High volatility may cause rapid price movements")
        
        if recommendation != Signal.HOLD and news_count == 0:
            warnings.append("Trading without fundamental context increases risk")
        
        if recommendation == Signal.HOLD:
            warnings.append("Wait for clearer directional signals before entering trades")
        
        return warnings
    
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
                recommendation=analysis_data.get('recommendation', 'HOLD'),
                confidence_score=min(1.0, max(0.0, analysis_data.get('confidence_score', 0.5))),
                key_factors=analysis_data.get('key_factors', []),
                warnings=analysis_data.get('warnings', [])
            )
            
            logger.info("AI response parsed successfully")
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response JSON: {str(e)}")
            logger.debug(f"Problematic JSON: {json_str if 'json_str' in locals() else 'N/A'}")
            return self._create_enhanced_fallback_analysis(technical_data, fundamental_news)
            
        except Exception as e:
            logger.error(f"Unexpected error parsing AI response: {str(e)}")
            return self._create_enhanced_fallback_analysis(technical_data, fundamental_news)
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string to handle common formatting issues"""
        # Remove extra whitespace and newlines
        json_str = ' '.join(json_str.split())
        
        # Fix common JSON issues
        json_str = json_str.replace('\\"', '"')
        json_str = json_str.replace('\\n', ' ')
        json_str = json_str.replace('\\t', ' ')
        
        # Ensure proper quotation
        json_str = json_str.replace("'", '"')
        
        return json_str
    
    def _create_enhanced_fallback_analysis(self, 
                                         technical_data: Dict[str, Any],
                                         fundamental_news: List[Dict]) -> AIAnalysis:
        """
        Create comprehensive fallback analysis when AI fails
        """
        logger.info("Creating enhanced fallback analysis")
        
        current_price = 1950.0  # Default fallback price
        support_levels = technical_data.get('support_levels', [])
        resistance_levels = technical_data.get('resistance_levels', [])
        
        news_sentiment = self._calculate_news_sentiment_score(fundamental_news)
        
        return AIAnalysis(
            technical_summary="Comprehensive technical analysis based on multiple indicators and price action patterns. "
                            "Current market conditions require careful monitoring of key support and resistance levels.",
            fundamental_impact="Market fundamentals suggest typical gold price drivers are in play. "
                             "Monitor USD strength, inflation expectations, and geopolitical developments for directional cues.",
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
            risk_assessment="MEDIUM",
            recommendation="HOLD",
            confidence_score=technical_data.get('confidence', 0.5),
            key_factors=[
                "Technical pattern confirmation required",
                "Fundamental catalyst development needed",
                "Market sentiment and positioning analysis",
                "Key support/resistance level behavior",
                "Volume and volatility conditions"
            ],
            warnings=[
                "Exercise caution in current market conditions",
                "Ensure proper risk management with position sizing",
                "Monitor for unexpected market-moving events",
                "Verify signals across multiple timeframes"
            ]
        )


# Global instance with context manager support
ai_analysis_service = EnhancedAIAnalysisService()
