import requests
import json
import logging
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)

class AIPredictor:
    """
    Integrasi dengan DeepSeek AI untuk analisis forex
    """
    
    def __init__(self, api_key: str = "sk-73d83584fd614656926e1d8860eae9ca"):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.max_retries = 3
        self.timeout = 30
    
    def analyze_forex_pair(self, pair: str, technical_data: Dict, 
                          news_summary: str, price_data: Dict) -> Optional[Dict]:
        """
        Analisis pair forex menggunakan AI
        """
        try:
            prompt = self._build_analysis_prompt(pair, technical_data, news_summary, price_data)
            response = self._call_deepseek_api(prompt)
            
            if response:
                return self._parse_ai_response(response, pair)
            else:
                return self._generate_fallback_analysis(technical_data)
                
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return self._generate_fallback_analysis(technical_data)
    
    def _build_analysis_prompt(self, pair: str, technical_data: Dict, 
                              news_summary: str, price_data: Dict) -> str:
        """
        Bangun prompt untuk AI analysis
        """
        current_price = price_data.get('current', 0)
        previous_price = price_data.get('previous', current_price)
        change = ((current_price - previous_price) / previous_price * 100) if previous_price else 0
        
        prompt = f"""
        Analisis pair forex {pair} dengan data berikut:
        
        DATA TEKNIKAL:
        - Harga terkini: {current_price:.4f}
        - Perubahan: {change:+.2f}%
        - Trend: {technical_data.get('trend', 'N/A')}
        - RSI: {technical_data.get('rsi', 'N/A'):.1f}
        - MACD Signal: {technical_data.get('macd_signal', 'N/A')}
        - Support: {technical_data.get('support', 'N/A'):.4f}
        - Resistance: {technical_data.get('resistance', 'N/A'):.4f}
        
        ANALISIS BERITA:
        {news_summary}
        
        Berikan analisis dalam bahasa Indonesia dengan format:
        1. **ANALISIS TEKNIKAL**: Evaluasi kondisi teknikal saat ini
        2. **PENGARUH BERITA**: Dampak berita terhadap pair ini
        3. **PREDIKSI**: Perkiraan pergerakan 24-48 jam ke depan
        4. **REKOMENDASI TRADING**: BUY/SELL/HOLD dengan alasan jelas
        5. **MANAJEMEN RISIKO**: Level Take Profit dan Stop Loss
        6. **KEY LEVELS**: Support dan resistance penting
        
        Berikan analisis yang objektif dan berdasarkan data.
        """
        
        return prompt
    
    def _call_deepseek_api(self, prompt: str) -> Optional[str]:
        """
        Panggil DeepSeek API dengan retry mechanism
        """
        for attempt in range(self.max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Anda adalah analis forex profesional dengan pengalaman 10 tahun. Berikan analisis yang objektif, detail, dan mudah dipahami."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "top_p": 0.9
                }
                
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=data, 
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    logger.warning(f"API call failed (attempt {attempt + 1}): {response.status_code}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2)  # Wait before retry
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API timeout (attempt {attempt + 1})")
                if attempt < self.max_retries - 1:
                    time.sleep(3)
            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
        
        return None
    
    def _parse_ai_response(self, response: str, pair: str) -> Dict:
        """
        Parse response AI menjadi structured data
        """
        # Extract key information from AI response
        # This is a simplified parser - you might want to make it more sophisticated
        
        analysis = {
            'pair': pair,
            'analysis_text': response,
            'recommendation': self._extract_recommendation(response),
            'confidence': self._extract_confidence(response),
            'timeframe': '24-48 hours',
            'key_levels': self._extract_key_levels(response),
            'risk_level': self._extract_risk_level(response)
        }
        
        return analysis
    
    def _extract_recommendation(self, text: str) -> str:
        """Extract recommendation dari text AI"""
        text_lower = text.lower()
        if 'buy' in text_lower and 'sell' not in text_lower:
            return 'BUY'
        elif 'sell' in text_lower and 'buy' not in text_lower:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence level dari text AI"""
        # Simple heuristic - bisa diperbaiki dengan NLP yang lebih advanced
        confidence_words = ['high', 'medium', 'low', 'strong', 'weak']
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['high', 'strong', 'very']):
            return 0.8
        elif any(word in text_lower for word in ['medium', 'moderate']):
            return 0.6
        else:
            return 0.4
    
    def _extract_key_levels(self, text: str) -> Dict:
        """Extract key levels dari text AI"""
        # Simple pattern matching untuk angka (support/resistance levels)
        import re
        numbers = re.findall(r'\d+\.\d+', text)
        return {
            'support': numbers[0] if len(numbers) > 0 else None,
            'resistance': numbers[1] if len(numbers) > 1 else None
        }
    
    def _extract_risk_level(self, text: str) -> str:
        """Extract risk level dari text AI"""
        text_lower = text.lower()
        if 'high risk' in text_lower:
            return 'HIGH'
        elif 'medium risk' in text_lower:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_fallback_analysis(self, technical_data: Dict) -> Dict:
        """
        Generate fallback analysis jika AI tidak available
        """
        trend = technical_data.get('trend', 'NEUTRAL')
        rsi = technical_data.get('rsi', 50)
        
        if trend == 'BULLISH' and rsi < 70:
            recommendation = 'BUY'
            confidence = 0.7
        elif trend == 'BEARISH' and rsi > 30:
            recommendation = 'SELL'
            confidence = 0.7
        else:
            recommendation = 'HOLD'
            confidence = 0.5
        
        return {
            'pair': 'UNKNOWN',
            'analysis_text': f'Fallback analysis: Trend {trend}, RSI {rsi:.1f}',
            'recommendation': recommendation,
            'confidence': confidence,
            'timeframe': '24 hours',
            'key_levels': {},
            'risk_level': 'MEDIUM',
            'is_fallback': True
        }
    
    def batch_analyze_pairs(self, pairs_data: Dict) -> Dict:
        """
        Analisis multiple pairs sekaligus
        """
        results = {}
        
        for pair, data in pairs_data.items():
            logger.info(f"Analyzing {pair}...")
            analysis = self.analyze_forex_pair(
                pair, 
                data.get('technical', {}),
                data.get('news', ''),
                data.get('price', {})
            )
            results[pair] = analysis
            
            # Delay antara requests untuk avoid rate limiting
            time.sleep(1)
        
        return results

# Global instance
AI_PREDICTOR = AIPredictor()
