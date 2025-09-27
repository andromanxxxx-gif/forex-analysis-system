import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Generator sinyal trading berdasarkan analisis teknikal dan AI
    """
    
    def __init__(self):
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.reward_ratio = 2.0     # Risk:Reward = 1:2
    
    def generate_signals(self, data: pd.DataFrame, ai_analysis: Optional[Dict] = None) -> Dict:
        """
        Generate sinyal trading komprehensif
        
        Args:
            data: DataFrame dengan data teknikal
            ai_analysis: Analisis dari DeepSeek AI (optional)
            
        Returns:
            Dictionary dengan sinyal trading
        """
        if data is None or data.empty:
            return self._get_default_signal("Data tidak tersedia")
        
        try:
            # Dapatkan data terbaru
            current_data = data.iloc[-1]
            previous_data = data.iloc[-2] if len(data) > 1 else current_data
            
            # Technical signals
            tech_signals = self._get_technical_signals(current_data, previous_data)
            
            # AI signals (jika ada)
            ai_signals = self._get_ai_signals(ai_analysis) if ai_analysis else {}
            
            # Combine signals
            final_signal = self._combine_signals(tech_signals, ai_signals)
            
            # Calculate position sizing
            position_info = self._calculate_position_size(final_signal, current_data['Close'])
            
            return {
                **final_signal,
                **position_info,
                'timestamp': pd.Timestamp.now(),
                'technical_indicators': tech_signals,
                'ai_analysis': ai_signals
            }
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return self._get_default_signal(f"Error: {str(e)}")
    
    def _get_technical_signals(self, current: pd.Series, previous: pd.Series) -> Dict:
        """Generate sinyal dari indikator teknikal"""
        signals = {}
        
        # Trend signals
        signals['ema_trend'] = 'BULLISH' if current['Close'] > current.get('EMA_200', current['Close']) else 'BEARISH'
        signals['price_vs_ema'] = current['Close'] - current.get('EMA_200', current['Close'])
        
        # MACD signals
        macd = current.get('MACD', 0)
        macd_signal = current.get('MACD_Signal', 0)
        macd_histogram = current.get('MACD_Histogram', 0)
        
        signals['macd_signal'] = 'BULLISH' if macd > macd_signal else 'BEARISH'
        signals['macd_cross'] = 'BULLISH_CROSS' if (macd > macd_signal and previous.get('MACD', 0) <= previous.get('MACD_Signal', 0)) else \
                               'BEARISH_CROSS' if (macd < macd_signal and previous.get('MACD', 0) >= previous.get('MACD_Signal', 0)) else 'NO_CROSS'
        
        # RSI signals
        rsi = current.get('RSI', 50)
        if rsi < 30:
            signals['rsi_signal'] = 'OVERSOLD'
        elif rsi > 70:
            signals['rsi_signal'] = 'OVERBOUGHT'
        else:
            signals['rsi_signal'] = 'NEUTRAL'
        
        # Volume signals
        obv = current.get('OBV', 0)
        obv_prev = previous.get('OBV', 0)
        signals['volume_trend'] = 'BULLISH' if obv > obv_prev else 'BEARISH'
        
        return signals
    
    def _get_ai_signals(self, ai_analysis: Dict) -> Dict:
        """Extract sinyal dari analisis AI"""
        if not ai_analysis:
            return {}
        
        return {
            'ai_recommendation': ai_analysis.get('recommendation', 'HOLD'),
            'ai_confidence': ai_analysis.get('confidence', 0.5),
            'ai_analysis_text': ai_analysis.get('analysis_text', ''),
            'ai_risk_level': ai_analysis.get('risk_level', 'MEDIUM')
        }
    
    def _combine_signals(self, tech_signals: Dict, ai_signals: Dict) -> Dict:
        """Kombinasikan sinyal teknikal dan AI"""
        # Hitung skor teknikal
        tech_score = 0
        max_tech_score = 0
        
        # Trend scoring
        if tech_signals.get('ema_trend') == 'BULLISH':
            tech_score += 1
        else:
            tech_score -= 1
        max_tech_score += 1
        
        # MACD scoring
        if tech_signals.get('macd_signal') == 'BULLISH':
            tech_score += 1
        else:
            tech_score -= 1
        max_tech_score += 1
        
        # RSI scoring
        rsi_signal = tech_signals.get('rsi_signal')
        if rsi_signal == 'OVERSOLD':
            tech_score += 1
        elif rsi_signal == 'OVERBOUGHT':
            tech_score -= 1
        max_tech_score += 1
        
        # AI scoring (jika ada)
        ai_score = 0
        max_ai_score = 0
        
        if ai_signals:
            ai_recommendation = ai_signals.get('ai_recommendation', 'HOLD')
            ai_confidence = ai_signals.get('ai_confidence', 0.5)
            
            if ai_recommendation == 'BUY':
                ai_score = ai_confidence
            elif ai_recommendation == 'SELL':
                ai_score = -ai_confidence
            max_ai_score = 1
        
        # Combine scores dengan weighting
        tech_weight = 0.6
        ai_weight = 0.4 if ai_signals else 0
        
        total_score = (tech_score/max_tech_score * tech_weight + 
                      ai_score/max_ai_score * ai_weight) if max_ai_score > 0 else tech_score/max_tech_score
        
        # Tentukan action berdasarkan total score
        if total_score > 0.1:
            action = 'BUY'
            confidence = min(total_score, 0.95)
        elif total_score < -0.1:
            action = 'SELL'
            confidence = min(-total_score, 0.95)
        else:
            action = 'HOLD'
            confidence = 0.5
        
        return {
            'action': action,
            'confidence': confidence,
            'combined_score': total_score
        }
    
    def _calculate_position_size(self, signal: Dict, current_price: float) -> Dict:
        """Hitung position sizing dan risk management"""
        if signal['action'] == 'HOLD':
            return {
                'position_size': 0,
                'take_profit': current_price,
                'stop_loss': current_price,
                'risk_reward_ratio': 0
            }
        
        # Hitung stop loss berdasarkan volatilitas
        atr_multiplier = 2.0  # 2x ATR untuk stop loss
        stop_loss_pct = 0.02  # 2% default stop loss
        
        # Calculate TP/SL levels
        if signal['action'] == 'BUY':
            stop_loss = current_price * (1 - stop_loss_pct)
            take_profit = current_price * (1 + stop_loss_pct * self.reward_ratio)
        else:  # SELL
            stop_loss = current_price * (1 + stop_loss_pct)
            take_profit = current_price * (1 - stop_loss_pct * self.reward_ratio)
        
        # Calculate position size berdasarkan risk
        risk_amount = self.risk_per_trade  # 2% of account
        price_diff = abs(current_price - stop_loss)
        position_size = risk_amount / price_diff if price_diff > 0 else 0
        
        risk_reward = (abs(take_profit - current_price) / 
                      abs(stop_loss - current_price)) if abs(stop_loss - current_price) > 0 else 0
        
        return {
            'position_size': position_size,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'risk_reward_ratio': risk_reward,
            'risk_per_trade': self.risk_per_trade
        }
    
    def _get_default_signal(self, message: str) -> Dict:
        """Default signal ketika ada error"""
        return {
            'action': 'HOLD',
            'confidence': 0.5,
            'message': message,
            'position_size': 0,
            'take_profit': 0,
            'stop_loss': 0,
            'risk_reward_ratio': 0
        }
    
    def generate_multiple_timeframe_analysis(self, data_dict: Dict) -> Dict:
        """Analisis multiple timeframe"""
        timeframe_signals = {}
        
        for timeframe, data in data_dict.items():
            if data is not None and not data.empty:
                signals = self.generate_signals(data)
                timeframe_signals[timeframe] = signals
        
        # Combine signals dari berbagai timeframe
        return self._combine_timeframe_signals(timeframe_signals)
    
    def _combine_timeframe_signals(self, timeframe_signals: Dict) -> Dict:
        """Kombinasikan sinyal dari berbagai timeframe"""
        if not timeframe_signals:
            return self._get_default_signal("No timeframe data")
        
        # Weighting berdasarkan timeframe (higher timeframe lebih berat)
        weights = {
            '1D': 0.4,
            '4H': 0.3, 
            '2H': 0.2,
            '1H': 0.1
        }
        
        total_weight = 0
        weighted_score = 0
        
        for timeframe, signals in timeframe_signals.items():
            weight = weights.get(timeframe, 0.1)
            
            if signals['action'] == 'BUY':
                score = signals['confidence'] * weight
            elif signals['action'] == 'SELL':
                score = -signals['confidence'] * weight
            else:
                score = 0
            
            weighted_score += score
            total_weight += weight
        
        if total_weight == 0:
            return self._get_default_signal("No valid timeframe signals")
        
        final_score = weighted_score / total_weight
        
        if final_score > 0.1:
            action = 'BUY'
            confidence = min(final_score, 0.95)
        elif final_score < -0.1:
            action = 'SELL' 
            confidence = min(-final_score, 0.95)
        else:
            action = 'HOLD'
            confidence = 0.5
        
        # Gunakan sinyal dari timeframe tertinggi untuk TP/SL
        highest_timeframe = max(weights.keys(), key=lambda x: weights.get(x, 0))
        highest_signal = timeframe_signals.get(highest_timeframe, {})
        
        return {
            'action': action,
            'confidence': confidence,
            'combined_score': final_score,
            'timeframe_analysis': timeframe_signals,
            'take_profit': highest_signal.get('take_profit', 0),
            'stop_loss': highest_signal.get('stop_loss', 0),
            'primary_timeframe': highest_timeframe
        }

# Global instance
SIGNAL_GENERATOR = SignalGenerator()
