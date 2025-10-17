import pandas as pd
import numpy as np
from talib import abstract
import logging
from typing import List, Tuple, Dict, Any  # ✅ ADD THIS IMPORT

# Direct imports
try:
    from models.analysis import TechnicalAnalysis, TechnicalIndicator, Trend, Signal
except ImportError:
    # Fallback classes
    from enum import Enum
    
    class Signal(Enum):
        BUY = "BUY"
        SELL = "SELL"
        NEUTRAL = "NEUTRAL"
    
    class Trend(Enum):
        BULLISH = "BULLISH"
        BEARISH = "BEARISH"
        NEUTRAL = "NEUTRAL"
    
    class TechnicalIndicator:
        def __init__(self, name, value, signal, description=None, previous_value=None, strength=None):
            self.name = name
            self.value = value
            self.signal = signal
            self.description = description
            self.previous_value = previous_value
            self.strength = strength
        
        def dict(self):
            return {
                "name": self.name,
                "value": self.value,
                "signal": self.signal.value if hasattr(self.signal, 'value') else self.signal,
                "description": self.description,
                "previous_value": self.previous_value,
                "strength": self.strength
            }
    
    class TechnicalAnalysis:
        def __init__(self, indicators, summary, confidence, support_levels, resistance_levels, trend_strength=None, volatility=None):
            self.indicators = indicators
            self.summary = summary
            self.confidence = confidence
            self.support_levels = support_levels
            self.resistance_levels = resistance_levels
            self.trend_strength = trend_strength
            self.volatility = volatility
        
        def dict(self):
            return {
                "indicators": [ind.dict() for ind in self.indicators],
                "summary": self.summary.value if hasattr(self.summary, 'value') else self.summary,
                "confidence": self.confidence,
                "support_levels": self.support_levels,
                "resistance_levels": self.resistance_levels,
                "trend_strength": self.trend_strength,
                "volatility": self.volatility
            }

logger = logging.getLogger(__name__)

class EnhancedTechnicalAnalyzer:
    def __init__(self):
        self.indicators_config = {
            'sma': [20, 50, 200],
            'ema': [12, 26],
            'rsi': [14],
            'macd': [12, 26, 9],
            'bbands': [20, 2],
            'stoch': [14, 3, 3],
            'atr': [14],
            'obv': [],
            'adx': [14]
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators with enhanced features"""
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        try:
            # Price-based indicators
            for period in self.indicators_config['sma']:
                df[f'sma_{period}'] = abstract.SMA(df['close'], timeperiod=period)
            
            for period in self.indicators_config['ema']:
                df[f'ema_{period}'] = abstract.EMA(df['close'], timeperiod=period)
            
            # Momentum indicators
            df['rsi_14'] = abstract.RSI(df['close'], timeperiod=14)
            
            macd, macd_signal, macd_hist = abstract.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            
            # Volatility indicators
            bb_upper, bb_middle, bb_lower = abstract.BBANDS(df['close'])
            df['bollinger_upper'] = bb_upper
            df['bollinger_lower'] = bb_lower
            df['bollinger_middle'] = bb_middle
            
            # Stochastic
            stoch_k, stoch_d = abstract.STOCH(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Volatility
            df['atr_14'] = abstract.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Volume indicators
            df['volume_sma_20'] = abstract.SMA(df['volume'], timeperiod=20)
            df['obv'] = abstract.OBV(df['close'], df['volume'])
            
            # Trend strength
            df['adx_14'] = abstract.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
        except Exception as e:
            logger.warning(f"TA-Lib indicator calculation failed: {e}. Using fallback methods.")
            # Fallback calculations if TA-Lib fails
            df = self._calculate_fallback_indicators(df)
        
        # Calculate additional derived metrics
        df = self._calculate_derived_metrics(df)
        
        return df
    
    def _calculate_fallback_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback indicator calculations without TA-Lib"""
        # Simple SMA fallback
        for period in self.indicators_config['sma']:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Simple EMA fallback
        for period in self.indicators_config['ema']:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI fallback
        df['rsi_14'] = self._calculate_rsi_fallback(df['close'], 14)
        
        # MACD fallback
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _calculate_rsi_fallback(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI without TA-Lib"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived technical metrics"""
        if len(df) < 2:
            return df
        
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        # Volatility (rolling standard deviation)
        df['volatility_20'] = df['close'].rolling(window=20).std()
        
        # Trend strength based on multiple indicators
        if 'adx_14' in df.columns:
            df['trend_strength'] = df['adx_14'] / 100  # Normalize to 0-1
        
        # Support and resistance strength
        df['support_strength'] = self._calculate_support_resistance_strength(df)
        
        return df
    
    def _calculate_support_resistance_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate strength of support/resistance levels"""
        if len(df) < 20:
            return pd.Series([0.5] * len(df), index=df.index)
        
        # Simple implementation - can be enhanced
        strength = []
        for i in range(len(df)):
            if i < 10:
                strength.append(0.5)
                continue
                
            # Look back window for strength calculation
            window = df.iloc[max(0, i-10):i]
            current_price = df.iloc[i]['close']
            
            # Calculate how many times price bounced from similar levels
            similar_levels = window[
                (window['low'] <= current_price * 1.01) & 
                (window['low'] >= current_price * 0.99)
            ]
            
            strength_value = min(1.0, len(similar_levels) / 5)  # Normalize
            strength.append(strength_value)
        
        return pd.Series(strength, index=df.index)
    
    def generate_signals(self, df: pd.DataFrame) -> List[TechnicalIndicator]:
        """Generate enhanced trading signals based on indicators"""
        if len(df) < 2:
            return []
        
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        indicators = []
        
        # RSI Signal with strength
        rsi = current.get('rsi_14')
        if rsi is not None and not pd.isna(rsi):
            if rsi < 30:
                rsi_signal = Signal.BUY
                rsi_strength = (30 - rsi) / 30  # 0-1 strength
            elif rsi > 70:
                rsi_signal = Signal.SELL
                rsi_strength = (rsi - 70) / 30
            else:
                rsi_signal = Signal.NEUTRAL
                rsi_strength = 0.5
                
            indicators.append(TechnicalIndicator(
                name="RSI_14",
                value=rsi,
                signal=rsi_signal,
                description="Relative Strength Index",
                previous_value=previous.get('rsi_14'),
                strength=rsi_strength
            ))
        
        # MACD Signal
        macd = current.get('macd')
        macd_signal = current.get('macd_signal')
        if macd is not None and not pd.isna(macd) and macd_signal is not None and not pd.isna(macd_signal):
            macd_histogram = current.get('macd_histogram', 0)
            prev_macd_histogram = previous.get('macd_histogram', 0)
            
            if macd > macd_signal and macd_histogram > prev_macd_histogram:
                macd_signal_type = Signal.BUY
                macd_strength = abs(macd_histogram) / (abs(macd) + 0.001)
            elif macd < macd_signal and macd_histogram < prev_macd_histogram:
                macd_signal_type = Signal.SELL
                macd_strength = abs(macd_histogram) / (abs(macd) + 0.001)
            else:
                macd_signal_type = Signal.NEUTRAL
                macd_strength = 0.5
                
            indicators.append(TechnicalIndicator(
                name="MACD",
                value=macd,
                signal=macd_signal_type,
                description="Moving Average Convergence Divergence",
                previous_value=previous.get('macd'),
                strength=macd_strength
            ))
        
        # Moving Average Signals
        sma_20 = current.get('sma_20')
        sma_50 = current.get('sma_50')
        if sma_20 is not None and not pd.isna(sma_20) and sma_50 is not None and not pd.isna(sma_50):
            price = current['close']
            
            # Multiple MA comparisons
            above_20 = price > sma_20
            above_50 = price > sma_50
            ma_trend = sma_20 > sma_50
            
            if above_20 and above_50 and ma_trend:
                ma_signal = Signal.BUY
                ma_strength = 0.8
            elif not above_20 and not above_50 and not ma_trend:
                ma_signal = Signal.SELL
                ma_strength = 0.8
            else:
                ma_signal = Signal.NEUTRAL
                ma_strength = 0.5
                
            indicators.append(TechnicalIndicator(
                name="MA_Crossover",
                value=sma_20 - sma_50,
                signal=ma_signal,
                description="Moving Average Crossover System",
                previous_value=previous.get('sma_20', 0) - previous.get('sma_50', 0),
                strength=ma_strength
            ))
        
        # Bollinger Bands Signal
        close = current['close']
        bb_upper = current.get('bollinger_upper')
        bb_lower = current.get('bollinger_lower')
        if bb_upper is not None and not pd.isna(bb_upper) and bb_lower is not None and not pd.isna(bb_lower):
            bb_middle = current.get('bollinger_middle', (bb_upper + bb_lower) / 2)
            
            if close <= bb_lower:
                bb_signal = Signal.BUY
                bb_strength = (bb_middle - close) / (bb_middle - bb_lower)
            elif close >= bb_upper:
                bb_signal = Signal.SELL
                bb_strength = (close - bb_middle) / (bb_upper - bb_middle)
            else:
                bb_signal = Signal.NEUTRAL
                bb_strength = 0.5
                
            indicators.append(TechnicalIndicator(
                name="Bollinger_Bands",
                value=close,
                signal=bb_signal,
                description="Bollinger Bands Position",
                previous_value=previous['close'],
                strength=bb_strength
            ))
        
        # Stochastic Signal
        stoch_k = current.get('stoch_k')
        stoch_d = current.get('stoch_d')
        if stoch_k is not None and not pd.isna(stoch_k) and stoch_d is not None and not pd.isna(stoch_d):
            if stoch_k < 20 and stoch_d < 20:
                stoch_signal = Signal.BUY
                stoch_strength = (20 - min(stoch_k, stoch_d)) / 20
            elif stoch_k > 80 and stoch_d > 80:
                stoch_signal = Signal.SELL
                stoch_strength = (min(stoch_k, stoch_d) - 80) / 20
            else:
                stoch_signal = Signal.NEUTRAL
                stoch_strength = 0.5
                
            indicators.append(TechnicalIndicator(
                name="Stochastic",
                value=stoch_k,
                signal=stoch_signal,
                description="Stochastic Oscillator",
                previous_value=previous.get('stoch_k'),
                strength=stoch_strength
            ))
        
        return indicators
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Calculate enhanced support and resistance levels"""
        if len(df) < 20:
            return [], []
        
        # Use recent data for levels (last 100 periods)
        recent_data = df.tail(100)
        highs = recent_data['high']
        lows = recent_data['low']
        closes = recent_data['close']
        
        # Find significant levels using rolling windows
        window_size = 10
        resistance_levels = []
        support_levels = []
        
        # Resistance levels (local highs)
        for i in range(window_size, len(highs) - window_size):
            window_highs = highs.iloc[i-window_size:i+window_size]
            if highs.iloc[i] == window_highs.max():
                resistance_levels.append(highs.iloc[i])
        
        # Support levels (local lows)
        for i in range(window_size, len(lows) - window_size):
            window_lows = lows.iloc[i-window_size:i+window_size]
            if lows.iloc[i] == window_lows.min():
                support_levels.append(lows.iloc[i])
        
        # Add moving averages as dynamic support/resistance
        if 'sma_50' in df.columns:
            sma_50_current = df['sma_50'].iloc[-1]
            if not pd.isna(sma_50_current):
                if sma_50_current < closes.iloc[-1]:
                    resistance_levels.append(sma_50_current)
                else:
                    support_levels.append(sma_50_current)
        
        if 'sma_200' in df.columns:
            sma_200_current = df['sma_200'].iloc[-1]
            if not pd.isna(sma_200_current):
                if sma_200_current < closes.iloc[-1]:
                    resistance_levels.append(sma_200_current)
                else:
                    support_levels.append(sma_200_current)
        
        # Remove duplicates and sort, keep only significant levels
        support_levels = sorted(list(set([round(level, 2) for level in support_levels])))
        resistance_levels = sorted(list(set([round(level, 2) for level in resistance_levels])))
        
        # Filter levels that are too close to current price
        current_price = df['close'].iloc[-1]
        price_range = current_price * 0.02  # 2% range
        
        support_levels = [level for level in support_levels if level < current_price - price_range]
        resistance_levels = [level for level in resistance_levels if level > current_price + price_range]
        
        # Return top 3-5 levels
        return support_levels[-5:], resistance_levels[:5]
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate overall trend strength (0-1)"""
        if len(df) < 20:
            return 0.5
        
        current = df.iloc[-1]
        
        strength_indicators = []
        
        # ADX trend strength
        if 'adx_14' in df.columns:
            adx_value = current['adx_14']
            if not pd.isna(adx_value):
                adx_strength = min(adx_value / 100, 1.0)
                strength_indicators.append(adx_strength)
        
        # Moving average alignment
        if all(col in df.columns for col in ['sma_20', 'sma_50', 'sma_200']):
            sma_20_val = current['sma_20']
            sma_50_val = current['sma_50']
            sma_200_val = current['sma_200']
            
            if not pd.isna(sma_20_val) and not pd.isna(sma_50_val) and not pd.isna(sma_200_val):
                ma_alignment = 0
                if (sma_20_val > sma_50_val > sma_200_val):
                    ma_alignment = 1.0
                elif (sma_20_val < sma_50_val < sma_200_val):
                    ma_alignment = 1.0
                strength_indicators.append(ma_alignment)
        
        # Price position relative to MAs
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            sma_20_val = current['sma_20']
            sma_50_val = current['sma_50']
            
            if not pd.isna(sma_20_val) and not pd.isna(sma_50_val):
                price_vs_ma = abs(current['close'] - sma_20_val) / sma_20_val
                ma_distance = abs(sma_20_val - sma_50_val) / sma_50_val
                combined_strength = min(price_vs_ma + ma_distance, 1.0)
                strength_indicators.append(combined_strength)
        
        return np.mean(strength_indicators) if strength_indicators else 0.5
    
    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate market volatility"""
        if len(df) < 20:
            return 0.0
        
        # Use ATR normalized by price
        if 'atr_14' in df.columns:
            current_atr = df['atr_14'].iloc[-1]
            current_price = df['close'].iloc[-1]
            if not pd.isna(current_atr):
                return (current_atr / current_price) * 100  # Return as percentage
        
        # Fallback to standard deviation
        returns = df['close'].pct_change().dropna()
        if len(returns) > 0:
            return returns.std() * 100  # Annualized percentage
        
        return 0.0
    
    def analyze(self, df: pd.DataFrame) -> TechnicalAnalysis:
        """Perform complete enhanced technical analysis"""
        try:
            df_with_indicators = self.calculate_indicators(df)
            indicators = self.generate_signals(df_with_indicators)
            support_levels, resistance_levels = self.calculate_support_resistance(df_with_indicators)
            trend_strength = self.calculate_trend_strength(df_with_indicators)
            volatility = self.calculate_volatility(df_with_indicators)
            
            # Generate enhanced summary
            buy_signals = len([i for i in indicators if i.signal == Signal.BUY])
            sell_signals = len([i for i in indicators if i.signal == Signal.SELL])
            signal_strengths = [i.strength or 0.5 for i in indicators if i.strength is not None]
            avg_signal_strength = np.mean(signal_strengths) if signal_strengths else 0.5
            
            if buy_signals > sell_signals:
                summary = Trend.BULLISH
                confidence = min(0.9, avg_signal_strength * (buy_signals / len(indicators)))
            elif sell_signals > buy_signals:
                summary = Trend.BEARISH
                confidence = min(0.9, avg_signal_strength * (sell_signals / len(indicators)))
            else:
                summary = Trend.NEUTRAL
                confidence = 0.5
            
            # Adjust confidence with trend strength
            confidence = (confidence + trend_strength) / 2
            
            return TechnicalAnalysis(
                indicators=indicators,
                summary=summary,
                confidence=confidence,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_strength=trend_strength,
                volatility=volatility
            )
            
        except Exception as e:
            logger.error(f"Technical analysis failed: {e}")
            # Return basic analysis as fallback
            return TechnicalAnalysis(
                indicators=[],
                summary=Trend.NEUTRAL,
                confidence=0.5,
                support_levels=[],
                resistance_levels=[],
                trend_strength=0.5,
                volatility=0.0
            )

# Update the global instance
technical_analyzer = EnhancedTechnicalAnalyzer()
