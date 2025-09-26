import ta
import pandas as pd

class TechnicalAnalyzer:
    def __init__(self):
        pass
    
    def calculate_indicators(self, data):
        """Hitung indikator teknikal"""
        # EMA 200
        data['EMA_200'] = ta.trend.ema_indicator(data['Close'], window=200)
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
        
        # OBV
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        
        return data
    
    def get_analysis(self, data):
        """Dapatkan analisis teknikal"""
        if len(data) < 50:
            return "Data tidak cukup untuk analisis"
        
        current_price = data['Close'].iloc[-1]
        ema_200 = data['EMA_200'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        obv = data['OBV'].iloc[-1]
        obv_prev = data['OBV'].iloc[-5]  # 5 periode sebelumnya
        
        trend = "BULLISH" if current_price > ema_200 else "BEARISH"
        macd_signal_str = "BULLISH" if macd > macd_signal else "BEARISH"
        obv_trend = "BULLISH" if obv > obv_prev else "BEARISH"
        
        analysis = f"""
        **Analisis Teknikal:**
        - Trend: {trend}
        - Harga relatif terhadap EMA 200: {current_price - ema_200:.4f}
        - Sinyal MACD: {macd_signal_str}
        - Trend OBV: {obv_trend}
        """
        
        return analysis
