import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import ta
from datetime import datetime, timedelta
import requests
import json
from bs4 import BeautifulSoup
import time

# Konfigurasi DeepSeek API
DEEPSEEK_API_KEY = "sk-73d83584fd614656926e1d8860eae9ca"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

class ForexAnalysisSystem:
    def __init__(self):
        self.pairs = ['GBPJPY=X', 'USDJPY=X', 'CHFJPY=X', 'EURJPY=X', 'EURNZD=X']
        self.timeframes = ['2h', '4h', '1d']
        
    def get_data(self, pair, period='60d'):
        """Mendapatkan data harga dari Yahoo Finance"""
        try:
            data = yf.download(pair, period=period, interval='1h')
            if data.empty:
                st.error(f"Data untuk {pair} tidak tersedia")
                return None
            return data
        except Exception as e:
            st.error(f"Error mendapatkan data: {e}")
            return None
    
    def calculate_indicators(self, data):
        """Menghitung indikator teknikal"""
        if data is None or data.empty:
            return None
            
        # EMA 200
        data['EMA_200'] = ta.trend.ema_indicator(data['Close'], window=200)
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
        
        # On Balance Volume (OBV)
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        
        return data
    
    def resample_data(self, data, timeframe):
        """Resample data berdasarkan timeframe"""
        if timeframe == '2h':
            return data.resample('2H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif timeframe == '4h':
            return data.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif timeframe == '1d':
            return data.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            return data
    
    def get_technical_analysis(self, data):
        """Analisis teknikal berdasarkan indikator"""
        if data is None or len(data) < 50:
            return "Data tidak cukup untuk analisis"
        
        current_price = data['Close'].iloc[-1]
        ema_200 = data['EMA_200'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_Signal'].iloc[-1]
        obv_trend = data['OBV'].iloc[-1] > data['OBV'].iloc[-20]
        
        # Analisis trend
        if current_price > ema_200:
            trend = "BULLISH"
        else:
            trend = "BEARISH"
        
        # Analisis MACD
        if macd > macd_signal:
            macd_signal_str = "BULLISH"
        else:
            macd_signal_str = "BEARISH"
        
        # Analisis OBV
        obv_signal = "BULLISH" if obv_trend else "BEARISH"
        
        return {
            'trend': trend,
            'macd_signal': macd_signal_str,
            'obv_signal': obv_signal,
            'current_price': current_price,
            'ema_200': ema_200
        }
    
    def scrape_news(self):
        """Scraping berita forex dari sumber terpercaya"""
        news_items = []
        
        try:
            # Contoh scraping dari sumber berita (disesuaikan dengan sumber yang diinginkan)
            sources = [
                "https://www.forexfactory.com/news",
                "https://www.investing.com/news/forex-news"
            ]
            
            # Simulasi data berita (dalam implementasi nyata, gunakan BeautifulSoup)
            sample_news = [
                {"title": "Bank of Japan Pertahankan Kebijakan Moneternya", "impact": "High", "currency": "JPY"},
                {"title": "Fed Signal Kemungkinan Kenaikan Suku Bunga", "impact": "Medium", "currency": "USD"},
                {"title": "ECB Bahas Kebijakan Monetari Eropa", "impact": "Medium", "currency": "EUR"},
                {"title": "GDP Inggris Lebih Baik dari Ekspektasi", "impact": "High", "currency": "GBP"}
            ]
            
            news_items = sample_news
            
        except Exception as e:
            st.warning(f"Error scraping news: {e}")
            # Fallback news
            news_items = [
                {"title": "Analisis Teknikal Mendominasi Pergerakan Pasar", "impact": "Medium", "currency": "ALL"}
            ]
        
        return news_items
    
    def get_ai_analysis(self, pair, technical_analysis, news_analysis):
        """Mendapatkan analisis dari DeepSeek AI"""
        
        prompt = f"""
        Analisis pair forex {pair} dengan kondisi berikut:
        
        ANALISIS TEKNIKAL:
        - Trend: {technical_analysis['trend']}
        - Sinyal MACD: {technical_analysis['macd_signal']}
        - Sinyal OBV: {technical_analysis['obv_signal']}
        - Harga saat ini: {technical_analysis['current_price']}
        - EMA 200: {technical_analysis['ema_200']}
        
        BERITA TERKINI:
        {news_analysis}
        
        Berikan rekomendasi trading yang detail termasuk:
        1. Prediksi pergerakan harga
        2. Rekomendasi BUY/SELL
        3. Level Take Profit dan Stop Loss
        4. Analisis risiko
        5. Prediksi untuk beberapa waktu ke depan
        
        Format respons dalam bahasa Indonesia.
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Anda adalah analis forex profesional."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error API: {response.status_code}"
                
        except Exception as e:
            return f"Analisis AI tidak tersedia: {e}"
    
    def create_interactive_chart(self, data, pair, timeframe):
        """Membuat chart interaktif dengan Plotly"""
        if data is None or data.empty:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f'{pair} - Price Chart', 'MACD', 'On Balance Volume'),
            vertical_spacing=0.08,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ), row=1, col=1
        )
        
        # EMA 200
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['EMA_200'],
                line=dict(color='orange', width=2),
                name='EMA 200'
            ), row=1, col=1
        )
        
        # MACD
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD'), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal'), row=2, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
        fig.add_trace(
            go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram', marker_color=colors),
            row=2, col=1
        )
        
        # OBV
        fig.add_trace(
            go.Scatter(x=data.index, y=data['OBV'], name='OBV', line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'Analisis Teknikal {pair} ({timeframe})',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_recommendation(self, technical_analysis):
        """Generate rekomendasi trading otomatis"""
        signals = []
        
        if technical_analysis['trend'] == 'BULLISH':
            signals.append('BULLISH Trend')
        else:
            signals.append('BEARISH Trend')
            
        if technical_analysis['macd_signal'] == 'BULLISH':
            signals.append('MACD BULLISH')
        else:
            signals.append('MACD BEARISH')
            
        if technical_analysis['obv_signal'] == 'BULLISH':
            signals.append('OBV BULLISH')
        else:
            signals.append('OBV BEARISH')
        
        # Logika rekomendasi sederhana
        bull_count = signals.count('BULLISH Trend') + signals.count('MACD BULLISH') + signals.count('OBV BULLISH')
        bear_count = signals.count('BEARISH Trend') + signals.count('MACD BEARISH') + signals.count('OBV BEARISH')
        
        if bull_count > bear_count:
            action = "BUY"
            confidence = bull_count / 3.0
        else:
            action = "SELL"
            confidence = bear_count / 3.0
        
        # Calculate TP/SL levels
        current_price = technical_analysis['current_price']
        if action == "BUY":
            take_profit = current_price * 1.015  # 1.5% TP
            stop_loss = current_price * 0.99     # 1% SL
        else:
            take_profit = current_price * 0.985  # 1.5% TP
            stop_loss = current_price * 1.01     # 1% SL
        
        return {
            'action': action,
            'confidence': confidence,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'signals': signals
        }

def main():
    st.set_page_config(
        page_title="FOREX ANALYSIS SYSTEM",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 FOREX ANALYSIS SYSTEM")
    st.markdown("Sistem Analisis Teknikal dan Fundamental untuk Pasangan GBPJPY, USDJPY, CHFJPY, EURJPY, EURNZD")
    
    # Inisialisasi sistem
    system = ForexAnalysisSystem()
    
    # Sidebar
    st.sidebar.header("Konfigurasi Analisis")
    selected_pair = st.sidebar.selectbox("Pilih Pasangan Forex:", system.pairs)
    selected_timeframe = st.sidebar.selectbox("Pilih Timeframe:", system.timeframes)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Analisis untuk {selected_pair} - {selected_timeframe}")
        
        # Get data
        with st.spinner("Mengambil data..."):
            data = system.get_data(selected_pair)
            
        if data is not None:
            # Resample data berdasarkan timeframe
            resampled_data = system.resample_data(data, selected_timeframe)
            
            # Calculate indicators
            analyzed_data = system.calculate_indicators(resampled_data)
            
            if analyzed_data is not None:
                # Display chart
                fig = system.create_interactive_chart(analyzed_data.tail(100), selected_pair, selected_timeframe)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Technical analysis
                tech_analysis = system.get_technical_analysis(analyzed_data)
                
                if tech_analysis:
                    # Display technical analysis
                    st.subheader("Analisis Teknikal")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Trend", tech_analysis['trend'])
                        st.metric("Harga Saat Ini", f"{tech_analysis['current_price']:.4f}")
                    
                    with col2:
                        st.metric("Sinyal MACD", tech_analysis['macd_signal'])
                        st.metric("EMA 200", f"{tech_analysis['ema_200']:.4f}")
                    
                    with col3:
                        st.metric("Sinyal OBV", tech_analysis['obv_signal'])
                        st.metric("Selisih Harga-EMA", f"{(tech_analysis['current_price'] - tech_analysis['ema_200']):.4f}")
    
    with col2:
        st.subheader("Rekomendasi Trading")
        
        if 'tech_analysis' in locals():
            # Generate recommendation
            recommendation = system.generate_recommendation(tech_analysis)
            
            # Display recommendation
            color = "green" if recommendation['action'] == "BUY" else "red"
            st.markdown(f"<h2 style='color: {color};'>{recommendation['action']}</h2>", unsafe_allow_html=True)
            st.metric("Confidence Level", f"{recommendation['confidence']*100:.1f}%")
            
            st.subheader("Level Trading")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Take Profit", f"{recommendation['take_profit']:.4f}")
            with col2:
                st.metric("Stop Loss", f"{recommendation['stop_loss']:.4f}")
            
            st.subheader("Sinyal yang Terdeteksi")
            for signal in recommendation['signals']:
                st.write(f"✅ {signal}")
        
        # News analysis
        st.subheader("Berita Terkini")
        with st.spinner("Mengumpulkan berita..."):
            news = system.scrape_news()
            
        for item in news[:5]:  # Tampilkan 5 berita terbaru
            with st.expander(f"{item['title']} ({item['impact']})"):
                st.write(f"**Mata Uang:** {item['currency']}")
                st.write(f"**Dampak:** {item['impact']}")
    
    # AI Analysis Section
    st.markdown("---")
    st.subheader("🤖 Analisis Mendalam oleh AI")
    
    if st.button("Dapatkan Analisis AI Lengkap"):
        with st.spinner("AI sedang menganalisis..."):
            # Get news analysis
            news_analysis = system.scrape_news()
            
            # Get AI analysis
            ai_analysis = system.get_ai_analysis(selected_pair, tech_analysis, news_analysis)
            
            # Display AI analysis
            st.text_area("Analisis AI:", ai_analysis, height=300)
    
    # Multiple Pairs Analysis
    st.markdown("---")
    st.subheader("📊 Analisis Semua Pasangan")
    
    if st.button("Analisis Semua Pasangan"):
        progress_bar = st.progress(0)
        results = []
        
        for i, pair in enumerate(system.pairs):
            progress_bar.progress((i + 1) / len(system.pairs))
            
            data = system.get_data(pair)
            if data is not None:
                resampled_data = system.resample_data(data, selected_timeframe)
                analyzed_data = system.calculate_indicators(resampled_data)
                
                if analyzed_data is not None:
                    tech_analysis = system.get_technical_analysis(analyzed_data)
                    recommendation = system.generate_recommendation(tech_analysis)
                    
                    results.append({
                        'Pair': pair,
                        'Trend': tech_analysis['trend'],
                        'Action': recommendation['action'],
                        'Confidence': f"{recommendation['confidence']*100:.1f}%",
                        'Current Price': f"{tech_analysis['current_price']:.4f}"
                    })
            
            time.sleep(1)  # Delay untuk menghindari rate limiting
        
        # Display results
        if results:
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

if __name__ == "__main__":
    main()
