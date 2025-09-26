# Di bagian atas app.py tambahkan:
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_fetcher import ForexDataFetcher
from src.technical_analyzer import TechnicalAnalyzer
from src.news_analyzer import NewsAnalyzer
from src.ai_predictor import AIPredictor
from src.dummy_data_generator import ForexDataGenerator
from src.backtester import ForexBacktester
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import tempfile
from datetime import datetime, timedelta
import requests
import yfinance as yf
import ta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.gdrive_config import GDRIVE_MANAGER

class CompleteForexSystem:
    def __init__(self):
        self.gdrive = GDRIVE_MANAGER
        self.pairs = ['GBPJPY', 'USDJPY', 'CHFJPY', 'EURJPY', 'EURNZD']
        self.timeframes = ['2H', '4H', '1D']
        self.api_key = "sk-73d83584fd614656926e1d8860eae9ca"
    
    def get_real_forex_data(self, pair, period='30d'):
        """Mendapatkan data real dari Yahoo Finance"""
        try:
            pair_mapping = {
                'GBPJPY': 'GBPJPY=X',
                'USDJPY': 'USDJPY=X',
                'CHFJPY': 'CHFJPY=X',
                'EURJPY': 'EURJPY=X',
                'EURNZD': 'EURNZD=X'
            }
            
            yf_symbol = pair_mapping.get(pair, pair)
            data = yf.download(yf_symbol, period=period, interval='1h')
            
            if data.empty:
                st.warning(f"Data real untuk {pair} tidak tersedia, menggunakan data simulasi")
                return self.get_simulated_data(pair)
            
            return data
        except Exception as e:
            st.warning(f"Error mengambil data real: {e}, menggunakan data simulasi")
            return self.get_simulated_data(pair)
    
    def get_simulated_data(self, pair, periods=200):
        """Data simulasi sebagai fallback"""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        
        # Simulasi yang lebih realistis
        np.random.seed(hash(pair) % 10000)
        base_price = 150.0 if 'JPY' in pair else 1.5
        
        prices = [base_price]
        for i in range(periods - 1):
            change = np.random.normal(0.0005, 0.008)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0.001, 0.003))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0.001, 0.003))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(10000, 100000, periods)
        }, index=dates)
        
        return data
    
    def calculate_technical_indicators(self, data):
        """Menghitung semua indikator teknikal"""
        # EMA 200
        data['EMA_200'] = ta.trend.ema_indicator(data['Close'], window=200)
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()
        
        # OBV
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        
        # RSI
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(data['Close'])
        data['BB_Upper'] = bollinger.bollinger_hband()
        data['BB_Lower'] = bollinger.bollinger_lband()
        
        return data
    
    def load_data(self, pair, timeframe):
        """Load data dari Google Drive atau sumber lain"""
        try:
            # Coba load dari Google Drive dulu
            file_name = f"{pair}_{timeframe}_data.csv"
            file_id = self.gdrive.get_file_id_by_name(file_name)
            
            if file_id:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                    temp_path = temp_file.name
                
                if self.gdrive.download_file(file_id, temp_path):
                    data = pd.read_csv(temp_path, index_col=0, parse_dates=True)
                    os.unlink(temp_path)
                    st.success(f"✅ Data loaded dari Google Drive: {file_name}")
                    return data
            
            # Jika tidak ada di Google Drive, ambil data baru
            st.info("🔄 Mengambil data terbaru...")
            data = self.get_real_forex_data(pair)
            
            # Resample berdasarkan timeframe
            data = self.resample_data(data, timeframe)
            
            # Hitung indikator
            data = self.calculate_technical_indicators(data)
            
            # Simpan ke Google Drive
            self.save_data(data, pair, timeframe)
            
            return data
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return self.get_simulated_data(pair)
    
    def resample_data(self, data, timeframe):
        """Resample data berdasarkan timeframe"""
        if timeframe == '2H':
            return data.resample('2H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif timeframe == '4H':
            return data.resample('4H').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        elif timeframe == '1D':
            return data.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        else:
            return data
    
    def save_data(self, data, pair, timeframe):
        """Simpan data ke Google Drive"""
        try:
            file_name = f"{pair}_{timeframe}_data.csv"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                temp_path = temp_file.name
            
            data.to_csv(temp_path)
            file_id = self.gdrive.upload_file(temp_path, file_name)
            os.unlink(temp_path)
            
            return file_id
        except Exception as e:
            st.error(f"Error saving data: {e}")
            return None
    
    def create_advanced_chart(self, data, pair, timeframe):
        """Membuat chart interaktif lengkap"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                f'{pair} - Price Chart ({timeframe})', 
                'MACD', 
                'RSI',
                'On Balance Volume'
            ),
            vertical_spacing=0.06,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Candlestick dengan Bollinger Bands
        fig.add_trace(go.Candlestick(
            x=data.index, open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'], name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['EMA_200'], name='EMA 200',
            line=dict(color='orange', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Upper'], name='BB Upper',
            line=dict(color='gray', width=1, dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, y=data['BB_Lower'], name='BB Lower',
            line=dict(color='gray', width=1, dash='dash')
        ), row=1, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD'), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal'), row=2, col=1)
        
        colors_histogram = ['green' if x >= 0 else 'red' for x in data['MACD_Histogram']]
        fig.add_trace(go.Bar(
            x=data.index, y=data['MACD_Histogram'], name='Histogram',
            marker_color=colors_histogram
        ), row=2, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_color="gray", row=3, col=1)
        
        # OBV
        fig.add_trace(go.Scatter(x=data.index, y=data['OBV'], name='OBV'), row=4, col=1)
        
        fig.update_layout(
            height=900,
            title=f"Advanced Technical Analysis - {pair}",
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        
        return fig
    
    def get_ai_analysis(self, pair, technical_data, news_data):
        """Mendapatkan analisis dari DeepSeek AI"""
        prompt = f"""
        Analisis pair forex {pair} dengan data teknikal berikut:
        
        DATA TEKNIKAL:
        - Harga terkini: {technical_data['current_price']:.4f}
        - Trend vs EMA 200: {technical_data['trend']}
        - Sinyal MACD: {technical_data['macd_signal']}
        - RSI: {technical_data['rsi']:.1f}
        - Sinyal OBV: {technical_data['obv_signal']}
        
        BERITA TERKINI: {news_data}
        
        Berikan analisis mendalam dalam bahasa Indonesia dengan format:
        1. **ANALISIS TEKNIKAL**: Evaluasi kondisi teknikal
        2. **PREDIKSI**: Perkiraan pergerakan 1-3 hari ke depan
        3. **REKOMENDASI**: BUY/SELL/HOLD dengan alasan
        4. **MANAJEMEN RISIKO**: TP/SL levels dan risk assessment
        5. **KEY LEVELS**: Level support dan resistance penting
        """
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Anda adalah analis forex profesional dengan pengalaman 10 tahun."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1500
            }
            
            response = requests.post("https://api.deepseek.com/v1/chat/completions", 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Analisis AI tidak tersedia: {str(e)}"
    
    def generate_trading_signals(self, data):
        """Generate sinyal trading komprehensif"""
        if data is None or len(data) < 20:
            return {"action": "HOLD", "confidence": 0.5, "reason": "Data tidak cukup"}
        
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # Hitung semua sinyal
        signals = {
            'ema_trend': 1 if current['Close'] > current['EMA_200'] else -1,
            'macd_cross': 1 if current['MACD'] > current['MACD_Signal'] else -1,
            'rsi_signal': 1 if current['RSI'] < 30 else (-1 if current['RSI'] > 70 else 0),
            'obv_trend': 1 if current['OBV'] > prev['OBV'] else -1,
            'bb_position': 1 if current['Close'] < current['BB_Lower'] else 
                          (-1 if current['Close'] > current['BB_Upper'] else 0)
        }
        
        total_score = sum(signals.values())
        max_score = len([x for x in signals.values() if x != 0])
        
        confidence = abs(total_score) / max_score if max_score > 0 else 0.5
        
        if total_score > 0:
            action = "BUY"
            tp = current['Close'] * 1.015  # 1.5% TP
            sl = current['Close'] * 0.985  # 1.5% SL
        elif total_score < 0:
            action = "SELL" 
            tp = current['Close'] * 0.985  # 1.5% TP
            sl = current['Close'] * 1.015  # 1.5% SL
        else:
            action = "HOLD"
            tp = current['Close']
            sl = current['Close']
        
        return {
            "action": action,
            "confidence": min(confidence, 0.95),  # Cap at 95%
            "take_profit": tp,
            "stop_loss": sl,
            "current_price": current['Close'],
            "signals": signals,
            "score": total_score
        }
    
    def get_technical_summary(self, data):
        """Ringkasan analisis teknikal"""
        if data is None or len(data) < 10:
            return {}
        
        current = data.iloc[-1]
        return {
            'current_price': current['Close'],
            'ema_200': current['EMA_200'],
            'trend': "BULLISH" if current['Close'] > current['EMA_200'] else "BEARISH",
            'macd_signal': "BULLISH" if current['MACD'] > current['MACD_Signal'] else "BEARISH",
            'rsi': current['RSI'],
            'obv_signal': "BULLISH" if current['OBV'] > data['OBV'].iloc[-5] else "BEARISH"
        }
    
    def run_complete_dashboard(self):
        """Menjalankan dashboard lengkap"""
        st.set_page_config(
            page_title="FOREX AI ANALYSIS SYSTEM - PRO",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center;}
        .section-header {font-size: 1.5rem; color: #ff7f0e; margin-top: 2rem;}
        .positive {color: #2ecc71;}
        .negative {color: #e74c3c;}
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<h1 class="main-header">🎯 FOREX AI ANALYSIS SYSTEM - PRO</h1>', 
                   unsafe_allow_html=True)
        st.markdown("**Google Drive Integrated | Multi-Timeframe | AI-Powered | Personal Use**")
        
        # Sidebar
        st.sidebar.header("⚙️ KONFIGURASI")
        selected_pair = st.sidebar.selectbox("Pilih Pasangan:", self.pairs)
        selected_timeframe = st.sidebar.selectbox("Pilih Timeframe:", self.timeframes)
        # Tambahkan di bagian sidebar dashboard app.py

# TESTING SECTION
st.sidebar.markdown("---")
st.sidebar.header("🧪 TESTING & DEVELOPMENT")

if st.sidebar.button("Generate Dummy Data"):
    from src.dummy_data_generator import ForexDataGenerator
    generator = ForexDataGenerator()
    generator.save_dummy_data()
    st.sidebar.success("✅ Dummy data generated!")

if st.sidebar.button("Run Quick Backtest"):
    from src.backtester import quick_backtest_test
    results = quick_backtest_test()
    st.sidebar.write("Backtest Results:", results)

if st.sidebar.button("Run System Tests"):
    from tests.test_technical_analysis import run_all_tests
    if run_all_tests():
        st.sidebar.success("✅ All tests passed!")
    else:
        st.sidebar.error("❌ Some tests failed!")
        # Main Content
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown('<div class="section-header">📊 CHART ANALYSIS</div>', 
                       unsafe_allow_html=True)
            
            # Load data
            with st.spinner("🔄 Memuat data..."):
                data = self.load_data(selected_pair, selected_timeframe)
            
            if data is not None:
                # Display advanced chart
                chart = self.create_advanced_chart(data.tail(100), selected_pair, selected_timeframe)
                st.plotly_chart(chart, use_container_width=True)
                
                # Technical Analysis Summary
                st.markdown('<div class="section-header">📈 SUMMARY TEKNIKAL</div>', 
                           unsafe_allow_html=True)
                
                tech_summary = self.get_technical_summary(data)
                if tech_summary:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Harga", f"{tech_summary['current_price']:.4f}")
                        trend_color = "positive" if tech_summary['trend'] == "BULLISH" else "negative"
                        st.markdown(f"Trend: <span class='{trend_color}'>{tech_summary['trend']}</span>", 
                                  unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("EMA 200", f"{tech_summary['ema_200']:.4f}")
                        st.metric("RSI", f"{tech_summary['rsi']:.1f}")
                    
                    with col3:
                        macd_color = "positive" if tech_summary['macd_signal'] == "BULLISH" else "negative"
                        st.markdown(f"MACD: <span class='{macd_color}'>{tech_summary['macd_signal']}</span>", 
                                  unsafe_allow_html=True)
                        
                        obv_color = "positive" if tech_summary['obv_signal'] == "BULLISH" else "negative"
                        st.markdown(f"OBV: <span class='{obv_color}'>{tech_summary['obv_signal']}</span>", 
                                  unsafe_allow_html=True)
                    
                    with col4:
                        price_vs_ema = tech_summary['current_price'] - tech_summary['ema_200']
                        st.metric("Selisih Harga-EMA", f"{price_vs_ema:.4f}")
        
        with col2:
            st.markdown('<div class="section-header">💡 TRADING SIGNALS</div>', 
                       unsafe_allow_html=True)
            
            if data is not None:
                signals = self.generate_trading_signals(data)
                
                # Display signals
                action_color = "positive" if signals["action"] == "BUY" else "negative"
                st.markdown(f"<h2 style='color: {action_color}; text-align: center;'>{signals['action']}</h2>", 
                           unsafe_allow_html=True)
                
                st.metric("Confidence", f"{signals['confidence']*100:.1f}%")
                st.progress(signals['confidence'])
                
                st.metric("Current Price", f"{signals['current_price']:.4f}")
                st.metric("Take Profit", f"{signals['take_profit']:.4f}")
                st.metric("Stop Loss", f"{signals['stop_loss']:.4f}")
                
                # Signal details
                with st.expander("Detail Sinyal"):
                    for signal_name, value in signals['signals'].items():
                        emoji = "🟢" if value > 0 else "🔴" if value < 0 else "⚪"
                        st.write(f"{emoji} {signal_name}: {value}")
                    st.write(f"**Total Score:** {signals['score']}")
        
        # AI Analysis Section
        st.markdown("---")
        st.markdown('<div class="section-header">🤖 AI DEEP ANALYSIS</div>', 
                   unsafe_allow_html=True)
        
        if st.button("🚀 Dapatkan Analisis AI Lengkap", type="primary"):
            with st.spinner("AI sedang menganalisis..."):
                tech_data = self.get_technical_summary(data)
                news_data = "Berita forex terkini: BOJ pertahankan kebijakan, Fed signal kenaikan suku bunga"
                
                ai_analysis = self.get_ai_analysis(selected_pair, tech_data, news_data)
                
                st.text_area("Analisis AI:", ai_analysis, height=300, key="ai_analysis")
        
        # Data Management
        st.sidebar.markdown("---")
        st.sidebar.header("💾 DATA MANAGEMENT")
        
        if st.sidebar.button("🔄 Update Semua Data"):
            with st.spinner("Mengupdate data semua pairs..."):
                for pair in self.pairs:
                    for tf in self.timeframes:
                        data = self.get_real_forex_data(pair)
                        if data is not None:
                            data_resampled = self.resample_data(data, tf)
                            data_with_indicators = self.calculate_technical_indicators(data_resampled)
                            self.save_data(data_with_indicators, pair, tf)
                st.success("✅ Semua data berhasil diupdate!")
        
        if st.sidebar.button("📁 Lihat File di Google Drive"):
            files = self.gdrive.list_files()
            if files:
                st.sidebar.write("**File tersimpan:**")
                for file in files[:10]:  # Tampilkan 10 file pertama
                    st.sidebar.write(f"📄 {file['name']}")
            else:
                st.sidebar.info("Belum ada file di Google Drive")
        
        # News Section
        st.sidebar.markdown("---")
        st.sidebar.header("📰 MARKET NEWS")
        st.sidebar.info("""
        **Update Terkini:**
        - BOJ: Pertahankan kebijakan moneter
        - Fed: Signal kenaikan suku bunga  
        - ECB: Bahas inflasi Eropa
        - GDP Inggris: Lebih baik dari ekspektasi
        """)

def main():
    try:
        system = CompleteForexSystem()
        system.run_complete_dashboard()
    except Exception as e:
        st.error(f"Error menjalankan sistem: {e}")
        st.info("Pastikan file konfigurasi Google Drive sudah benar dan koneksi internet tersedia.")

if __name__ == "__main__":
    main()
