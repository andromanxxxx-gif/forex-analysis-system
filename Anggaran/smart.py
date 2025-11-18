import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pdfplumber
import re
import requests
import io

# Set page config
st.set_page_config(
    page_title="SMART BNNP DKI - IKPA Analyzer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .excellent { color: #28a745; font-weight: bold; }
    .good { color: #ffc107; font-weight: bold; }
    .fair { color: #fd7e14; font-weight: bold; }
    .poor { color: #dc3545; font-weight: bold; }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IKPACalculator:
    def __init__(self):
        self.ikpa_data = None
        
    def calculate_ikpa(self, penyerapan_data, capaian_output_data, deviasi_rpd=0):
        """
        Calculate IKPA based on actual formula from PER-5/PB/2022
        Asumsi: Tidak ada kontraktual, UP/TUP optimal
        """
        try:
            # Hitung komponen IKPA
            revisi_dipa = 100.00  # Asumsi optimal - tidak ada revisi
            deviasi_halaman_iii = max(0, 100 - (deviasi_rpd * 2))  # Konversi deviasi ke skala 0-100
            penyerapan_anggaran = penyerapan_data
            belanja_kontraktual = 0.00  # Tidak ada kontraktual
            penyelesaian_tagihan = 0.00  # Tidak ada kontraktual
            pengelolaan_up_tup = 100.00  # Asumsi optimal
            capaian_output = capaian_output_data
            dispensasi_spm = 0.00  # Asumsi optimal
            
            # Perhitungan nilai total berdasarkan bobot
            nilai_total = (
                (revisi_dipa * 0.10) +
                (deviasi_halaman_iii * 0.15) +
                (penyerapan_anggaran * 0.20) +
                (belanja_kontraktual * 0.00) +
                (penyelesaian_tagihan * 0.00) +
                (pengelolaan_up_tup * 0.10) +
                (capaian_output * 0.25)
            )
            
            # Konversi ke skala 100 (karena total bobot efektif = 80%)
            nilai_konversi = (nilai_total / 0.80) * 100
            
            # Kurangi dispensasi SPM
            nilai_akhir = nilai_konversi - dispensasi_spm
            
            # Tentukan kategori
            if nilai_akhir >= 95:
                kategori = "Sangat Baik"
                color_class = "excellent"
            elif nilai_akhir >= 89:
                kategori = "Baik"
                color_class = "good"
            elif nilai_akhir >= 70:
                kategori = "Cukup"
                color_class = "fair"
            else:
                kategori = "Kurang"
                color_class = "poor"
            
            self.ikpa_data = {
                'nilai_total': nilai_total,
                'nilai_konversi': nilai_konversi,
                'nilai_akhir': nilai_akhir,
                'kategori': kategori,
                'color_class': color_class,
                'components': {
                    'revisi_dipa': revisi_dipa,
                    'deviasi_halaman_iii': deviasi_halaman_iii,
                    'penyerapan_anggaran': penyerapan_anggaran,
                    'belanja_kontraktual': belanja_kontraktual,
                    'penyelesaian_tagihan': penyelesaian_tagihan,
                    'pengelolaan_up_tup': pengelolaan_up_tup,
                    'capaian_output': capaian_output,
                    'dispensasi_spm': dispensasi_spm
                },
                'bobot': {
                    'revisi_dipa': 10,
                    'deviasi_halaman_iii': 15,
                    'penyerapan_anggaran': 20,
                    'belanja_kontraktual': 0,
                    'penyelesaian_tagihan': 0,
                    'pengelolaan_up_tup': 10,
                    'capaian_output': 25,
                    'total_efektif': 80
                }
            }
            
            return self.ikpa_data
            
        except Exception as e:
            return {"error": f"Error calculating IKPA: {str(e)}"}

class BudgetProcessor:
    def __init__(self):
        self.penyerapan_data = None
        
    def process_budget_file(self, file_buffer):
        """Process budget Excel file"""
        try:
            # Baca file Excel
            df = pd.read_excel(file_buffer)
            
            # Deteksi kolom secara fleksibel
            column_mapping = self._detect_columns(df)
            
            # Rename kolom ke standar
            df = df.rename(columns=column_mapping)
            
            # Pastikan kolom numerik
            numeric_cols = ['Harga Satuan', 'Jumlah', 'Realisasi', 'Sisa Anggaran']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Hitung metrik utama
            total_alokasi = df['Jumlah'].sum() if 'Jumlah' in df.columns else 0
            total_realisasi = df['Realisasi'].sum() if 'Realisasi' in df.columns else 0
            penyerapan_persen = (total_realisasi / total_alokasi * 100) if total_alokasi > 0 else 0
            
            # Hitung deviasi RPD (simplified)
            deviasi_rpd = self._calculate_deviation_rpd(df)
            
            # Data per bidang
            if 'Bidang' in df.columns:
                bidang_summary = df.groupby('Bidang').agg({
                    'Jumlah': 'sum',
                    'Realisasi': 'sum'
                }).reset_index()
                bidang_summary['Penyerapan_Persen'] = (bidang_summary['Realisasi'] / bidang_summary['Jumlah'] * 100)
            else:
                bidang_summary = pd.DataFrame({
                    'Bidang': ['All'],
                    'Jumlah': [total_alokasi],
                    'Realisasi': [total_realisasi],
                    'Penyerapan_Persen': [penyerapan_persen]
                })
            
            self.penyerapan_data = {
                'total_alokasi': total_alokasi,
                'total_realisasi': total_realisasi,
                'penyerapan_persen': penyerapan_persen,
                'deviasi_rpd': deviasi_rpd,
                'bidang_summary': bidang_summary,
                'raw_data': df,
                'columns_found': list(column_mapping.values())
            }
            
            return self.penyerapan_data
            
        except Exception as e:
            return {"error": f"Error processing budget file: {str(e)}"}
    
    def _detect_columns(self, df):
        """Detect and map columns flexibly"""
        column_mapping = {}
        
        # Mapping patterns
        patterns = {
            'Kode': ['kode', 'KODE', 'Kode Rekening', 'KODE REKENING'],
            'Uraian Volume': ['uraian', 'Uraian', 'Kegiatan', 'KEGIATAN', 'Uraian Volume', 'URAIAN VOLUME'],
            'Satuan': ['satuan', 'SATUAN', 'Unit', 'UNIT'],
            'Harga Satuan': ['harga', 'Harga', 'Harga Satuan', 'HARGA SATUAN'],
            'Jumlah': ['jumlah', 'Jumlah', 'Anggaran', 'ANGGARAN', 'Jumlah Anggaran'],
            'Realisasi': ['realisasi', 'Realisasi', 'REALISASI', 'Realisasi Anggaran'],
            'Sisa Anggaran': ['sisa', 'Sisa', 'Sisa Anggaran', 'SISA ANGGARAN'],
            'Bidang': ['bidang', 'Bidang', 'BIDANG', 'Unit Kerja'],
            'Triwulan': ['triwulan', 'Triwulan', 'TRIWULAN', 'Periode']
        }
        
        for standard_name, possible_names in patterns.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    column_mapping[standard_name] = possible_name
                    break
        
        return column_mapping
    
    def _calculate_deviation_rpd(self, df):
        """Calculate RPD deviation (simplified)"""
        try:
            if 'Realisasi' in df.columns and 'Jumlah' in df.columns:
                # Asumsi: deviasi dihitung dari variasi penyerapan
                avg_deviation = abs(df['Realisasi'] / df['Jumlah'].replace(0, 1) - 1).mean() * 100
                return min(avg_deviation, 50)  # Cap at 50%
            return 5.0  # Default value
        except:
            return 5.0  # Default value

class DeepSeekAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        
    def analyze_ikpa(self, ikpa_data, budget_data, capaian_output):
        """Analyze IKPA and generate recommendations using DeepSeek API"""
        try:
            prompt = self._build_analysis_prompt(ikpa_data, budget_data, capaian_output)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": """Anda adalah ahli monitoring dan evaluasi kinerja pemerintah. 
                        Berikan analisis mendalam dan rekomendasi praktis untuk meningkatkan nilai IKPA.
                        Format output: ANALISIS, REKOMENDASI STRATEGIS, ACTION PLAN, MONITORING"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error API: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"
    
    def _build_analysis_prompt(self, ikpa_data, budget_data, capaian_output):
        """Build analysis prompt for DeepSeek"""
        
        return f"""
        ANALISIS KINERJA IKPA BNNP DKI JAKARTA:

        DATA KINERJA:
        - Nilai IKPA: {ikpa_data['nilai_akhir']:.2f} ({ikpa_data['kategori']})
        - Target IKPA: 95.00 (Sangat Baik)
        - Gap: {95 - ikpa_data['nilai_akhir']:.2f} poin

        DETAIL KOMPONEN IKPA:
        1. Revisi DIPA: {ikpa_data['components']['revisi_dipa']:.2f}% (Bobot: 10%)
        2. Deviasi Halaman III: {ikpa_data['components']['deviasi_halaman_iii']:.2f}% (Bobot: 15%)
        3. Penyerapan Anggaran: {ikpa_data['components']['penyerapan_anggaran']:.2f}% (Bobot: 20%)
        4. Pengelolaan UP/TUP: {ikpa_data['components']['pengelolaan_up_tup']:.2f}% (Bobot: 10%)
        5. Capaian Output: {ikpa_data['components']['capaian_output']:.2f}% (Bobot: 25%)

        DATA ANGGARAN:
        - Total Alokasi: Rp {budget_data['total_alokasi']:,.0f}
        - Total Realisasi: Rp {budget_data['total_realisasi']:,.0f}
        - Penyerapan: {budget_data['penyerapan_persen']:.1f}%
        - Deviasi RPD: {budget_data['deviasi_rpd']:.1f}%

        CAPAIAN OUTPUT: {capaian_output}%

        TUGAS ANDA:
        1. ANALISIS PENYEBAB: Identifikasi akar masalah dari nilai IKPA saat ini
        2. REKOMENDASI STRATEGIS: Berikan 3-5 rekomendasi utama untuk meningkatkan IKPA
        3. ACTION PLAN: Rencana aksi konkret dengan timeline
        4. TARGET PERBAIKAN: Estimasi peningkatan yang bisa dicapai

        FORMAT OUTPUT:
        ## ANALISIS KONDISI
        [analisis mendalam]

        ## REKOMENDASI STRATEGIS
        1. [Rekomendasi 1]
        2. [Rekomendasi 2] 
        3. [Rekomendasi 3]

        ## ACTION PLAN
        - Jangka Pendek (1-2 minggu): [aksi]
        - Jangka Menengah (3-4 minggu): [aksi]
        - Jangka Panjang (1-2 bulan): [aksi]

        ## TARGET CAPAIAN
        - Target IKPA: ≥95 (Sangat Baik)
        - Estimasi Peningkatan: [estimasi]
        """

def create_ikpa_gauge(value, category):
    """Create IKPA gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"IKPA - {category}", 'font': {'size': 24}},
        delta = {'reference': 95, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': 'red'},
                {'range': [70, 89], 'color': 'orange'},
                {'range': [89, 95], 'color': 'yellow'},
                {'range': [95, 100], 'color': 'green'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 95}
        }
    ))
    
    fig.update_layout(height=300, margin=dict(t=50, b=10))
    return fig

def create_component_chart(ikpa_data):
    """Create component breakdown chart"""
    components = list(ikpa_data['components'].keys())
    values = [ikpa_data['components'][k] for k in components]
    bobot = [ikpa_data['bobot'][k] for k in components]
    
    # Filter hanya komponen yang memiliki bobot
    filtered_data = [(c, v, b) for c, v, b in zip(components, values, bobot) if b > 0]
    components, values, bobot = zip(*filtered_data)
    
    fig = go.Figure(data=[
        go.Bar(name='Nilai', x=components, y=values, marker_color='blue'),
        go.Bar(name='Target', x=components, y=[95]*len(components), marker_color='red', opacity=0.3)
    ])
    
    fig.update_layout(
        title='Perbandingan Komponen IKPA vs Target',
        barmode='overlay',
        height=400
    )
    
    return fig

def main():
    st.markdown('<h1 class="main-header">🏢 SMART - Analisis IKPA & Rekomendasi</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Badan Narkotika Nasional Provinsi DKI Jakarta</h3>', unsafe_allow_html=True)
    
    # Initialize classes
    budget_processor = BudgetProcessor()
    ikpa_calculator = IKPACalculator()
    
    # Sidebar configuration
    st.sidebar.header("🔑 Konfigurasi")
    
    # DeepSeek API Key
    api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="Masukkan API Key...",
        help="Dapatkan dari https://platform.deepseek.com/"
    )
    
    deepseek_analyzer = None
    if api_key:
        deepseek_analyzer = DeepSeekAnalyzer(api_key)
        st.sidebar.success("✅ API Terkoneksi")
    
    # File uploads
    st.sidebar.header("📁 Upload Data")
    
    budget_file = st.sidebar.file_uploader(
        "Laporan Penyerapan Anggaran (Excel)",
        type=['xlsx', 'xls'],
        help="Upload file Excel laporan realisasi anggaran"
    )
    
    capaian_output = st.sidebar.number_input(
        "Capaian Output (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        help="Masukkan persentase capaian output"
    )
    
    # Process data
    budget_data = None
    ikpa_result = None
    
    if budget_file:
        with st.spinner("Memproses data anggaran..."):
            budget_data = budget_processor.process_budget_file(budget_file)
    
    if budget_data and "error" not in budget_data:
        # Calculate IKPA
        with st.spinner("Menghitung nilai IKPA..."):
            ikpa_result = ikpa_calculator.calculate_ikpa(
                budget_data['penyerapan_persen'],
                capaian_output,
                budget_data['deviasi_rpd']
            )
    
    # Display results
    if ikpa_result and "error" not in ikpa_result:
        # Header with IKPA Score
        st.header("📊 Hasil Analisis IKPA")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            fig_gauge = create_ikpa_gauge(ikpa_result['nilai_akhir'], ikpa_result['kategori'])
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            st.metric("Nilai IKPA", f"{ikpa_result['nilai_akhir']:.2f}")
            st.metric("Kategori", ikpa_result['kategori'])
        
        with col3:
            gap = 95 - ikpa_result['nilai_akhir']
            st.metric("Gap Target", f"{gap:+.2f}")
            status = "✅ Optimal" if ikpa_result['nilai_akhir'] >= 95 else "⚠️ Perlu Perbaikan"
            st.metric("Status", status)
        
        # Component Analysis
        st.header("🔍 Analisis Komponen IKPA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_components = create_component_chart(ikpa_result)
            st.plotly_chart(fig_components, use_container_width=True)
        
        with col2:
            # Component details
            st.subheader("Detail Komponen")
            components_df = pd.DataFrame({
                'Komponen': list(ikpa_result['components'].keys()),
                'Nilai': [ikpa_result['components'][k] for k in ikpa_result['components'].keys()],
                'Bobot': [ikpa_result['bobot'][k] for k in ikpa_result['components'].keys()],
                'Kontribusi': [ikpa_result['components'][k] * ikpa_result['bobot'][k] / 100 for k in ikpa_result['components'].keys()]
            })
            
            # Filter hanya yang ada bobotnya
            components_df = components_df[components_df['Bobot'] > 0]
            components_df['Kontribusi'] = components_df['Kontribusi'].round(2)
            
            st.dataframe(components_df, use_container_width=True)
            
            # Summary
            st.subheader("📈 Summary")
            st.write(f"**Total Bobot Efektif:** {ikpa_result['bobot']['total_efektif']}%")
            st.write(f"**Nilai Total:** {ikpa_result['nilai_total']:.2f}")
            st.write(f"**Nilai Konversi:** {ikpa_result['nilai_konversi']:.2f}")
        
        # Budget Analysis
        if budget_data:
            st.header("💰 Analisis Anggaran")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Alokasi", f"Rp {budget_data['total_alokasi']:,.0f}")
            with col2:
                st.metric("Total Realisasi", f"Rp {budget_data['total_realisasi']:,.0f}")
            with col3:
                st.metric("Penyerapan", f"{budget_data['penyerapan_persen']:.1f}%")
            with col4:
                st.metric("Deviasi RPD", f"{budget_data['deviasi_rpd']:.1f}%")
            
            # Bidang analysis
            if len(budget_data['bidang_summary']) > 1:
                st.subheader("📋 Penyerapan per Bidang")
                fig_bidang = px.bar(
                    budget_data['bidang_summary'],
                    x='Bidang',
                    y='Penyerapan_Persen',
                    title='Persentase Penyerapan per Bidang',
                    color='Penyerapan_Persen',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_bidang, use_container_width=True)
        
        # AI Recommendations
        if deepseek_analyzer and budget_data:
            st.header("🤖 Rekomendasi AI DeepSeek")
            
            if st.button("🔄 Generate Rekomendasi Mendalam", type="primary"):
                with st.spinner("DeepSeek AI menganalisis data dan menghasilkan rekomendasi..."):
                    recommendations = deepseek_analyzer.analyze_ikpa(
                        ikpa_result, 
                        budget_data, 
                        capaian_output
                    )
                    
                    st.markdown("### 📋 Hasil Analisis DeepSeek AI")
                    st.markdown(f'<div class="recommendation-box">{recommendations}</div>', unsafe_allow_html=True)
        
        # Manual Recommendations based on analysis
        st.header("🎯 Rekomendasi Strategis")
        
        # Analyze weaknesses and generate recommendations
        weaknesses = []
        
        if ikpa_result['components']['penyerapan_anggaran'] < 70:
            weaknesses.append("Penyerapan anggaran di bawah target triwulan 3 (70%)")
        
        if ikpa_result['components']['deviasi_halaman_iii'] < 95:
            weaknesses.append("Deviasi RPD melebihi batas optimal 5%")
        
        if ikpa_result['components']['capaian_output'] < 80:
            weaknesses.append("Capaian output perlu ditingkatkan")
        
        if weaknesses:
            st.warning("**Area Perbaikan:**")
            for weakness in weaknesses:
                st.write(f"• {weakness}")
            
            st.success("**Rekomendasi Aksi:**")
            
            if ikpa_result['components']['penyerapan_anggaran'] < 70:
                st.write("""
                **🚀 Akselerasi Penyerapan Anggaran:**
                - Prioritaskan item dengan sisa anggaran besar
                - Koordinasi intensif dengan bidang terkait
                - Monitoring mingguan progres penyerapan
                """)
            
            if ikpa_result['components']['deviasi_halaman_iii'] < 95:
                st.write("""
                **📊 Optimalkan Perencanaan RPD:**
                - Review akurasi Rencana Penarikan Dana
                - Sesuaikan dengan realisasi bulanan
                - Minimalkan deviasi ≤5%
                """)
            
            if ikpa_result['components']['capaian_output'] < 80:
                st.write("""
                **🎯 Tingkatkan Capaian Output:**
                - Perkuat dokumentasi outcome program
                - Percepat pelaporan capaian
                - Verifikasi kualitas output
                """)
        
        else:
            st.success("✅ Semua komponen IKPA dalam kondisi optimal!")
            st.write("Pertahankan kinerja yang sudah baik dan fokus pada konsistensi.")
    
    elif budget_data and "error" in budget_data:
        st.error(f"❌ {budget_data['error']}")
    
    else:
        # Welcome screen
        st.header("🚀 Selamat Datang di Sistem Analisis IKPA")
        
        st.markdown("""
        ### 📋 Tentang Sistem
        
        Sistem ini membantu menganalisis **Indikator Kinerja Pelaksanaan Anggaran (IKPA)** 
        berdasarkan Peraturan Dirjen Perbendaharaan No. PER-5/PB/2022.
        
        ### 🎯 Fitur Utama:
        - **Analisis Komprehensif** nilai IKPA
        - **Identifikasi Area Perbaikan** berdasarkan komponen IKPA
        - **Rekomendasi AI** menggunakan DeepSeek
        - **Visualisasi Interaktif** kinerja anggaran
        - **Action Plan** untuk peningkatan IKPA
        
        ### 📊 Komponen IKPA yang Dianalisis:
        1. **Revisi DIPA** (10%) - Asumsi optimal
        2. **Deviasi Halaman III DIPA** (15%) - Dari data RPD
        3. **Penyerapan Anggaran** (20%) - Dari realisasi anggaran
        4. **Pengelolaan UP/TUP** (10%) - Asumsi optimal
        5. **Capaian Output** (25%) - Input manual/upload PDF
        
        ### 🚀 Cara Menggunakan:
        1. **Upload file Excel** laporan penyerapan anggaran
        2. **Input persentase capaian output**
        3. **Masukkan API Key DeepSeek** (opsional untuk rekomendasi AI)
        4. **Lihat hasil analisis** dan rekomendasi
        
        ### 📁 Format Data yang Didukung:
        - File Excel dengan data realisasi anggaran
        - Kolom fleksibel (sistem otomatis deteksi)
        - Data numerik untuk alokasi dan realisasi
        """)
        
        # Demo visualization
        st.header("📈 Contoh Visualisasi IKPA")
        
        # Create demo IKPA data
        demo_ikpa = ikpa_calculator.calculate_ikpa(75.0, 80.0, 6.0)
        
        if demo_ikpa and "error" not in demo_ikpa:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_demo = create_ikpa_gauge(demo_ikpa['nilai_akhir'], demo_ikpa['kategori'])
                st.plotly_chart(fig_demo, use_container_width=True)
            
            with col2:
                st.metric("Nilai IKPA Demo", f"{demo_ikpa['nilai_akhir']:.2f}")
                st.metric("Kategori", demo_ikpa['kategori'])
                st.metric("Status", "Contoh Analisis")

if __name__ == "__main__":
    main()
