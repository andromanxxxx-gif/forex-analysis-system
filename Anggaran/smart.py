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
import base64

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
    .file-upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
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
        Dengan normalisasi yang benar agar maksimal 100
        """
        try:
            # Normalisasi input data antara 0-100
            revisi_dipa = 100.00  # Asumsi optimal - tidak ada revisi
            
            # Deviasi RPD: semakin kecil deviasi semakin baik, maksimal 100 jika deviasi <= 5%
            deviasi_halaman_iii = max(0, 100 - (min(deviasi_rpd, 50) * 2))  # Normalisasi deviasi
            
            # Pastikan penyerapan antara 0-100
            penyerapan_anggaran = max(0, min(100, penyerapan_data))
            
            belanja_kontraktual = 0.00  # Tidak ada kontraktual
            penyelesaian_tagihan = 0.00  # Tidak ada kontraktual
            pengelolaan_up_tup = 100.00  # Asumsi optimal
            
            # Pastikan capaian output antara 0-100
            capaian_output = max(0, min(100, capaian_output_data))
            
            dispensasi_spm = 0.00  # Asumsi optimal
            
            # PERHITUNGAN YANG BENAR - sesuai rumus resmi
            # Nilai per komponen dikalikan bobot, lalu dijumlahkan
            nilai_per_komponen = (
                (revisi_dipa * 10) +          # Bobot 10%
                (deviasi_halaman_iii * 15) +  # Bobot 15%  
                (penyerapan_anggaran * 20) +  # Bobot 20%
                (belanja_kontraktual * 0) +   # Bobot 0%
                (penyelesaian_tagihan * 0) +  # Bobot 0%
                (pengelolaan_up_tup * 10) +   # Bobot 10%
                (capaian_output * 25)         # Bobot 25%
            )
            
            # Total bobot efektif = 80% (karena beberapa indikator 0%)
            nilai_total = nilai_per_komponen / 100  # Konversi ke skala 1
            
            # Konversi ke skala 100
            nilai_konversi = (nilai_total / 80) * 100
            
            # Kurangi dispensasi SPM
            nilai_akhir = max(0, min(100, nilai_konversi - dispensasi_spm))
            
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
                    'dispensasi_spm': 0,
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

class PDFProcessor:
    def __init__(self):
        self.extracted_data = {}
    
    def process_capaian_output_pdf(self, file_buffer):
        """Extract achievement output from PDF"""
        try:
            text = ""
            with pdfplumber.open(file_buffer) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            # Multiple patterns to find achievement percentage
            patterns = [
                r'capaian\s*output\s*:?\s*(\d+[.,]?\d*)%',
                r'persentase\s*capaian\s*:?\s*(\d+[.,]?\d*)%',
                r'capaian\s*:?\s*(\d+[.,]?\d*)%',
                r'realisasi\s*output\s*:?\s*(\d+[.,]?\d*)%',
                r'(\d+[.,]?\d*)%\s*.*capaian',
                r'capaian.*?(\d+[.,]?\d*)%'
            ]
            
            persentase = 0
            for pattern in patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    # Handle comma as decimal separator
                    value = matches[0].replace(',', '.')
                    persentase = float(value)
                    break
            
            self.extracted_data['capaian_output'] = persentase
            self.extracted_data['capaian_text'] = text[:2000]  # Store first 2000 chars
            
            return {
                'persentase_capaian': persentase,
                'text_sample': text[:1000],
                'status': 'success'
            }
            
        except Exception as e:
            return {"error": f"Error processing capaian output PDF: {str(e)}"}
    
    def process_rencana_penarikan_pdf(self, file_buffer):
        """Extract RPD data from PDF"""
        try:
            text = ""
            with pdfplumber.open(file_buffer) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            # Look for RPD patterns
            patterns = [
                r'rencana\s*penarikan\s*dana.*?(\d+[.,]?\d*)%',
                r'rpd.*?(\d+[.,]?\d*)%',
                r'deviasi.*?rpd.*?(\d+[.,]?\d*)%',
                r'penarikan\s*dana.*?(\d+[.,]?\d*)%'
            ]
            
            # Also try to extract tabular data if available
            rpd_data = {}
            
            # Simple extraction for demo
            for pattern in patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    value = matches[0].replace(',', '.')
                    rpd_data['deviasi_rpd'] = float(value)
                    break
            
            self.extracted_data['rpd_info'] = rpd_data
            self.extracted_data['rpd_text'] = text[:2000]
            
            return {
                'rpd_data': rpd_data,
                'text_sample': text[:1000],
                'status': 'success'
            }
            
        except Exception as e:
            return {"error": f"Error processing RPD PDF: {str(e)}"}
    
    def process_ikpa_previous_pdf(self, file_buffer):
        """Extract previous IKPA data from PDF"""
        try:
            text = ""
            with pdfplumber.open(file_buffer) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            # Look for IKPA value patterns
            patterns = [
                r'ikpa.*?(\d+[.,]?\d*)',
                r'nilai\s*ikpa.*?(\d+[.,]?\d*)',
                r'indikator\s*kinerja.*?(\d+[.,]?\d*)',
                r'(\d+[.,]?\d*).*ikpa'
            ]
            
            ikpa_value = 0
            for pattern in patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    value = matches[0].replace(',', '.')
                    ikpa_value = float(value)
                    break
            
            # Also look for category
            kategori_patterns = [
                r'sangat\s+baik',
                r'baik',
                r'cukup',
                r'kurang'
            ]
            
            kategori = "Tidak Diketahui"
            for pattern in kategori_patterns:
                if re.search(pattern, text.lower()):
                    kategori = pattern.upper()
                    break
            
            self.extracted_data['ikpa_previous'] = ikpa_value
            self.extracted_data['ikpa_kategori'] = kategori
            self.extracted_data['ikpa_text'] = text[:2000]
            
            return {
                'nilai_ikpa': ikpa_value,
                'kategori': kategori,
                'text_sample': text[:1000],
                'status': 'success'
            }
            
        except Exception as e:
            return {"error": f"Error processing IKPA PDF: {str(e)}"}

class DeepSeekAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        
    def analyze_ikpa(self, ikpa_data, budget_data, capaian_output, previous_ikpa=None):
        """Analyze IKPA and generate recommendations using DeepSeek API"""
        try:
            prompt = self._build_analysis_prompt(ikpa_data, budget_data, capaian_output, previous_ikpa)
            
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
    
    def _build_analysis_prompt(self, ikpa_data, budget_data, capaian_output, previous_ikpa):
        """Build analysis prompt for DeepSeek"""
        
        previous_text = ""
        if previous_ikpa:
            previous_text = f"- IKPA Bulan Sebelumnya: {previous_ikpa['nilai_ikpa']:.2f} ({previous_ikpa['kategori']})"
        
        return f"""
        ANALISIS KINERJA IKPA BNNP DKI JAKARTA:

        DATA KINERJA:
        - Nilai IKPA: {ikpa_data['nilai_akhir']:.2f} ({ikpa_data['kategori']})
        - Target IKPA: 95.00 (Sangat Baik)
        - Gap: {95 - ikpa_data['nilai_akhir']:.2f} poin
        {previous_text}

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
    """Create component breakdown chart - FIXED VERSION"""
    try:
        # Components with non-zero weights only
        components_to_show = ['revisi_dipa', 'deviasi_halaman_iii', 'penyerapan_anggaran', 'pengelolaan_up_tup', 'capaian_output']
        
        components = []
        values = []
        bobot = []
        
        for comp in components_to_show:
            if comp in ikpa_data['components'] and comp in ikpa_data['bobot']:
                components.append(comp)
                values.append(ikpa_data['components'][comp])
                bobot.append(ikpa_data['bobot'][comp])
        
        if not components:
            # Return empty figure if no data
            fig = go.Figure()
            fig.update_layout(title="No IKPA component data available", height=400)
            return fig
        
        # Create readable labels
        readable_labels = {
            'revisi_dipa': 'Revisi DIPA',
            'deviasi_halaman_iii': 'Deviasi RPD', 
            'penyerapan_anggaran': 'Penyerapan',
            'pengelolaan_up_tup': 'UP/TUP',
            'capaian_output': 'Capaian Output'
        }
        
        labels = [readable_labels.get(comp, comp) for comp in components]
        
        fig = go.Figure(data=[
            go.Bar(name='Nilai', x=labels, y=values, marker_color='blue'),
            go.Bar(name='Target', x=labels, y=[95]*len(components), marker_color='red', opacity=0.3)
        ])
        
        fig.update_layout(
            title='Perbandingan Komponen IKPA vs Target',
            barmode='overlay',
            height=400
        )
        
        return fig
        
    except Exception as e:
        # Return empty figure in case of error
        fig = go.Figure()
        fig.update_layout(title=f"Error creating chart: {str(e)}", height=400)
        return fig

def main():
    st.markdown('<h1 class="main-header">🏢 SMART - Analisis IKPA & Rekomendasi</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Badan Narkotika Nasional Provinsi DKI Jakarta</h3>', unsafe_allow_html=True)
    
    # Initialize classes
    budget_processor = BudgetProcessor()
    ikpa_calculator = IKPACalculator()
    pdf_processor = PDFProcessor()
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {
            'budget': None,
            'capaian_output': None,
            'rpd': None,
            'ikpa_previous': None
        }
    
    # Sidebar configuration
    st.sidebar.header("🔑 Konfigurasi")
    
    # DeepSeek API Key
    api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="Masukkan API Key...",
        help="Dapatkan dari https://platform.deepseek.com/",
        key="api_key_input"  # UNIQUE KEY
    )
    
    deepseek_analyzer = None
    if api_key:
        deepseek_analyzer = DeepSeekAnalyzer(api_key)
        st.sidebar.success("✅ API Terkoneksi")
    
    # File uploads section
    st.sidebar.header("📁 Upload Data")
    
    # Budget Excel Upload
    st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
    st.sidebar.subheader("💰 Data Penyerapan Anggaran")
    budget_file = st.sidebar.file_uploader(
        "Laporan Penyerapan Anggaran (Excel)",
        type=['xlsx', 'xls'],
        help="Upload file Excel laporan realisasi anggaran",
        key="budget_upload"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # PDF Uploads
    st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
    st.sidebar.subheader("📊 Data Tambahan (PDF)")
    
    capaian_output_file = st.sidebar.file_uploader(
        "Capaian Output Bulan Terakhir (PDF)",
        type=['pdf'],
        help="Upload PDF laporan capaian output",
        key="capaian_upload"
    )
    
    rencana_penarikan_file = st.sidebar.file_uploader(
        "Rencana Penarikan Dana (PDF)",
        type=['pdf'],
        help="Upload PDF rencana penarikan dana",
        key="rpd_upload"
    )
    
    ikpa_previous_file = st.sidebar.file_uploader(
        "IKPA Bulan Sebelumnya (PDF)",
        type=['pdf'],
        help="Upload PDF laporan IKPA bulan sebelumnya",
        key="ikpa_previous_upload"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Manual input fallback
    st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
    st.sidebar.subheader("✍️ Input Manual")
    manual_capaian_output = st.sidebar.number_input(
        "Capaian Output (%) - Manual",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        help="Masukkan persentase capaian output jika tidak upload PDF",
        key="manual_capaian_input"  # UNIQUE KEY
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded files
    budget_data = None
    capaian_output_data = manual_capaian_output
    rpd_data = None
    ikpa_previous_data = None
    
    # Process budget file
    if budget_file:
        with st.spinner("Memproses data anggaran..."):
            budget_data = budget_processor.process_budget_file(budget_file)
            if "error" not in budget_data:
                st.session_state.processed_data['budget'] = budget_data
                st.sidebar.success("✅ Data anggaran berhasil diproses")
            else:
                st.sidebar.error(f"❌ {budget_data['error']}")
    
    # Process capaian output PDF
    if capaian_output_file:
        with st.spinner("Memproses capaian output..."):
            result = pdf_processor.process_capaian_output_pdf(capaian_output_file)
            if "error" not in result:
                capaian_output_data = result['persentase_capaian']
                st.session_state.processed_data['capaian_output'] = result
                st.sidebar.success(f"✅ Capaian output: {capaian_output_data}%")
            else:
                st.sidebar.error(f"❌ {result['error']}")
    
    # Process RPD PDF
    if rencana_penarikan_file:
        with st.spinner("Memproses Rencana Penarikan Dana..."):
            result = pdf_processor.process_rencana_penarikan_pdf(rencana_penarikan_file)
            if "error" not in result:
                st.session_state.processed_data['rpd'] = result
                st.sidebar.success("✅ Data RPD berhasil diproses")
    
    # Process previous IKPA PDF
    if ikpa_previous_file:
        with st.spinner("Memproses IKPA sebelumnya..."):
            result = pdf_processor.process_ikpa_previous_pdf(ikpa_previous_file)
            if "error" not in result:
                ikpa_previous_data = result
                st.session_state.processed_data['ikpa_previous'] = result
                st.sidebar.success(f"✅ IKPA sebelumnya: {result['nilai_ikpa']:.2f}")
    
    # Use stored data if available
    if st.session_state.processed_data['budget']:
        budget_data = st.session_state.processed_data['budget']
    
    # Calculate IKPA if we have budget data
    ikpa_result = None
    if budget_data and "error" not in budget_data:
        with st.spinner("Menghitung nilai IKPA..."):
            ikpa_result = ikpa_calculator.calculate_ikpa(
                budget_data['penyerapan_persen'],
                capaian_output_data,
                budget_data['deviasi_rpd']
            )
    
    # Display results
    if ikpa_result and "error" not in ikpa_result:
        # Header with IKPA Score
        st.header("📊 Hasil Analisis IKPA")
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            fig_gauge = create_ikpa_gauge(ikpa_result['nilai_akhir'], ikpa_result['kategori'])
            st.plotly_chart(fig_gauge, width='stretch')
        
        with col2:
            st.metric("Nilai IKPA", f"{ikpa_result['nilai_akhir']:.2f}")
            st.metric("Kategori", ikpa_result['kategori'])
        
        with col3:
            gap = 95 - ikpa_result['nilai_akhir']
            st.metric("Gap Target", f"{gap:+.2f}")
            status = "✅ Optimal" if ikpa_result['nilai_akhir'] >= 95 else "⚠️ Perlu Perbaikan"
            st.metric("Status", status)
        
        with col4:
            if ikpa_previous_data:
                change = ikpa_result['nilai_akhir'] - ikpa_previous_data['nilai_ikpa']
                st.metric("IKPA Sebelumnya", f"{ikpa_previous_data['nilai_ikpa']:.2f}", f"{change:+.2f}")
            else:
                st.metric("IKPA Sebelumnya", "N/A")
        
        # Component Analysis
        st.header("🔍 Analisis Komponen IKPA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_components = create_component_chart(ikpa_result)
            st.plotly_chart(fig_components, width='stretch')
        
        with col2:
            # Component details
            st.subheader("Detail Komponen")
            components_data = []
            for comp, weight in ikpa_result['bobot'].items():
                if weight > 0 and comp in ikpa_result['components']:
                    components_data.append({
                        'Komponen': comp.replace('_', ' ').title(),
                        'Nilai': ikpa_result['components'][comp],
                        'Bobot': f"{weight}%",
                        'Kontribusi': f"{ikpa_result['components'][comp] * weight / 100:.2f}"
                    })
            
            components_df = pd.DataFrame(components_data)
            st.dataframe(components_df, width='stretch', hide_index=True)
            
            # Summary
            st.subheader("📈 Summary")
            st.write(f"**Total Bobot Efektif:** {ikpa_result['bobot']['total_efektif']}%")
            st.write(f"**Nilai Total:** {ikpa_result['nilai_total']:.2f}")
            st.write(f"**Nilai Konversi:** {ikpa_result['nilai_konversi']:.2f}")
        
        # Data Sources Section
        st.header("📁 Sumber Data yang Diproses")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Anggaran", "✅ Tersedia" if budget_data else "❌ Tidak Ada")
        
        with col2:
            st.metric("Capaian Output", f"{capaian_output_data}%")
        
        with col3:
            deviasi = budget_data['deviasi_rpd'] if budget_data else 0
            st.metric("Deviasi RPD", f"{deviasi:.1f}%")
        
        with col4:
            prev_ikpa = ikpa_previous_data['nilai_ikpa'] if ikpa_previous_data else "N/A"
            st.metric("IKPA Sebelumnya", f"{prev_ikpa}")
        
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
                st.plotly_chart(fig_bidang, width='stretch')
        
        # Extracted Data from PDFs
        if (st.session_state.processed_data['capaian_output'] or 
            st.session_state.processed_data['rpd'] or 
            st.session_state.processed_data['ikpa_previous']):
            
            st.header("📄 Data yang Diekstrak dari PDF")
            
            cols = st.columns(3)
            
            with cols[0]:
                if st.session_state.processed_data['capaian_output']:
                    data = st.session_state.processed_data['capaian_output']
                    st.subheader("Capaian Output")
                    st.write(f"**Persentase:** {data['persentase_capaian']}%")
                    with st.expander("Lihat teks yang diekstrak"):
                        # FIXED: Added unique key
                        st.text_area("Teks Capaian Output", data['text_sample'], height=200, key="capaian_output_text")
            
            with cols[1]:
                if st.session_state.processed_data['rpd']:
                    data = st.session_state.processed_data['rpd']
                    st.subheader("Rencana Penarikan")
                    if 'rpd_data' in data and 'deviasi_rpd' in data['rpd_data']:
                        st.write(f"**Deviasi RPD:** {data['rpd_data']['deviasi_rpd']}%")
                    with st.expander("Lihat teks yang diekstrak"):
                        # FIXED: Added unique key
                        st.text_area("Teks RPD", data['text_sample'], height=200, key="rpd_text")
            
            with cols[2]:
                if st.session_state.processed_data['ikpa_previous']:
                    data = st.session_state.processed_data['ikpa_previous']
                    st.subheader("IKPA Sebelumnya")
                    st.write(f"**Nilai:** {data['nilai_ikpa']:.2f}")
                    st.write(f"**Kategori:** {data['kategori']}")
                    with st.expander("Lihat teks yang diekstrak"):
                        # FIXED: Added unique key
                        st.text_area("Teks IKPA Sebelumnya", data['text_sample'], height=200, key="ikpa_previous_text")
        
        # AI Recommendations
        if deepseek_analyzer and budget_data:
            st.header("🤖 Rekomendasi AI DeepSeek")
            
            if st.button("🔄 Generate Rekomendasi Mendalam", type="primary", key="generate_recommendations_btn"):
                with st.spinner("DeepSeek AI menganalisis data dan menghasilkan rekomendasi..."):
                    recommendations = deepseek_analyzer.analyze_ikpa(
                        ikpa_result, 
                        budget_data, 
                        capaian_output_data,
                        ikpa_previous_data
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
        - **Analisis Komprehensif** nilai IKPA dengan normalisasi yang benar
        - **Multi-file Upload** untuk data lengkap
        - **Ekstraksi Otomatis** dari file PDF
        - **Rekomendasi AI** menggunakan DeepSeek
        - **Visualisasi Interaktif** kinerja anggaran
        
        ### 📊 Sumber Data yang Didukung:
        1. **Excel Anggaran** - Data penyerapan anggaran
        2. **PDF Capaian Output** - Laporan capaian output 
        3. **PDF Rencana Penarikan** - Data RPD dan deviasi
        4. **PDF IKPA Sebelumnya** - Data historis IKPA
        
        ### 🚀 Cara Menggunakan:
        1. **Upload file Excel** laporan penyerapan anggaran
        2. **Upload file PDF** tambahan (opsional)
        3. **Input manual** data yang tidak tersedia
        4. **Lihat hasil analisis** dan rekomendasi otomatis
        """)
        
        # Demo visualization
        st.header("📈 Contoh Visualisasi IKPA")
        
        # Create demo IKPA data
        demo_ikpa = ikpa_calculator.calculate_ikpa(75.0, 80.0, 6.0)
        
        if demo_ikpa and "error" not in demo_ikpa:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_demo = create_ikpa_gauge(demo_ikpa['nilai_akhir'], demo_ikpa['kategori'])
                st.plotly_chart(fig_demo, width='stretch')
            
            with col2:
                st.metric("Nilai IKPA Demo", f"{demo_ikpa['nilai_akhir']:.2f}", key="demo_ikpa_metric")
                st.metric("Kategori", demo_ikpa['kategori'], key="demo_kategori_metric")
                st.metric("Status", "Contoh Analisis", key="demo_status_metric")

if __name__ == "__main__":
    main()
