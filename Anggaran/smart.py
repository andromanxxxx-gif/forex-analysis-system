import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pdfplumber
import re
import requests
import io
import base64
import traceback

# Set page config
st.set_page_config(
    page_title="SMART BNNP DKI - Monitoring IKPA 2024",
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
    .regulation-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IKPA2024Calculator:
    def __init__(self):
        self.ikpa_data = None
        self.regulation_info = {
            'bobot': {
                'revisi_dipa': 10,
                'deviasi_halaman_iii': 15,
                'penyerapan_anggaran': 20,
                'belanja_kontraktual': 0,
                'penyelesaian_tagihan': 0,
                'pengelolaan_up_tup': 10,
                'capaian_output': 25,
                'dispensasi_spm': 0
            },
            'target_kkp': {
                'triwulan1': 1.0,
                'triwulan2': 5.0,
                'triwulan3': 9.0,
                'triwulan4': 12.5
            }
        }
    
    def calculate_ikpa_2024(self, data_input):
        """
        Calculate IKPA 2024 berdasarkan PER-5/PB/2024
        """
        try:
            # Ekstrak data input
            revisi_dipa = data_input.get('revisi_dipa', 100.0)
            deviasi_halaman_iii = data_input.get('deviasi_halaman_iii', 0.0)
            penyerapan_anggaran = data_input.get('penyerapan_anggaran', 0.0)
            belanja_kontraktual = data_input.get('belanja_kontraktual', 0.0)
            penyelesaian_tagihan = data_input.get('penyelesaian_tagihan', 0.0)
            pengelolaan_up_tup = data_input.get('pengelolaan_up_tup', 100.0)
            capaian_output = data_input.get('capaian_output', 0.0)
            dispensasi_spm = data_input.get('dispensasi_spm', 0.0)
            
            # Normalisasi sesuai formula 2024
            # 1. Revisi DIPA - semakin sedikit revisi semakin baik
            nilai_revisi_dipa = max(0, 100 - (revisi_dipa * 10))
            
            # 2. Deviasi Halaman III - menggunakan rata-rata tertimbang
            nilai_deviasi = max(0, 100 - min(deviasi_halaman_iii, 10) * 10)
            
            # 3. Penyerapan Anggaran - dengan rata-rata tertimbang
            nilai_penyerapan = min(100, penyerapan_anggaran)
            
            # 4. Belanja Kontraktual dengan distribusi akselerasi
            nilai_belanja_kontraktual = self._calculate_belanja_kontraktual(belanja_kontraktual)
            
            # 5. Penyelesaian Tagihan
            nilai_penyelesaian_tagihan = min(100, penyelesaian_tagihan)
            
            # 6. Pengelolaan UP/TUP dengan KKP
            nilai_pengelolaan_up_tup = self._calculate_pengelolaan_up_tup(pengelolaan_up_tup, data_input.get('triwulan', 1))
            
            # 7. Capaian Output dengan komponen ketepatan waktu (30%) dan capaian RO (70%)
            nilai_capaian_output = self._calculate_capaian_output(capaian_output, data_input.get('ketepatan_waktu', 100))
            
            # 8. Dispensasi SPM sebagai pengurang
            pengurangan_dispensasi = dispensasi_spm
            
            # PERHITUNGAN NILAI TOTAL
            bobot = self.regulation_info['bobot']
            nilai_total = (
                (nilai_revisi_dipa * bobot['revisi_dipa']) +
                (nilai_deviasi * bobot['deviasi_halaman_iii']) +
                (nilai_penyerapan * bobot['penyerapan_anggaran']) +
                (nilai_belanja_kontraktual * bobot['belanja_kontraktual']) +
                (nilai_penyelesaian_tagihan * bobot['penyelesaian_tagihan']) +
                (nilai_pengelolaan_up_tup * bobot['pengelolaan_up_tup']) +
                (nilai_capaian_output * bobot['capaian_output'])
            ) / 100
            
            # KONVERSI BOBOT
            total_bobot_efektif = sum([bobot[k] for k in bobot if k not in ['dispensasi_spm']])
            nilai_konversi = (nilai_total / total_bobot_efektif) * 100
            
            # NILAI AKHIR (dikurangi dispensasi SPM)
            nilai_akhir = max(0, nilai_konversi - pengurangan_dispensasi)
            
            # KATEGORI
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
                'gap_target': 95 - nilai_akhir,
                'components': {
                    'revisi_dipa': nilai_revisi_dipa,
                    'deviasi_halaman_iii': nilai_deviasi,
                    'penyerapan_anggaran': nilai_penyerapan,
                    'belanja_kontraktual': nilai_belanja_kontraktual,
                    'penyelesaian_tagihan': nilai_penyelesaian_tagihan,
                    'pengelolaan_up_tup': nilai_pengelolaan_up_tup,
                    'capaian_output': nilai_capaian_output,
                    'dispensasi_spm': pengurangan_dispensasi
                },
                'bobot': bobot,
                'improvement_areas': self._identify_improvement_areas({
                    'penyerapan': nilai_penyerapan,
                    'capaian_output': nilai_capaian_output,
                    'deviasi': nilai_deviasi,
                    'pengelolaan_up_tup': nilai_pengelolaan_up_tup
                })
            }
            
            return self.ikpa_data
            
        except Exception as e:
            return {"error": f"Error calculating IKPA 2024: {str(e)}"}
    
    def _calculate_belanja_kontraktual(self, belanja_kontraktual):
        """Calculate belanja kontraktual dengan distribusi akselerasi"""
        # Asumsi: jika tidak ada data, beri nilai optimal
        if belanja_kontraktual == 0:
            return 100.0
        return min(100, belanja_kontraktual)
    
    def _calculate_pengelolaan_up_tup(self, pengelolaan_up_tup, triwulan):
        """Calculate pengelolaan UP/TUP dengan target KKP"""
        target_kkp = self.regulation_info['target_kkp'].get(f'triwulan{triwulan}', 1.0)
        
        # Asumsi: jika penggunaan KKP memenuhi target, nilai optimal
        if pengelolaan_up_tup >= target_kkp:
            return 100.0
        else:
            # Proporsional terhadap target
            return (pengelolaan_up_tup / target_kkp) * 100
    
    def _calculate_capaian_output(self, capaian_output, ketepatan_waktu=100):
        """Calculate capaian output dengan komponen ketepatan waktu (30%) dan capaian RO (70%)"""
        nilai_ketepatan_waktu = min(100, ketepatan_waktu) * 0.3
        nilai_capaian_ro = min(100, capaian_output) * 0.7
        return nilai_ketepatan_waktu + nilai_capaian_ro
    
    def _identify_improvement_areas(self, components):
        """Identify areas for improvement"""
        areas = []
        
        if components['penyerapan'] < 80:
            areas.append({
                'area': 'Penyerapan Anggaran',
                'current': components['penyerapan'],
                'target': 95,
                'priority': 'High',
                'description': 'Perlu akselerasi penyerapan anggaran sesuai proyeksi triwulan'
            })
        
        if components['capaian_output'] < 85:
            areas.append({
                'area': 'Capaian Output',
                'current': components['capaian_output'],
                'target': 95,
                'priority': 'High',
                'description': 'Tingkatkan kualitas pelaporan capaian output dan ketepatan waktu'
            })
        
        if components['deviasi'] < 95:
            areas.append({
                'area': 'Deviasi RPD',
                'current': components['deviasi'],
                'target': 95,
                'priority': 'Medium',
                'description': 'Optimalkan perencanaan penarikan dana untuk minimalkan deviasi'
            })
        
        if components['pengelolaan_up_tup'] < 90:
            areas.append({
                'area': 'Pengelolaan UP/TUP',
                'current': components['pengelolaan_up_tup'],
                'target': 100,
                'priority': 'Medium',
                'description': 'Tingkatkan penggunaan KKP sesuai target triwulan'
            })
        
        return areas

class DataProcessor:
    def __init__(self):
        self.processed_data = {}
    
    def process_anggaran_excel(self, file_buffer):
        """Process Excel file for budget data"""
        try:
            df = pd.read_excel(file_buffer)
            
            # Normalize column names
            df = self._normalize_columns(df)
            
            # Calculate metrics
            total_alokasi = df['Jumlah'].sum() if 'Jumlah' in df.columns else 0
            total_realisasi = df['Realisasi'].sum() if 'Realisasi' in df.columns else 0
            penyerapan_persen = (total_realisasi / total_alokasi * 100) if total_alokasi > 0 else 0
            
            # Calculate deviasi RPD
            deviasi_rpd = self._calculate_deviation_rpd(df)
            
            # Group by bidang
            if 'Bidang' in df.columns:
                bidang_summary = df.groupby('Bidang').agg({
                    'Jumlah': 'sum',
                    'Realisasi': 'sum'
                }).reset_index()
                bidang_summary['Penyerapan_Persen'] = (bidang_summary['Realisasi'] / bidang_summary['Jumlah'] * 100)
            else:
                bidang_summary = pd.DataFrame()
            
            self.processed_data['anggaran'] = {
                'total_alokasi': total_alokasi,
                'total_realisasi': total_realisasi,
                'penyerapan_persen': penyerapan_persen,
                'deviasi_rpd': deviasi_rpd,
                'bidang_summary': bidang_summary,
                'raw_data': df
            }
            
            return self.processed_data['anggaran']
            
        except Exception as e:
            return {"error": f"Error processing anggaran Excel: {str(e)}"}
    
    def process_capaian_output_excel(self, file_buffer):
        """Process Excel file for capaian output data"""
        try:
            df = pd.read_excel(file_buffer)
            
            # Normalize column names for capaian output
            column_mapping = {
                'Kode Satker': 'kode_satker',
                'Nama Satker': 'nama_satker',
                'Program': 'program',
                'Kegiatan': 'kegiatan',
                'KRO/RO': 'kro_ro',
                'Uraian RO': 'uraian_ro',
                'Pagu': 'pagu',
                'Realisasi': 'realisasi_anggaran',
                'Target Output': 'target_output',
                'Satuan': 'satuan',
                'RVRO Bulan Ini': 'rvro_bulan_ini',
                'TPCRO Bulan Ini': 'tpcro_bulan_ini',
                'PCRO Bulan Ini': 'pcro_bulan_ini',
                'RVRO s.d Bulan Ini': 'rvro_kumulatif',
                'TPCRO s.d Bulan Ini': 'tpcro_kumulatif',
                'PCRO s.d Bulan Ini': 'pcro_kumulatif',
                'GAP': 'gap'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Calculate capaian output metrics
            if 'pcro_kumulatif' in df.columns:
                avg_capaian_output = df['pcro_kumulatif'].mean()
            else:
                avg_capaian_output = 0
            
            # Calculate ketepatan waktu (asumsi: semua tepat waktu)
            ketepatan_waktu = 100
            
            self.processed_data['capaian_output'] = {
                'avg_capaian_output': avg_capaian_output,
                'ketepatan_waktu': ketepatan_waktu,
                'total_ro': len(df),
                'raw_data': df
            }
            
            return self.processed_data['capaian_output']
            
        except Exception as e:
            return {"error": f"Error processing capaian output Excel: {str(e)}"}
    
    def process_indikator_pelaksanaan_excel(self, file_buffer):
        """Process Excel file for indikator pelaksanaan anggaran"""
        try:
            df = pd.read_excel(file_buffer)
            
            # Normalize column names
            column_mapping = {
                'Kode Satker': 'kode_satker',
                'Uraian Satker': 'nama_satker',
                'Revisi DIPA': 'revisi_dipa',
                'Deviasi Halaman III DIPA': 'deviasi_halaman_iii',
                'Penyerapan Anggaran': 'penyerapan_anggaran',
                'Belanja Kontraktual': 'belanja_kontraktual',
                'Penyelesaian Tagihan': 'penyelesaian_tagihan',
                'Pengelolaan UP/TUP': 'pengelolaan_up_tup',
                'Capaian Output': 'capaian_output',
                'Nilai Total': 'nilai_total',
                'Konversi Bobot': 'konversi_bobot',
                'Dispensasi SPM': 'dispensasi_spm',
                'Nilai Akhir': 'nilai_akhir'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Extract the first row as current data
            if len(df) > 0:
                current_data = df.iloc[0].to_dict()
            else:
                current_data = {}
            
            self.processed_data['indikator_pelaksanaan'] = {
                'current_data': current_data,
                'raw_data': df
            }
            
            return self.processed_data['indikator_pelaksanaan']
            
        except Exception as e:
            return {"error": f"Error processing indikator pelaksanaan Excel: {str(e)}"}
    
    def _normalize_columns(self, df):
        """Normalize column names"""
        column_mapping = {
            'kode': 'Kode',
            'KODE': 'Kode',
            'uraian': 'Uraian',
            'Uraian': 'Uraian',
            'jumlah': 'Jumlah',
            'Jumlah': 'Jumlah',
            'anggaran': 'Jumlah',
            'realisasi': 'Realisasi',
            'Realisasi': 'Realisasi',
            'sisa': 'Sisa',
            'Sisa': 'Sisa',
            'bidang': 'Bidang',
            'Bidang': 'Bidang',
            'triwulan': 'Triwulan',
            'Triwulan': 'Triwulan'
        }
        
        new_columns = []
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in column_mapping:
                new_columns.append(column_mapping[col_lower])
            else:
                new_columns.append(col)
        
        df.columns = new_columns
        return df
    
    def _calculate_deviation_rpd(self, df):
        """Calculate RPD deviation"""
        try:
            if 'Realisasi' in df.columns and 'Jumlah' in df.columns:
                planned = df['Jumlah'].sum()
                actual = df['Realisasi'].sum()
                if planned > 0:
                    deviation = abs((actual - planned) / planned) * 100
                    return min(deviation, 50)
            return 5.0
        except:
            return 5.0

class PDFExtractor:
    def __init__(self):
        self.extracted_data = {}
    
    def extract_ikpa_from_pdf(self, file_buffer):
        """Extract IKPA data from PDF"""
        try:
            text = ""
            with pdfplumber.open(file_buffer) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            
            # Extract nilai akhir
            patterns = [
                r'nilai\s*akhir\s*:?\s*(\d+[.,]?\d*)',
                r'ikpa.*?(\d+[.,]?\d*)',
                r'nilai\s*total.*?(\d+[.,]?\d*)'
            ]
            
            nilai_akhir = 0
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    nilai_akhir = float(matches[0].replace(',', '.'))
                    break
            
            # Extract components
            components = {}
            component_patterns = {
                'revisi_dipa': r'revisi\s*dipa.*?(\d+[.,]?\d*)',
                'deviasi_halaman_iii': r'deviasi.*halaman.*iii.*?(\d+[.,]?\d*)',
                'penyerapan_anggaran': r'penyerapan.*anggaran.*?(\d+[.,]?\d*)',
                'capaian_output': r'capaian.*output.*?(\d+[.,]?\d*)'
            }
            
            for key, pattern in component_patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    components[key] = float(matches[0].replace(',', '.'))
            
            self.extracted_data['ikpa'] = {
                'nilai_akhir': nilai_akhir,
                'components': components,
                'raw_text': text[:2000]
            }
            
            return self.extracted_data['ikpa']
            
        except Exception as e:
            return {"error": f"Error extracting IKPA from PDF: {str(e)}"}

class StrategicAdvisor:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
    
    def generate_recommendations(self, ikpa_data, budget_data, capaian_data, triwulan):
        """Generate strategic recommendations"""
        
        recommendations = {
            'penarikan_dana': self._generate_penarikan_dana_recommendation(budget_data, triwulan),
            'capaian_output': self._generate_capaian_output_recommendation(capaian_data, triwulan),
            'ikpa_improvement': self._generate_ikpa_improvement_recommendation(ikpa_data)
        }
        
        return recommendations
    
    def _generate_penarikan_dana_recommendation(self, budget_data, triwulan):
        """Generate recommendations for penarikan dana"""
        
        triwulan_plan = {
            1: {"target": "15-20%", "focus": "Kegiatan persiapan dan administrasi"},
            2: {"target": "35-40%", "focus": "Akselerasi kegiatan inti"},
            3: {"target": "30-35%", "focus": "Penyelesaian kegiatan"},
            4: {"target": "10-15%", "focus": "Evaluasi dan penyesuaian"}
        }
        
        current_triwulan = triwulan_plan.get(triwulan, {})
        
        return {
            'triwulan': triwulan,
            'target_penarikan': current_triwulan.get('target', 'N/A'),
            'fokus_kegiatan': current_triwulan.get('focus', 'N/A'),
            'strategi': [
                "Alokasikan dana berdasarkan prioritas kegiatan",
                "Monitor realisasi mingguan",
                "Koordinasi dengan PPK untuk percepatan penyerapan",
                "Antisipasi bottleneck dengan contingency plan"
            ]
        }
    
    def _generate_capaian_output_recommendation(self, capaian_data, triwulan):
        """Generate recommendations for capaian output"""
        
        timeline_targets = {
            1: {"target": "20%", "focus": "Penyelesaian assessment RO"},
            2: {"target": "45%", "focus": "Akselerasi pencapaian output"},
            3: {"target": "75%", "focus": "Penyelesaian major output"},
            4: {"target": "100%", "focus": "Finalisasi dan evaluasi"}
        }
        
        current_target = timeline_targets.get(triwulan, {})
        
        return {
            'triwulan': triwulan,
            'target_capaian': current_target.get('target', 'N/A'),
            'fokus_output': current_target.get('focus', 'N/A'),
            'strategi': [
                "Input data capaian output sebelum batas 5 hari kerja",
                "Tingkatkan kualitas dokumentasi output",
                "Koordinasi intensif dengan pengelola kegiatan",
                "Monitor status data di OMSPAN secara berkala"
            ]
        }
    
    def _generate_ikpa_improvement_recommendation(self, ikpa_data):
        """Generate recommendations for IKPA improvement"""
        
        improvement_areas = ikpa_data.get('improvement_areas', [])
        
        strategies = []
        for area in improvement_areas:
            if area['priority'] == 'High':
                strategies.append(f"FOKUS: {area['area']} - {area['description']}")
        
        return {
            'target_ikpa': "≥95 (Sangat Baik)",
            'strategi_prioritas': strategies,
            'timeline': [
                {"periode": "1-2 minggu", "aksi": "Quick wins - perbaikan administrasi"},
                {"periode": "3-4 minggu", "aksi": "Akselerasi penyerapan anggaran"},
                {"periode": "1-2 bulan", "aksi": "Peningkatan kualitas capaian output"}
            ]
        }

# Visualization functions
def create_ikpa_gauge(value, category):
    """Create IKPA gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta+goal",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"IKPA 2024 - {category}", 'font': {'size': 24}},
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
    components = ['Revisi DIPA', 'Deviasi Halaman III', 'Penyerapan Anggaran', 'Pengelolaan UP/TUP', 'Capaian Output']
    values = [
        ikpa_data['components']['revisi_dipa'],
        ikpa_data['components']['deviasi_halaman_iii'],
        ikpa_data['components']['penyerapan_anggaran'],
        ikpa_data['components']['pengelolaan_up_tup'],
        ikpa_data['components']['capaian_output']
    ]
    
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
    try:
        st.markdown('<h1 class="main-header">🏢 SMART BNNP DKI - Monitoring IKPA 2024</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666;">Sistem Monitoring Indikator Kinerja Pelaksanaan Anggaran</h3>', unsafe_allow_html=True)
        
        # Regulation Information
        with st.expander("📚 Informasi Regulasi PER-5/PB/2024"):
            st.markdown("""
            ### Reformulasi IKPA 2024
            - **Bobot Capaian Output**: 25% (tertinggi)
            - **Deviasi Halaman III**: Bobot naik dari 10% menjadi 15%
            - **Pengelolaan UP/TUP**: Penambahan penilaian KKP
            - **Dispensasi SPM**: Pengurang nilai IKPA
            
            ### Komponen Capaian Output:
            - **Ketepatan Waktu (30%)**: Pengiriman sebelum 5 hari kerja
            - **Capaian RO (70%)**: Berdasarkan realisasi vs target
            """)
        
        # Initialize processors
        data_processor = DataProcessor()
        pdf_extractor = PDFExtractor()
        ikpa_calculator = IKPA2024Calculator()
        strategic_advisor = StrategicAdvisor()
        
        # Sidebar configuration
        st.sidebar.header("⚙️ Konfigurasi Sistem")
        
        # Triwulan selection
        triwulan = st.sidebar.selectbox(
            "Pilih Triwulan",
            options=[1, 2, 3, 4],
            index=0,
            help="Pilih triwulan untuk analisis"
        )
        
        # File upload section
        st.sidebar.header("📁 Upload Data")
        
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("💰 Data Anggaran (Excel)")
        budget_file = st.sidebar.file_uploader(
            "Upload Realisasi Anggaran",
            type=['xlsx', 'xls'],
            key="budget_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📊 Capaian Output (Excel)")
        capaian_file = st.sidebar.file_uploader(
            "Upload Capaian Output",
            type=['xlsx', 'xls'],
            key="capaian_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📈 Indikator Pelaksanaan (Excel)")
        indikator_file = st.sidebar.file_uploader(
            "Upload Indikator Pelaksanaan",
            type=['xlsx', 'xls'],
            key="indikator_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📋 IKPA Sebelumnya (PDF)")
        ikpa_previous_file = st.sidebar.file_uploader(
            "Upload IKPA Previous",
            type=['pdf'],
            key="ikpa_previous_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Manual input fallback
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("✍️ Input Manual")
        
        manual_penyerapan = st.sidebar.number_input(
            "Penyerapan Anggaran (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            key="manual_penyerapan"
        )
        
        manual_capaian = st.sidebar.number_input(
            "Capaian Output (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            key="manual_capaian"
        )
        
        manual_deviasi = st.sidebar.number_input(
            "Deviasi RPD (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            key="manual_deviasi"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Process data
        budget_data = None
        capaian_data = None
        indikator_data = None
        ikpa_previous_data = None
        
        # Process budget file
        if budget_file:
            with st.spinner("Memproses data anggaran..."):
                budget_data = data_processor.process_anggaran_excel(budget_file)
        
        # Process capaian output file
        if capaian_file:
            with st.spinner("Memproses capaian output..."):
                capaian_data = data_processor.process_capaian_output_excel(capaian_file)
        
        # Process indikator file
        if indikator_file:
            with st.spinner("Memproses indikator pelaksanaan..."):
                indikator_data = data_processor.process_indikator_pelaksanaan_excel(indikator_file)
        
        # Process previous IKPA file
        if ikpa_previous_file:
            with st.spinner("Memproses IKPA sebelumnya..."):
                ikpa_previous_data = pdf_extractor.extract_ikpa_from_pdf(ikpa_previous_file)
        
        # Prepare data for IKPA calculation
        ikpa_input = {
            'penyerapan_anggaran': manual_penyerapan,
            'capaian_output': manual_capaian,
            'deviasi_halaman_iii': manual_deviasi,
            'triwulan': triwulan
        }
        
        # Use file data if available
        if budget_data and 'error' not in budget_data:
            ikpa_input['penyerapan_anggaran'] = budget_data['penyerapan_persen']
            ikpa_input['deviasi_halaman_iii'] = budget_data['deviasi_rpd']
        
        if capaian_data and 'error' not in capaian_data:
            ikpa_input['capaian_output'] = capaian_data['avg_capaian_output']
        
        if indikator_data and 'error' not in indikator_data:
            current_data = indikator_data['current_data']
            for key in ['revisi_dipa', 'deviasi_halaman_iii', 'penyerapan_anggaran', 'capaian_output']:
                if key in current_data:
                    ikpa_input[key] = current_data[key]
        
        # Calculate IKPA
        with st.spinner("Menghitung nilai IKPA 2024..."):
            ikpa_result = ikpa_calculator.calculate_ikpa_2024(ikpa_input)
        
        # Display results
        if ikpa_result and 'error' not in ikpa_result:
            # IKPA Dashboard
            st.header("📊 Dashboard IKPA 2024")
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                fig_gauge = create_ikpa_gauge(ikpa_result['nilai_akhir'], ikpa_result['kategori'])
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.metric("Nilai IKPA", f"{ikpa_result['nilai_akhir']:.2f}")
                st.metric("Kategori", ikpa_result['kategori'])
            
            with col3:
                st.metric("Gap Target", f"{ikpa_result['gap_target']:+.2f}")
                status = "✅ Optimal" if ikpa_result['nilai_akhir'] >= 95 else "⚠️ Perlu Perbaikan"
                st.metric("Status", status)
            
            with col4:
                if ikpa_previous_data and 'error' not in ikpa_previous_data:
                    change = ikpa_result['nilai_akhir'] - ikpa_previous_data['nilai_akhir']
                    st.metric("IKPA Sebelumnya", f"{ikpa_previous_data['nilai_akhir']:.2f}", f"{change:+.2f}")
                else:
                    st.metric("IKPA Sebelumnya", "N/A")
            
            # Component Analysis
            st.header("🔍 Analisis Komponen IKPA")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_components = create_component_chart(ikpa_result)
                st.plotly_chart(fig_components, use_container_width=True)
            
            with col2:
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
                st.dataframe(components_df, use_container_width=True, hide_index=True)
            
            # Data Summary
            st.header("📈 Summary Data")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Penyerapan Anggaran", f"{ikpa_input['penyerapan_anggaran']:.1f}%")
            
            with col2:
                st.metric("Capaian Output", f"{ikpa_input['capaian_output']:.1f}%")
            
            with col3:
                st.metric("Deviasi RPD", f"{ikpa_input['deviasi_halaman_iii']:.1f}%")
            
            with col4:
                st.metric("Triwulan", f"{triwulan}")
            
            # Strategic Recommendations
            st.header("🎯 Rekomendasi Strategis")
            
            recommendations = strategic_advisor.generate_recommendations(
                ikpa_result, budget_data, capaian_data, triwulan
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Pengaturan Penarikan Dana")
                penarikan = recommendations['penarikan_dana']
                st.metric("Target Triwulan", penarikan['target_penarikan'])
                st.write(f"**Fokus:** {penarikan['fokus_kegiatan']}")
                st.write("**Strategi:**")
                for strategy in penarikan['strategi']:
                    st.write(f"- {strategy}")
            
            with col2:
                st.subheader("📊 Target Capaian Output")
                capaian = recommendations['capaian_output']
                st.metric("Target Triwulan", capaian['target_capaian'])
                st.write(f"**Fokus:** {capaian['fokus_output']}")
                st.write("**Strategi:**")
                for strategy in capaian['strategi']:
                    st.write(f"- {strategy}")
            
            # Improvement Areas
            st.header("🚀 Area Perbaikan Prioritas")
            
            if ikpa_result.get('improvement_areas'):
                for area in ikpa_result['improvement_areas']:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.subheader(f"{area['area']} ({area['priority']} Priority)")
                            st.write(area['description'])
                        with col2:
                            st.metric("Current", f"{area['current']:.1f}%")
                        with col3:
                            st.metric("Target", f"{area['target']}%")
                        
                        progress = min(area['current'] / area['target'], 1.0)
                        st.progress(progress)
            
            # Raw Data Preview
            if st.checkbox("Tampilkan Preview Data"):
                st.header("📋 Preview Data")
                
                if budget_data and 'error' not in budget_data:
                    with st.expander("Data Anggaran"):
                        st.dataframe(budget_data['raw_data'].head(10), use_container_width=True)
                
                if capaian_data and 'error' not in capaian_data:
                    with st.expander("Data Capaian Output"):
                        st.dataframe(capaian_data['raw_data'].head(10), use_container_width=True)
        
        else:
            # Welcome screen
            st.header("🚀 Selamat Datang di Sistem Monitoring IKPA 2024")
            
            st.info("""
            ### 📋 Panduan Penggunaan:
            
            1. **Pilih Triwulan** yang akan dianalisis
            2. **Upload file Excel** data anggaran, capaian output, dan indikator pelaksanaan
            3. **Upload file PDF** IKPA sebelumnya (opsional)
            4. **Input manual** data yang tidak tersedia via file
            5. **Sistem akan menghitung** nilai IKPA 2024 otomatis
            6. **Dapatkan rekomendasi** strategis untuk peningkatan IKPA
            
            ### 🎯 Fitur Utama:
            - Perhitungan IKPA 2024 berdasarkan PER-5/PB/2024
            - Analisis komponen dengan bobot terbaru
            - Rekomendasi penarikan dana triwulan depan
            - Target waktu pencapaian output
            - Monitoring area perbaikan prioritas
            """)
            
            # Demo visualization
            st.header("📊 Contoh Dashboard IKPA")
            demo_input = {
                'penyerapan_anggaran': 75.0,
                'capaian_output': 80.0,
                'deviasi_halaman_iii': 5.0,
                'triwulan': 1
            }
            demo_ikpa = ikpa_calculator.calculate_ikpa_2024(demo_input)
            
            if demo_ikpa and 'error' not in demo_ikpa:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_demo = create_ikpa_gauge(demo_ikpa['nilai_akhir'], demo_ikpa['kategori'])
                    st.plotly_chart(fig_demo, use_container_width=True)
                
                with col2:
                    st.metric("Nilai IKPA Contoh", f"{demo_ikpa['nilai_akhir']:.2f}")
                    st.metric("Kategori", demo_ikpa['kategori'])
                    st.metric("Status", "Contoh Analisis")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
