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

# CSS Custom (tetap sama)
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
        padding: 1.rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .data-preview {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
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
        if belanja_kontraktual == 0:
            return 100.0
        return min(100, belanja_kontraktual)
    
    def _calculate_pengelolaan_up_tup(self, pengelolaan_up_tup, triwulan):
        """Calculate pengelolaan UP/TUP dengan target KKP"""
        target_kkp = self.regulation_info['target_kkp'].get(f'triwulan{triwulan}', 1.0)
        
        if pengelolaan_up_tup >= target_kkp:
            return 100.0
        else:
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
    
    def process_realisasi_anggaran(self, file_buffer):
        """
        Process Excel file for budget realization data
        """
        try:
            # Baca file Excel
            df = pd.read_excel(file_buffer)
            
            # Tampilkan preview data
            st.sidebar.info(f"📊 Kolom Realisasi Anggaran: {list(df.columns)}")
            
            # Normalize column names
            df = self._normalize_realisasi_columns(df)
            
            # Clean numeric data
            df = self._clean_realisasi_numeric_data(df)
            
            # Calculate metrics
            metrics = self._calculate_realisasi_metrics(df)
            
            # Calculate deviasi RPD
            deviasi_rpd = self._calculate_deviation_rpd(df)
            
            self.processed_data['realisasi_anggaran'] = {
                'metrics': metrics,
                'deviasi_rpd': deviasi_rpd,
                'raw_data': df,
                'columns_info': {
                    'all_columns': list(df.columns),
                    'pagu_columns': [col for col in df.columns if 'pagu' in col.lower()],
                    'realisasi_columns': [col for col in df.columns if 'realisasi' in col.lower()]
                }
            }
            
            return self.processed_data['realisasi_anggaran']
            
        except Exception as e:
            error_msg = f"Error processing realisasi anggaran: {str(e)}"
            st.sidebar.error(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def _normalize_realisasi_columns(self, df):
        """Normalize columns for realisasi anggaran data"""
        column_mapping = {
            'nama kegiatan': 'Nama Kegiatan',
            'kegiatan': 'Nama Kegiatan',
            'uraian': 'Nama Kegiatan',
            'program': 'Nama Kegiatan',
            'pagu belanja pegawai': 'Pagu Belanja Pegawai',
            'pagu pegawai': 'Pagu Belanja Pegawai',
            'belanja pegawai': 'Pagu Belanja Pegawai',
            'realisasi belanja pegawai': 'Realisasi Belanja Pegawai',
            'realisasi pegawai': 'Realisasi Belanja Pegawai',
            'pagu belanja barang': 'Pagu Belanja Barang',
            'pagu barang': 'Pagu Belanja Barang',
            'belanja barang': 'Pagu Belanja Barang',
            'realisasi belanja barang': 'Realisasi Belanja Barang',
            'realisasi barang': 'Realisasi Belanja Barang',
            'pagu belanja modal': 'Pagu Belanja Modal',
            'pagu modal': 'Pagu Belanja Modal',
            'belanja modal': 'Pagu Belanja Modal',
            'realisasi belanja modal': 'Realisasi Belanja Modal',
            'realisasi modal': 'Realisasi Belanja Modal',
            'total pagu': 'Total Pagu',
            'total anggaran': 'Total Pagu',
            'pagu total': 'Total Pagu',
            'total realisasi': 'Total Realisasi',
            'realisasi total': 'Total Realisasi'
        }
        
        new_columns = []
        for col in df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            
            mapped = False
            for pattern, standard_name in column_mapping.items():
                if pattern.lower() == col_lower or col_lower in pattern.lower():
                    new_columns.append(standard_name)
                    mapped = True
                    break
            
            if not mapped:
                new_columns.append(col_str)
        
        df.columns = new_columns
        return df
    
    def _clean_realisasi_numeric_data(self, df):
        """Clean numeric data for realisasi anggaran"""
        numeric_columns = [
            'Pagu Belanja Pegawai', 'Realisasi Belanja Pegawai',
            'Pagu Belanja Barang', 'Realisasi Belanja Barang',
            'Pagu Belanja Modal', 'Realisasi Belanja Modal',
            'Total Pagu', 'Total Realisasi'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(r'[^\d,-.]', '', regex=True)
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _calculate_realisasi_metrics(self, df):
        """Calculate metrics for realisasi anggaran"""
        metrics = {}
        
        if 'Total Pagu' in df.columns and 'Total Realisasi' in df.columns:
            metrics['total_pagu'] = df['Total Pagu'].sum()
            metrics['total_realisasi'] = df['Total Realisasi'].sum()
        else:
            pagu_columns = [col for col in df.columns if 'pagu' in col.lower() and 'realisasi' not in col.lower()]
            realisasi_columns = [col for col in df.columns if 'realisasi' in col.lower()]
            
            metrics['total_pagu'] = df[pagu_columns].sum().sum() if pagu_columns else 0
            metrics['total_realisasi'] = df[realisasi_columns].sum().sum() if realisasi_columns else 0
        
        if metrics['total_pagu'] > 0:
            metrics['penyerapan_persen'] = (metrics['total_realisasi'] / metrics['total_pagu']) * 100
        else:
            metrics['penyerapan_persen'] = 0
        
        if 'Pagu Belanja Pegawai' in df.columns and 'Realisasi Belanja Pegawai' in df.columns:
            metrics['pagu_pegawai'] = df['Pagu Belanja Pegawai'].sum()
            metrics['realisasi_pegawai'] = df['Realisasi Belanja Pegawai'].sum()
            metrics['penyerapan_pegawai'] = (metrics['realisasi_pegawai'] / metrics['pagu_pegawai'] * 100) if metrics['pagu_pegawai'] > 0 else 0
        
        if 'Pagu Belanja Barang' in df.columns and 'Realisasi Belanja Barang' in df.columns:
            metrics['pagu_barang'] = df['Pagu Belanja Barang'].sum()
            metrics['realisasi_barang'] = df['Realisasi Belanja Barang'].sum()
            metrics['penyerapan_barang'] = (metrics['realisasi_barang'] / metrics['pagu_barang'] * 100) if metrics['pagu_barang'] > 0 else 0
        
        if 'Pagu Belanja Modal' in df.columns and 'Realisasi Belanja Modal' in df.columns:
            metrics['pagu_modal'] = df['Pagu Belanja Modal'].sum()
            metrics['realisasi_modal'] = df['Realisasi Belanja Modal'].sum()
            metrics['penyerapan_modal'] = (metrics['realisasi_modal'] / metrics['pagu_modal'] * 100) if metrics['pagu_modal'] > 0 else 0
        
        return metrics

    # TAMBAHKAN METHOD UNTUK PROSES REVISI DIPA DI SINI
    def process_revisi_dipa(self, file_buffer):
        """
        Process Excel file for Revisi DIPA data
        """
        try:
            df = pd.read_excel(file_buffer)
            st.sidebar.info(f"📋 Kolom Revisi DIPA: {list(df.columns)}")
            
            df = self._normalize_revisi_columns(df)
            df = self._clean_revisi_numeric_data(df)
            metrics = self._calculate_revisi_metrics(df)
            
            self.processed_data['revisi_dipa'] = {
                'metrics': metrics,
                'raw_data': df,
                'deviasi_analysis': self._analyze_deviation_patterns(df)
            }
            
            return self.processed_data['revisi_dipa']
            
        except Exception as e:
            error_msg = f"Error processing revisi DIPA: {str(e)}"
            st.sidebar.error(f"❌ {error_msg}")
            return {"error": error_msg}

    def _normalize_revisi_columns(self, df):
        """Normalize columns for revisi DIPA data"""
        column_mapping = {
            'rencana belanja pegawai': 'Rencana Belanja Pegawai',
            'rencana pegawai': 'Rencana Belanja Pegawai',
            'plan pegawai': 'Rencana Belanja Pegawai',
            'rencana belanja barang': 'Rencana Belanja Barang',
            'rencana barang': 'Rencana Belanja Barang',
            'plan barang': 'Rencana Belanja Barang',
            'rencana belanja modal': 'Rencana Belanja Modal',
            'rencana modal': 'Rencana Belanja Modal',
            'plan modal': 'Rencana Belanja Modal',
            'penyerapan belanja pegawai': 'Penyerapan Belanja Pegawai',
            'realisasi pegawai': 'Penyerapan Belanja Pegawai',
            'penyerapan belanja barang': 'Penyerapan Belanja Barang',
            'realisasi barang': 'Penyerapan Belanja Barang',
            'penyerapan belanja modal': 'Penyerapan Belanja Modal',
            'realisasi modal': 'Penyerapan Belanja Modal',
            'deviasi belanja pegawai': 'Deviasi Belanja Pegawai',
            'deviasi pegawai': 'Deviasi Belanja Pegawai',
            'deviasi belanja barang': 'Deviasi Belanja Barang',
            'deviasi barang': 'Deviasi Belanja Barang',
            'deviasi belanja modal': 'Deviasi Belanja Modal',
            'deviasi modal': 'Deviasi Belanja Modal',
            '% deviasi belanja pegawai': '% Deviasi Belanja Pegawai',
            'persentase deviasi pegawai': '% Deviasi Belanja Pegawai',
            '% deviasi belanja barang': '% Deviasi Belanja Barang',
            'persentase deviasi barang': '% Deviasi Belanja Barang',
            '% deviasi belanja modal': '% Deviasi Belanja Modal',
            'persentase deviasi modal': '% Deviasi Belanja Modal',
            'nilai revisi halaman iii dipa': 'Nilai Revisi Halaman III DIPA',
            'revisi halaman iii': 'Nilai Revisi Halaman III DIPA',
            'nilai revisi': 'Nilai Revisi Halaman III DIPA'
        }
        
        new_columns = []
        for col in df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            
            mapped = False
            for pattern, standard_name in column_mapping.items():
                if pattern.lower() == col_lower or col_lower in pattern.lower():
                    new_columns.append(standard_name)
                    mapped = True
                    break
            
            if not mapped:
                new_columns.append(col_str)
        
        df.columns = new_columns
        return df

    def _clean_revisi_numeric_data(self, df):
        """Clean numeric data for revisi DIPA"""
        numeric_columns = [
            'Rencana Belanja Pegawai', 'Rencana Belanja Barang', 'Rencana Belanja Modal',
            'Penyerapan Belanja Pegawai', 'Penyerapan Belanja Barang', 'Penyerapan Belanja Modal',
            'Deviasi Belanja Pegawai', 'Deviasi Belanja Barang', 'Deviasi Belanja Modal',
            '% Deviasi Belanja Pegawai', '% Deviasi Belanja Barang', '% Deviasi Belanja Modal',
            'Nilai Revisi Halaman III DIPA'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = df[col].str.replace(r'[^\d,-.]', '', regex=True)
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df

    def _calculate_revisi_metrics(self, df):
        """Calculate metrics for revisi DIPA"""
        metrics = {}
        
        deviasi_columns = [col for col in df.columns if '% Deviasi' in col]
        if deviasi_columns:
            metrics['avg_deviasi_persen'] = df[deviasi_columns].mean().mean()
            metrics['max_deviasi_persen'] = df[deviasi_columns].max().max()
            metrics['min_deviasi_persen'] = df[deviasi_columns].min().min()
        else:
            metrics['avg_deviasi_persen'] = 0
            metrics['max_deviasi_persen'] = 0
            metrics['min_deviasi_persen'] = 0
        
        if 'Nilai Revisi Halaman III DIPA' in df.columns:
            metrics['nilai_revisi_halaman_iii'] = df['Nilai Revisi Halaman III DIPA'].mean()
        else:
            metrics['nilai_revisi_halaman_iii'] = 0
        
        rencana_columns = [col for col in df.columns if 'Rencana' in col]
        penyerapan_columns = [col for col in df.columns if 'Penyerapan' in col]
        
        if rencana_columns and penyerapan_columns:
            metrics['total_rencana'] = df[rencana_columns].sum().sum()
            metrics['total_penyerapan'] = df[penyerapan_columns].sum().sum()
            metrics['effisiensi_penyerapan'] = (metrics['total_penyerapan'] / metrics['total_rencana'] * 100) if metrics['total_rencana'] > 0 else 0
        
        return metrics

    def _analyze_deviation_patterns(self, df):
        """Analyze deviation patterns for recommendations"""
        analysis = {}
        
        deviasi_persen_columns = [col for col in df.columns if '% Deviasi' in col]
        
        if deviasi_persen_columns:
            avg_deviasi = df[deviasi_persen_columns].mean()
            analysis['highest_deviation_type'] = avg_deviasi.idxmax()
            analysis['highest_deviation_value'] = avg_deviasi.max()
            analysis['lowest_deviation_type'] = avg_deviasi.idxmin() 
            analysis['lowest_deviation_value'] = avg_deviasi.min()
        
        return analysis

    # Method lainnya tetap sama...
    def process_capaian_output(self, file_buffer):
        """Process capaian output data"""
        try:
            df = pd.read_excel(file_buffer)
            st.sidebar.info(f"📈 Kolom Capaian Output: {list(df.columns)}")
            
            df = self._normalize_capaian_columns(df)
            df = self._clean_capaian_numeric_data(df)
            metrics = self._calculate_capaian_metrics(df)
            
            self.processed_data['capaian_output'] = {
                'metrics': metrics,
                'raw_data': df,
                'columns_info': {
                    'all_columns': list(df.columns),
                    'total_ro': len(df),
                    'ro_with_data': len(df[df['PCRO sampai Bulan ini'] > 0])
                }
            }
            
            return self.processed_data['capaian_output']
            
        except Exception as e:
            error_msg = f"Error processing capaian output: {str(e)}"
            st.sidebar.error(f"❌ {error_msg}")
            return {"error": error_msg}

    def _normalize_capaian_columns(self, df):
        """Normalize columns for capaian output data"""
        column_mapping = {
            'kode kegiatan': 'Kode Kegiatan',
            'kode kro/ro': 'Kode KRO/RO',
            'kro/ro': 'Kode KRO/RO',
            'uraian ro': 'Uraian RO',
            'kegiatan': 'Uraian RO',
            'pagu anggaran ro': 'Pagu Anggaran RO',
            'pagu ro': 'Pagu Anggaran RO',
            'anggaran ro': 'Pagu Anggaran RO',
            'realisasi anggaran ro': 'Realisasi Anggaran RO',
            'realisasi ro': 'Realisasi Anggaran RO',
            'target output ro': 'Target Output RO',
            'target ro': 'Target Output RO',
            'volume target': 'Target Output RO',
            'satuan output': 'Satuan Output',
            'satuan': 'Satuan Output',
            'rvro bulan ini': 'RVRO Bulan ini',
            'realisasi volume ro bulan ini': 'RVRO Bulan ini',
            'tpcro bulan ini': 'TPCRO Bulan ini',
            'target progress capaian ro bulan ini': 'TPCRO Bulan ini',
            'pcro bulan ini': 'PCRO Bulan ini',
            'progress capaian ro bulan ini': 'PCRO Bulan ini',
            'rvro sampai bulan ini': 'RVRO sampai Bulan ini',
            'realisasi volume ro sampai bulan ini': 'RVRO sampai Bulan ini',
            'tpcro sampai bulan ini': 'TPCRO sampai Bulan ini',
            'target progress capaian ro sampai bulan ini': 'TPCRO sampai Bulan ini',
            'pcro sampai bulan ini': 'PCRO sampai Bulan ini',
            'progress capaian ro sampai bulan ini': 'PCRO sampai Bulan ini'
        }
        
        new_columns = []
        for col in df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            
            mapped = False
            for pattern, standard_name in column_mapping.items():
                if pattern.lower() == col_lower or col_lower in pattern.lower():
                    new_columns.append(standard_name)
                    mapped = True
                    break
            
            if not mapped:
                new_columns.append(col_str)
        
        df.columns = new_columns
        return df

    def _clean_capaian_numeric_data(self, df):
        """Clean numeric data for capaian output"""
        numeric_columns = [
            'Pagu Anggaran RO', 'Realisasi Anggaran RO', 'Target Output RO',
            'RVRO Bulan ini', 'TPCRO Bulan ini', 'PCRO Bulan ini',
            'RVRO sampai Bulan ini', 'TPCRO sampai Bulan ini', 'PCRO sampai Bulan ini'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                if 'TPCRO' in col or 'PCRO' in col:
                    df[col] = df[col].str.replace('%', '', regex=False)
                df[col] = df[col].str.replace(r'[^\d,-.]', '', regex=True)
                df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df

    def _calculate_capaian_metrics(self, df):
        """Calculate metrics for capaian output"""
        metrics = {}
        
        if 'PCRO sampai Bulan ini' in df.columns:
            metrics['avg_capaian_output'] = df['PCRO sampai Bulan ini'].mean()
            metrics['ro_above_80'] = len(df[df['PCRO sampai Bulan ini'] >= 80])
            metrics['ro_below_80'] = len(df[df['PCRO sampai Bulan ini'] < 80])
            metrics['total_ro'] = len(df)
            
            if metrics['total_ro'] > 0:
                metrics['persentase_ro_above_80'] = (metrics['ro_above_80'] / metrics['total_ro']) * 100
            else:
                metrics['persentase_ro_above_80'] = 0
            
            if 'PCRO Bulan ini' in df.columns:
                metrics['avg_capaian_bulan_ini'] = df['PCRO Bulan ini'].mean()
            else:
                metrics['avg_capaian_bulan_ini'] = 0
        else:
            metrics['avg_capaian_output'] = 0
            metrics['ro_above_80'] = 0
            metrics['ro_below_80'] = len(df)
            metrics['total_ro'] = len(df)
            metrics['persentase_ro_above_80'] = 0
            metrics['avg_capaian_bulan_ini'] = 0
        
        metrics['ketepatan_waktu'] = 100
        
        if 'Pagu Anggaran RO' in df.columns:
            metrics['total_pagu_ro'] = df['Pagu Anggaran RO'].sum()
        else:
            metrics['total_pagu_ro'] = 0
            
        if 'Realisasi Anggaran RO' in df.columns:
            metrics['total_realisasi_ro'] = df['Realisasi Anggaran RO'].sum()
        else:
            metrics['total_realisasi_ro'] = 0
        
        return metrics

    def _calculate_deviation_rpd(self, df):
        """Calculate RPD deviation"""
        try:
            pagu_columns = [col for col in df.columns if 'pagu' in col.lower() and 'realisasi' not in col.lower()]
            realisasi_columns = [col for col in df.columns if 'realisasi' in col.lower()]
            
            if pagu_columns and realisasi_columns:
                total_pagu = df[pagu_columns].sum().sum()
                total_realisasi = df[realisasi_columns].sum().sum()
                
                if total_pagu > 0:
                    deviation = abs((total_realisasi - total_pagu) / total_pagu) * 100
                    return min(deviation, 50)
            return 5.0
        except:
            return 5.0

class PDFExtractor:
    def __init__(self):
        self.extracted_data = {}
    
    def extract_ikpa_previous(self, file_buffer):
        """Extract previous IKPA data from PDF"""
        try:
            text = ""
            tables_data = []
            
            with pdfplumber.open(file_buffer) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:
                            tables_data.append(table)
            
            nilai_ikpa = self._extract_nilai_total_konversi_bobot(text, tables_data)
            
            if nilai_ikpa == 0:
                nilai_ikpa = self._extract_nilai_akhir(text)
            
            kategori = self._extract_kategori(text)
            
            self.extracted_data['ikpa_previous'] = {
                'nilai_akhir': nilai_ikpa,
                'kategori': kategori,
                'tables_found': len(tables_data)
            }
            
            st.sidebar.success(f"✅ Nilai IKPA sebelumnya: {nilai_ikpa:.2f}")
            
            return self.extracted_data['ikpa_previous']
            
        except Exception as e:
            error_msg = f"Error extracting IKPA from PDF: {str(e)}"
            st.sidebar.error(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def _extract_nilai_total_konversi_bobot(self, text, tables_data):
        """Extract nilai dari Nilai Total atau Konversi Bobot"""
        patterns = [
            r'nilai\s*total\s*:?\s*(\d+[.,]\d+)',
            r'konversi\s*bobot\s*:?\s*(\d+[.,]\d+)',
            r'nilai\s*total\s*\/\s*konversi\s*bobot\s*:?\s*(\d+[.,]\d+)',
            r'total\s*\/\s*konversi\s*:?\s*(\d+[.,]\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0].replace(',', '.'))
                    if 0 <= value <= 100:
                        return value
                except ValueError:
                    continue
        
        return 0
    
    def _extract_nilai_akhir(self, text):
        """Extract nilai akhir sebagai fallback"""
        patterns = [
            r'nilai\s*akhir\s*:?\s*(\d+[.,]\d+)',
            r'ikpa.*?(\d+[.,]\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0].replace(',', '.'))
                    if 0 <= value <= 100:
                        return value
                except ValueError:
                    continue
        return 0
    
    def _extract_kategori(self, text):
        """Extract category"""
        kategori_patterns = [
            (r'sangat\s+baik', 'Sangat Baik'),
            (r'baik', 'Baik'),
            (r'cukup', 'Cukup'),
            (r'kurang', 'Kurang')
        ]
        
        for pattern, kategori in kategori_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return kategori
        
        return "Tidak Diketahui"

class StrategicAdvisor:
    def __init__(self):
        pass
    
    def generate_recommendations(self, ikpa_data, realisasi_data, capaian_data, revisi_dipa_data, triwulan):
        """Generate comprehensive strategic recommendations"""
        
        recommendations = {
            'penarikan_dana': self._generate_penarikan_dana_recommendation(realisasi_data, triwulan),
            'capaian_output': self._generate_capaian_output_recommendation(capaian_data, triwulan),
            'revisi_dipa': self._generate_revisi_dipa_recommendation(revisi_dipa_data, triwulan),
            'rencana_penyerapan': self._generate_rencana_penyerapan_recommendation(realisasi_data, revisi_dipa_data, triwulan),
            'ikpa_improvement': self._generate_ikpa_improvement_recommendation(ikpa_data)
        }
        
        return recommendations
    
    def _generate_penarikan_dana_recommendation(self, realisasi_data, triwulan):
        """Generate recommendations for penarikan dana triwulan depan"""
        
        triwulan_plan = {
            1: {"target": "15-20%", "focus": "Kegiatan persiapan dan administrasi"},
            2: {"target": "35-40%", "focus": "Akselerasi kegiatan inti"},
            3: {"target": "30-35%", "focus": "Penyelesaian kegiatan"},
            4: {"target": "10-15%", "focus": "Evaluasi dan penyesuaian"}
        }
        
        current_triwulan = triwulan_plan.get(triwulan, {})
        
        penyerapan_info = ""
        if realisasi_data and 'metrics' in realisasi_data:
            metrics = realisasi_data['metrics']
            penyerapan_info = f" (Current: {metrics.get('penyerapan_persen', 0):.1f}%)"
        
        return {
            'triwulan': triwulan,
            'target_penarikan': current_triwulan.get('target', 'N/A') + penyerapan_info,
            'fokus_kegiatan': current_triwulan.get('focus', 'N/A'),
            'strategi': [
                f"Alokasikan {current_triwulan.get('target', 'N/A')} untuk triwulan {triwulan}",
                "Prioritaskan kegiatan dengan dampak IKPA tinggi",
                "Koordinasi intensif dengan PPK dan pengelola kegiatan",
                "Monitoring realisasi mingguan dan antisipasi bottleneck"
            ]
        }
    
    def _generate_capaian_output_recommendation(self, capaian_data, triwulan):
        """Generate recommendations for capaian output"""
        
        timeline_targets = {
            1: {"target": "20-25%", "focus": "Penyelesaian assessment RO dan baseline"},
            2: {"target": "45-50%", "focus": "Akselerasi pencapaian output utama"},
            3: {"target": "75-80%", "focus": "Penyelesaian major output dan quality check"},
            4: {"target": "95-100%", "focus": "Finalisasi, evaluasi, dan pelaporan"}
        }
        
        current_target = timeline_targets.get(triwulan, {})
        
        capaian_info = ""
        if capaian_data and 'metrics' in capaian_data:
            metrics = capaian_data['metrics']
            capaian_info = f" (Current: {metrics.get('avg_capaian_output', 0):.1f}%)"
        
        return {
            'triwulan': triwulan,
            'target_capaian': current_target.get('target', 'N/A') + capaian_info,
            'fokus_output': current_target.get('focus', 'N/A'),
            'strategi': [
                "Input data capaian output maksimal 5 hari kerja setelah bulan berakhir",
                "Tingkatkan kualitas dokumentasi realisasi output",
                "Koordinasi dengan pengelola kegiatan untuk validasi capaian",
                "Monitor status data di OMSPAN secara berkala"
            ]
        }

    def _generate_revisi_dipa_recommendation(self, revisi_data, triwulan):
        """Generate recommendations for Revisi DIPA optimization"""
        
        if not revisi_data or 'metrics' not in revisi_data:
            return {
                'status': 'Data tidak tersedia',
                'rekomendasi': [
                    "Lakukan pemantauan berkala rencana vs realisasi belanja",
                    "Minimalkan revisi DIPA - maksimal 1x revisi per triwulan",
                    "Optimalkan perencanaan awal untuk hindari deviasi >5%"
                ]
            }
        
        metrics = revisi_data['metrics']
        analysis = revisi_data.get('deviasi_analysis', {})
        
        rekomendasi = []
        
        if metrics.get('avg_deviasi_persen', 0) > 10:
            rekomendasi.append(f"🚨 **PRIORITAS TINGGI**: Turunkan deviasi rata-rata dari {metrics['avg_deviasi_persen']:.1f}% menjadi <5%")
            rekomendasi.append("• Perbaiki akurasi perencanaan bulanan")
            rekomendasi.append("• Lakukan forecasting yang lebih realistis")
            rekomendasi.append("• Koordinasi intensif dengan pengguna anggaran")
        
        if metrics.get('max_deviasi_persen', 0) > 20:
            rekomendasi.append(f"⚠️ **Koreksi Segera**: Deviasi maksimal {metrics['max_deviasi_persen']:.1f}% perlu penanganan khusus")
        
        if 'highest_deviation_type' in analysis:
            rekomendasi.append(f"🔍 **Fokus Perbaikan**: {analysis['highest_deviation_type']} (deviasi: {analysis['highest_deviation_value']:.1f}%)")
        
        return {
            'avg_deviasi': f"{metrics.get('avg_deviasi_persen', 0):.1f}%",
            'target_deviasi': "<5%",
            'revisi_halaman_iii': f"{metrics.get('nilai_revisi_halaman_iii', 0):.1f}",
            'target_revisi': "0",
            'rekomendasi': rekomendasi if rekomendasi else [
                "✅ Pertahankan kinerja optimal deviasi dan revisi",
                "📊 Monitoring rutin perbandingan rencana vs realisasi",
                "🔒 Standardisasi proses perencanaan anggaran"
            ]
        }

    def _generate_rencana_penyerapan_recommendation(self, realisasi_data, revisi_data, triwulan):
        """Generate detailed penyerapan recommendations for next period"""
        
        triwulan_projection = {
            1: {"target": "20-25%", "bulan": ["Jan: 5-7%", "Feb: 7-9%", "Mar: 8-9%"]},
            2: {"target": "45-50%", "bulan": ["Apr: 10-12%", "Mei: 15-17%", "Jun: 20-21%"]},
            3: {"target": "75-80%", "bulan": ["Jul: 15-17%", "Agu: 25-27%", "Sep: 35-36%"]},
            4: {"target": "95-100%", "bulan": ["Okt: 15-17%", "Nov: 35-37%", "Des: 45-46%"]}
        }
        
        current_projection = triwulan_projection.get(triwulan, {})
        
        current_penyerapan = 0
        if realisasi_data and 'metrics' in realisasi_data:
            current_penyerapan = realisasi_data['metrics'].get('penyerapan_persen', 0)
        
        rekomendasi_penyerapan = [
            f"🎯 **Target Triwulan {triwulan}**: {current_projection.get('target', 'N/A')}",
            f"📊 **Realitas Saat Ini**: {current_penyerapan:.1f}%"
        ]
        
        gap_analysis = []
        if current_penyerapan < 80:
            gap_analysis.extend([
                "🚀 **Akselerasi Diperlukan**:",
                "• Front-loading penyerapan di awal bulan",
                "• Prioritaskan kegiatan dengan SPM mudah",
                "• Daily monitoring oleh pengelola kegiatan"
            ])
        else:
            gap_analysis.extend([
                "✅ **Kinerja Optimal**:",
                "• Pertahankan momentum penyerapan",
                "• Antisipasi bottleneck di akhir periode",
                "• Optimalkan belanja modal yang tertinggal"
            ])
        
        return {
            'target_triwulan': current_projection.get('target', 'N/A'),
            'current_performance': f"{current_penyerapan:.1f}%",
            'bulanan_breakdown': current_projection.get('bulan', []),
            'strategic_actions': gap_analysis,
            'compliance_rules': [
                "📅 **Wajib Patuh**: Input realisasi max 3 hari kerja setelah bulan berakhir",
                "📝 **Dokumentasi**: Lengkapi supporting documents sebelum SPP",
                "🔍 **Review**: Validasi oleh PPK sebelum pengajuan SPM",
                "📊 **Reporting**: Laporan harian ke Kuasa BUN"
            ]
        }
    
    def _generate_ikpa_improvement_recommendation(self, ikpa_data):
        """Generate recommendations for IKPA improvement"""
        
        improvement_areas = ikpa_data.get('improvement_areas', [])
        
        strategies = []
        for area in improvement_areas:
            if area['priority'] == 'High':
                strategies.append(f"🎯 {area['area']}: {area['description']}")
        
        if not strategies:
            strategies = [
                "🎯 Pertahankan kinerja optimal semua komponen IKPA",
                "📊 Monitoring berkala untuk early detection issues",
                "🔄 Continuous improvement proses kerja"
            ]
        
        return {
            'target_ikpa': "≥95 (Sangat Baik)",
            'current_ikpa': f"{ikpa_data['nilai_akhir']:.2f} ({ikpa_data['kategori']})",
            'gap': f"{ikpa_data['gap_target']:.2f} poin",
            'strategi_prioritas': strategies,
            'timeline': [
                {"periode": "1-2 minggu", "aksi": "Quick wins - optimasi administrasi"},
                {"periode": "3-4 minggu", "aksi": "Akselerasi penyerapan anggaran"},
                {"periode": "1-2 bulan", "aksi": "Peningkatan kualitas capaian output"}
            ]
        }

# Visualization functions
def create_ikpa_gauge(value, category):
    """Create IKPA gauge chart"""
    try:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"IKPA 2024 - {category}", 'font': {'size': 24}},
            delta={'reference': 95, 'increasing': {'color': "green"}},
            gauge={
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
    except Exception as e:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"IKPA - {category}"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        fig.update_layout(height=300)
        return fig

def create_component_chart(ikpa_data):
    """Create component breakdown chart"""
    try:
        components = ['Revisi DIPA', 'Deviasi RPD', 'Penyerapan Anggaran', 'Pengelolaan UP/TUP', 'Capaian Output']
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
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title="Error creating chart", height=400)
        return fig

def create_penyerapan_chart(realisasi_data):
    """Create chart for penyerapan anggaran by jenis belanja"""
    try:
        if not realisasi_data or 'metrics' not in realisasi_data:
            return go.Figure()
        
        metrics = realisasi_data['metrics']
        
        categories = ['Belanja Pegawai', 'Belanja Barang', 'Belanja Modal', 'Total']
        values = [
            metrics.get('penyerapan_pegawai', 0),
            metrics.get('penyerapan_barang', 0),
            metrics.get('penyerapan_modal', 0),
            metrics.get('penyerapan_persen', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(name='Penyerapan', x=categories, y=values, marker_color='green')
        ])
        
        fig.update_layout(
            title='Persentase Penyerapan per Jenis Belanja',
            yaxis_title='Persentase (%)',
            height=300
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title="Error creating chart", height=300)
        return fig

def create_revisi_dipa_chart(revisi_data):
    """Create visualization for revisi DIPA data"""
    try:
        if not revisi_data or 'metrics' not in revisi_data:
            return go.Figure()
        
        metrics = revisi_data['metrics']
        
        categories = ['Pegawai', 'Barang', 'Modal']
        deviasi_values = [
            metrics.get('avg_deviasi_pegawai', 0),
            metrics.get('avg_deviasi_barang', 0), 
            metrics.get('avg_deviasi_modal', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(name='% Deviasi Rata-rata', x=categories, y=deviasi_values, marker_color='red'),
            go.Bar(name='Target (<5%)', x=categories, y=[5, 5, 5], marker_color='green', opacity=0.3)
        ])
        
        fig.update_layout(
            title='Rata-rata % Deviasi per Jenis Belanja',
            yaxis_title='Persentase Deviasi (%)',
            barmode='overlay',
            height=300
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title="Error creating revisi chart", height=300)
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
            
            ### Periode Pelaporan:
            - **Bulan Ini**: Capaian bulan terakhir (bulan laporan)
            - **Sampai Bulan Ini**: Capaian kumulatif hingga bulan laporan
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
            help="Pilih triwulan untuk analisis dan rekomendasi"
        )
        
        # File upload section - TAMBAHKAN UPLOAD REVISI DIPA DI SINI
        st.sidebar.header("📁 Upload Data")
        
        # 1. Data Realisasi Anggaran
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("💰 Data Realisasi Anggaran")
        st.sidebar.write("**Format:** Nama Kegiatan, Pagu/Realisasi Belanja Pegawai/Barang/Modal, Total")
        realisasi_file = st.sidebar.file_uploader(
            "Upload Realisasi Anggaran (Excel)",
            type=['xlsx', 'xls'],
            key="realisasi_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # 2. Data Capaian Output
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📊 Data Capaian Output")
        st.sidebar.write("**Format:** Kertas Kerja Capaian RO dengan PCRO bulan ini & kumulatif")
        capaian_file = st.sidebar.file_uploader(
            "Upload Capaian Output (Excel)", 
            type=['xlsx', 'xls'],
            key="capaian_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # 3. Data Revisi DIPA - INI YANG DITAMBAHKAN
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📋 Data Revisi DIPA")
        st.sidebar.write("**Format:** Rencana Belanja, Penyerapan, Deviasi, % Deviasi per Jenis Belanja")
        revisi_dipa_file = st.sidebar.file_uploader(
            "Upload Data Revisi DIPA (Excel)",
            type=['xlsx', 'xls'],
            key="revisi_dipa_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # 4. Data IKPA Sebelumnya
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📈 IKPA Sebelumnya")
        st.sidebar.write("**Format:** PDF laporan IKPA periode sebelumnya")
        ikpa_previous_file = st.sidebar.file_uploader(
            "Upload IKPA Sebelumnya (PDF)",
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
        
        # Process data - INISIALISASI VARIABEL REVISI_DIPA_DATA DI SINI
        realisasi_data = None
        capaian_data = None
        revisi_dipa_data = None  # INI YANG DITAMBAHKAN
        ikpa_previous_data = None
        
        # Process realisasi anggaran
        if realisasi_file:
            with st.spinner("🔄 Memproses data realisasi anggaran..."):
                realisasi_data = data_processor.process_realisasi_anggaran(realisasi_file)
                if "error" not in realisasi_data:
                    st.sidebar.success("✅ Data realisasi anggaran berhasil diproses")
                else:
                    st.sidebar.error(f"❌ {realisasi_data['error']}")
        
        # Process capaian output
        if capaian_file:
            with st.spinner("🔄 Memproses data capaian output..."):
                capaian_data = data_processor.process_capaian_output(capaian_file)
                if "error" not in capaian_data:
                    st.sidebar.success("✅ Data capaian output berhasil diproses")
                else:
                    st.sidebar.error(f"❌ {capaian_data['error']}")
        
        # Process revisi DIPA - INI YANG DITAMBAHKAN
        if revisi_dipa_file:
            with st.spinner("🔄 Memproses data revisi DIPA..."):
                revisi_dipa_data = data_processor.process_revisi_dipa(revisi_dipa_file)
                if "error" not in revisi_dipa_data:
                    st.sidebar.success("✅ Data revisi DIPA berhasil diproses")
                else:
                    st.sidebar.error(f"❌ {revisi_dipa_data['error']}")
        
        # Process previous IKPA
        if ikpa_previous_file:
            with st.spinner("🔄 Memproses IKPA sebelumnya..."):
                ikpa_previous_data = pdf_extractor.extract_ikpa_previous(ikpa_previous_file)
                if "error" not in ikpa_previous_data:
                    st.sidebar.success("✅ Data IKPA sebelumnya berhasil diproses")
                else:
                    st.sidebar.error(f"❌ {ikpa_previous_data['error']}")
        
        # Prepare data for IKPA calculation
        ikpa_input = {
            'penyerapan_anggaran': manual_penyerapan,
            'capaian_output': manual_capaian,
            'deviasi_halaman_iii': manual_deviasi,
            'triwulan': triwulan
        }
        
        # Use file data if available
        if realisasi_data and 'error' not in realisasi_data:
            ikpa_input['penyerapan_anggaran'] = realisasi_data['metrics']['penyerapan_persen']
            ikpa_input['deviasi_halaman_iii'] = realisasi_data['deviasi_rpd']
            st.sidebar.info(f"📊 Penyerapan dari file: {realisasi_data['metrics']['penyerapan_persen']:.1f}%")
        
        if capaian_data and 'error' not in capaian_data:
            ikpa_input['capaian_output'] = capaian_data['metrics']['avg_capaian_output']
            st.sidebar.info(f"📊 Capaian output dari file: {capaian_data['metrics']['avg_capaian_output']:.1f}%")
        
        # Use revisi DIPA data if available - INI YANG DITAMBAHKAN
        if revisi_dipa_data and 'error' not in revisi_dipa_data:
            ikpa_input['revisi_dipa'] = revisi_dipa_data['metrics'].get('nilai_revisi_halaman_iii', 0)
            st.sidebar.info(f"📋 Revisi DIPA dari file: {revisi_dipa_data['metrics'].get('nilai_revisi_halaman_iii', 0):.1f}")
        
        # Calculate IKPA
        ikpa_result = None
        if any([realisasi_data, capaian_data, revisi_dipa_data]) or any([manual_penyerapan > 0, manual_capaian > 0]):
            with st.spinner("🔄 Menghitung nilai IKPA 2024..."):
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
                            'Nilai': f"{ikpa_result['components'][comp]:.2f}%",
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
            
            # Tampilkan data Revisi DIPA jika ada - INI YANG DITAMBAHKAN
            if revisi_dipa_data and 'error' not in revisi_dipa_data:
                st.header("📋 Analisis Revisi DIPA & Deviasi")
                
                metrics = revisi_dipa_data['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rata-rata % Deviasi", f"{metrics.get('avg_deviasi_persen', 0):.1f}%")
                with col2:
                    st.metric("Deviasi Maksimal", f"{metrics.get('max_deviasi_persen', 0):.1f}%")
                with col3:
                    st.metric("Nilai Revisi Halaman III", f"{metrics.get('nilai_revisi_halaman_iii', 0):.1f}")
                with col4:
                    st.metric("Efisiensi Penyerapan", f"{metrics.get('effisiensi_penyerapan', 0):.1f}%")
                
                fig_revisi = create_revisi_dipa_chart(revisi_dipa_data)
                st.plotly_chart(fig_revisi, use_container_width=True)
            
            # Detail Data Section
            if realisasi_data and 'error' not in realisasi_data:
                st.header("💰 Detail Realisasi Anggaran")
                
                metrics = realisasi_data['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pagu", f"Rp {metrics['total_pagu']:,.0f}")
                with col2:
                    st.metric("Total Realisasi", f"Rp {metrics['total_realisasi']:,.0f}")
                with col3:
                    st.metric("Penyerapan", f"{metrics['penyerapan_persen']:.1f}%")
                with col4:
                    st.metric("Deviasi RPD", f"{realisasi_data['deviasi_rpd']:.1f}%")
                
                fig_penyerapan = create_penyerapan_chart(realisasi_data)
                st.plotly_chart(fig_penyerapan, use_container_width=True)
            
            if capaian_data and 'error' not in capaian_data:
                st.header("📊 Detail Capaian Output")
                
                metrics = capaian_data['metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rata-rata Capaian", f"{metrics['avg_capaian_output']:.1f}%")
                with col2:
                    st.metric("RO ≥80%", f"{metrics['ro_above_80']}/{metrics['total_ro']}")
                with col3:
                    st.metric("RO <80%", f"{metrics['ro_below_80']}/{metrics['total_ro']}")
                with col4:
                    st.metric("Ketepatan Waktu", f"{metrics['ketepatan_waktu']:.1f}%")
            
            # Strategic Recommendations - UPDATE PEMANGGILAN DENGAN REVISI_DIPA_DATA
            st.header("🎯 Rekomendasi Strategis")
            
            recommendations = strategic_advisor.generate_recommendations(
                ikpa_result, realisasi_data, capaian_data, revisi_dipa_data, triwulan
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Pengaturan Penarikan Dana Triwulan Depan")
                penarikan = recommendations['penarikan_dana']
                st.metric(f"Target Triwulan {triwulan}", penarikan['target_penarikan'])
                st.write(f"**Fokus Kegiatan:** {penarikan['fokus_kegiatan']}")
                st.write("**Strategi Implementasi:**")
                for strategy in penarikan['strategi']:
                    st.write(f"• {strategy}")
            
            with col2:
                st.subheader("📊 Target Waktu Pencapaian Output")
                capaian = recommendations['capaian_output']
                st.metric(f"Target Triwulan {triwulan}", capaian['target_capaian'])
                st.write(f"**Fokus Output:** {capaian['fokus_output']}")
                st.write("**Strategi Implementasi:**")
                for strategy in capaian['strategi']:
                    st.write(f"• {strategy}")
            
            # Tampilkan rekomendasi revisi DIPA - INI YANG DITAMBAHKAN
            st.header("🔄 Rekomendasi Optimasi Revisi DIPA")
            revisi_rec = recommendations['revisi_dipa']
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Deviasi Saat Ini", revisi_rec['avg_deviasi'])
                st.metric("Target Deviasi", revisi_rec['target_deviasi'])
            with col2:
                st.metric("Revisi Halaman III", revisi_rec['revisi_halaman_iii'])
                st.metric("Target Revisi", revisi_rec['target_revisi'])

            st.write("**Rekomendasi Implementasi:**")
            for rec in revisi_rec['rekomendasi']:
                st.write(f"• {rec}")

            # Tampilkan rekomendasi rencana penyerapan - INI YANG DITAMBAHKAN
            st.header("📅 Rencana Penyerapan Periode Berikutnya")
            penyerapan_rec = recommendations['rencana_penyerapan']
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Target Triwulan {triwulan}", penyerapan_rec['target_triwulan'])
                st.metric("Kinerja Saat Ini", penyerapan_rec['current_performance'])
                
                st.write("**Rencana Bulanan:**")
                for bulan in penyerapan_rec['bulanan_breakdown']:
                    st.write(f"• {bulan}")

            with col2:
                st.write("**Strategi Akselerasi:**")
                for action in penyerapan_rec['strategic_actions']:
                    st.write(f"• {action}")
                
                st.write("**Aturan Wajib Patuh:**")
                for rule in penyerapan_rec['compliance_rules']:
                    st.write(f"• {rule}")
            
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
            else:
                st.success("✅ Semua komponen IKPA dalam kondisi optimal!")
            
            # IKPA Improvement Timeline
            st.header("📅 Timeline Peningkatan IKPA")
            improvement = recommendations['ikpa_improvement']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Target IKPA", improvement['target_ikpa'])
                st.metric("Current IKPA", improvement['current_ikpa'])
                st.metric("Gap", improvement['gap'])
            
            with col2:
                st.write("**Strategi Prioritas:**")
                for strategy in improvement['strategi_prioritas']:
                    st.write(f"• {strategy}")
            
            st.write("**Timeline Implementasi:**")
            for timeline in improvement['timeline']:
                st.write(f"• **{timeline['periode']}**: {timeline['aksi']}")
            
            # Data Preview
            if st.checkbox("📋 Tampilkan Preview Data"):
                st.header("📋 Preview Data")
                
                if realisasi_data and 'error' not in realisasi_data:
                    with st.expander("Data Realisasi Anggaran"):
                        st.write(f"**Total Records:** {len(realisasi_data['raw_data'])}")
                        st.dataframe(realisasi_data['raw_data'].head(10), use_container_width=True)
                
                if capaian_data and 'error' not in capaian_data:
                    with st.expander("Data Capaian Output"):
                        st.write(f"**Total RO:** {capaian_data['metrics']['total_ro']}")
                        st.dataframe(capaian_data['raw_data'].head(10), use_container_width=True)
                
                # Tampilkan preview revisi DIPA - INI YANG DITAMBAHKAN
                if revisi_dipa_data and 'error' not in revisi_dipa_data:
                    with st.expander("Data Revisi DIPA"):
                        st.write(f"**Total Records:** {len(revisi_dipa_data['raw_data'])}")
                        st.dataframe(revisi_dipa_data['raw_data'].head(10), use_container_width=True)
        
        else:
            # Welcome screen
            st.header("🚀 Selamat Datang di Sistem Monitoring IKPA 2024")
            
            st.info("""
            ### 📋 Panduan Penggunaan:
            
            **1. Upload Data Realisasi Anggaran (Excel)**
            - Format: Nama Kegiatan, Pagu/Realisasi Belanja Pegawai/Barang/Modal, Total
            
            **2. Upload Data Capaian Output (Excel)** 
            - Format: Kertas Kerja Capaian RO dengan PCRO bulan ini & kumulatif
            
            **3. Upload Data Revisi DIPA (Excel) - BARU**
            - Format: Rencana Belanja, Penyerapan, Deviasi, % Deviasi per Jenis Belanja
            
            **4. Upload IKPA Sebelumnya (PDF) - Opsional**
            - Format: PDF laporan IKPA periode sebelumnya
            
            **5. Input Manual - Opsional**
            - Digunakan jika tidak ada file yang diupload
            
            ### 🎯 Fitur Utama:
            - Perhitungan IKPA 2024 berdasarkan PER-5/PB/2024
            - Analisis komponen dengan bobot terbaru
            - Rekomendasi penarikan dana triwulan depan
            - Target waktu pencapaian output
            - Monitoring area perbaikan prioritas
            - Analisis Revisi DIPA & Deviasi - FITUR BARU
            """)
            
            # Demo visualization
            if st.button("Lihat Contoh Dashboard"):
                st.header("📊 Contoh Dashboard IKPA")
                demo_input = {
                    'penyerapan_anggaran': 78.5,
                    'capaian_output': 82.3,
                    'deviasi_halaman_iii': 4.2,
                    'triwulan': triwulan
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
