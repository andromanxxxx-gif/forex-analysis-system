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
    .debug-info {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
        font-size: 0.9rem;
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
            # Ekstrak data input dengan default values yang lebih baik
            revisi_dipa = data_input.get('revisi_dipa', 0.0)
            deviasi_halaman_iii = data_input.get('deviasi_halaman_iii', 0.0)
            penyerapan_anggaran = data_input.get('penyerapan_anggaran', 0.0)
            belanja_kontraktual = data_input.get('belanja_kontraktual', 0.0)
            penyelesaian_tagihan = data_input.get('penyelesaian_tagihan', 0.0)
            pengelolaan_up_tup = data_input.get('pengelolaan_up_tup', 100.0)
            capaian_output = data_input.get('capaian_output', 0.0)
            dispensasi_spm = data_input.get('dispensasi_spm', 0.0)
            triwulan = data_input.get('triwulan', 1)
            
            # Debug info - tampilkan input yang diterima
            debug_info = f"""
            🔍 INPUT UNTUK PERHITUNGAN IKPA:
            • Revisi DIPA: {revisi_dipa}
            • Deviasi Halaman III: {deviasi_halaman_iii}%
            • Penyerapan: {penyerapan_anggaran}%
            • Capaian Output: {capaian_output}%
            • Triwulan: {triwulan}
            """
            st.sidebar.markdown(f'<div class="debug-info">{debug_info}</div>', unsafe_allow_html=True)
            
            # Normalisasi sesuai formula 2024
            # 1. Revisi DIPA - semakin sedikit revisi semakin baik
            nilai_revisi_dipa = max(0, 100 - (revisi_dipa * 10))
            
            # 2. Deviasi Halaman III - semakin kecil deviasi semakin baik
            nilai_deviasi = max(0, 100 - min(deviasi_halaman_iii, 10) * 10)
            
            # 3. Penyerapan Anggaran - langsung digunakan sebagai persentase
            nilai_penyerapan = min(100, penyerapan_anggaran)
            
            # 4. Belanja Kontraktual dengan distribusi akselerasi
            nilai_belanja_kontraktual = self._calculate_belanja_kontraktual(belanja_kontraktual)
            
            # 5. Penyelesaian Tagihan
            nilai_penyelesaian_tagihan = min(100, penyelesaian_tagihan)
            
            # 6. Pengelolaan UP/TUP dengan KKP
            nilai_pengelolaan_up_tup = self._calculate_pengelolaan_up_tup(pengelolaan_up_tup, triwulan)
            
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
            
            # Tampilkan debug info hasil perhitungan
            result_debug = f"""
            📊 HASIL PERHITUNGAN IKPA:
            • Nilai Akhir: {nilai_akhir:.2f} ({kategori})
            • Komponen Revisi DIPA: {nilai_revisi_dipa:.2f}%
            • Komponen Deviasi: {nilai_deviasi:.2f}%
            • Komponen Penyerapan: {nilai_penyerapan:.2f}%
            • Komponen Capaian: {nilai_capaian_output:.2f}%
            """
            st.sidebar.markdown(f'<div class="debug-info">{result_debug}</div>', unsafe_allow_html=True)
            
            return self.ikpa_data
            
        except Exception as e:
            st.error(f"❌ Error dalam perhitungan IKPA: {str(e)}")
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

    def process_revisi_dipa(self, file_buffer):
        """
        Process Excel file for Revisi DIPA data - VERSI YANG DIPERBAIKI
        """
        try:
            df = pd.read_excel(file_buffer)
            
            # DEBUG: Tampilkan semua kolom yang ada
            st.sidebar.info(f"📋 Kolom Revisi DIPA yang terbaca: {list(df.columns)}")
            
            # Jika dataframe kosong, return error
            if df.empty:
                return {"error": "File Revisi DIPA kosong atau tidak berisi data"}
            
            # Normalize column names
            df = self._normalize_revisi_columns(df)
            
            # DEBUG: Tampilkan kolom setelah normalisasi
            st.sidebar.info(f"📋 Kolom setelah normalisasi: {list(df.columns)}")
            
            # Clean numeric data
            df = self._clean_revisi_numeric_data(df)
            
            # Calculate metrics
            metrics = self._calculate_revisi_metrics(df)
            
            # DEBUG: Tampilkan metrics yang dihitung
            st.sidebar.info(f"📋 Metrics Revisi DIPA: Jumlah Revisi={metrics.get('jumlah_revisi', 0)}, Deviasi Halaman III={metrics.get('nilai_revisi_halaman_iii', 0):.2f}%")
            
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
        """Normalize columns for revisi DIPA data - MAPPING YANG LEBIH LUAS"""
        column_mapping = {
            # Rencana Belanja
            'rencana belanja pegawai': 'Rencana Belanja Pegawai',
            'rencana pegawai': 'Rencana Belanja Pegawai',
            'plan pegawai': 'Rencana Belanja Pegawai',
            'pagu pegawai': 'Rencana Belanja Pegawai',
            'belanja pegawai': 'Rencana Belanja Pegawai',
            'rencana belanja barang': 'Rencana Belanja Barang',
            'rencana barang': 'Rencana Belanja Barang', 
            'plan barang': 'Rencana Belanja Barang',
            'pagu barang': 'Rencana Belanja Barang',
            'belanja barang': 'Rencana Belanja Barang',
            'rencana belanja modal': 'Rencana Belanja Modal',
            'rencana modal': 'Rencana Belanja Modal',
            'plan modal': 'Rencana Belanja Modal',
            'pagu modal': 'Rencana Belanja Modal',
            'belanja modal': 'Rencana Belanja Modal',
            
            # Penyerapan/Realisasi
            'penyerapan belanja pegawai': 'Penyerapan Belanja Pegawai',
            'realisasi pegawai': 'Penyerapan Belanja Pegawai',
            'realisasi belanja pegawai': 'Penyerapan Belanja Pegawai',
            'penyerapan belanja barang': 'Penyerapan Belanja Barang',
            'realisasi barang': 'Penyerapan Belanja Barang',
            'realisasi belanja barang': 'Penyerapan Belanja Barang',
            'penyerapan belanja modal': 'Penyerapan Belanja Modal',
            'realisasi modal': 'Penyerapan Belanja Modal',
            'realisasi belanja modal': 'Penyerapan Belanja Modal',
            
            # Deviasi Absolute
            'deviasi belanja pegawai': 'Deviasi Belanja Pegawai',
            'deviasi pegawai': 'Deviasi Belanja Pegawai',
            'deviasi belanja barang': 'Deviasi Belanja Barang',
            'deviasi barang': 'Deviasi Belanja Barang',
            'deviasi belanja modal': 'Deviasi Belanja Modal',
            'deviasi modal': 'Deviasi Belanja Modal',
            
            # % Deviasi
            '% deviasi belanja pegawai': '% Deviasi Belanja Pegawai',
            'persentase deviasi pegawai': '% Deviasi Belanja Pegawai',
            'deviasi % pegawai': '% Deviasi Belanja Pegawai',
            '% deviasi belanja barang': '% Deviasi Belanja Barang',
            'persentase deviasi barang': '% Deviasi Belanja Barang',
            'deviasi % barang': '% Deviasi Belanja Barang',
            '% deviasi belanja modal': '% Deviasi Belanja Modal',
            'persentase deviasi modal': '% Deviasi Belanja Modal',
            'deviasi % modal': '% Deviasi Belanja Modal',
            'deviasi': '% Deviasi',
            
            # Nilai Revisi Halaman III
            'nilai revisi halaman iii dipa': 'Nilai Revisi Halaman III DIPA',
            'revisi halaman iii': 'Nilai Revisi Halaman III DIPA',
            'nilai revisi': 'Nilai Revisi Halaman III DIPA',
            'deviasi halaman iii': 'Nilai Revisi Halaman III DIPA',
            'nilai deviasi halaman iii': 'Nilai Revisi Halaman III DIPA',
            'deviasi halaman': 'Nilai Revisi Halaman III DIPA',
            'revisi': 'Nilai Revisi Halaman III DIPA',
            
            # Jumlah Revisi
            'jumlah revisi': 'Jumlah Revisi',
            'revisi dipa': 'Jumlah Revisi',
            'total revisi': 'Jumlah Revisi',
            'revisi': 'Jumlah Revisi'
        }
        
        new_columns = []
        for col in df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            
            mapped = False
            for pattern, standard_name in column_mapping.items():
                if pattern in col_lower or col_lower in pattern:
                    new_columns.append(standard_name)
                    mapped = True
                    break
            
            if not mapped:
                new_columns.append(col_str)
        
        df.columns = new_columns
        return df

    def _clean_revisi_numeric_data(self, df):
        """Clean numeric data for revisi DIPA"""
        # Semua kolom yang mungkin berisi angka
        for col in df.columns:
            try:
                # Coba konversi ke numeric
                df[col] = pd.to_numeric(df[col], errors='ignore')
                
                # Jika masih string, coba clean
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
                    # Hapus karakter non-numeric kecuali titik dan minus
                    df[col] = df[col].str.replace(r'[^\d,-.]', '', regex=True)
                    df[col] = df[col].str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                continue
        
        return df

    def _calculate_revisi_metrics(self, df):
        """Calculate metrics for revisi DIPA - LOGIC YANG LEBIH ROBUST"""
        metrics = {}
        
        # DEBUG: Tampilkan data yang akan diproses
        st.sidebar.info(f"📋 Sample data Revisi DIPA (3 baris pertama):")
        for i in range(min(3, len(df))):
            st.sidebar.info(f"   Baris {i+1}: {dict(df.iloc[i])}")
        
        # 1. Hitung Jumlah Revisi DIPA
        # Prioritas: Kolom khusus -> jumlah baris -> default 0
        if 'Jumlah Revisi' in df.columns:
            metrics['jumlah_revisi'] = df['Jumlah Revisi'].sum()
        else:
            # Cek jika ada kolom yang menunjukkan revisi
            revisi_columns = [col for col in df.columns if 'revisi' in col.lower() and 'jumlah' not in col.lower() and 'nilai' not in col.lower()]
            if revisi_columns:
                # Jika ada kolom revisi, hitung baris yang memiliki nilai
                metrics['jumlah_revisi'] = len(df[df[revisi_columns[0]] != 0])
            else:
                # Default: setiap baris = 1 revisi
                metrics['jumlah_revisi'] = len(df)
        
        # 2. Hitung Nilai Revisi Halaman III DIPA
        if 'Nilai Revisi Halaman III DIPA' in df.columns:
            metrics['nilai_revisi_halaman_iii'] = df['Nilai Revisi Halaman III DIPA'].mean()
        else:
            # Fallback 1: Cari kolom deviasi
            deviasi_columns = [col for col in df.columns if 'deviasi' in col.lower() and '%' in col.lower()]
            if deviasi_columns:
                metrics['nilai_revisi_halaman_iii'] = abs(df[deviasi_columns].mean().mean())
            else:
                # Fallback 2: Cari kolom deviasi tanpa %
                deviasi_abs_columns = [col for col in df.columns if 'deviasi' in col.lower() and '%' not in col.lower()]
                if deviasi_abs_columns:
                    metrics['nilai_revisi_halaman_iii'] = abs(df[deviasi_abs_columns].mean().mean())
                else:
                    # Fallback 3: Hitung dari rencana vs penyerapan
                    total_rencana = 0
                    total_penyerapan = 0
                    
                    rencana_cols = [col for col in df.columns if 'rencana' in col.lower()]
                    penyerapan_cols = [col for col in df.columns if 'penyerapan' in col.lower() or 'realisasi' in col.lower()]
                    
                    if rencana_cols and penyerapan_cols:
                        total_rencana = df[rencana_cols].sum().sum()
                        total_penyerapan = df[penyerapan_cols].sum().sum()
                        
                        if total_rencana > 0:
                            metrics['nilai_revisi_halaman_iii'] = abs((total_penyerapan - total_rencana) / total_rencana) * 100
                        else:
                            metrics['nilai_revisi_halaman_iii'] = 0
                    else:
                        # Fallback 4: Default value
                        metrics['nilai_revisi_halaman_iii'] = 0
        
        # 3. Hitung metrik tambahan untuk analisis
        deviasi_columns = [col for col in df.columns if '% Deviasi' in col]
        if deviasi_columns:
            metrics['avg_deviasi_persen'] = df[deviasi_columns].mean().mean()
            metrics['max_deviasi_persen'] = df[deviasi_columns].max().max()
            metrics['min_deviasi_persen'] = df[deviasi_columns].min().min()
            
            # Deviasi per jenis belanja
            if '% Deviasi Belanja Pegawai' in df.columns:
                metrics['avg_deviasi_pegawai'] = df['% Deviasi Belanja Pegawai'].mean()
            if '% Deviasi Belanja Barang' in df.columns:
                metrics['avg_deviasi_barang'] = df['% Deviasi Belanja Barang'].mean()
            if '% Deviasi Belanja Modal' in df.columns:
                metrics['avg_deviasi_modal'] = df['% Deviasi Belanja Modal'].mean()
        else:
            metrics['avg_deviasi_persen'] = 0
            metrics['max_deviasi_persen'] = 0
            metrics['min_deviasi_persen'] = 0
        
        # Total Rencana vs Penyerapan
        rencana_columns = [col for col in df.columns if 'Rencana' in col]
        penyerapan_columns = [col for col in df.columns if 'Penyerapan' in col]
        
        if rencana_columns and penyerapan_columns:
            metrics['total_rencana'] = df[rencana_columns].sum().sum()
            metrics['total_penyerapan'] = df[penyerapan_columns].sum().sum()
            if metrics['total_rencana'] > 0:
                metrics['effisiensi_penyerapan'] = (metrics['total_penyerapan'] / metrics['total_rencana']) * 100
            else:
                metrics['effisiensi_penyerapan'] = 0
        
        # DEBUG: Tampilkan metrics akhir
        st.sidebar.success(f"✅ FINAL - Jumlah Revisi: {metrics.get('jumlah_revisi', 0)}")
        st.sidebar.success(f"✅ FINAL - Deviasi Halaman III: {metrics.get('nilai_revisi_halaman_iii', 0):.2f}%")
        
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
    # ... [kode untuk process_capaian_output dan method lainnya]

def main():
    try:
        st.markdown('<h1 class="main-header">🏢 SMART BNNP DKI - Monitoring IKPA 2024</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666;">Sistem Monitoring Indikator Kinerja Pelaksanaan Anggaran</h3>', unsafe_allow_html=True)
        
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
        
        # File upload section
        st.sidebar.header("📁 Upload Data")
        
        # 1. Data Realisasi Anggaran
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("💰 Data Realisasi Anggaran")
        realisasi_file = st.sidebar.file_uploader(
            "Upload Realisasi Anggaran (Excel)",
            type=['xlsx', 'xls'],
            key="realisasi_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # 2. Data Capaian Output
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📊 Data Capaian Output")
        capaian_file = st.sidebar.file_uploader(
            "Upload Capaian Output (Excel)", 
            type=['xlsx', 'xls'],
            key="capaian_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # 3. Data Revisi DIPA - PERBAIKAN DESKRIPSI
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📋 Data Revisi DIPA")
        st.sidebar.write("""
        **Format yang didukung:**
        - Rencana Belanja Pegawai/Barang/Modal
        - Penyerapan Belanja Pegawai/Barang/Modal  
        - Deviasi Belanja Pegawai/Barang/Modal
        - % Deviasi Belanja Pegawai/Barang/Modal
        - Nilai Revisi Halaman III DIPA
        - Jumlah Revisi
        """)
        revisi_dipa_file = st.sidebar.file_uploader(
            "Upload Data Revisi DIPA (Excel)",
            type=['xlsx', 'xls'],
            key="revisi_dipa_file"
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Manual input fallback
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("✍️ Input Manual")
        
        manual_revisi = st.sidebar.number_input(
            "Jumlah Revisi DIPA",
            min_value=0,
            max_value=10,
            value=0,
            key="manual_revisi"
        )
        
        manual_deviasi_halaman_iii = st.sidebar.number_input(
            "Deviasi Halaman III DIPA (%)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            key="manual_deviasi_halaman_iii"
        )
        
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
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Process data
        realisasi_data = None
        capaian_data = None
        revisi_dipa_data = None
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
        
        # Process revisi DIPA - DENGAN DEBUGGING EXTENSIF
        if revisi_dipa_file:
            with st.spinner("🔄 Memproses data revisi DIPA..."):
                revisi_dipa_data = data_processor.process_revisi_dipa(revisi_dipa_file)
                if "error" not in revisi_dipa_data:
                    st.sidebar.success("✅ Data revisi DIPA berhasil diproses")
                    
                    # Tampilkan preview data
                    with st.sidebar.expander("🔍 Preview Data Revisi DIPA"):
                        st.write("Data mentah dari file:")
                        st.dataframe(revisi_dipa_data['raw_data'].head(3))
                        st.write("Metrics yang dihitung:")
                        st.json(revisi_dipa_data['metrics'])
                    
                    # Tampilkan detail metrics
                    metrics = revisi_dipa_data['metrics']
                    st.sidebar.success(f"📊 Jumlah Revisi: {metrics.get('jumlah_revisi', 0)}")
                    st.sidebar.success(f"📊 Deviasi Halaman III: {metrics.get('nilai_revisi_halaman_iii', 0):.2f}%")
                    
                else:
                    st.sidebar.error(f"❌ {revisi_dipa_data['error']}")
        
        # Prepare data for IKPA calculation - DENGAN LOGGING DETAIL
        ikpa_input = {
            'revisi_dipa': manual_revisi,
            'deviasi_halaman_iii': manual_deviasi_halaman_iii,
            'penyerapan_anggaran': manual_penyerapan,
            'capaian_output': manual_capaian,
            'triwulan': triwulan
        }
        
        # Tampilkan status data sebelum override
        st.sidebar.markdown('<div class="debug-info">📝 STATUS AWAL DATA:</div>', unsafe_allow_html=True)
        st.sidebar.info(f"   • Revisi: {ikpa_input['revisi_dipa']}")
        st.sidebar.info(f"   • Deviasi: {ikpa_input['deviasi_halaman_iii']}%")
        st.sidebar.info(f"   • Penyerapan: {ikpa_input['penyerapan_anggaran']}%")
        st.sidebar.info(f"   • Capaian: {ikpa_input['capaian_output']}%")
        
        # Override dengan data dari file (jika ada) - PRIORITAS FILE
        if realisasi_data and 'error' not in realisasi_data:
            ikpa_input['penyerapan_anggaran'] = realisasi_data['metrics']['penyerapan_persen']
            st.sidebar.success(f"📊 Penyerapan dari file: {realisasi_data['metrics']['penyerapan_persen']:.1f}%")
        
        if capaian_data and 'error' not in capaian_data:
            ikpa_input['capaian_output'] = capaian_data['metrics']['avg_capaian_output']
            st.sidebar.success(f"📊 Capaian output dari file: {capaian_data['metrics']['avg_capaian_output']:.1f}%")
        
        # PRIORITAS TERTINGGI: Data dari file Revisi DIPA
        if revisi_dipa_data and 'error' not in revisi_dipa_data:
            metrics = revisi_dipa_data['metrics']
            
            # Ambil nilai dengan fallback yang lebih baik
            jumlah_revisi = metrics.get('jumlah_revisi', 0)
            deviasi_halaman_iii = metrics.get('nilai_revisi_halaman_iii', 0)
            
            # Assign ke IKPA input
            ikpa_input['revisi_dipa'] = jumlah_revisi
            ikpa_input['deviasi_halaman_iii'] = deviasi_halaman_iii
            
            # Debug info detail
            st.sidebar.success(f"📋 REVISI DIPA TERBACA DAN DIGUNAKAN:")
            st.sidebar.success(f"   • Jumlah Revisi: {jumlah_revisi}")
            st.sidebar.success(f"   • Deviasi Halaman III: {deviasi_halaman_iii:.2f}%")
            
        else:
            st.sidebar.warning("⚠️ Data Revisi DIPA tidak digunakan")
            st.sidebar.info(f"📝 Menggunakan nilai manual: Revisi={manual_revisi}, Deviasi={manual_deviasi_halaman_iii}%")
        
        # Tampilkan status data setelah override
        st.sidebar.markdown('<div class="debug-info">🎯 DATA UNTUK PERHITUNGAN IKPA:</div>', unsafe_allow_html=True)
        st.sidebar.success(f"   • Revisi: {ikpa_input['revisi_dipa']}")
        st.sidebar.success(f"   • Deviasi: {ikpa_input['deviasi_halaman_iii']}%")
        st.sidebar.success(f"   • Penyerapan: {ikpa_input['penyerapan_anggaran']}%")
        st.sidebar.success(f"   • Capaian: {ikpa_input['capaian_output']}%")
        
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
                st.metric("Triwulan", f"{triwulan}")
            
            # Component Analysis - DENGAN INFO DATA YANG DIGUNAKAN
            st.header("🔍 Analisis Komponen IKPA")
            
            # Tampilkan data input yang digunakan
            st.info(f"🎯 **Data yang digunakan dalam perhitungan:**")
            st.info(f"• Jumlah Revisi: {ikpa_input['revisi_dipa']} → Nilai Komponen: {ikpa_result['components']['revisi_dipa']:.2f}%")
            st.info(f"• Deviasi Halaman III: {ikpa_input['deviasi_halaman_iii']:.2f}% → Nilai Komponen: {ikpa_result['components']['deviasi_halaman_iii']:.2f}%")
            st.info(f"• Penyerapan Anggaran: {ikpa_input['penyerapan_anggaran']:.2f}% → Nilai Komponen: {ikpa_result['components']['penyerapan_anggaran']:.2f}%")
            st.info(f"• Capaian Output: {ikpa_input['capaian_output']:.2f}% → Nilai Komponen: {ikpa_result['components']['capaian_output']:.2f}%")
            
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
            
            # ... [bagian lainnya tetap sama]

if __name__ == "__main__":
    main()
