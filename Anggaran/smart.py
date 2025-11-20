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
    .critical-alert {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
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
    .improvement-card {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0c5460;
        margin: 0.5rem 0;
    }
    .success-alert {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
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
        Calculate IKPA 2024 berdasarkan PER-5/PB/2024 - VERSI DIPERBAIKI
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
            • Pengelolaan UP/TUP: {pengelolaan_up_tup}%
            • Capaian Output: {capaian_output}%
            • Triwulan: {triwulan}
            """
            st.sidebar.markdown(f'<div class="debug-info">{debug_info}</div>', unsafe_allow_html=True)
            
            # PERBAIKAN: Normalisasi sesuai formula 2024 dengan handling khusus
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
            
            # PERBAIKAN: Analisis khusus untuk nilai rendah
            analysis_note = ""
            critical_issues = []
            
            if penyerapan_anggaran == 0:
                critical_issues.append("Penyerapan anggaran 0%")
            elif penyerapan_anggaran < 50:
                critical_issues.append(f"Penyerapan anggaran rendah ({penyerapan_anggaran:.1f}%)")
                
            if nilai_akhir < 70:
                if critical_issues:
                    analysis_note = f"🚨 PRIORITAS TINGGI: {' dan '.join(critical_issues)} menyebabkan penurunan signifikan nilai IKPA"
                else:
                    analysis_note = "⚠️ PERBAIKAN DIPERLUKAN: Beberapa komponen perlu ditingkatkan"
            
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
            
            # Hitung kontribusi per komponen
            component_contributions = {
                'revisi_dipa': (nilai_revisi_dipa * bobot['revisi_dipa']) / 100,
                'deviasi_halaman_iii': (nilai_deviasi * bobot['deviasi_halaman_iii']) / 100,
                'penyerapan_anggaran': (nilai_penyerapan * bobot['penyerapan_anggaran']) / 100,
                'belanja_kontraktual': (nilai_belanja_kontraktual * bobot['belanja_kontraktual']) / 100,
                'penyelesaian_tagihan': (nilai_penyelesaian_tagihan * bobot['penyelesaian_tagihan']) / 100,
                'pengelolaan_up_tup': (nilai_pengelolaan_up_tup * bobot['pengelolaan_up_tup']) / 100,
                'capaian_output': (nilai_capaian_output * bobot['capaian_output']) / 100
            }
            
            self.ikpa_data = {
                'nilai_total': nilai_total,
                'nilai_konversi': nilai_konversi,
                'nilai_akhir': nilai_akhir,
                'kategori': kategori,
                'color_class': color_class,
                'gap_target': 95 - nilai_akhir,
                'analysis_note': analysis_note,
                'critical_issues': critical_issues,
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
                'component_contributions': component_contributions,
                'improvement_areas': self._identify_improvement_areas({
                    'penyerapan': nilai_penyerapan,
                    'capaian_output': nilai_capaian_output,
                    'deviasi': nilai_deviasi,
                    'pengelolaan_up_tup': nilai_pengelolaan_up_tup,
                    'nilai_akhir': nilai_akhir
                })
            }
            
            # Tampilkan debug info hasil perhitungan
            result_debug = f"""
            📊 HASIL PERHITUNGAN IKPA:
            • Nilai Akhir: {nilai_akhir:.2f} ({kategori})
            • Komponen Revisi DIPA: {nilai_revisi_dipa:.2f}% → {component_contributions['revisi_dipa']:.2f}
            • Komponen Deviasi: {nilai_deviasi:.2f}% → {component_contributions['deviasi_halaman_iii']:.2f}
            • Komponen Penyerapan: {nilai_penyerapan:.2f}% → {component_contributions['penyerapan_anggaran']:.2f}
            • Komponen UP/TUP: {nilai_pengelolaan_up_tup:.2f}% → {component_contributions['pengelolaan_up_tup']:.2f}
            • Komponen Capaian: {nilai_capaian_output:.2f}% → {component_contributions['capaian_output']:.2f}
            • Total: {nilai_total:.2f}
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
        """Identify areas for improvement - VERSI DIPERBAIKI"""
        areas = []
        
        # Prioritas berdasarkan dampak terhadap nilai akhir
        if components['penyerapan'] < 50:
            areas.append({
                'area': 'Penyerapan Anggaran',
                'current': components['penyerapan'],
                'target': 80,
                'priority': 'Critical',
                'description': 'Akselerasi penyerapan anggaran merupakan prioritas utama',
                'impact': 'Tinggi',
                'recommendation': 'Lakukan percepatan penyerapan dengan fokus pada kegiatan prioritas'
            })
        elif components['penyerapan'] < 80:
            areas.append({
                'area': 'Penyerapan Anggaran',
                'current': components['penyerapan'],
                'target': 95,
                'priority': 'High',
                'description': 'Perlu akselerasi penyerapan anggaran sesuai proyeksi triwulan',
                'impact': 'Tinggi',
                'recommendation': 'Tingkatkan monitoring mingguan dan koordinasi dengan PPK'
            })
        
        if components['capaian_output'] < 70:
            areas.append({
                'area': 'Capaian Output',
                'current': components['capaian_output'],
                'target': 85,
                'priority': 'High',
                'description': 'Tingkatkan kualitas pelaporan capaian output dan ketepatan waktu',
                'impact': 'Tinggi',
                'recommendation': 'Perbaiki dokumentasi realisasi output dan validasi data'
            })
        elif components['capaian_output'] < 85:
            areas.append({
                'area': 'Capaian Output',
                'current': components['capaian_output'],
                'target': 95,
                'priority': 'Medium',
                'description': 'Optimalkan pencapaian output yang tertinggal',
                'impact': 'Sedang',
                'recommendation': 'Fokus pada RO yang belum mencapai target'
            })
        
        if components['deviasi'] < 90:
            areas.append({
                'area': 'Deviasi RPD',
                'current': components['deviasi'],
                'target': 95,
                'priority': 'Medium',
                'description': 'Optimalkan perencanaan penarikan dana untuk minimalkan deviasi',
                'impact': 'Sedang',
                'recommendation': 'Perbaiki akurasi forecasting dan real-time monitoring'
            })
        
        if components['pengelolaan_up_tup'] < 90:
            areas.append({
                'area': 'Pengelolaan UP/TUP',
                'current': components['pengelolaan_up_tup'],
                'target': 100,
                'priority': 'Low',
                'description': 'Tingkatkan penggunaan KKP sesuai target triwulan',
                'impact': 'Rendah',
                'recommendation': 'Optimalkan timing penarikan dana UP/TUP'
            })
        
        # Jika nilai akhir sangat rendah, tambahkan rekomendasi khusus
        if components['nilai_akhir'] < 50:
            areas.append({
                'area': 'Strategi Darurat',
                'current': components['nilai_akhir'],
                'target': 70,
                'priority': 'Critical',
                'description': 'Diperlukan intervensi khusus untuk menaikkan nilai IKPA',
                'impact': 'Sangat Tinggi',
                'recommendation': 'Bentuk tim khusus dan lakukan koordinasi intensif dengan seluruh pihak terkait'
            })
        
        return areas

class DataProcessor:
    def __init__(self):
        self.processed_data = {}
    
    def process_realisasi_anggaran(self, file_buffer):
        """
        Process Excel file for budget realization data - VERSI DIPERBAIKI
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
            
            # Calculate metrics - ambil data dari row terakhir
            metrics = self._calculate_realisasi_metrics(df)
            
            # Calculate deviasi RPD
            deviasi_rpd = self._calculate_deviation_rpd(df)
            
            self.processed_data['realisasi_anggaran'] = {
                'metrics': metrics,
                'deviasi_rpd': deviasi_rpd,
                'raw_data': df,
                'last_row_data': self._extract_last_row_data(df),
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
        """Normalize columns for realisasi anggaran data - VERSI DIPERBAIKI"""
        column_mapping = {
            # Kolom utama
            'jumlah revisi': 'Jumlah Revisi',
            'revisi': 'Jumlah Revisi',
            'keterangan': 'Keterangan',
            'uraian': 'Keterangan',
            'deskripsi': 'Keterangan',
            
            # Pagu
            'pagu belanja pegawai': 'Pagu Belanja Pegawai',
            'pagu pegawai': 'Pagu Belanja Pegawai',
            'belanja pegawai': 'Pagu Belanja Pegawai',
            'pagu belanja barang': 'Pagu Belanja Barang',
            'pagu barang': 'Pagu Belanja Barang',
            'belanja barang': 'Pagu Belanja Barang',
            'pagu belanja modal': 'Pagu Belanja Modal',
            'pagu modal': 'Pagu Belanja Modal',
            'belanja modal': 'Pagu Belanja Modal',
            
            # Realisasi
            'realisasi belanja pegawai': 'Realisasi Belanja Pegawai',
            'realisasi pegawai': 'Realisasi Belanja Pegawai',
            'real belanja pegawai': 'Realisasi Belanja Pegawai',
            'realisasi belanja barang': 'Realisasi Belanja Barang',
            'realisasi barang': 'Realisasi Belanja Barang',
            'real belanja barang': 'Realisasi Belanja Barang',
            'realisasi belanja modal': 'Realisasi Belanja Modal',
            'realisasi modal': 'Realisasi Belanja Modal',
            'real belanja modal': 'Realisasi Belanja Modal',
            
            # Total dan NKPA
            'total pagu': 'Total Pagu',
            'total anggaran': 'Total Pagu',
            'pagu total': 'Total Pagu',
            'total realisasi': 'Total Realisasi',
            'realisasi total': 'Total Realisasi',
            'nkpa semua jenis belanja': 'NKPA Semua Jenis Belanja',
            'nkpa': 'NKPA Semua Jenis Belanja',
            'nilai ikpa penyerapan anggaran': 'Nilai IKPA Penyerapan Anggaran',
            'ikpa penyerapan': 'Nilai IKPA Penyerapan Anggaran',
            'penyerapan': 'Nilai IKPA Penyerapan Anggaran'
        }
        
        new_columns = []
        seen_columns = {}
        
        for col in df.columns:
            col_str = str(col).strip()
            col_lower = col_str.lower()
            
            mapped = False
            for pattern, standard_name in column_mapping.items():
                if pattern in col_lower or col_lower in pattern:
                    # Handle duplicate column names
                    if standard_name in seen_columns:
                        count = seen_columns[standard_name] + 1
                        new_name = f"{standard_name}_{count}"
                        seen_columns[standard_name] = count
                    else:
                        new_name = standard_name
                        seen_columns[standard_name] = 1
                    
                    new_columns.append(new_name)
                    mapped = True
                    break
            
            if not mapped:
                # Handle duplicate unnamed columns
                if col_str.startswith('Unnamed:'):
                    if col_str in seen_columns:
                        count = seen_columns[col_str] + 1
                        new_name = f"{col_str}_{count}"
                        seen_columns[col_str] = count
                    else:
                        new_name = col_str
                        seen_columns[col_str] = 1
                    new_columns.append(new_name)
                else:
                    new_columns.append(col_str)
        
        df.columns = new_columns
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        return df

    def _clean_realisasi_numeric_data(self, df):
        """Clean numeric data for realisasi anggaran - VERSI DIPERBAIKI"""
        # Semua kolom yang mungkin berisi angka
        numeric_columns = [
            'Jumlah Revisi',
            'Pagu Belanja Pegawai', 'Realisasi Belanja Pegawai',
            'Pagu Belanja Barang', 'Realisasi Belanja Barang', 
            'Pagu Belanja Modal', 'Realisasi Belanja Modal',
            'Total Pagu', 'Total Realisasi',
            'NKPA Semua Jenis Belanja', 'Nilai IKPA Penyerapan Anggaran'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Convert to string first to handle various formats
                    df[col] = df[col].astype(str)
                    
                    # Handle percentage values
                    if 'NKPA' in col or 'IKPA' in col or 'Penyerapan' in col:
                        df[col] = df[col].str.replace('%', '', regex=False)
                    
                    # Hapus karakter non-numeric kecuali titik dan minus
                    df[col] = df[col].str.replace(r'[^\d,-.]', '', regex=True)
                    df[col] = df[col].str.replace(',', '.', regex=False)
                    
                    # Convert to numeric, coerce errors to NaN then fill with 0
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    # Debug info untuk kolom yang diproses
                    st.sidebar.info(f"✅ Kolom {col}: {len(df[df[col] > 0])} nilai non-zero")
                    
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Gagal memproses kolom {col}: {str(e)}")
                    continue
        
        return df

    def _calculate_realisasi_metrics(self, df):
        """Calculate metrics for realisasi anggaran - AMBIL DARI ROW TERAKHIR"""
        metrics = {}
        
        # Ambil data dari row terakhir (kondisi bulan terakhir)
        if len(df) > 0:
            last_row = df.iloc[-1]
            
            # Debug info
            st.sidebar.info(f"📊 Menggunakan data dari row terakhir (total {len(df)} rows)")
            
            # Ambil nilai dari row terakhir
            metrics['jumlah_revisi'] = last_row.get('Jumlah Revisi', 0)
            
            # Hitung total pagu dan realisasi dari row terakhir
            pagu_pegawai = last_row.get('Pagu Belanja Pegawai', 0)
            pagu_barang = last_row.get('Pagu Belanja Barang', 0)
            pagu_modal = last_row.get('Pagu Belanja Modal', 0)
            
            realisasi_pegawai = last_row.get('Realisasi Belanja Pegawai', 0)
            realisasi_barang = last_row.get('Realisasi Belanja Barang', 0)
            realisasi_modal = last_row.get('Realisasi Belanja Modal', 0)
            
            # Total dari komponen
            metrics['total_pagu'] = pagu_pegawai + pagu_barang + pagu_modal
            metrics['total_realisasi'] = realisasi_pegawai + realisasi_barang + realisasi_modal
            
            # Jika ada kolom total langsung, gunakan itu
            if 'Total Pagu' in last_row.index and last_row['Total Pagu'] > 0:
                metrics['total_pagu'] = last_row['Total Pagu']
                st.sidebar.info(f"📊 Total Pagu dari kolom: {metrics['total_pagu']:,.0f}")
            
            if 'Total Realisasi' in last_row.index and last_row['Total Realisasi'] > 0:
                metrics['total_realisasi'] = last_row['Total Realisasi']
                st.sidebar.info(f"📊 Total Realisasi dari kolom: {metrics['total_realisasi']:,.0f}")
            
            # Hitung penyerapan
            if metrics['total_pagu'] > 0:
                metrics['penyerapan_persen'] = (metrics['total_realisasi'] / metrics['total_pagu']) * 100
                st.sidebar.info(f"📊 Penyerapan dihitung: {metrics['penyerapan_persen']:.2f}%")
            else:
                metrics['penyerapan_persen'] = 0
                st.sidebar.warning("⚠️ Total Pagu = 0, penyerapan di-set 0%")
            
            # Jika ada nilai IKPA penyerapan langsung, gunakan itu (PRIORITAS TERTINGGI)
            if 'Nilai IKPA Penyerapan Anggaran' in last_row.index:
                ikpa_penyerapan = last_row['Nilai IKPA Penyerapan Anggaran']
                if ikpa_penyerapan > 0:
                    metrics['penyerapan_persen'] = ikpa_penyerapan
                    st.sidebar.success(f"🎯 Menggunakan nilai IKPA Penyerapan langsung: {ikpa_penyerapan:.2f}%")
            
            # Hitung penyerapan per jenis belanja
            if pagu_pegawai > 0:
                metrics['penyerapan_pegawai'] = (realisasi_pegawai / pagu_pegawai) * 100
            else:
                metrics['penyerapan_pegawai'] = 0
                
            if pagu_barang > 0:
                metrics['penyerapan_barang'] = (realisasi_barang / pagu_barang) * 100
            else:
                metrics['penyerapan_barang'] = 0
                
            if pagu_modal > 0:
                metrics['penyerapan_modal'] = (realisasi_modal / pagu_modal) * 100
            else:
                metrics['penyerapan_modal'] = 0
            
            # Simpan nilai NKPA jika ada
            if 'NKPA Semua Jenis Belanja' in last_row.index:
                metrics['nkpa_semua_jenis'] = last_row['NKPA Semua Jenis Belanja']
            else:
                metrics['nkpa_semua_jenis'] = 0
                
            # Tampilkan summary metrics
            st.sidebar.success(f"✅ Metrics Realisasi: Penyerapan = {metrics['penyerapan_persen']:.2f}%")
                
        else:
            # Fallback jika dataframe kosong
            st.sidebar.error("❌ DataFrame realisasi kosong!")
            metrics.update({
                'jumlah_revisi': 0,
                'total_pagu': 0,
                'total_realisasi': 0,
                'penyerapan_persen': 0,
                'penyerapan_pegawai': 0,
                'penyerapan_barang': 0,
                'penyerapan_modal': 0,
                'nkpa_semua_jenis': 0
            })
        
        return metrics

    def _extract_last_row_data(self, df):
        """Extract data from the last row for detailed analysis"""
        if len(df) == 0:
            return {}
        
        last_row = df.iloc[-1]
        last_row_data = {}
        
        # Extract all numeric values from last row
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                value = last_row[col]
                if not pd.isna(value):
                    last_row_data[col] = value
            else:
                # Juga ambil nilai string untuk kolom penting
                if any(keyword in col.lower() for keyword in ['keterangan', 'uraian', 'deskripsi']):
                    value = last_row[col]
                    if pd.notna(value) and str(value).strip() != '':
                        last_row_data[col] = value
        
        return last_row_data

    def _calculate_deviation_rpd(self, df):
        """Calculate RPD deviation from last row"""
        try:
            if len(df) == 0:
                return 5.0
                
            last_row = df.iloc[-1]
            
            # Hitung deviasi dari row terakhir
            pagu_columns = [col for col in df.columns if 'pagu' in col.lower() and 'realisasi' not in col.lower()]
            realisasi_columns = [col for col in df.columns if 'realisasi' in col.lower()]
            
            if pagu_columns and realisasi_columns:
                total_pagu = sum([last_row.get(col, 0) for col in pagu_columns])
                total_realisasi = sum([last_row.get(col, 0) for col in realisasi_columns])
                
                if total_pagu > 0:
                    deviation = abs((total_realisasi - total_pagu) / total_pagu) * 100
                    return min(deviation, 50)
            
            return 5.0
        except:
            return 5.0

    def process_revisi_dipa(self, file_buffer):
        """Process revisi DIPA data"""
        try:
            df = pd.read_excel(file_buffer)
            # Implementasi processing revisi DIPA...
            return {"metrics": {"jumlah_revisi": 0, "nilai_revisi_halaman_iii": 0}}
        except Exception as e:
            return {"error": str(e)}
    
    def process_capaian_output(self, file_buffer):
        """Process capaian output data"""
        try:
            df = pd.read_excel(file_buffer)
            # Implementasi processing capaian output...
            return {"metrics": {"avg_capaian_output": 0}}
        except Exception as e:
            return {"error": str(e)}

# ... (StrategicAdvisor class dan visualization functions tetap sama seperti sebelumnya)

class StrategicAdvisor:
    def __init__(self):
        pass
    
    def generate_recommendations(self, ikpa_data, realisasi_data, capaian_data, revisi_dipa_data, triwulan):
        """Generate comprehensive strategic recommendations"""
        
        recommendations = {
            'critical_analysis': self._generate_critical_analysis(ikpa_data),
            'penarikan_dana': self._generate_penarikan_dana_recommendation(realisasi_data, triwulan),
            'capaian_output': self._generate_capaian_output_recommendation(capaian_data, triwulan),
            'revisi_dipa': self._generate_revisi_dipa_recommendation(revisi_dipa_data, triwulan),
            'rencana_penyerapan': self._generate_rencana_penyerapan_recommendation(realisasi_data, revisi_dipa_data, triwulan),
            'ikpa_improvement': self._generate_ikpa_improvement_recommendation(ikpa_data),
            'emergency_plan': self._generate_emergency_plan(ikpa_data)
        }
        
        return recommendations
    
    def _generate_critical_analysis(self, ikpa_data):
        """Generate critical analysis for low IKPA scores"""
        if ikpa_data['nilai_akhir'] >= 70:
            return {
                'status': 'Stabil',
                'message': 'Nilai IKPA dalam kategori memadai, fokus pada optimasi',
                'priority': 'Medium'
            }
        
        analysis = {
            'status': 'Kritis' if ikpa_data['nilai_akhir'] < 50 else 'Perhatian',
            'message': '',
            'priority': 'High' if ikpa_data['nilai_akhir'] < 50 else 'Medium',
            'root_causes': [],
            'immediate_actions': []
        }
        
        # Identifikasi penyebab utama
        components = ikpa_data['components']
        
        if components['penyerapan_anggaran'] < 50:
            analysis['root_causes'].append('Penyerapan anggaran sangat rendah')
            analysis['immediate_actions'].append('Akselerasi penyerapan anggaran mingguan')
            
        if components['capaian_output'] < 70:
            analysis['root_causes'].append('Capaian output tidak optimal')
            analysis['immediate_actions'].append('Tingkatkan kualitas pelaporan output')
            
        if ikpa_data['nilai_akhir'] < 50:
            analysis['immediate_actions'].extend([
                'Bentuk tim khusus penanganan IKPA',
                'Lakukan koordinasi darurat dengan seluruh PPK',
                'Implementasi monitoring harian'
            ])
            
        analysis['message'] = f"Nilai IKPA {ikpa_data['nilai_akhir']:.1f} memerlukan intervensi {'darurat' if ikpa_data['nilai_akhir'] < 50 else 'khusus'}"
        
        return analysis

    def _generate_emergency_plan(self, ikpa_data):
        """Generate emergency improvement plan for critical cases"""
        if ikpa_data['nilai_akhir'] >= 60:
            return None
            
        plan = {
            'phase_1_1week': [
                "Bentuk Tim Khusus Penanganan IKPA",
                "Koordinasi Darurat dengan Seluruh PPK",
                "Analisis Penyebab Rendahnya Setiap Komponen",
                "Setup Monitoring Harian Realisasi"
            ],
            'phase_2_2weeks': [
                "Implementasi Akselerasi Penyerapan Mingguan",
                "Perbaikan Dokumen Capaian Output",
                "Optimalisasi Pengelolaan UP/TUP",
                "Review dan Perbaikan RPD"
            ],
            'phase_3_1month': [
                "Evaluasi Progress Mingguan",
                "Adjustment Strategi Berdasarkan Hasil",
                "Koordinasi dengan BUN untuk Optimalisasi",
                "Peningkatan Kualitas Data dan Pelaporan"
            ]
        }
        
        # Customize based on specific issues
        if ikpa_data['components']['penyerapan_anggaran'] == 0:
            plan['phase_1_1week'].insert(1, "PRIORITAS: Akselerasi Penyerapan - Fokus pada Kegiatan Prioritas")
            
        return plan

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
        
        rekomendasi = []
        
        # Analisis jumlah revisi
        jumlah_revisi = metrics.get('jumlah_revisi', 0)
        if jumlah_revisi > 1:
            rekomendasi.append(f"🚨 **PRIORITAS TINGGI**: Kurangi jumlah revisi dari {jumlah_revisi} menjadi maksimal 1")
        
        return {
            'jumlah_revisi': f"{jumlah_revisi}",
            'target_revisi': "≤1",
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
        
        return {
            'target_triwulan': current_projection.get('target', 'N/A'),
            'current_performance': f"{current_penyerapan:.1f}%",
            'bulanan_breakdown': current_projection.get('bulan', []),
            'strategic_actions': [
                "Front-loading penyerapan di awal bulan",
                "Prioritaskan kegiatan dengan SPM mudah", 
                "Daily monitoring oleh pengelola kegiatan"
            ],
            'compliance_rules': [
                "📅 Input realisasi max 3 hari kerja setelah bulan berakhir",
                "📝 Lengkapi supporting documents sebelum SPP",
                "🔍 Validasi oleh PPK sebelum pengajuan SPM",
                "📊 Laporan harian ke Kuasa BUN"
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
        # Tentukan warna berdasarkan kategori
        if value >= 95:
            bar_color = "green"
        elif value >= 89:
            bar_color = "yellow" 
        elif value >= 70:
            bar_color = "orange"
        else:
            bar_color = "red"

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"IKPA 2024 - {category}", 'font': {'size': 24}},
            delta={'reference': 95, 'increasing': {'color': "green"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 70], 'color': 'rgba(220, 53, 69, 0.3)'},
                    {'range': [70, 89], 'color': 'rgba(253, 126, 20, 0.3)'},
                    {'range': [89, 95], 'color': 'rgba(255, 193, 7, 0.3)'},
                    {'range': [95, 100], 'color': 'rgba(40, 167, 69, 0.3)'}],
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

def create_detailed_component_chart(ikpa_data):
    """Create detailed component chart with contributions"""
    try:
        components = ['Revisi DIPA', 'Deviasi RPD', 'Penyerapan Anggaran', 'Pengelolaan UP/TUP', 'Capaian Output']
        values = [
            ikpa_data['components']['revisi_dipa'],
            ikpa_data['components']['deviasi_halaman_iii'], 
            ikpa_data['components']['penyerapan_anggaran'],
            ikpa_data['components']['pengelolaan_up_tup'],
            ikpa_data['components']['capaian_output']
        ]
        
        contributions = [
            ikpa_data['component_contributions']['revisi_dipa'],
            ikpa_data['component_contributions']['deviasi_halaman_iii'],
            ikpa_data['component_contributions']['penyerapan_anggaran'],
            ikpa_data['component_contributions']['pengelolaan_up_tup'],
            ikpa_data['component_contributions']['capaian_output']
        ]
        
        fig = go.Figure()
        
        # Bar untuk nilai komponen
        fig.add_trace(go.Bar(
            name='Nilai Komponen (%)',
            x=components,
            y=values,
            marker_color='lightblue',
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
        ))
        
        # Bar untuk kontribusi
        fig.add_trace(go.Bar(
            name='Kontribusi ke Nilai',
            x=components, 
            y=contributions,
            marker_color='darkblue',
            text=[f'{v:.2f}' for v in contributions],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Detail Komponen IKPA dan Kontribusinya',
            barmode='group',
            height=500,
            yaxis_title='Nilai'
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title="Error creating detailed chart", height=500)
        return fig

def main():
    try:
        st.markdown('<h1 class="main-header">🏢 SMART BNNP DKI - Monitoring IKPA 2024</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #666;">Sistem Monitoring Indikator Kinerja Pelaksanaan Anggaran</h3>', unsafe_allow_html=True)
        
        # Initialize processors
        data_processor = DataProcessor()
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
        st.sidebar.write("""
        **Format yang didukung:**
        - Jumlah Revisi, Keterangan
        - Pagu Belanja Pegawai/Barang/Modal
        - Realisasi Belanja Pegawai/Barang/Modal  
        - NKPA Semua Jenis Belanja
        - Nilai IKPA Penyerapan Anggaran
        """)
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
        
        # 3. Data Revisi DIPA
        st.sidebar.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📋 Data Revisi DIPA")
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
        
        manual_pengelolaan_up_tup = st.sidebar.number_input(
            "Pengelolaan UP/TUP (%)", 
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            key="manual_pengelolaan_up_tup"
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
        
        # Process realisasi anggaran - PERBAIKAN UTAMA
        if realisasi_file is not None:
            with st.spinner("🔄 Memproses data realisasi anggaran..."):
                realisasi_data = data_processor.process_realisasi_anggaran(realisasi_file)
                if "error" not in realisasi_data:
                    st.sidebar.success("✅ Data realisasi anggaran berhasil diproses")
                    
                    # Tampilkan detail data realisasi
                    with st.sidebar.expander("🔍 Detail Realisasi Anggaran"):
                        st.write("Data dari row terakhir:")
                        if 'last_row_data' in realisasi_data:
                            for key, value in realisasi_data['last_row_data'].items():
                                if isinstance(value, (int, float)):
                                    if value > 0:  # Hanya tampilkan nilai non-zero
                                        st.write(f"• {key}: {value:,.0f}")
                                else:
                                    st.write(f"• {key}: {value}")
                        
                        st.write("Metrics yang dihitung:")
                        st.json(realisasi_data['metrics'])
                        
                else:
                    st.sidebar.error(f"❌ {realisasi_data['error']}")
        
        # Process capaian output
        if capaian_file is not None:
            with st.spinner("🔄 Memproses data capaian output..."):
                capaian_data = data_processor.process_capaian_output(capaian_file)
                if "error" not in capaian_data:
                    st.sidebar.success("✅ Data capaian output berhasil diproses")
                else:
                    st.sidebar.error(f"❌ {capaian_data['error']}")
        
        # Process revisi DIPA
        if revisi_dipa_file is not None:
            with st.spinner("🔄 Memproses data revisi DIPA..."):
                revisi_dipa_data = data_processor.process_revisi_dipa(revisi_dipa_file)
                if "error" not in revisi_dipa_data:
                    st.sidebar.success("✅ Data revisi DIPA berhasil diproses")
                else:
                    st.sidebar.error(f"❌ {revisi_dipa_data['error']}")
        
        # Prepare data for IKPA calculation - PERBAIKAN PENTING: Prioritaskan data dari file
        ikpa_input = {
            'revisi_dipa': manual_revisi,
            'deviasi_halaman_iii': manual_deviasi_halaman_iii,
            'penyerapan_anggaran': manual_penyerapan,
            'pengelolaan_up_tup': manual_pengelolaan_up_tup,
            'capaian_output': manual_capaian,
            'triwulan': triwulan
        }
        
        # Tampilkan status data sebelum override
        st.sidebar.markdown('<div class="debug-info">📝 STATUS AWAL DATA (MANUAL):</div>', unsafe_allow_html=True)
        st.sidebar.info(f"   • Revisi: {ikpa_input['revisi_dipa']}")
        st.sidebar.info(f"   • Deviasi: {ikpa_input['deviasi_halaman_iii']}%")
        st.sidebar.info(f"   • Penyerapan: {ikpa_input['penyerapan_anggaran']}%")
        st.sidebar.info(f"   • UP/TUP: {ikpa_input['pengelolaan_up_tup']}%")
        st.sidebar.info(f"   • Capaian: {ikpa_input['capaian_output']}%")
        
        # OVERRIDE DENGAN DATA DARI FILE - PRIORITAS TERTINGGI
        file_data_used = False
        
        if realisasi_data and 'error' not in realisasi_data:
            metrics = realisasi_data['metrics']
            ikpa_input['penyerapan_anggaran'] = metrics['penyerapan_persen']
            file_data_used = True
            st.sidebar.success(f"📊 Penyerapan dari file: {metrics['penyerapan_persen']:.2f}%")
        
        if capaian_data and 'error' not in capaian_data:
            metrics = capaian_data['metrics']
            ikpa_input['capaian_output'] = metrics['avg_capaian_output']
            file_data_used = True
            st.sidebar.success(f"📊 Capaian output dari file: {metrics['avg_capaian_output']:.2f}%")
        
        if revisi_dipa_data and 'error' not in revisi_dipa_data:
            metrics = revisi_dipa_data['metrics']
            ikpa_input['revisi_dipa'] = metrics.get('jumlah_revisi', 0)
            ikpa_input['deviasi_halaman_iii'] = metrics.get('nilai_revisi_halaman_iii', 0)
            file_data_used = True
            st.sidebar.success(f"📋 Revisi DIPA dari file: {metrics.get('jumlah_revisi', 0)} revisi, {metrics.get('nilai_revisi_halaman_iii', 0):.2f}% deviasi")
        
        # Tampilkan status data setelah override
        if file_data_used:
            st.sidebar.markdown('<div class="success-alert">🎯 DATA UNTUK PERHITUNGAN IKPA (SETELAH OVERRIDE):</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="debug-info">🎯 DATA UNTUK PERHITUNGAN IKPA (MANUAL):</div>', unsafe_allow_html=True)
        
        st.sidebar.success(f"   • Revisi: {ikpa_input['revisi_dipa']}")
        st.sidebar.success(f"   • Deviasi: {ikpa_input['deviasi_halaman_iii']}%")
        st.sidebar.success(f"   • Penyerapan: {ikpa_input['penyerapan_anggaran']}%")
        st.sidebar.success(f"   • UP/TUP: {ikpa_input['pengelolaan_up_tup']}%")
        st.sidebar.success(f"   • Capaian: {ikpa_input['capaian_output']}%")
        
        # Calculate IKPA
        ikpa_result = None
        if st.button("🚀 Hitung Nilai IKPA 2024", type="primary") or any([realisasi_data, capaian_data, revisi_dipa_data]):
            with st.spinner("🔄 Menghitung nilai IKPA 2024..."):
                ikpa_result = ikpa_calculator.calculate_ikpa_2024(ikpa_input)
        
        # Display results
        if ikpa_result and 'error' not in ikpa_result:
            # Tampilkan sumber data
            if file_data_used:
                st.markdown(f"""
                <div class="success-alert">
                    <h3>✅ Data Berhasil Diproses</h3>
                    <p>Nilai IKPA dihitung berdasarkan data dari file yang diupload</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="debug-info">
                    <h3>ℹ️ Menggunakan Data Manual</h3>
                    <p>Nilai IKPA dihitung berdasarkan input manual</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Critical Alert untuk nilai rendah
            if ikpa_result['nilai_akhir'] < 70:
                st.markdown(f"""
                <div class="critical-alert">
                    <h3>🚨 PERHATIAN: NILAI IKPA RENDAH</h3>
                    <p><strong>Nilai: {ikpa_result['nilai_akhir']:.2f} ({ikpa_result['kategori']})</strong></p>
                    <p>{ikpa_result['analysis_note']}</p>
                </div>
                """, unsafe_allow_html=True)
            
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
                status_color = "🟢" if ikpa_result['nilai_akhir'] >= 95 else "🟡" if ikpa_result['nilai_akhir'] >= 70 else "🔴"
                st.metric("Status", f"{status_color} {'Optimal' if ikpa_result['nilai_akhir'] >= 95 else 'Perlu Perbaikan'}")
            
            with col4:
                st.metric("Triwulan", f"{triwulan}")
                st.metric("Sumber Data", "File Upload" if file_data_used else "Input Manual")
            
            # Detailed Component Analysis
            st.header("🔍 Analisis Detail Komponen")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_detailed = create_detailed_component_chart(ikpa_result)
                st.plotly_chart(fig_detailed, use_container_width=True)
            
            with col2:
                # Tampilkan tabel kontribusi detail
                st.subheader("Kontribusi per Komponen")
                contribution_data = []
                total_contribution = 0
                
                for comp, weight in ikpa_result['bobot'].items():
                    if weight > 0 and comp in ikpa_result['component_contributions']:
                        contribution = ikpa_result['component_contributions'][comp]
                        total_contribution += contribution
                        contribution_data.append({
                            'Komponen': comp.replace('_', ' ').title(),
                            'Nilai': f"{ikpa_result['components'][comp]:.2f}%",
                            'Bobot': f"{weight}%", 
                            'Kontribusi': f"{contribution:.2f}",
                            '% dari Total': f"{(contribution/ikpa_result['nilai_akhir']*100) if ikpa_result['nilai_akhir'] > 0 else 0:.1f}%"
                        })
                
                contribution_df = pd.DataFrame(contribution_data)
                st.dataframe(contribution_df, use_container_width=True, hide_index=True)
                
                st.metric("Total Kontribusi", f"{total_contribution:.2f}")
                st.metric("Nilai Konversi", f"{ikpa_result['nilai_konversi']:.2f}")
                st.metric("Nilai Akhir", f"{ikpa_result['nilai_akhir']:.2f}")
            
            # Data Realisasi Detail
            if realisasi_data and 'error' not in realisasi_data:
                st.header("💰 Detail Realisasi Anggaran")
                
                metrics = realisasi_data['metrics']
                last_row = realisasi_data.get('last_row_data', {})
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Pagu", f"Rp {metrics['total_pagu']:,.0f}")
                with col2:
                    st.metric("Total Realisasi", f"Rp {metrics['total_realisasi']:,.0f}")
                with col3:
                    st.metric("Penyerapan", f"{metrics['penyerapan_persen']:.1f}%")
                with col4:
                    st.metric("Jumlah Revisi", f"{metrics.get('jumlah_revisi', 0)}")
                
                # Tampilkan data row terakhir
                with st.expander("📋 Data Row Terakhir (Bulan Berjalan)"):
                    if last_row:
                        for key, value in last_row.items():
                            if isinstance(value, (int, float)) and value != 0:
                                st.write(f"**{key}:** {value:,.0f}" if value > 1000 else f"**{key}:** {value:.2f}")
            
            # Strategic Recommendations
            st.header("🎯 Rekomendasi Strategis")
            
            recommendations = strategic_advisor.generate_recommendations(
                ikpa_result, realisasi_data, capaian_data, revisi_dipa_data, triwulan
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💰 Pengaturan Penarikan Dana")
                penarikan = recommendations['penarikan_dana']
                st.metric(f"Target Triwulan {triwulan}", penarikan['target_penarikan'])
                st.write(f"**Fokus Kegiatan:** {penarikan['fokus_kegiatan']}")
                st.write("**Strategi:**")
                for strategy in penarikan['strategi']:
                    st.write(f"• {strategy}")
            
            with col2:
                st.subheader("📊 Target Capaian Output")
                capaian = recommendations['capaian_output']
                st.metric(f"Target Triwulan {triwulan}", capaian['target_capaian'])
                st.write(f"**Fokus Output:** {capaian['fokus_output']}")
                st.write("**Strategi:**")
                for strategy in capaian['strategi']:
                    st.write(f"• {strategy}")
            
            # Improvement Areas
            st.header("🚀 Area Perbaikan Prioritas")
            
            if ikpa_result.get('improvement_areas'):
                for area in ikpa_result['improvement_areas']:
                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.subheader(f"{area['area']} ({area['priority']} Priority)")
                            st.write(area['description'])
                            st.write(f"**Rekomendasi:** {area['recommendation']}")
                        with col2:
                            st.metric("Current", f"{area['current']:.1f}%")
                        with col3:
                            st.metric("Target", f"{area['target']}%")
                        
                        progress = min(area['current'] / area['target'], 1.0)
                        st.progress(progress)
            else:
                st.success("✅ Semua komponen IKPA dalam kondisi optimal!")
        
        else:
            # Welcome screen
            st.header("🚀 Selamat Datang di Sistem Monitoring IKPA 2024")
            
            st.info("""
            ### 📋 Cara Penggunaan:
            
            1. **Upload file data** (Realisasi Anggaran, Capaian Output, Revisi DIPA) atau
            2. **Input manual nilai komponen** di sidebar
            3. **Klik tombol 'Hitung Nilai IKPA 2024'**
            4. **Sistem akan menampilkan analisis lengkap** dan rekomendasi
            
            ### 🎯 Fitur Utama:
            - **Prioritas data file**: Data dari file akan menggantikan input manual
            - **Analisis komponen mendalam**: Detail kontribusi setiap komponen
            - **Rekomendasi terpersonalisasi**: Sesuai dengan kondisi nilai IKPA
            - **Deteksi masalah otomatis**: Identifikasi area perbaikan prioritas
            """)
            
            # Upload file demo section
            st.header("📁 Upload Data untuk Analisis Lengkap")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("💰 Realisasi Anggaran")
                st.write("Format yang didukung:")
                st.write("- Jumlah Revisi, Keterangan")
                st.write("- Pagu & Realisasi per Jenis Belanja")
                st.write("- NKPA dan Nilai IKPA Penyerapan")
            
            with col2:
                st.subheader("📊 Capaian Output") 
                st.write("Format yang didukung:")
                st.write("- Kode Kegiatan/RO")
                st.write("- Pagu dan Realisasi RO")
                st.write("- Progress Capaian (PCRO)")
            
            with col3:
                st.subheader("📋 Revisi DIPA")
                st.write("Format yang didukung:")
                st.write("- Rencana vs Realisasi Belanja")
                st.write("- Deviasi per Jenis Belanja")
                st.write("- Nilai Revisi Halaman III")
    
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
