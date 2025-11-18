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
    page_title="SMART BNNP DKI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedBudgetProcessor:
    def __init__(self):
        self.penyerapan_data = None
        self.recommendations = None
    
    def process_detailed_budget(self, file_buffer):
        """Process budget Excel file dengan handling worksheet yang fleksibel"""
        try:
            # Baca semua sheet names untuk debug
            excel_file = pd.ExcelFile(file_buffer)
            sheet_names = excel_file.sheet_names
            st.sidebar.info(f"Sheet yang tersedia: {', '.join(sheet_names)}")
            
            # Coba berbagai nama worksheet yang mungkin
            possible_sheet_names = [
                'Rincian_Anggaran', 
                'Rincian Anggaran',
                'Data',
                'Sheet1',
                'Laporan',
                sheet_names[0]  # Sheet pertama
            ]
            
            df = None
            used_sheet = None
            
            for sheet_name in possible_sheet_names:
                try:
                    if sheet_name in sheet_names:
                        df = pd.read_excel(file_buffer, sheet_name=sheet_name)
                        used_sheet = sheet_name
                        st.sidebar.success(f"✅ Menggunakan sheet: {sheet_name}")
                        break
                except:
                    continue
            
            if df is None:
                return {"error": f"Tidak dapat membaca file Excel. Sheet yang tersedia: {', '.join(sheet_names)}"}
            
            # Cek kolom yang diperlukan - lebih fleksibel
            required_columns_mapping = {
                'Kode': ['Kode', 'KODE', 'Kode Rekening', 'KODE REKENING'],
                'Uraian Volume': ['Uraian Volume', 'Uraian', 'Kegiatan', 'URAIAN'],
                'Satuan': ['Satuan', 'SATUAN', 'Unit'],
                'Harga Satuan': ['Harga Satuan', 'Harga', 'HARGA SATUAN', 'Harga Per Satuan'],
                'Jumlah': ['Jumlah', 'Jumlah Anggaran', 'Anggaran', 'JUMLAH'],
                'Realisasi': ['Realisasi', 'Realisasi Anggaran', 'REALISASI'],
                'Sisa Anggaran': ['Sisa Anggaran', 'Sisa', 'SISA ANGGARAN'],
                'Bidang': ['Bidang', 'BIDANG', 'Unit Kerja'],
                'Triwulan': ['Triwulan', 'TRIWULAN', 'Periode', 'TW']
            }
            
            # Mapping kolom aktual ke nama standar
            column_mapping = {}
            for standard_name, possible_names in required_columns_mapping.items():
                for possible_name in possible_names:
                    if possible_name in df.columns:
                        column_mapping[standard_name] = possible_name
                        break
            
            # Jika tidak ada kolom yang cocok, gunakan kolom yang ada
            if not column_mapping:
                st.warning("Format kolom tidak standar. Menggunakan kolom yang tersedia.")
                # Coba tebak berdasarkan posisi atau tipe data
                for i, col in enumerate(df.columns):
                    if i == 0: column_mapping['Kode'] = col
                    elif i == 1: column_mapping['Uraian Volume'] = col
                    elif i == 2: column_mapping['Satuan'] = col
                    elif i == 3: column_mapping['Harga Satuan'] = col
                    elif i == 4: column_mapping['Jumlah'] = col
                    elif i == 5: column_mapping['Realisasi'] = col
                    elif i == 6: column_mapping['Sisa Anggaran'] = col
                    elif i == 7: column_mapping['Bidang'] = col
                    elif i == 8: column_mapping['Triwulan'] = col
            
            # Rename kolom ke nama standar
            df = df.rename(columns={v: k for k, v in column_mapping.items()})
            
            # Tambahkan kolom default jika tidak ada
            if 'Bidang' not in df.columns:
                df['Bidang'] = 'Umum'
            if 'Triwulan' not in df.columns:
                df['Triwulan'] = 3  # Default triwulan 3
            
            # Pastikan kolom numerik
            numeric_columns = ['Harga Satuan', 'Jumlah', 'Realisasi', 'Sisa Anggaran']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Hitung Sisa Anggaran jika tidak ada
            if 'Sisa Anggaran' not in df.columns and 'Jumlah' in df.columns and 'Realisasi' in df.columns:
                df['Sisa Anggaran'] = df['Jumlah'] - df['Realisasi']
            
            # Hitung persentase penyerapan
            if 'Jumlah' in df.columns and 'Realisasi' in df.columns:
                df['Penyerapan_Persen'] = (df['Realisasi'] / df['Jumlah'].replace(0, np.nan)) * 100
                df['Penyerapan_Persen'] = df['Penyerapan_Persen'].replace([np.inf, -np.inf], 0).fillna(0)
            else:
                df['Penyerapan_Persen'] = 0
            
            # Group by bidang untuk summary
            if 'Bidang' in df.columns:
                summary_by_bidang = df.groupby('Bidang').agg({
                    'Jumlah': 'sum',
                    'Realisasi': 'sum',
                    'Sisa Anggaran': 'sum'
                }).reset_index()
                summary_by_bidang['Penyerapan_Persen'] = (
                    summary_by_bidang['Realisasi'] / summary_by_bidang['Jumlah'].replace(0, np.nan)
                ) * 100
                summary_by_bidang['Penyerapan_Persen'] = summary_by_bidang['Penyerapan_Persen'].fillna(0)
            else:
                summary_by_bidang = pd.DataFrame({
                    'Bidang': ['Umum'],
                    'Jumlah': [df['Jumlah'].sum()],
                    'Realisasi': [df['Realisasi'].sum()],
                    'Sisa Anggaran': [df['Sisa Anggaran'].sum()],
                    'Penyerapan_Persen': [df['Penyerapan_Persen'].mean()]
                })
            
            # Calculate overall metrics
            total_alokasi = df['Jumlah'].sum()
            total_realisasi = df['Realisasi'].sum()
            total_sisa = df['Sisa Anggaran'].sum()
            rata_penyerapan = (total_realisasi / total_alokasi) * 100 if total_alokasi > 0 else 0
            
            self.penyerapan_data = {
                'raw_data': df,
                'summary_by_bidang': summary_by_bidang,
                'total_alokasi': total_alokasi,
                'total_realisasi': total_realisasi,
                'total_sisa': total_sisa,
                'rata_penyerapan': rata_penyerapan,
                'used_sheet': used_sheet,
                'timestamp': datetime.now()
            }
            
            return self.penyerapan_data
            
        except Exception as e:
            return {"error": f"Error processing budget file: {str(e)}"}

def main():
    st.title("🏢 SMART - Sistem Monitoring dan Evaluasi Kinerja Terpadu")
    st.subheader("Badan Narkotika Nasional Provinsi DKI Jakarta")
    
    # Initialize processor
    processor = AdvancedBudgetProcessor()
    
    # Sidebar
    st.sidebar.header("📁 Upload Files")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Laporan Penyerapan Anggaran (Excel)", 
        type=['xlsx', 'xls'],
        help="Upload file Excel laporan penyerapan anggaran"
    )
    
    # Template download
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Template")
    
    # Create sample template
    sample_data = {
        'Kode': ['5.1.01.01.01', '5.1.02.01.01', '5.2.01.01.01'],
        'Uraian Volume': ['Pengadaan ATK', 'Bimbingan Teknis', 'Kendaraan Dinas'],
        'Satuan': ['Paket', 'Kegiatan', 'Unit'],
        'Harga Satuan': [50000000, 100000000, 300000000],
        'Jumlah': [50000000, 100000000, 300000000],
        'Realisasi': [45000000, 75000000, 0],
        'Sisa Anggaran': [5000000, 25000000, 300000000],
        'Bidang': ['Pemberantasan', 'P2M', 'Bagian Umum'],
        'Triwulan': [3, 3, 3]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Download template
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df_to_csv(sample_df)
    
    st.sidebar.download_button(
        label="📥 Download Template Excel",
        data=csv,
        file_name="template_penyerapan_anggaran.csv",
        mime="text/csv",
        help="Download template untuk input data"
    )
    
    # Process uploaded file
    if uploaded_file:
        with st.spinner("Memproses file Excel..."):
            result = processor.process_detailed_budget(uploaded_file)
            
        if "error" in result:
            st.error(f"❌ {result['error']}")
            
            st.info("""
            **Format Excel yang Diperlukan:**
            - File Excel dengan data penyerapan anggaran
            - Kolom yang diperlukan: Kode, Uraian Volume, Satuan, Harga Satuan, Jumlah, Realisasi, Sisa Anggaran
            - Kolom opsional: Bidang, Triwulan
            
            **Tips:**
            - Download template di sidebar untuk panduan
            - Pastikan data numerik (tanpa karakter selain angka)
            - Nama worksheet bisa apa saja, sistem akan otomatis mendeteksi
            """)
        else:
            st.success("✅ File berhasil diproses!")
            
            data = processor.penyerapan_data
            
            # Display metrics
            st.header("📊 Dashboard Penyerapan Anggaran")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Alokasi", f"Rp {data['total_alokasi']:,.0f}")
            with col2:
                st.metric("Total Realisasi", f"Rp {data['total_realisasi']:,.0f}")
            with col3:
                st.metric("Sisa Anggaran", f"Rp {data['total_sisa']:,.0f}")
            with col4:
                st.metric("Rata-rata Penyerapan", f"{data['rata_penyerapan']:.1f}%")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                if len(data['summary_by_bidang']) > 0:
                    fig1 = px.bar(
                        data['summary_by_bidang'], 
                        x='Bidang', 
                        y='Penyerapan_Persen',
                        title="Penyerapan per Bidang",
                        color='Penyerapan_Persen',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.info("Tidak ada data per bidang")
            
            with col2:
                if len(data['summary_by_bidang']) > 0:
                    fig2 = px.pie(
                        data['summary_by_bidang'], 
                        values='Realisasi', 
                        names='Bidang',
                        title='Distribusi Realisasi per Bidang'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Raw data
            with st.expander("📋 Lihat Data Mentah"):
                st.dataframe(data['raw_data'], use_container_width=True)
            
            # Analysis
            st.header("🎯 Analisis Kinerja")
            
            if data['rata_penyerapan'] < 70:
                st.error("**⚠️ PERINGATAN:** Penyerapan anggaran di bawah target 70% untuk Triwulan 3")
                st.write("**Rekomendasi:** Percepat penyerapan anggaran dengan fokus pada item-item prioritas")
            elif data['rata_penyerapan'] < 85:
                st.warning("**📊 CATATAN:** Penyerapan anggaran mendekati target namun perlu optimasi")
                st.write("**Rekomendasi:** Tingkatkan koordinasi antar bidang untuk percepatan penyerapan")
            else:
                st.success("**✅ OPTIMAL:** Penyerapan anggaran sesuai atau di atas target")
                st.write("**Rekomendasi:** Pertahankan kinerja dan fokus pada kualitas output")
    
    else:
        # Default dashboard when no file uploaded
        st.header("🚀 Selamat Datang di SMART System")
        
        st.markdown("""
        ### 📋 Tentang Sistem SMART
        
        **S**istem **M**onitoring dan **A**nalisis **R**eal-time **T**erpadu
        
        Sistem ini membantu BNNP DKI Jakarta dalam:
        - 📊 **Monitoring real-time** penyerapan anggaran
        - 🎯 **Analisis kinerja** per bidang
        - 🤖 **Rekomendasi AI** untuk optimalisasi IKPA
        - 📈 **Forecasting** kinerja hingga akhir tahun
        
        ### 🚀 Cara Menggunakan:
        1. **Upload file Excel** laporan penyerapan anggaran di sidebar
        2. **Sistem akan otomatis** menganalisis data
        3. **Lihat dashboard** dengan visualisasi interaktif
        4. **Dapatkan rekomendasi** untuk peningkatan kinerja
        
        ### 📁 Format Data yang Didukung:
        - File Excel (.xlsx, .xls) dengan data penyerapan anggaran
        - Kolom fleksibel - sistem akan otomatis menyesuaikan
        - Download template di sidebar untuk panduan
        
        ### 🎯 Target Kinerja:
        - **IKPA ≥95** (Sangat Baik)
        - **Penyerapan anggaran** sesuai target triwulanan
        - **Capaian output** optimal
        """)
        
        # Sample visualization for demo
        st.header("📊 Demo Visualisasi")
        
        # Create sample data for demo
        demo_data = pd.DataFrame({
            'Bidang': ['Pemberantasan', 'Rehabilitasi', 'P2M', 'Bagian Umum'],
            'Penyerapan': [80, 65, 75, 70],
            'Target': [70, 70, 70, 70]
        })
        
        fig_demo = px.bar(
            demo_data, 
            x='Bidang', 
            y=['Penyerapan', 'Target'],
            title='Contoh Visualisasi Penyerapan per Bidang',
            barmode='group'
        )
        st.plotly_chart(fig_demo, use_container_width=True)

if __name__ == "__main__":
    main()
