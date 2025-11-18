import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pdfplumber
import re
import io
import requests
import json

# Suppress specific warnings
import logging
logging.getLogger().setLevel(logging.ERROR)

# Konfigurasi halaman
st.set_page_config(
    page_title="SMART BNNP DKI",
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
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin-bottom: 1rem;
    }
    .ai-response {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DeepSeekAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_recommendation(self, prompt, max_tokens=1500):
        """Generate recommendations using DeepSeek API"""
        try:
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "Anda adalah ahli monitoring dan evaluasi kinerja pemerintah khususnya di bidang pengelolaan anggaran. Berikan rekomendasi yang praktis, terstruktur, dan dapat ditindaklanjuti."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error calling DeepSeek API: {str(e)}"

class AdvancedBudgetProcessor:
    def __init__(self):
        self.penyerapan_data = None
        self.capaian_output_data = None
        self.recommendations = None
    
    def process_detailed_budget(self, file_buffer):
        """Process detailed budget Excel file with the specified format"""
        try:
            # Read the detailed budget sheet
            df = pd.read_excel(file_buffer, sheet_name='Rincian_Anggaran')
            
            # Validate required columns
            required_columns = ['Kode', 'Uraian Volume', 'Satuan', 'Harga Satuan', 
                              'Jumlah', 'Realisasi', 'Sisa Anggaran', 'Bidang', 'Triwulan']
            
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                return {"error": f"Kolom yang hilang: {missing}"}
            
            # Calculate additional metrics
            df['Penyerapan_Persen'] = (df['Realisasi'] / df['Jumlah']) * 100
            df['Penyerapan_Persen'] = df['Penyerapan_Persen'].replace([np.inf, -np.inf], 0)
            df['Penyerapan_Persen'] = df['Penyerapan_Persen'].fillna(0)
            
            # Group by bidang for summary
            summary_by_bidang = df.groupby('Bidang').agg({
                'Jumlah': 'sum',
                'Realisasi': 'sum',
                'Sisa Anggaran': 'sum'
            }).reset_index()
            
            summary_by_bidang['Penyerapan_Persen'] = (
                summary_by_bidang['Realisasi'] / summary_by_bidang['Jumlah']
            ) * 100
            
            # Group by triwulan and bidang
            triwulan_bidang = df.groupby(['Triwulan', 'Bidang']).agg({
                'Jumlah': 'sum',
                'Realisasi': 'sum'
            }).reset_index()
            triwulan_bidang['Penyerapan_Persen'] = (
                triwulan_bidang['Realisasi'] / triwulan_bidang['Jumlah']
            ) * 100
            
            # Calculate overall metrics
            total_alokasi = df['Jumlah'].sum()
            total_realisasi = df['Realisasi'].sum()
            total_sisa = df['Sisa Anggaran'].sum()
            rata_penyerapan = (total_realisasi / total_alokasi) * 100 if total_alokasi > 0 else 0
            
            # Calculate RPD deviation (simplified)
            deviasi_rpd_rata = df['Penyerapan_Persen'].std() / 10
            
            self.penyerapan_data = {
                'raw_data': df,
                'summary_by_bidang': summary_by_bidang,
                'triwulan_bidang': triwulan_bidang,
                'total_alokasi': total_alokasi,
                'total_realisasi': total_realisasi,
                'total_sisa': total_sisa,
                'rata_penyerapan': rata_penyerapan,
                'deviasi_rpd_rata': deviasi_rpd_rata,
                'timestamp': datetime.now()
            }
            
            # Generate recommendations
            self.generate_recommendations()
            
            return self.penyerapan_data
            
        except Exception as e:
            return {"error": f"Error processing budget file: {str(e)}"}
    
    def generate_recommendations(self):
        """Generate spending recommendations for each bidang"""
        if not self.penyerapan_data:
            return
        
        df = self.penyerapan_data['raw_data']
        current_quarter = 3  # Asumsi triwulan 3
        target_penyerapan = {1: 15, 2: 50, 3: 70, 4: 90}
        
        recommendations = {}
        
        for bidang in df['Bidang'].unique():
            bidang_data = df[df['Bidang'] == bidang]
            total_alokasi = bidang_data['Jumlah'].sum()
            total_realisasi = bidang_data['Realisasi'].sum()
            penyerapan_sekarang = (total_realisasi / total_alokasi) * 100
            
            # Items with remaining budget
            sisa_items = bidang_data[bidang_data['Sisa Anggaran'] > 0]
            
            # Calculate required acceleration
            target_now = target_penyerapan[current_quarter]
            gap = target_now - penyerapan_sekarang
            required_spending = (gap / 100) * total_alokasi if gap > 0 else 0
            
            # Priority items for acceleration
            priority_items = sisa_items.nlargest(5, 'Sisa Anggaran')
            
            recommendations[bidang] = {
                'penyerapan_sekarang': penyerapan_sekarang,
                'target_triwulan': target_now,
                'gap': gap,
                'required_spending': required_spending,
                'sisa_anggaran': sisa_items['Sisa Anggaran'].sum(),
                'priority_items': priority_items[['Kode', 'Uraian Volume', 'Sisa Anggaran']].to_dict('records'),
                'rekomendasi_aksi': self.generate_action_plan(bidang, gap, required_spending, priority_items)
            }
        
        self.recommendations = recommendations
        return recommendations
    
    def generate_action_plan(self, bidang, gap, required_spending, priority_items):
        """Generate specific action plans for each bidang"""
        actions = []
        
        if gap > 0:
            actions.append({
                'priority': 'HIGH',
                'action': f"Percepat penyerapan sebesar Rp {required_spending:,.0f}",
                'deadline': "Akhir Triwulan",
                'target_peningkatan': f"{gap:.1f}%"
            })
            
            if len(priority_items) > 0:
                actions.append({
                    'priority': 'HIGH',
                    'action': "Fokus pada item dengan sisa anggaran terbesar",
                    'items': [f"{item['Kode']} - {item['Uraian Volume']} (Rp {item['Sisa Anggaran']:,.0f})" 
                             for item in priority_items],
                    'deadline': "2 minggu"
                })
        
        # Bidang-specific recommendations
        if bidang == "Bagian Umum":
            actions.append({
                'priority': 'MEDIUM',
                'action': "Akselerasi belanja modal dan pengadaan barang",
                'deadline': "4 minggu",
                'catatan': "Perhatikan proses pengadaan yang membutuhkan waktu panjang"
            })
        elif bidang in ["Pemberantasan", "Rehabilitasi", "P2M"]:
            actions.append({
                'priority': 'MEDIUM', 
                'action': "Optimalkan belanja barang operasional",
                'deadline': "3 minggu",
                'catatan': "Koordinasi dengan Bagian Umum untuk percepatan proses"
            })
        
        return actions

class SatuanKerjaIKPACalculator:
    def __init__(self):
        self.satuan_kerja_data = None
        self.bidang_data = {}
    
    def calculate_satuan_kerja_ikpa(self, all_bidang_data, capaian_output_satker):
        """Calculate IKPA for entire satuan kerja"""
        # Aggregate data from all bidang
        total_alokasi = sum(data['total_alokasi'] for data in all_bidang_data.values())
        total_realisasi = sum(data['total_realisasi'] for data in all_bidang_data.values())
        
        # Calculate weighted averages for each indicator
        aggregated_metrics = self.aggregate_metrics(all_bidang_data)
        
        # Calculate IKPA for satuan kerja
        ikpa_data = {
            'revisi_dipa': 100.00,
            'deviasi_halaman_iii': aggregated_metrics['avg_deviasi_rpd'],
            'penyerapan_anggaran': aggregated_metrics['avg_penyerapan'],
            'belanja_kontraktual': 0.00,
            'penyelesaian_tagihan': 0.00,
            'pengelolaan_up_tup': 100.00,
            'capaian_output': capaian_output_satker,
            'dispensasi_spm': 0.00
        }
        
        hasil_ikpa = self.calculate_ikpa_actual(ikpa_data)
        
        self.satuan_kerja_data = {
            'ikpa': hasil_ikpa,
            'aggregated_metrics': aggregated_metrics,
            'total_alokasi': total_alokasi,
            'total_realisasi': total_realisasi,
            'total_sisa': total_alokasi - total_realisasi,
            'timestamp': datetime.now()
        }
        
        return self.satuan_kerja_data
    
    def aggregate_metrics(self, all_bidang_data):
        """Aggregate metrics from all bidang"""
        total_weight = 0
        weighted_penyerapan = 0
        weighted_deviasi = 0
        
        for bidang, data in all_bidang_data.items():
            weight = data['total_alokasi']
            total_weight += weight
            weighted_penyerapan += data['rata_penyerapan'] * weight
            weighted_deviasi += data.get('deviasi_rpd_rata', 0) * weight
        
        return {
            'avg_penyerapan': weighted_penyerapan / total_weight if total_weight > 0 else 0,
            'avg_deviasi_rpd': max(0, 100 - (weighted_deviasi / total_weight)) if total_weight > 0 else 100,
            'total_bidang': len(all_bidang_data)
        }
    
    def calculate_ikpa_actual(self, data):
        """Calculate actual IKPA score"""
        nilai_total = (
            data['revisi_dipa'] * 10 +
            data['deviasi_halaman_iii'] * 15 +
            data['penyerapan_anggaran'] * 20 +
            data['belanja_kontraktual'] * 0 +
            data['penyelesaian_tagihan'] * 0 +
            data['pengelolaan_up_tup'] * 10 +
            data['capaian_output'] * 25
        ) / 100
        
        nilai_konversi = (nilai_total / 80) * 100
        nilai_akhir = nilai_konversi - data['dispensasi_spm']
        
        if nilai_akhir >= 95:
            kategori = "Sangat Baik"
        elif nilai_akhir >= 89:
            kategori = "Baik"
        elif nilai_akhir >= 70:
            kategori = "Cukup"
        else:
            kategori = "Kurang"
        
        return {
            'nilai_total': nilai_total,
            'nilai_konversi': nilai_konversi,
            'nilai_akhir': nilai_akhir,
            'kategori': kategori
        }

# Visualization Functions
def create_ikpa_radar_chart(ikpa_data):
    """Create radar chart for IKPA components"""
    
    categories = ['Revisi DIPA', 'Deviasi RPD', 'Penyerapan', 'UP/TUP', 'Capaian Output']
    values = [
        ikpa_data.get('revisi_dipa', 0),
        ikpa_data.get('deviasi_halaman_iii', 0),
        ikpa_data.get('penyerapan_anggaran', 0),
        ikpa_data.get('pengelolaan_up_tup', 0),
        ikpa_data.get('capaian_output', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='IKPA Components',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Radar Chart Komponen IKPA",
        height=400
    )
    
    return fig

def create_ikpa_gauge_chart(current_value, category):
    """Create gauge chart for IKPA score"""
    
    if category == "Sangat Baik":
        color = "green"
    elif category == "Baik":
        color = "blue" 
    elif category == "Cukup":
        color = "orange"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = current_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"IKPA Score - {category}"},
        delta = {'reference': 95, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
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
    
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏢 SMART - Sistem Monitoring dan Evaluasi Kinerja Terpadu</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Badan Narkotika Nasional Provinsi DKI Jakarta</h3>', unsafe_allow_html=True)
    
    # Initialize processors
    processor = AdvancedBudgetProcessor()
    satuan_kerja_calculator = SatuanKerjaIKPACalculator()
    
    # DeepSeek API Configuration
    st.sidebar.header("🔑 Konfigurasi DeepSeek API")
    
    api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="Masukkan API Key DeepSeek Anda...",
        help="Dapatkan API key dari https://platform.deepseek.com/"
    )
    
    # Initialize DeepSeek API
    deepseek_api = None
    if api_key:
        deepseek_api = DeepSeekAPI(api_key)
        st.sidebar.success("✅ DeepSeek API terkoneksi")
    else:
        st.sidebar.warning("⚠️ Masukkan DeepSeek API Key untuk menggunakan fitur AI")
    
    # Sidebar for file uploads
    st.sidebar.header("📁 Upload Files")
    
    with st.sidebar.expander("📊 Laporan Penyerapan Anggaran", expanded=True):
        budget_file = st.sidebar.file_uploader(
            "Upload Excel Penyerapan Anggaran", 
            type=['xlsx', 'xls'],
            help="Format: Kode, Uraian Volume, Satuan, Harga Satuan, Jumlah, Realisasi, Sisa Anggaran, Bidang, Triwulan"
        )
    
    with st.sidebar.expander("📈 Capaian Output"):
        pdf_file = st.sidebar.file_uploader(
            "Upload PDF Capaian Output", 
            type=['pdf'],
            help="Laporan capaian output dalam format PDF"
        )
    
    # Process files
    capaian_output_data = None
    
    if budget_file is not None:
        with st.spinner("Memproses laporan penyerapan anggaran..."):
            budget_results = processor.process_detailed_budget(budget_file)
            
        if "error" not in budget_results:
            st.sidebar.success("✅ Data penyerapan berhasil diproses")
            
            # Display budget dashboard
            st.header("📊 Dashboard Penyerapan Anggaran")
            
            data = processor.penyerapan_data
            
            # Key Metrics
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
                fig1 = px.bar(data['summary_by_bidang'], 
                             x='Bidang', y='Penyerapan_Persen',
                             title="Penyerapan per Bidang",
                             color='Penyerapan_Persen',
                             color_continuous_scale='RdYlGn')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.pie(data['summary_by_bidang'], 
                             values='Realisasi', names='Bidang',
                             title='Distribusi Realisasi per Bidang')
                st.plotly_chart(fig2, use_container_width=True)
            
            # Recommendations per bidang
            if processor.recommendations:
                st.header("🎯 Rekomendasi per Bidang")
                for bidang, rec in processor.recommendations.items():
                    with st.expander(f"{bidang} - Penyerapan: {rec['penyerapan_sekarang']:.1f}%", expanded=True):
                        st.metric("Gap", f"{rec['gap']:.1f}%")
                        st.metric("Dibutuhkan", f"Rp {rec['required_spending']:,.0f}")
                        
                        for action in rec['rekomendasi_aksi']:
                            st.write(f"**{action['action']}**")
                            st.write(f"Deadline: {action['deadline']}")
        else:
            st.error(f"❌ {budget_results['error']}")
    
    if pdf_file is not None:
        with st.spinner("Memproses laporan capaian output..."):
            # Process PDF
            try:
                text = ""
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                
                # Extract percentage
                patterns = [r'capaian\s*output\s*:?\s*(\d+\.?\d*)%', r'persentase\s*capaian\s*:?\s*(\d+\.?\d*)%']
                persentase = 0
                for pattern in patterns:
                    matches = re.findall(pattern, text.lower())
                    if matches:
                        persentase = float(matches[0])
                        break
                
                capaian_output_data = {
                    'persentase_capaian': persentase,
                    'text_analysis': text[:1000]
                }
                
                st.sidebar.success("✅ Data capaian output berhasil diproses")
                
                # Display capaian output
                st.header("🎯 Dashboard Capaian Output")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Persentase Capaian", f"{persentase}%")
                with col2:
                    status = "✅ Optimal" if persentase >= 80 else "⚠️ Perlu Perbaikan"
                    st.metric("Status", status)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    # Calculate IKPA if both files are available
    if (processor.penyerapan_data is not None and 
        capaian_output_data is not None and
        deepseek_api is not None):
        
        st.header("🧮 Perhitungan IKPA Terintegrasi")
        
        # Calculate IKPA
        all_bidang_data = {'overall': processor.penyerapan_data}
        satuan_kerja_results = satuan_kerja_calculator.calculate_satuan_kerja_ikpa(
            all_bidang_data,
            capaian_output_data['persentase_capaian']
        )
        
        data_satker = satuan_kerja_results
        
        # Display IKPA Results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nilai IKPA", f"{data_satker['ikpa']['nilai_akhir']:.2f}")
        with col2:
            st.metric("Kategori", data_satker['ikpa']['kategori'])
        with col3:
            gap = 95 - data_satker['ikpa']['nilai_akhir']
            st.metric("Selisih Target", f"{gap:+.2f}")
        
        # IKPA Visualization
        col1, col2 = st.columns(2)
        with col1:
            fig_gauge = create_ikpa_gauge_chart(
                data_satker['ikpa']['nilai_akhir'], 
                data_satker['ikpa']['kategori']
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            fig_radar = create_ikpa_radar_chart(data_satker['ikpa'])
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # AI Recommendations
        st.header("🤖 Rekomendasi AI DeepSeek")
        
        if st.button("🔄 Generate Rekomendasi AI", type="primary"):
            with st.spinner("DeepSeek AI sedang menganalisis..."):
                prompt = f"""
                ANALISIS KINERJA BNNP DKI JAKARTA:

                DATA:
                - IKPA: {data_satker['ikpa']['nilai_akhir']:.2f} ({data_satker['ikpa']['kategori']})
                - Penyerapan: {data_satker['aggregated_metrics']['avg_penyerapan']:.1f}%
                - Capaian Output: {capaian_output_data['persentase_capaian']}%
                - Alokasi: Rp {data_satker['total_alokasi']:,.0f}
                - Realisasi: Rp {data_satker['total_realisasi']:,.0f}

                TARGET: Mencapai IKPA ≥95 (Sangat Baik)

                BERIKAN REKOMENDASI STRATEGIS YANG:
                1. SPESIFIK dan dapat ditindaklanjuti
                2. TERSTRUKTUR dengan prioritas jelas
                3. REALISTIS untuk triwulan 3-4
                4. TERUKUR dengan target kuantitatif

                FORMAT: Analisis, Rekomendasi Utama, Action Plan, Timeline
                """
                
                ai_response = deepseek_api.generate_recommendation(prompt)
                
                st.markdown("### 📋 Rekomendasi DeepSeek AI")
                st.markdown(f'<div class="ai-response">{ai_response}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
