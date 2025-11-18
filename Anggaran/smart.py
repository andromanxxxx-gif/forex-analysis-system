import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pdfplumber
import re
import io
import requests
import json

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
    
    def process_detailed_budget(self, file_path):
        """Process detailed budget Excel file with the specified format"""
        try:
            # Read the detailed budget sheet
            df = pd.read_excel(file_path, sheet_name='Rincian_Anggaran')
            
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
            sisa_per_triwulan = sisa_items.groupby('Triwulan')['Sisa Anggaran'].sum()
            
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
        self.ikpa_trend = []
    
    def calculate_satuan_kerja_ikpa(self, all_bidang_data, capaian_output_satker):
        """Calculate IKPA for entire satuan kerja"""
        # Aggregate data from all bidang
        total_alokasi = sum(data['total_alokasi'] for data in all_bidang_data.values())
        total_realisasi = sum(data['total_realisasi'] for data in all_bidang_data.values())
        
        # Calculate weighted averages for each indicator
        aggregated_metrics = self.aggregate_metrics(all_bidang_data)
        
        # Calculate IKPA for satuan kerja
        ikpa_data = {
            'revisi_dipa': 100.00,  # Asumsi optimal untuk satker
            'deviasi_halaman_iii': aggregated_metrics['avg_deviasi_rpd'],
            'penyerapan_anggaran': aggregated_metrics['avg_penyerapan'],
            'belanja_kontraktual': 0.00,
            'penyelesaian_tagihan': 0.00,
            'pengelolaan_up_tup': 100.00,  # Asumsi optimal
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

def create_penyerapan_pie_chart(penyerapan_data):
    """Create pie chart for budget absorption by bidang"""
    
    labels = penyerapan_data['summary_by_bidang']['Bidang'].tolist()
    values = penyerapan_data['summary_by_bidang']['Realisasi'].tolist()
    
    fig = px.pie(
        values=values, 
        names=labels,
        title='Distribusi Realisasi Anggaran per Bidang',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    
    fig.update_layout(height=400)
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

def display_satuan_kerja_dashboard(processor, satuan_kerja_calculator, capaian_output_data, deepseek_api):
    """Display comprehensive dashboard for entire satuan kerja"""
    
    st.header("🏢 DASHBOARD SATUAN KERJA - BNN PROVINSI DKI JAKARTA")
    
    if not satuan_kerja_calculator.satuan_kerja_data:
        st.warning("Data satuan kerja belum tersedia")
        return
    
    data_satker = satuan_kerja_calculator.satuan_kerja_data
    
    # Key Performance Indicators for Satuan Kerja
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Nilai IKPA Satuan Kerja", 
            f"{data_satker['ikpa']['nilai_akhir']:.2f}",
            data_satker['ikpa']['kategori']
        )
    with col2:
        st.metric("Total Alokasi Anggaran", f"Rp {data_satker['total_alokasi']:,.0f}")
    with col3:
        st.metric("Total Realisasi", f"Rp {data_satker['total_realisasi']:,.0f}")
    with col4:
        st.metric("Rata-rata Penyerapan", f"{data_satker['aggregated_metrics']['avg_penyerapan']:.1f}%")
    
    # Tab untuk berbagai level analisis
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview Satuan Kerja", 
        "🔍 Analisis Per Bidang", 
        "📈 Trend & Forecasting",
        "🎯 Rekomendasi AI"
    ])
    
    with tab1:
        display_satuan_kerja_overview(data_satker, processor)
    
    with tab2:
        display_per_bidang_analysis(processor)
    
    with tab3:
        display_trend_forecasting(processor, satuan_kerja_calculator)
    
    with tab4:
        display_ai_recommendations(data_satker, processor, capaian_output_data, deepseek_api)

def display_satuan_kerja_overview(data_satker, processor):
    """Display overview charts for entire satuan kerja"""
    
    st.subheader("📊 Overview Kinerja Satuan Kerja")
    
    # Chart 1: IKPA Breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # IKPA Components Radar Chart
        fig_radar = create_ikpa_radar_chart(data_satker['ikpa'])
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Penyerapan per Jenis Belanja
        if processor.penyerapan_data:
            fig_pie = create_penyerapan_pie_chart(processor.penyerapan_data)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Chart 2: Performance vs Target
    fig_gauge = create_ikpa_gauge_chart(
        data_satker['ikpa']['nilai_akhir'], 
        data_satker['ikpa']['kategori']
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

def display_per_bidang_analysis(processor):
    """Display detailed analysis per bidang"""
    
    st.subheader("🔍 Analisis Detail Per Bidang")
    
    if not processor.penyerapan_data:
        st.warning("Data bidang belum tersedia")
        return
    
    data = processor.penyerapan_data
    
    # Chart 1: Perbandingan Penyerapan per Bidang
    fig_comparison = px.bar(
        data['summary_by_bidang'], 
        x='Bidang', 
        y='Penyerapan_Persen',
        title='Perbandingan Penyerapan Anggaran per Bidang',
        color='Penyerapan_Persen',
        color_continuous_scale='RdYlGn',
        text='Penyerapan_Persen'
    )
    fig_comparison.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Chart 2: Trend Triwulanan per Bidang
    col1, col2 = st.columns(2)
    
    with col1:
        fig_trend = px.line(
            data['triwulan_bidang'],
            x='Triwulan',
            y='Penyerapan_Persen',
            color='Bidang',
            title='Trend Penyerapan per Triwulan',
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Alokasi vs Realisasi per Bidang
        fig_bubble = px.scatter(
            data['summary_by_bidang'],
            x='Jumlah',
            y='Penyerapan_Persen',
            size='Realisasi',
            color='Bidang',
            title='Alokasi vs Penyerapan per Bidang',
            hover_data=['Bidang', 'Penyerapan_Persen']
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Detailed Table per Bidang
    with st.expander("📋 Detail Data per Bidang"):
        st.dataframe(data['summary_by_bidang'], use_container_width=True)

def display_trend_forecasting(processor, satuan_kerja_calculator):
    """Display trend analysis and forecasting"""
    
    st.subheader("📈 Analisis Trend & Forecasting")
    
    if not processor.penyerapan_data:
        return
    
    # Forecasting untuk akhir tahun
    current_data = processor.penyerapan_data
    current_penyerapan = current_data['rata_penyerapan']
    current_triwulan = 3  # Asumsi triwulan 3
    
    # Calculate projection
    months_remaining = (4 - current_triwulan) * 3
    if months_remaining > 0:
        monthly_rate = (100 - current_penyerapan) / months_remaining
        projected_penyerapan = min(100, current_penyerapan + (monthly_rate * months_remaining))
    else:
        projected_penyerapan = current_penyerapan
    
    # Projection Chart
    periods = ['Triwulan 1', 'Triwulan 2', 'Triwulan 3', 'Triwulan 4 (Proyeksi)']
    actual_penyerapan = [15, 50, current_penyerapan, projected_penyerapan]  # Example data
    target_penyerapan = [15, 50, 70, 90]
    
    fig_projection = go.Figure()
    
    fig_projection.add_trace(go.Scatter(
        x=periods, y=actual_penyerapan,
        mode='lines+markers',
        name='Realisasi',
        line=dict(color='blue', width=3)
    ))
    
    fig_projection.add_trace(go.Scatter(
        x=periods, y=target_penyerapan,
        mode='lines+markers',
        name='Target',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    fig_projection.update_layout(
        title='Proyeksi Penyerapan Anggaran hingga Akhir Tahun',
        xaxis_title='Periode',
        yaxis_title='Penyerapan (%)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_projection, use_container_width=True)
    
    # Risk Analysis
    st.subheader("⚠️ Analisis Risiko")
    
    risk_factors = analyze_risk_factors(processor, current_penyerapan, projected_penyerapan)
    
    for risk in risk_factors:
        with st.container():
            st.write(f"**{risk['factor']}** - {risk['level']}")
            st.progress(risk['severity'] / 100)
            st.write(f"*{risk['description']}*")
            st.write("---")

def analyze_risk_factors(processor, current_penyerapan, projected_penyerapan):
    """Analyze risk factors for IKPA achievement"""
    
    risks = []
    
    # Risk 1: Penyerapan rendah
    if current_penyerapan < 70:
        risks.append({
            'factor': 'Penyerapan Anggaran',
            'level': 'TINGGI',
            'severity': 80,
            'description': f'Penyerapan saat ini {current_penyerapan:.1f}% di bawah target 70% untuk triwulan 3'
        })
    
    # Risk 2: Deviasi RPD
    if processor.penyerapan_data and processor.penyerapan_data.get('deviasi_rpd_rata', 0) > 5:
        risks.append({
            'factor': 'Deviasi RPD',
            'level': 'SEDANG', 
            'severity': 60,
            'description': f'Deviasi RPD {processor.penyerapan_data["deviasi_rpd_rata"]:.1f}% melebihi batas 5%'
        })
    
    # Risk 3: Proyeksi tidak mencapai target
    if projected_penyerapan < 90:
        risks.append({
            'factor': 'Proyeksi Akhir Tahun',
            'level': 'TINGGI',
            'severity': 75,
            'description': f'Proyeksi penyerapan akhir tahun {projected_penyerapan:.1f}% di bawah target 90%'
        })
    
    return risks

def display_ai_recommendations(data_satker, processor, capaian_output_data, deepseek_api):
    """Display AI-powered recommendations using DeepSeek API"""
    
    st.subheader("🤖 Rekomendasi AI DeepSeek untuk Optimalisasi IKPA")
    
    # Current Status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("IKPA Saat Ini", f"{data_satker['ikpa']['nilai_akhir']:.2f}")
    with col2:
        st.metric("Target", "95.00", delta_color="off")
    with col3:
        gap = 95 - data_satker['ikpa']['nilai_akhir']
        st.metric("Gap", f"{gap:+.2f}")
    
    # Generate AI Recommendations
    if st.button("🔄 Generate Rekomendasi AI", type="primary"):
        with st.spinner("DeepSeek AI sedang menganalisis data dan menghasilkan rekomendasi..."):
            
            # Prepare prompt for DeepSeek
            prompt = f"""
            ANALISIS KINERJA DAN REKOMENDASI UNTUK BNN PROVINSI DKI JAKARTA:

            DATA KINERJA SATUAN KERJA:
            - Nilai IKPA: {data_satker['ikpa']['nilai_akhir']:.2f} ({data_satker['ikpa']['kategori']})
            - Rata-rata Penyerapan Anggaran: {data_satker['aggregated_metrics']['avg_penyerapan']:.1f}%
            - Total Alokasi: Rp {data_satker['total_alokasi']:,.0f}
            - Total Realisasi: Rp {data_satker['total_realisasi']:,.0f}
            - Sisa Anggaran: Rp {data_satker['total_sisa']:,.0f}
            - Capaian Output: {capaian_output_data['persentase_capaian']}%

            INDIKATOR KRITIS IKPA:
            1. Revisi DIPA: 100.00% (Optimal)
            2. Deviasi Halaman III: {data_satker['ikpa']['deviasi_halaman_iii']:.2f}%
            3. Penyerapan Anggaran: {data_satker['ikpa']['penyerapan_anggaran']:.2f}%
            4. Pengelolaan UP/TUP: 100.00% (Optimal)
            5. Capaian Output: {data_satker['ikpa']['capaian_output']:.2f}%

            TARGET: Mencapai IKPA ≥95 (Sangat Baik)

            BERDASARKAN DATA DI ATAS, BERIKAN REKOMENDASI STRATEGIS YANG:
            1. SPESIFIK: Rekomendasi konkret dan dapat ditindaklanjuti
            2. TERSTRUKTUR: Prioritaskan berdasarkan urgensi dan impact
            3. REALISTIS: Mempertimbangkan waktu triwulan 3-4
            4. TERUKUR: Dengan target kuantitatif yang jelas
            5. TERINTEGRASI: Koordinasi antar bidang/bagian

            FORMAT OUTPUT:
            - Analisis kondisi terkini
            - 3-5 rekomendasi strategis utama
            - Action plan per bidang
            - Timeline dan target
            - Monitoring dan evaluasi
            """

            # Call DeepSeek API
            ai_response = deepseek_api.generate_recommendation(prompt)
            
            # Display AI Response
            st.markdown('<div class="ai-response">', unsafe_allow_html=True)
            st.markdown("### 📋 Rekomendasi DeepSeek AI")
            st.markdown(ai_response)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save recommendation to session state
            st.session_state.ai_recommendation = ai_response

    # Display previous recommendation if exists
    if 'ai_recommendation' in st.session_state:
        st.markdown("---")
        st.subheader("📖 Rekomendasi Sebelumnya")
        st.markdown(st.session_state.ai_recommendation)

def display_budget_dashboard(processor):
    """Display comprehensive budget dashboard with recommendations"""
    
    st.header("📊 Dashboard Detail Penyerapan Anggaran")
    
    if not processor.penyerapan_data:
        st.error("Data penyerapan belum diproses")
        return
    
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
    
    # Visualization 1: Penyerapan per Bidang
    st.subheader("📈 Penyerapan Anggaran per Bidang")
    fig1 = px.bar(data['summary_by_bidang'], 
                 x='Bidang', y='Penyerapan_Persen',
                 title="Persentase Penyerapan per Bidang",
                 color='Penyerapan_Persen',
                 color_continuous_scale='RdYlGn')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Visualization 2: Trend Triwulanan
    st.subheader("📅 Trend Penyerapan per Triwulan")
    fig2 = px.line(data['triwulan_bidang'], 
                  x='Triwulan', y='Penyerapan_Persen',
                  color='Bidang',
                  title="Perkembangan Penyerapan per Triwulan")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed Data Table
    st.subheader("📋 Data Rinci Anggaran")
    st.dataframe(data['raw_data'], use_container_width=True)
    
    # Recommendations Section
    display_recommendations(processor)

def display_recommendations(processor):
    """Display spending recommendations for each bidang"""
    st.header("🎯 Rekomendasi Rencana Penyerapan per Bidang")
    
    if not processor.recommendations:
        st.warning("Belum ada rekomendasi yang dihasilkan")
        return
    
    for bidang, rec in processor.recommendations.items():
        with st.expander(f"📌 {bidang} - Penyerapan: {rec['penyerapan_sekarang']:.1f}% (Target: {rec['target_triwulan']}%)", expanded=True):
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gap Penyerapan", f"{rec['gap']:.1f}%")
            with col2:
                st.metric("Dibutuhkan Penyerapan", f"Rp {rec['required_spending']:,.0f}")
            with col3:
                st.metric("Sisa Anggaran", f"Rp {rec['sisa_anggaran']:,.0f}")
            
            # Action Plan
            st.subheader("📋 Rencana Aksi")
            for i, action in enumerate(rec['rekomendasi_aksi'], 1):
                priority_color = {
                    'HIGH': '🔴', 
                    'MEDIUM': '🟡', 
                    'LOW': '🟢'
                }
                
                with st.container():
                    st.write(f"**{priority_color[action['priority']]} Aksi {i}: {action['action']}**")
                    st.write(f"**Deadline:** {action['deadline']}")
                    if 'target_peningkatan' in action:
                        st.write(f"**Target Peningkatan:** {action['target_peningkatan']}")
                    if 'items' in action:
                        st.write("**Item Prioritas:**")
                        for item in action['items']:
                            st.write(f"- {item}")
                    if 'catatan' in action:
                        st.info(f"Catatan: {action['catatan']}")
                    st.write("---")
            
            # Priority Items Table
            if rec['priority_items']:
                st.subheader("🎯 Item Prioritas untuk Diselesaikan")
                priority_df = pd.DataFrame(rec['priority_items'])
                st.dataframe(priority_df, use_container_width=True)

def process_pdf_capaian_output(file_buffer):
    """Process PDF file for output achievement"""
    try:
        text = ""
        with pdfplumber.open(file_buffer) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        # Extract percentage using various patterns
        patterns = [
            r'capaian\s*output\s*:?\s*(\d+\.?\d*)%',
            r'persentase\s*capaian\s*:?\s*(\d+\.?\d*)%',
            r'capaian\s*:?\s*(\d+\.?\d*)%',
            r'realisasi\s*output\s*:?\s*(\d+\.?\d*)%'
        ]
        
        persentase = 0
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                persentase = float(matches[0])
                break
        
        return {
            'persentase_capaian': persentase,
            'text_analysis': text[:1500],
            'status': 'success'
        }
        
    except Exception as e:
        return {"error": f"Error processing PDF: {str(e)}"}

def display_capaian_dashboard(data):
    """Display output achievement dashboard"""
    st.header("🎯 Dashboard Capaian Output")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Persentase Capaian Output", f"{data['persentase_capaian']}%")
    with col2:
        st.metric("Status", "✅ Terpenuhi" if data['persentase_capaian'] >= 80 else "⚠️ Perlu Perbaikan")
    
    with st.expander("📋 Analisis Konten Laporan"):
        st.text_area("Teks yang diekstrak:", data['text_analysis'], height=200)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏢 SMART - Sistem Monitoring dan Evaluasi Kinerja Terpadu</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Badan Narkotika Nasional Provinsi DKI Jakarta</h3>', unsafe_allow_html=True)
    
    # Initialize processors
    processor = AdvancedBudgetProcessor()
    satuan_kerja_calculator = SatuanKerjaIKPACalculator()
    
    # DeepSeek API Configuration
    st.sidebar.header("🔑 Konfigurasi DeepSeek API")
    
    # Option 1: Input API Key manually
    api_key = st.sidebar.text_input(
        "DeepSeek API Key",
        type="password",
        placeholder="Masukkan API Key DeepSeek Anda...",
        help="Dapatkan API key dari https://platform.deepseek.com/"
    )
    
    # Option 2: Use secrets (for deployment)
    if not api_key and 'DEEPSEEK_API_KEY' in st.secrets:
        api_key = st.secrets['DEEPSEEK_API_KEY']
    
    # Initialize DeepSeek API
    deepseek_api = None
    if api_key:
        deepseek_api = DeepSeekAPI(api_key)
        st.sidebar.success("✅ DeepSeek API terkoneksi")
    else:
        st.sidebar.warning("⚠️ Masukkan DeepSeek API Key untuk menggunakan fitur AI")
    
    # Sidebar for file uploads
    st.sidebar.header("📁 Upload Files untuk Satuan Kerja")
    
    with st.sidebar.expander("📊 Laporan Penyerapan Anggaran", expanded=True):
        budget_file = st.sidebar.file_uploader(
            "Upload Excel Penyerapan seluruh Satuan Kerja", 
            type=['xlsx', 'xls'],
            help="Format harus mencakup semua bidang: Pemberantasan, Rehabilitasi, P2M, Bagian Umum"
        )
    
    with st.sidebar.expander("📈 Capaian Output Satuan Kerja"):
        pdf_file = st.sidebar.file_uploader(
            "Upload PDF Capaian Output Satuan Kerja", 
            type=['pdf'],
            help="Laporan capaian output keseluruhan BNNP DKI Jakarta"
        )
    
    # Process files
    capaian_output_data = None
    
    if budget_file:
        with st.spinner("Memproses laporan penyerapan anggaran seluruh satuan kerja..."):
            budget_results = processor.process_detailed_budget(budget_file)
            
        if "error" not in budget_results:
            st.sidebar.success("✅ Data penyerapan berhasil diproses")
        else:
            st.sidebar.error(f"❌ {budget_results['error']}")
    
    if pdf_file:
        with st.spinner("Memproses laporan capaian output satuan kerja..."):
            capaian_output_data = process_pdf_capaian_output(pdf_file)
            
        if "error" not in capaian_output_data:
            st.sidebar.success("✅ Data capaian output berhasil diproses")
        else:
            st.sidebar.error(f"❌ {capaian_output_data['error']}")
    
    # Calculate and display satuan kerja analysis
    if (processor.penyerapan_data and capaian_output_data and 
        "error" not in capaian_output_data):
        
        # Calculate IKPA for entire satuan kerja
        all_bidang_data = {
            'overall': processor.penyerapan_data
        }
        
        satuan_kerja_results = satuan_kerja_calculator.calculate_satuan_kerja_ikpa(
            all_bidang_data,
            capaian_output_data['persentase_capaian']
        )
        
        # Display comprehensive dashboard
        display_satuan_kerja_dashboard(processor, satuan_kerja_calculator, capaian_output_data, deepseek_api)
    
    elif budget_file and "error" not in budget_results:
        # Display bidang-level analysis if only budget data is available
        display_budget_dashboard(processor)
    
    # Additional AI Features
    if deepseek_api and processor.penyerapan_data:
        st.sidebar.header("🤖 Fitur AI Tambahan")
        
        with st.sidebar.expander("💡 Konsultasi AI"):
            user_question = st.text_area(
                "Pertanyaan spesifik untuk AI:",
                placeholder="Contoh: Bagaimana strategi meningkatkan penyerapan di Bidang Pemberantasan?",
                height=100
            )
            
            if st.button("Tanyakan AI", key="ask_ai"):
                with st.spinner("AI sedang menganalisis..."):
                    custom_prompt = f"""
                    Pertanyaan: {user_question}
                    
                    Konteks Data BNNP DKI Jakarta:
                    - Rata-rata penyerapan: {processor.penyerapan_data['rata_penyerapan']:.1f}%
                    - Total alokasi: Rp {processor.penyerapan_data['total_alokasi']:,.0f}
                    - Total realisasi: Rp {processor.penyerapan_data['total_realisasi']:,.0f}
                    
                    Berikan jawaban yang spesifik dan praktis berdasarkan data di atas.
                    """
                    
                    ai_response = deepseek_api.generate_recommendation(custom_prompt)
                    st.info(f"**Jawaban AI:** {ai_response}")

if __name__ == "__main__":
    main()
