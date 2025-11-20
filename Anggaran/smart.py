# Tampilkan data Revisi DIPA jika ada
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
    
    # Chart deviasi
    fig_revisi = create_revisi_dipa_chart(revisi_dipa_data)
    st.plotly_chart(fig_revisi, use_container_width=True)

# Update pemanggilan recommendations
recommendations = strategic_advisor.generate_recommendations(
    ikpa_result, realisasi_data, capaian_data, revisi_dipa_data, triwulan
)

# Tampilkan rekomendasi revisi DIPA
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

# Tampilkan rekomendasi rencana penyerapan
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
