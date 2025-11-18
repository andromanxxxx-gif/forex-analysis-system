def process_ikpa_previous_pdf(self, file_buffer):
    """Extract previous IKPA data from PDF - FOCUS ON NILAI AKHIR"""
    try:
        text = ""
        with pdfplumber.open(file_buffer) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        # Debug: Tampilkan sample teks untuk analisis
        st.sidebar.info(f"Sample teks IKPA: {text[:500]}...")
        
        # PATTERN UTAMA: Fokus pada "Nilai Akhir" dan format serupa
        patterns = [
            # Pattern khusus untuk "Nilai Akhir" dengan berbagai format
            r'nilai\s*akhir\s*:?\s*(\d+[.,]?\d*)',
            r'nilai\s*total\s*:?\s*(\d+[.,]?\d*)',
            r'konversi\s*bobot\s*:?\s*(\d+[.,]?\d*)',
            r'total\s*\/\s*konversi\s*bobot\s*:?\s*(\d+[.,]?\d*)',
            
            # Pattern untuk format tabel dengan pemisah
            r'nilai\s*akhir[:\s]*([0-9]+[.,][0-9]+)',
            r'nilai\s*total[:\s]*([0-9]+[.,][0-9]+)',
            
            # Pattern umum sebagai fallback
            r'ikpa.*?(\d+[.,]?\d*)',
            r'nilai\s*ikpa\s*:?\s*(\d+[.,]?\d*)',
            r'(\d+[.,]?\d*).*ikpa',
        ]
        
        ikpa_value = 0
        best_match = None
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        value = float(match.replace(',', '.'))
                        # Prioritaskan nilai antara 0-100 (range normal IKPA)
                        if 0 <= value <= 100:
                            if "nilai akhir" in pattern.lower() or "nilai total" in pattern.lower():
                                # Ini pattern prioritas tinggi
                                ikpa_value = value
                                best_match = f"Pattern: {pattern} -> {value}"
                                st.sidebar.success(f"✅ Nilai Akhir ditemukan: {value}")
                                break
                            elif ikpa_value == 0:
                                # Fallback ke pattern lain jika belum ada nilai
                                ikpa_value = value
                                best_match = f"Pattern: {pattern} -> {value}"
                    except ValueError:
                        continue
                if ikpa_value > 0 and ("nilai akhir" in pattern.lower() or "nilai total" in pattern.lower()):
                    break  # Stop jika sudah dapat nilai dari pattern prioritas
        
        # Jika belum ketemu, coba ekstrak dari konteks tabel
        if ikpa_value == 0:
            # Cari pattern tabel dengan header dan nilai
            table_patterns = [
                r'(?:nilai\s*akhir|nilai\s*total)[\s:]*([0-9]+[.,][0-9]+)',
                r'(\d+[.,]\d+)\s*(?:nilai\s*akhir|nilai\s*total)',
            ]
            
            for pattern in table_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    value = float(matches[0].replace(',', '.'))
                    if 0 <= value <= 100:
                        ikpa_value = value
                        best_match = f"Tabel Pattern: {pattern} -> {value}"
                        st.sidebar.success(f"✅ Nilai dari tabel: {value}")
                        break
        
        # Fallback akhir: cari semua angka dan ambil yang paling mungkin
        if ikpa_value == 0:
            all_numbers = re.findall(r'(\d+[.,]?\d*)', text)
            possible_ikpa = []
            for num in all_numbers:
                try:
                    value = float(num.replace(',', '.'))
                    # Filter angka yang masuk akal untuk IKPA
                    if 50 <= value <= 100:  # IKPA biasanya di range 50-100
                        possible_ikpa.append(value)
                except:
                    continue
            
            if possible_ikpa:
                # Ambil nilai tertinggi yang reasonable
                ikpa_value = max(possible_ikpa)
                best_match = f"Fallback: {ikpa_value} dari {len(possible_ikpa)} angka"
                st.sidebar.info(f"ℹ️ Nilai fallback: {ikpa_value}")

        # Tampilkan debug info
        if best_match:
            st.sidebar.info(f"Match: {best_match}")

        # EKSTRAKSI KATEGORI - Pattern yang lebih komprehensif
        kategori_patterns = [
            (r'sangat\s+baik', 'Sangat Baik'),
            (r'baik', 'Baik'),
            (r'cukup', 'Cukup'),
            (r'kurang', 'Kurang'),
            (r'memuaskan', 'Baik'),
            (r'optimal', 'Sangat Baik'),
            (r'excellent', 'Sangat Baik'),
            (r'good', 'Baik'),
            (r'fair', 'Cukup'),
            (r'poor', 'Kurang')
        ]
        
        kategori = "Tidak Diketahui"
        for pattern, kategori_name in kategori_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                kategori = kategori_name
                st.sidebar.info(f"Kategori: {kategori}")
                break
        
        # Simpan data lengkap untuk analisis
        self.extracted_data['ikpa_previous'] = ikpa_value
        self.extracted_data['ikpa_kategori'] = kategori
        self.extracted_data['ikpa_text'] = text[:2000]
        self.extracted_data['debug_info'] = {
            'best_match': best_match,
            'text_sample': text[:500]
        }
        
        return {
            'nilai_ikpa': ikpa_value,
            'kategori': kategori,
            'text_sample': text[:1000],
            'debug_info': best_match,
            'status': 'success'
        }
        
    except Exception as e:
        st.sidebar.error(f"Error detail: {str(e)}")
        return {"error": f"Error processing IKPA PDF: {str(e)}"}
