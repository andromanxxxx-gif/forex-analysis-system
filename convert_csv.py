import csv
import os
import glob

def convert_tab_to_comma_delimiter():
    # Dapatkan direktori tempat script ini berada
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path folder sumber dan tujuan RELATIF terhadap script
    source_folder = os.path.join(script_dir, "data_historis_sumber")
    target_folder = os.path.join(script_dir, "data_historis")
    
    print(f"Script location: {script_dir}")
    print(f"Source folder: {source_folder}")
    print(f"Target folder: {target_folder}")
    
    # Cek apakah folder sumber ada
    if not os.path.exists(source_folder):
        print(f"❌ ERROR: Folder sumber tidak ditemukan: {source_folder}")
        print("\nBuat folder 'data_historis_sumber' dan taruh file CSV Anda di sana!")
        return
    
    # Buat folder tujuan jika belum ada
    os.makedirs(target_folder, exist_ok=True)
    
    # Cari semua file CSV di folder sumber
    source_files = glob.glob(os.path.join(source_folder, "*.csv"))
    
    if not source_files:
        print(f"❌ Tidak ditemukan file CSV di folder: {source_folder}")
        print("\nPastikan:")
        print("1. Folder 'data_historis_sumber' sudah ada")
        print("2. File CSV sudah ditaruh di folder tersebut")
        print("3. File CSV memiliki ekstensi .csv")
        return
    
    print(f"✅ Menemukan {len(source_files)} file CSV untuk dikonversi:")
    
    successful_conversions = 0
    
    for source_file in source_files:
        try:
            # Ambil nama file saja (tanpa path)
            filename = os.path.basename(source_file)
            target_file = os.path.join(target_folder, filename)
            
            print(f"\n📁 Memproses: {filename}")
            
            # Coba berbagai encoding yang umum
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            
            converted = False
            for encoding in encodings_to_try:
                try:
                    # Method 1: menggunakan csv module
                    with open(source_file, 'r', encoding=encoding) as f_in:
                        lines = f_in.readlines()
                    
                    # Process conversion
                    with open(target_file, 'w', encoding='utf-8', newline='') as f_out:
                        for line in lines:
                            # Replace tabs with commas
                            line_converted = line.replace('\t', ',')
                            f_out.write(line_converted)
                    
                    # Verify the file was created and has content
                    if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
                        print(f"  ✅ Berhasil dengan encoding: {encoding}")
                        print(f"  📂 Disimpan sebagai: {os.path.basename(target_file)}")
                        successful_conversions += 1
                        converted = True
                        break
                    
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"  ⚠️ Error dengan encoding {encoding}: {str(e)}")
                    continue
            
            if not converted:
                print(f"  ❌ Gagal mengkonversi {filename}")
                
        except Exception as e:
            print(f"  ❌ Error memproses {filename}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"📊 HASIL KONVERSI:")
    print(f"✅ Berhasil: {successful_conversions} dari {len(source_files)} file")
    print(f"📁 Folder hasil: {target_folder}")
    
    # Tampilkan file hasil
    if os.path.exists(target_folder):
        result_files = os.listdir(target_folder)
        if result_files:
            print(f"\n📋 File yang berhasil dikonversi:")
            for file in result_files:
                file_path = os.path.join(target_folder, file)
                file_size = os.path.getsize(file_path)
                print(f"   - {file} ({file_size} bytes)")

def create_sample_structure():
    """Buat struktur folder contoh jika belum ada"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_folder = os.path.join(script_dir, "data_historis_sumber")
    
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
        print(f"📁 Folder sumber dibuat: {source_folder}")
        print("📝 Silakan taruh file CSV Anda di folder tersebut dan jalankan script lagi!")
        
        # Buat file contoh
        sample_file = os.path.join(source_folder, "contoh_data.csv")
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("pair\tdate\topen\thigh\tlow\tclose\n")
            f.write("EURUSD\t2024-01-01\t1.1000\t1.1050\t1.0990\t1.1020\n")
            f.write("EURUSD\t2024-01-02\t1.1020\t1.1080\t1.1010\t1.1070\n")
        print(f"📄 File contoh dibuat: {sample_file}")

if __name__ == "__main__":
    print("🔄 SCRIPT KONVERSI CSV TAB → KOMA")
    print("=" * 60)
    
    # Cek dan buat struktur jika perlu
    create_sample_structure()
    
    # Jalankan konversi
    convert_tab_to_comma_delimiter()
    
    print(f"\n💡 Tips: Pastikan file CSV sumber menggunakan TAB sebagai pemisah, bukan koma atau titik koma.")
