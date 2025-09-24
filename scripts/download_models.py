# scripts/download_models.py
import os
import json
from pathlib import Path
from googleapiclient.http import MediaIoBaseDownload
import io
import sys

# Tambahkan path ke src agar bisa mengimpor modul
sys.path.append(str(Path(__file__).parent.parent))

from src.google_drive_auth import get_drive_service

# Hapus fungsi authenticate_drive dan get_auth_file_path karena sudah ada di modul

# ... (fungsi find_file_id dan download_file tetap sama)

def download_models():
    """Download semua model dari Google Drive menggunakan Google Drive API"""
    
    # Autentikasi ke Google Drive
    drive_service = get_drive_service()  # Gunakan fungsi dari modul
    if not drive_service:
        return False
    
    # ... (kode selanjutnya tetap sama)
