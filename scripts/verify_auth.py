# scripts/verify_auth.py
import sys
from pathlib import Path

# Tambahkan path ke src
sys.path.append(str(Path(__file__).parent.parent))

from src.google_drive_auth import drive_auth

def verify_auth_file():
    """Verifikasi bahwa file auth valid dan dapat mengakses Google Drive"""
    try:
        drive_service = drive_auth.get_service()
        
        # Lakukan permintaan sederhana untuk memverifikasi akses
        results = drive_service.files().list(
            pageSize=1,
            fields="files(id, name)"
        ).execute()
        
        print("✅ Auth file verification successful!")
        print(f"📁 Auth file location: {drive_auth.get_auth_file_path()}")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying auth file: {e}")
        return False

if __name__ == "__main__":
    verify_auth_file()
