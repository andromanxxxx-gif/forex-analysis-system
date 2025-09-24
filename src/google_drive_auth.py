# src/google_drive_auth.py
import os
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Scope untuk Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_auth_file_path():
    """
    Mendapatkan path file auth dari lokasi khusus di Windows
    Mengembalikan Path object atau None jika tidak ditemukan
    """
    # Path khusus yang Anda tentukan
    custom_path = Path("C:/hp/Json/google-auth.json")
    
    # Cek jika file ada di lokasi khusus
    if custom_path.exists():
        return custom_path
    
    # Cek lokasi alternatif (jika diperlukan)
    alt_paths = [
        Path.home() / "hp" / "Json" / "google-auth.json",
        Path("C:/hp/Json/google-auth.json"),
        Path(__file__).parent.parent / "config" / "google-auth.json"
    ]
    
    for path in alt_paths:
        if path.exists():
            return path
    
    return None

def get_drive_service():
    """Mengembalikan service Google Drive yang terautentikasi, atau None jika gagal."""
    auth_file = get_auth_file_path()
    
    if not auth_file or not auth_file.exists():
        print("Error: Google auth file not found.")
        print("Please ensure the file exists at C:\\hp\\Json\\google-auth.json")
        return None
    
    try:
        creds = service_account.Credentials.from_service_account_file(
            str(auth_file), scopes=SCOPES
        )
        drive_service = build('drive', 'v3', credentials=creds)
        print("Google Drive authentication successful")
        return drive_service
    except Exception as e:
        print(f"Error during authentication: {e}")
        return None
