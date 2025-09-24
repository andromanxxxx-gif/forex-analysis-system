# scripts/verify_auth.py
import os
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def verify_auth_file():
    """Verifikasi bahwa file auth valid dan dapat mengakses Google Drive"""
    
    auth_path = Path("C:/hp/Json/google-auth.json")
    
    if not auth_path.exists():
        print(f"Error: Auth file not found at {auth_path}")
        return False
    
    try:
        # Coba baca dan parse file auth
        creds = service_account.Credentials.from_service_account_file(
            str(auth_path), 
            scopes=['https://www.googleapis.com/auth/drive.metadata.readonly']
        )
        
        # Coba akses Google Drive API
        drive_service = build('drive', 'v3', credentials=creds)
        
        # Lakukan permintaan sederhana untuk memverifikasi akses
        results = drive_service.files().list(
            pageSize=1,  # Hanya butuh 1 hasil untuk verifikasi
            fields="files(id, name)"
        ).execute()
        
        print("Auth file verification successful!")
        print(f"Service account email: {creds.service_account_email}")
        return True
        
    except Exception as e:
        print(f"Error verifying auth file: {e}")
        return False

if __name__ == "__main__":
    verify_auth_file()
