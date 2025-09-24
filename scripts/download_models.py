# scripts/download_models.py
import os
import json
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import sys

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

def authenticate_drive():
    """Autentikasi ke Google Drive menggunakan service account credentials"""
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

def find_file_id(drive_service, filename):
    """Cari file ID berdasarkan nama file"""
    try:
        results = drive_service.files().list(
            q=f"name='{filename}' and trashed=false",
            fields="files(id, name)",
            pageSize=10
        ).execute()
        
        items = results.get('files', [])
        if not items:
            print(f"File {filename} not found in Google Drive")
            return None
        
        # Jika ada multiple files, ambil yang paling baru
        if len(items) > 1:
            print(f"Found multiple files with name {filename}. Using the first one.")
        
        return items[0]['id']
    except Exception as e:
        print(f"Error searching for file {filename}: {e}")
        return None

def download_file(drive_service, file_id, destination_path):
    """Download file dari Google Drive menggunakan ID file"""
    try:
        request = drive_service.files().get_media(fileId=file_id)
        with open(destination_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Download progress: {int(status.progress() * 100)}%")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def download_models():
    """Download semua model dari Google Drive menggunakan Google Drive API"""
    
    # Autentikasi ke Google Drive
    drive_service = authenticate_drive()
    if not drive_service:
        return False
    
    # Daftar file yang akan didownload
    files_to_download = {
        'gbpjpy': ['gbpjpy_model.h5', 'gbpjpy_scaler.joblib'],
        'chfjpy': ['chfjpy_model.h5', 'chfjpy_scaler.joblib'],
        'usdjpy': ['usdjpy_model.h5', 'usdjpy_scaler.joblib'],
        'eurjpy': ['eurjpy_model.h5', 'eurjpy_scaler.joblib']
    }
    
    # Buat folder jika belum ada
    model_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    
    for pair, files in files_to_download.items():
        model_filename, scaler_filename = files
        
        # Download model
        model_path = model_dir / model_filename
        if not model_path.exists():
            print(f'Searching for model file: {model_filename}')
            model_id = find_file_id(drive_service, model_filename)
            if model_id:
                print(f'Downloading model for {pair}...')
                if download_file(drive_service, model_id, str(model_path)):
                    downloaded_count += 1
                    print(f'Successfully downloaded model for {pair}')
                else:
                    print(f'Failed to download model for {pair}')
            else:
                print(f'Model file {model_filename} not found in Google Drive.')
        else:
            print(f'Model for {pair} already exists.')
        
        # Download scaler
        scaler_path = model_dir / scaler_filename
        if not scaler_path.exists():
            print(f'Searching for scaler file: {scaler_filename}')
            scaler_id = find_file_id(drive_service, scaler_filename)
            if scaler_id:
                print(f'Downloading scaler for {pair}...')
                if download_file(drive_service, scaler_id, str(scaler_path)):
                    downloaded_count += 1
                    print(f'Successfully downloaded scaler for {pair}')
                else:
                    print(f'Failed to download scaler for {pair}')
            else:
                print(f'Scaler file {scaler_filename} not found in Google Drive.')
        else:
            print(f'Scaler for {pair} already exists.')
    
    print(f'Download completed. {downloaded_count} files downloaded successfully!')
    return True

if __name__ == '__main__':
    download_models()
