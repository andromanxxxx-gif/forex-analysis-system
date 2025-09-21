# scripts/download_models.py
import gdown
import os
import json
from pathlib import Path

def download_models():
    """Download semua model dari Google Drive berdasarkan mapping file"""
    
    # Path ke file mapping - relative ke root project
    mapping_file = Path(__file__).parent.parent / 'config' / 'model_mapping.json'
    
    # Pastikan file mapping ada
    if not mapping_file.exists():
        print(f"Error: Mapping file {mapping_file} not found.")
        print("Please create the mapping file first.")
        return False
    
    # Load mapping dari file
    with open(mapping_file, 'r') as f:
        model_mapping = json.load(f)
    
    # Buat folder jika belum ada
    model_dir = Path(__file__).parent.parent / 'models' / 'saved_models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    
    # Download setiap model dan scaler
    for pair, ids in model_mapping.items():
        model_url = f'https://drive.google.com/uc?id={ids["model_id"]}'
        scaler_url = f'https://drive.google.com/uc?id={ids["scaler_id"]}'
        
        model_path = model_dir / f'{pair}_model.h5'
        scaler_path = model_dir / f'{pair}_scaler.joblib'
        
        # Download model
        if not model_path.exists():
            print(f'Downloading model for {pair}...')
            try:
                gdown.download(model_url, str(model_path), quiet=False)
                downloaded_count += 1
            except Exception as e:
                print(f'Error downloading model for {pair}: {e}')
        else:
            print(f'Model for {pair} already exists.')
        
        # Download scaler
        if not scaler_path.exists():
            print(f'Downloading scaler for {pair}...')
            try:
                gdown.download(scaler_url, str(scaler_path), quiet=False)
                downloaded_count += 1
            except Exception as e:
                print(f'Error downloading scaler for {pair}: {e}')
        else:
            print(f'Scaler for {pair} already exists.')
    
    print(f'Download completed. {downloaded_count} files downloaded successfully!')
    return True

if __name__ == '__main__':
    download_models()
