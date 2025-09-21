import gdown
import os
import json
from config import settings

def download_models():
    """Download semua model dari Google Drive berdasarkan mapping file"""
    
    # Path untuk file mapping
    mapping_file = 'config/model_mapping.json'
    
    # Load mapping dari file
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file {mapping_file} not found.")
        print("Please create the mapping file first.")
        return False
    
    with open(mapping_file, 'r') as f:
        model_mapping = json.load(f)
    
    # Buat folder jika belum ada
    model_dir = 'models/saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    downloaded_count = 0
    
    # Download setiap model dan scaler
    for pair, ids in model_mapping.items():
        model_url = f'https://drive.google.com/uc?id={ids["model_id"]}'
        scaler_url = f'https://drive.google.com/uc?id={ids["scaler_id"]}'
        
        model_path = os.path.join(model_dir, f'{pair}_model.h5')
        scaler_path = os.path.join(model_dir, f'{pair}_scaler.joblib')
        
        # Download model
        if not os.path.exists(model_path):
            print(f'Downloading model for {pair}...')
            try:
                gdown.download(model_url, model_path, quiet=False)
                downloaded_count += 1
            except Exception as e:
                print(f'Error downloading model for {pair}: {e}')
        else:
            print(f'Model for {pair} already exists.')
        
        # Download scaler
        if not os.path.exists(scaler_path):
            print(f'Downloading scaler for {pair}...')
            try:
                gdown.download(scaler_url, scaler_path, quiet=False)
                downloaded_count += 1
            except Exception as e:
                print(f'Error downloading scaler for {pair}: {e}')
        else:
            print(f'Scaler for {pair} already exists.')
    
    print(f'Download completed. {downloaded_count} files downloaded successfully!')
    return True

if __name__ == '__main__':
    download_models()
