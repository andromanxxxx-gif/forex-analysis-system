# scripts/download_models.py
import gdown
import os

def download_models():
    """Download model dari Google Drive"""
    model_files = {
        'gbpjpy': 'https://drive.google.com/uc?id=YOUR_GBPJPY_MODEL_ID',
        'chfjpy': 'https://drive.google.com/uc?id=YOUR_CHFJPY_MODEL_ID',
        'usdjpy': 'https://drive.google.com/uc?id=YOUR_USDJPY_MODEL_ID',
        'eurjpy': 'https://drive.google.com/uc?id=YOUR_EURJPY_MODEL_ID'
    }
    
    model_dir = 'models/saved_models'
    os.makedirs(model_dir, exist_ok=True)
    
    for pair, url in model_files.items():
        output_path = f'{model_dir}/{pair}_model.h5'
        if not os.path.exists(output_path):
            print(f'Downloading model for {pair}...')
            gdown.download(url, output_path, quiet=False)
        else:
            print(f'Model for {pair} already exists.')

if __name__ == '__main__':
    download_models()
