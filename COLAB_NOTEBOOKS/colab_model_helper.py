#!/usr/bin/env python3
"""
Google Colab Model Helper
Script untuk membantu proses training dan penyimpanan model di Google Colab
"""

import os
import json
import joblib
from google.colab import drive
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.client import GoogleCredentials

# Mount Google Drive
drive.mount('/content/drive')

class ColabModelHelper:
    def __init__(self, models_base_path='/content/drive/MyDrive/forex-models'):
        """
        Inisialisasi helper dengan path dasar untuk menyimpan model
        
        Args:
            models_base_path (str): Path dasar untuk menyimpan model di Google Drive
        """
        self.models_base_path = models_base_path
        self.drive_service = None
        self.setup_drive_service()
    
    def setup_drive_service(self):
        """Setup Google Drive service untuk mendapatkan file IDs"""
        try:
            credentials = GoogleCredentials.get_application_default()
            self.drive_service = build('drive', 'v3', credentials=credentials)
            print("Google Drive service setup successfully")
        except Exception as e:
            print(f"Error setting up Drive service: {e}")
            print("File IDs will need to be retrieved manually")
    
    def get_file_id(self, file_path):
        """
        Mendapatkan file ID dari path file di Google Drive
        
        Args:
            file_path (str): Path lengkap ke file di Google Drive
            
        Returns:
            str: File ID atau None jika tidak ditemukan
        """
        if self.drive_service is None:
            print("Drive service not available. Cannot get file ID.")
            return None
        
        try:
            # Extract filename from path
            filename = os.path.basename(file_path)
            
            # Search for the file by name
            results = self.drive_service.files().list(
                q=f"name='{filename}' and trashed=false",
                fields="files(id, name)"
            ).execute()
            
            items = results.get('files', [])
            
            if items:
                return items[0]['id']
            else:
                print(f"File {filename} not found in Google Drive")
                return None
                
        except Exception as e:
            print(f"Error getting file ID for {file_path}: {e}")
            return None
    
    def save_model(self, model, scaler, pair, model_name_suffix=""):
        """
        Menyimpan model dan scaler ke Google Drive dengan nama konsisten
        
        Args:
            model: Model TensorFlow/Keras yang akan disimpan
            scaler: Scaler sklearn yang akan disimpan
            pair (str): Pasangan forex (e.g., 'GBPJPY=X')
            model_name_suffix (str): Suffix untuk nama file (e.g., "_v2")
            
        Returns:
            dict: Dictionary berisi path dan ID file yang disimpan
        """
        # Buat nama file yang konsisten
        pair_name = pair.replace('=', '').lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        model_filename = f"{pair_name}_model{model_name_suffix}_{timestamp}.h5"
        scaler_filename = f"{pair_name}_scaler{model_name_suffix}_{timestamp}.joblib"
        
        # Buat path lengkap
        model_path = os.path.join(self.models_base_path, model_filename)
        scaler_path = os.path.join(self.models_base_path, scaler_filename)
        
        # Simpan model dan scaler
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"Saved model to: {model_path}")
        print(f"Saved scaler to: {scaler_path}")
        
        # Dapatkan file IDs
        model_id = self.get_file_id(model_path)
        scaler_id = self.get_file_id(scaler_path)
        
        return {
            'pair': pair_name,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'model_id': model_id,
            'scaler_id': scaler_id,
            'timestamp': timestamp
        }
    
    def update_mapping_file(self, file_info, mapping_file_path='/content/forex-analysis-system/config/model_mapping.json'):
        """
        Memperbarui file mapping dengan informasi file baru
        
        Args:
            file_info (dict): Informasi file yang dihasilkan oleh save_model
            mapping_file_path (str): Path ke file mapping di repo
        """
        # Baca mapping file yang ada
        if os.path.exists(mapping_file_path):
            with open(mapping_file_path, 'r') as f:
                mapping = json.load(f)
        else:
            mapping = {}
        
        # Update mapping
        pair = file_info['pair']
        if pair not in mapping:
            mapping[pair] = {}
        
        mapping[pair]['model_id'] = file_info['model_id']
        mapping[pair]['scaler_id'] = file_info['scaler_id']
        mapping[pair]['last_updated'] = datetime.now().isoformat()
        mapping[pair]['model_path'] = file_info['model_path']
        mapping[pair]['scaler_path'] = file_info['scaler_path']
        
        # Simpan mapping file
        os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)
        with open(mapping_file_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        
        print(f"Mapping file updated for {pair}")
    
    def create_symlink_to_latest(self, file_info):
        """
        Membuat symbolic link ke file terbaru untuk kemudahan akses
        
        Args:
            file_info (dict): Informasi file yang dihasilkan oleh save_model
        """
        pair = file_info['pair']
        
        # Buat symlink untuk model
        latest_model_path = os.path.join(self.models_base_path, f"{pair}_model_latest.h5")
        if os.path.exists(latest_model_path):
            os.remove(latest_model_path)
        os.symlink(file_info['model_path'], latest_model_path)
        
        # Buat symlink untuk scaler
        latest_scaler_path = os.path.join(self.models_base_path, f"{pair}_scaler_latest.joblib")
        if os.path.exists(latest_scaler_path):
            os.remove(latest_scaler_path)
        os.symlink(file_info['scaler_path'], latest_scaler_path)
        
        print(f"Created symlinks to latest files for {pair}")
    
    def training_pipeline(self, model, scaler, pair, update_mapping=True, create_symlinks=True):
        """
        Pipeline lengkap untuk training dan penyimpanan model
        
        Args:
            model: Model yang telah dilatih
            scaler: Scaler yang telah difit
            pair (str): Pasangan forex
            update_mapping (bool):是否更新mapping file
            create_symlinks (bool):是否创建symlinks
            
        Returns:
            dict: Informasi file yang disimpan
        """
        print(f"Starting training pipeline for {pair}...")
        
        # Simpan model dan scaler
        file_info = self.save_model(model, scaler, pair)
        
        # Update mapping file
        if update_mapping:
            self.update_mapping_file(file_info)
        
        # Buat symlinks
        if create_symlinks:
            self.create_symlink_to_latest(file_info)
        
        print(f"Training pipeline completed for {pair}")
        return file_info

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi helper
    helper = ColabModelHelper()
    
    # Contoh: Setelah training model
    # asumsikan kita sudah memiliki model dan scaler
    # model = ...  # Model yang telah dilatih
    # scaler = ... # Scaler yang telah difit
    
    # # Jalankan pipeline
    # file_info = helper.training_pipeline(
    #     model=model,
    #     scaler=scaler,
    #     pair="GBPJPY=X"
    # )
    
    # print(f"Model saved with ID: {file_info['model_id']}")
    # print(f"Scaler saved with ID: {file_info['scaler_id']}")
    
    print("Colab Model Helper initialized. Ready to use in your notebooks.")
