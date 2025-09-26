import os
import pandas as pd
import tempfile
from datetime import datetime
import logging
from config.gdrive_config import GDRIVE_MANAGER

logger = logging.getLogger(__name__)

class DataManager:
    """
    Manager untuk handling data antara lokal dan Google Drive
    """
    
    def __init__(self):
        self.gdrive = GDRIVE_MANAGER
        self.local_data_dir = 'data'
        
        # Buat folder lokal jika belum ada
        os.makedirs(self.local_data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.local_data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(self.local_data_dir, 'processed'), exist_ok=True)
        os.makedirs(os.path.join(self.local_data_dir, 'dummy'), exist_ok=True)
    
    def save_dataframe(self, df, filename, folder='processed', upload_to_drive=True):
        """
        Save DataFrame ke lokal dan optionally ke Google Drive
        """
        try:
            # Save ke lokal
            local_path = os.path.join(self.local_data_dir, folder, filename)
            df.to_csv(local_path, index=True)
            logger.info(f"Data saved locally: {local_path}")
            
            # Upload ke Google Drive
            if upload_to_drive and self.gdrive and self.gdrive.is_authenticated():
                drive_filename = f"{folder}_{filename}"
                file_id = self.gdrive.upload_file(local_path, drive_filename)
                if file_id:
                    logger.info(f"Data uploaded to Google Drive: {drive_filename}")
                    return file_id
            
            return local_path
            
        except Exception as e:
            logger.error(f"Error saving dataframe: {e}")
            return None
    
    def load_dataframe(self, filename, folder='processed', try_drive=True):
        """
        Load DataFrame dari lokal atau Google Drive
        """
        try:
            # Coba load dari lokal dulu
            local_path = os.path.join(self.local_data_dir, folder, filename)
            if os.path.exists(local_path):
                df = pd.read_csv(local_path, index_col=0, parse_dates=True)
                logger.info(f"Data loaded from local: {local_path}")
                return df
            
            # Jika tidak ada di lokal, coba dari Google Drive
            if try_drive and self.gdrive and self.gdrive.is_authenticated():
                drive_filename = f"{folder}_{filename}"
                file_id = self.gdrive.get_file_id_by_name(drive_filename)
                
                if file_id:
                    # Download dari Google Drive
                    temp_path = os.path.join(tempfile.gettempdir(), filename)
                    if self.gdrive.download_file(file_id, temp_path):
                        df = pd.read_csv(temp_path, index_col=0, parse_dates=True)
                        os.unlink(temp_path)  # Cleanup
                        
                        # Save ke lokal untuk下次使用
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        df.to_csv(local_path)
                        
                        logger.info(f"Data loaded from Google Drive: {drive_filename}")
                        return df
            
            logger.warning(f"Data not found: {filename} in {folder}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading dataframe: {e}")
            return None
    
    def list_available_data(self, folder='processed'):
        """
        List semua data yang available (lokal + drive)
        """
        data_files = {}
        
        # Data lokal
        local_dir = os.path.join(self.local_data_dir, folder)
        if os.path.exists(local_dir):
            local_files = [f for f in os.listdir(local_dir) if f.endswith('.csv')]
            data_files['local'] = local_files
        
        # Data di Google Drive
        if self.gdrive and self.gdrive.is_authenticated():
            try:
                drive_files = self.gdrive.list_files()
                csv_files = [f['name'] for f in drive_files if f['name'].endswith('.csv')]
                data_files['drive'] = csv_files
            except Exception as e:
                logger.error(f"Error listing drive files: {e}")
        
        return data_files
    
    def sync_data_to_drive(self):
        """
        Sync semua data lokal ke Google Drive
        """
        if not self.gdrive or not self.gdrive.is_authenticated():
            logger.error("Google Drive not authenticated")
            return False
        
        try:
            synced_files = []
            
            for folder in ['raw', 'processed', 'dummy']:
                local_dir = os.path.join(self.local_data_dir, folder)
                if not os.path.exists(local_dir):
                    continue
                
                for filename in os.listdir(local_dir):
                    if filename.endswith('.csv'):
                        local_path = os.path.join(local_dir, filename)
                        drive_filename = f"{folder}_{filename}"
                        
                        file_id = self.gdrive.upload_file(local_path, drive_filename)
                        if file_id:
                            synced_files.append(drive_filename)
            
            logger.info(f"Synced {len(synced_files)} files to Google Drive")
            return synced_files
            
        except Exception as e:
            logger.error(f"Error syncing data to drive: {e}")
            return False
    
    def cleanup_old_data(self, days_old=30):
        """
        Hapus data yang sudah lama (baik lokal maupun drive)
        """
        try:
            deleted_files = []
            cutoff_date = datetime.now() - pd.Timedelta(days=days_old)
            
            # Cleanup lokal
            for folder in ['raw', 'processed', 'dummy']:
                local_dir = os.path.join(self.local_data_dir, folder)
                if not os.path.exists(local_dir):
                    continue
                
                for filename in os.listdir(local_dir):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(local_dir, filename)
                        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        if file_time < cutoff_date:
                            os.remove(file_path)
                            deleted_files.append(f"local_{filename}")
            
            # Cleanup Google Drive (hanya yang ada pattern tanggal lama)
            if self.gdrive and self.gdrive.is_authenticated():
                drive_files = self.gdrive.list_files()
                for file in drive_files:
                    if file['name'].endswith('.csv') and 'old' in file['name'].lower():
                        file_id = file['id']
                        self.gdrive.delete_file(file_id)
                        deleted_files.append(f"drive_{file['name']}")
            
            logger.info(f"Cleaned up {len(deleted_files)} old files")
            return deleted_files
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False

# Global instance
DATA_MANAGER = DataManager()
