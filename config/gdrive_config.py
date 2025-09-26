import os
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveManager:
    def __init__(self, credentials_file=None, token_file='token.pickle', scopes=None):
        """
        Initialize Google Drive Manager
        
        Args:
            credentials_file: Path to credentials JSON file
            token_file: Path to store authentication token
            scopes: List of Google Drive scopes
        """
        if credentials_file is None:
            # Default path to credentials file
            credentials_file = os.path.join(
                os.path.dirname(__file__), 
                'client_secret_589203755993-9e04439ocp6bltt1b3o2eb3uev0ru9n0.apps.googleusercontent.com.json'
            )
        
        if scopes is None:
            scopes = ['https://www.googleapis.com/auth/drive']
        
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.scopes = scopes
        self.service = None
        self.folder_id = None
        
        # Initialize service
        self.authenticate()
        
        # Get or create main folder
        self.folder_id = self.get_or_create_folder('Forex_Analysis_Data')
        
        # Create subfolders
        self.subfolders = {
            'raw_data': self.get_or_create_folder('Raw_Data', parent_id=self.folder_id),
            'processed_data': self.get_or_create_folder('Processed_Data', parent_id=self.folder_id),
            'predictions': self.get_or_create_folder('Predictions', parent_id=self.folder_id),
            'backtest_results': self.get_or_create_folder('Backtest_Results', parent_id=self.folder_id),
            'news_data': self.get_or_create_folder('News_Data', parent_id=self.folder_id)
        }
    
    def authenticate(self):
        """Authenticate with Google Drive API"""
        creds = None
        
        # Check if token file exists
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
                logger.info("Token loaded from file")
            except Exception as e:
                logger.warning(f"Error loading token: {e}")
                creds = None
        
        # If there are no valid credentials available, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired token")
                creds.refresh(Request())
            else:
                logger.info("Starting new authentication flow")
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Credentials file not found: {self.credentials_file}\n"
                        "Please make sure your Google Drive API credentials JSON file is in the config folder."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.scopes)
                
                # Run local server for authentication
                creds = flow.run_local_server(port=0, open_browser=True)
            
            # Save the credentials for the next run
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
            logger.info("Token saved successfully")
        
        # Build the service
        self.service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive service initialized successfully")
    
    def get_or_create_folder(self, folder_name, parent_id=None):
        """
        Get existing folder or create new one
        
        Args:
            folder_name: Name of the folder
            parent_id: Parent folder ID (None for root)
        
        Returns:
            Folder ID
        """
        try:
            # Search for existing folder
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            
            results = self.service.files().list(q=query).execute()
            items = results.get('files', [])
            
            if items:
                folder_id = items[0]['id']
                logger.info(f"Folder '{folder_name}' found with ID: {folder_id}")
                return folder_id
            else:
                # Create new folder
                file_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                
                if parent_id:
                    file_metadata['parents'] = [parent_id]
                
                folder = self.service.files().create(body=file_metadata, fields='id').execute()
                folder_id = folder.get('id')
                logger.info(f"Folder '{folder_name}' created with ID: {folder_id}")
                return folder_id
                
        except Exception as e:
            logger.error(f"Error creating folder '{folder_name}': {e}")
            return None
    
    def upload_file(self, file_path, file_name, parent_folder_id=None, overwrite=True):
        """
        Upload file to Google Drive
        
        Args:
            file_path: Local path to file
            file_name: Name to save as in Drive
            parent_folder_id: Parent folder ID
            overwrite: Whether to overwrite existing file
        
        Returns:
            File ID of uploaded file, or None if failed
        """
        try:
            if parent_folder_id is None:
                parent_folder_id = self.folder_id
            
            # Check if file already exists
            existing_file_id = None
            if overwrite:
                existing_file_id = self.get_file_id_by_name(file_name, parent_folder_id)
            
            file_metadata = {
                'name': file_name,
                'parents': [parent_folder_id]
            }
            
            media = MediaFileUpload(file_path, resumable=True)
            
            if existing_file_id:
                # Update existing file
                file = self.service.files().update(
                    fileId=existing_file_id,
                    body=file_metadata,
                    media_body=media
                ).execute()
                logger.info(f"File updated: {file_name}")
            else:
                # Create new file
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                logger.info(f"File uploaded: {file_name}")
            
            return file.get('id')
            
        except Exception as e:
            logger.error(f"Error uploading file {file_name}: {e}")
            return None
    
    def download_file(self, file_id, destination_path):
        """
        Download file from Google Drive
        
        Args:
            file_id: ID of file to download
            destination_path: Local path to save file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            request = self.service.files().get_media(fileId=file_id)
            fh = io.FileIO(destination_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logger.info(f"Download progress: {int(status.progress() * 100)}%")
            
            logger.info(f"File downloaded to: {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file {file_id}: {e}")
            return False
    
    def download_file_by_name(self, file_name, destination_path, parent_folder_id=None):
        """
        Download file by name
        
        Args:
            file_name: Name of file to download
            destination_path: Local path to save file
            parent_folder_id: Parent folder ID to search in
        
        Returns:
            True if successful, False otherwise
        """
        try:
            file_id = self.get_file_id_by_name(file_name, parent_folder_id)
            if file_id:
                return self.download_file(file_id, destination_path)
            else:
                logger.warning(f"File not found: {file_name}")
                return False
        except Exception as e:
            logger.error(f"Error downloading file by name {file_name}: {e}")
            return False
    
    def get_file_id_by_name(self, file_name, parent_folder_id=None):
        """
        Get file ID by name
        
        Args:
            file_name: Name of file to find
            parent_folder_id: Parent folder ID to search in
        
        Returns:
            File ID if found, None otherwise
        """
        try:
            if parent_folder_id is None:
                parent_folder_id = self.folder_id
            
            query = f"name='{file_name}' and '{parent_folder_id}' in parents and trashed=false"
            results = self.service.files().list(q=query).execute()
            files = results.get('files', [])
            
            return files[0]['id'] if files else None
            
        except Exception as e:
            logger.error(f"Error getting file ID for {file_name}: {e}")
            return None
    
    def list_files(self, folder_id=None, file_type=None):
        """
        List files in folder
        
        Args:
            folder_id: Folder ID to list (None for main folder)
            file_type: Filter by file type (e.g., 'csv', 'json')
        
        Returns:
            List of file dictionaries
        """
        try:
            if folder_id is None:
                folder_id = self.folder_id
            
            query = f"'{folder_id}' in parents and trashed=false"
            if file_type:
                query += f" and name contains '.{file_type}'"
            
            results = self.service.files().list(q=query).execute()
            files = results.get('files', [])
            
            # Add additional info for each file
            for file in files:
                file['createdTime'] = file.get('createdTime', '')
                file['modifiedTime'] = file.get('modifiedTime', '')
                file['size'] = file.get('size', 0)
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    def delete_file(self, file_id):
        """
        Delete file from Google Drive
        
        Args:
            file_id: ID of file to delete
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.service.files().delete(fileId=file_id).execute()
            logger.info(f"File deleted: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            return False
    
    def file_exists(self, file_name, parent_folder_id=None):
        """
        Check if file exists
        
        Args:
            file_name: Name of file to check
            parent_folder_id: Parent folder ID
        
        Returns:
            True if exists, False otherwise
        """
        return self.get_file_id_by_name(file_name, parent_folder_id) is not None
    
    def get_file_info(self, file_id):
        """
        Get detailed information about a file
        
        Args:
            file_id: ID of file
        
        Returns:
            Dictionary with file information
        """
        try:
            file = self.service.files().get(
                fileId=file_id, 
                fields='id,name,createdTime,modifiedTime,size,parents'
            ).execute()
            return file
        except Exception as e:
            logger.error(f"Error getting file info for {file_id}: {e}")
            return None
    
    def create_folder_structure(self):
        """Create the complete folder structure for the Forex analysis system"""
        folders = {
            'Technical_Analysis': ['MACD', 'EMA', 'RSI', 'OBV'],
            'Fundamental_Analysis': ['News', 'Economic_Data'],
            'Predictions': ['Short_Term', 'Medium_Term', 'Long_Term'],
            'Backtesting': ['Results', 'Strategies'],
            'Exports': ['Charts', 'Reports']
        }
        
        structure = {}
        for main_folder, subfolders in folders.items():
            main_id = self.get_or_create_folder(main_folder, self.folder_id)
            structure[main_folder] = {
                'id': main_id,
                'subfolders': {}
            }
            
            for subfolder in subfolders:
                sub_id = self.get_or_create_folder(subfolder, main_id)
                structure[main_folder]['subfolders'][subfolder] = sub_id
        
        return structure
    
    def get_storage_usage(self):
        """Get storage usage information"""
        try:
            about = self.service.about().get(fields='storageQuota').execute()
            storage_quota = about.get('storageQuota', {})
            
            usage_info = {
                'limit': int(storage_quota.get('limit', 0)),
                'usage': int(storage_quota.get('usage', 0)),
                'usage_in_drive': int(storage_quota.get('usageInDrive', 0)),
                'usage_in_drive_trash': int(storage_quota.get('usageInDriveTrash', 0))
            }
            
            # Calculate percentages
            if usage_info['limit'] > 0:
                usage_info['usage_percent'] = (usage_info['usage'] / usage_info['limit']) * 100
                usage_info['usage_in_drive_percent'] = (usage_info['usage_in_drive'] / usage_info['limit']) * 100
            else:
                usage_info['usage_percent'] = 0
                usage_info['usage_in_drive_percent'] = 0
            
            return usage_info
            
        except Exception as e:
            logger.error(f"Error getting storage usage: {e}")
            return None

# Global instance for easy access
try:
    GDRIVE_MANAGER = GoogleDriveManager()
    logger.info("Google Drive Manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Google Drive Manager: {e}")
    GDRIVE_MANAGER = None

# Utility functions
def test_connection():
    """Test Google Drive connection"""
    try:
        if GDRIVE_MANAGER and GDRIVE_MANAGER.service:
            files = GDRIVE_MANAGER.list_files()
            print(f"✅ Google Drive connection successful! Found {len(files)} files.")
            return True
        else:
            print("❌ Google Drive connection failed!")
            return False
    except Exception as e:
        print(f"❌ Google Drive connection error: {e}")
        return False

def get_folder_structure():
    """Get current folder structure"""
    if GDRIVE_MANAGER:
        return GDRIVE_MANAGER.create_folder_structure()
    return None

if __name__ == "__main__":
    # Test the configuration
    print("Testing Google Drive configuration...")
    
    if test_connection():
        print("\nFolder structure:")
        structure = get_folder_structure()
        for main_folder, info in structure.items():
            print(f"📁 {main_folder}")
            for subfolder, sub_id in info['subfolders'].items():
                print(f"  └── 📂 {subfolder}")
        
        print("\nStorage usage:")
        usage = GDRIVE_MANAGER.get_storage_usage()
        if usage:
            print(f"Usage: {usage['usage_in_drive'] / (1024**3):.2f} GB / {usage['limit'] / (1024**3):.2f} GB")
            print(f"Percentage: {usage['usage_in_drive_percent']:.1f}%")
    else:
        print("Please check your credentials file and try again.")
