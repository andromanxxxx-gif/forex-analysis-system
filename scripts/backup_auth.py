# scripts/backup_auth.py
import shutil
from pathlib import Path
from datetime import datetime

def backup_auth_file():
    """Membuat backup dari file auth ke lokasi yang aman"""
    
    source_path = Path("C:/hp/Json/google-auth.json")
    
    if not source_path.exists():
        print("Error: Source auth file not found")
        return False
    
    # Buat folder backup jika belum ada
    backup_dir = Path("C:/hp/Json/backups")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate nama backup dengan timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"google-auth_backup_{timestamp}.json"
    
    try:
        # Salin file
        shutil.copy2(source_path, backup_path)
        print(f"Backup created successfully: {backup_path}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False

if __name__ == "__main__":
    backup_auth_file()
