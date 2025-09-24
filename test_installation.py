# test_structure.py - Simpan di root folder dan jalankan
import os
from pathlib import Path

def check_structure():
    base_path = Path(r"C:\Users\HP\forex-analysis-system")
    required_folders = ['src', 'config', 'dashboard', 'scripts', 'models']
    required_files = {
        'src': ['data_collection.py', 'technical_analysis.py', 'news_analyzer.py', 
                'signal_generator.py', 'ml_predictor.py', 'utils.py'],
        'config': ['settings.py'],
        'dashboard': ['app.py', 'requirements.txt'],
        'scripts': ['download_models.py', 'verify_auth.py']
    }
    
    print("Checking project structure...")
    print(f"Base path: {base_path}")
    print()
    
    # Check folders
    for folder in required_folders:
        folder_path = base_path / folder
        exists = folder_path.exists()
        print(f"📁 {folder}: {'✅' if exists else '❌'} {folder_path}")
        
        if exists and folder in required_files:
            for file in required_files[folder]:
                file_path = folder_path / file
                file_exists = file_path.exists()
                print(f"   📄 {file}: {'✅' if file_exists else '❌'}")
    
    # Check __init__.py files
    print("\nChecking __init__.py files:")
    for folder in ['src', 'config', 'dashboard', 'scripts']:
        init_file = base_path / folder / '__init__.py'
        exists = init_file.exists()
        print(f"📄 {folder}/__init__.py: {'✅' if exists else '❌'}")

if __name__ == "__main__":
    check_structure()
