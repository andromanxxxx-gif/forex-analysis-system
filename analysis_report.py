import os
import sys
from pathlib import Path

def collect_project_info():
    """Kumpulkan semua informasi project untuk analisis"""
    project_info = {
        'structure': {},
        'app_py': '',
        'index_html': '',
        'requirements': '',
        'error_logs': '',
        'system_info': {}
    }
    
    # Struktur project
    base_path = Path('.')
    for file_path in base_path.rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(base_path)
            project_info['structure'][str(relative_path)] = {
                'size': file_path.stat().st_size,
                'modified': file_path.stat().st_mtime
            }
    
    # Baca file penting
    try:
        with open('dashboard/app.py', 'r', encoding='utf-8') as f:
            project_info['app_py'] = f.read()
    except Exception as e:
        project_info['app_py'] = f"Error reading app.py: {e}"
    
    try:
        with open('dashboard/templates/index.html', 'r', encoding='utf-8') as f:
            project_info['index_html'] = f.read()
    except Exception as e:
        project_info['index_html'] = f"Error reading index.html: {e}"
    
    try:
        with open('dashboard/requirements.txt', 'r', encoding='utf-8') as f:
            project_info['requirements'] = f.read()
    except:
        project_info['requirements'] = "No requirements.txt found"
    
    # System info
    project_info['system_info'] = {
        'python_version': sys.version,
        'platform': sys.platform,
        'current_directory': os.getcwd(),
        'environment_variables': dict(os.environ)
    }
    
    return project_info

def create_analysis_report():
    """Buat laporan analisis lengkap"""
    info = collect_project_info()
    
    report = f"""
# FOREX ANALYSIS SYSTEM - COMPLETE ANALYSIS REPORT
Generated on: {import datetime; datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## PROJECT STRUCTURE:
