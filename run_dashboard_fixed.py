#!/usr/bin/env python3
"""
FOREX DASHBOARD - FIXED VERSION dengan error handling
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

class FixedDashboardLauncher:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.dashboard_dir = self.root_dir / "dashboard"
        
    def check_basic_imports(self):
        """Cek import dasar tanpa pkg_resources"""
        try:
            import streamlit
            import pandas
            import numpy
            print("✅ Basic imports OK")
            return True
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
    
    def run_dashboard_safe(self):
        """Jalankan dashboard dengan error handling"""
        print("🚀 Starting Forex Dashboard...")
        
        # Cek apakah app.py ada
        main_app = self.dashboard_dir / "app.py"
        simple_app = self.dashboard_dir / "app_simple.py"
        
        app_to_run = main_app if main_app.exists() else simple_app
        
        if not app_to_run.exists():
            print("❌ No dashboard app found!")
            return False
        
        print(f"📊 Running: {app_to_run.name}")
        
        # Buka browser setelah delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("✅ Dashboard will open in your browser...")
        print("🌐 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop\n")
        
        try:
            # Jalankan streamlit
            os.chdir(self.dashboard_dir)
            result = subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                app_to_run.name, "--server.port=8501", "--server.address=0.0.0.0"
            ])
            
            return result.returncode == 0
            
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
            return True
        except Exception as e:
            print(f"❌ Error running dashboard: {e}")
            return False

def main():
    print("🎯 FOREX ANALYSIS SYSTEM - FIXED LAUNCHER")
    print("=" * 50)
    
    launcher = FixedDashboardLauncher()
    
    # Cek basic imports
    if not launcher.check_basic_imports():
        print("\n⚠️  Please install required packages:")
        print("pip install streamlit pandas numpy plotly yfinance")
        return
    
    # Jalankan dashboard
    success = launcher.run_dashboard_safe()
    
    if success:
        print("✅ Dashboard session completed")
    else:
        print("❌ Dashboard failed to start")

if __name__ == "__main__":
    main()
