#!/usr/bin/env python3
"""
AUTO-LAUNCHER untuk Flask Dashboard
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path
import threading

class FlaskDashboardLauncher:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.dashboard_dir = self.root_dir / "dashboard"
        self.app_file = self.dashboard_dir / "app.py"
        self.url = "http://127.0.0.1:5000"

    def print_banner(self):
        banner = """
=====================================================
        🚀 FOREX ANALYSIS SYSTEM - DASHBOARD
=====================================================
"""
        print(banner)

    def check_app_file(self):
        """Cek apakah app.py ada"""
        if not self.app_file.exists():
            print(f"❌ File tidak ditemukan: {self.app_file}")
            return False
        return True

    def open_browser(self):
        """Tunggu sebentar lalu buka browser"""
        time.sleep(3)
        webbrowser.open(self.url)

    def launch(self):
        """Jalankan Flask dashboard"""
        if not self.check_app_file():
            return

        print(f"✅ Menjalankan dashboard Flask dari: {self.app_file}")
        print(f"🌐 Dashboard akan terbuka otomatis di {self.url}\n")

        # Thread untuk auto-buka browser
        browser_thread = threading.Thread(target=self.open_browser, daemon=True)
        browser_thread.start()

        # Jalankan Flask app
        try:
            subprocess.run([sys.executable, str(self.app_file)])
        except KeyboardInterrupt:
            print("\n🛑 Dashboard dihentikan oleh user.")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    launcher = FlaskDashboardLauncher()
    launcher.print_banner()
    launcher.launch()

if __name__ == "__main__":
    main()
