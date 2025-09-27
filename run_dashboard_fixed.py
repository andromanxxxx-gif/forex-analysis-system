#!/usr/bin/env python3
"""
🚀 FOREX ANALYSIS SYSTEM - DASHBOARD LAUNCHER
Menjalankan Flask dashboard dengan satu perintah
"""

import os
import sys
import threading
import time
import webbrowser

# Tambahkan root project ke sys.path supaya modul src bisa diimport
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Banner
BANNER = """
=====================================================
        🚀 FOREX ANALYSIS SYSTEM - DASHBOARD
=====================================================
"""

print(BANNER)

# Fungsi buka browser otomatis
def open_browser():
    time.sleep(2)  # Tunggu server Flask siap
    webbrowser.open("http://127.0.0.1:5000")

# Start browser thread
threading.Thread(target=open_browser).start()

# Jalankan Flask app
try:
    from dashboard import app
    # Pastikan debug=True untuk auto-reload saat development
    app.run(host="0.0.0.0", port=5000, debug=True)
except ModuleNotFoundError as e:
    print(f"❌ Module not found: {e}")
    print("Pastikan folder 'src' ada di root project dan sys.path sudah benar.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
