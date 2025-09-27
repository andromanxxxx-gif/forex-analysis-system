# run_dashboard.py
from pathlib import Path
import subprocess
import sys

dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
print("=====================================================")
print("        🚀 FOREX ANALYSIS SYSTEM - DASHBOARD")
print("=====================================================")
print(f"✅ Menjalankan dashboard Flask dari: {dashboard_path}")
print("🌐 Dashboard akan terbuka otomatis di http://127.0.0.1:5000\n")

subprocess.run([sys.executable, str(dashboard_path)])
