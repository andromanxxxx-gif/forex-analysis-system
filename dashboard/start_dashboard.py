import subprocess
import sys
import os

def start_dashboard():
    # Pastikan berada di direktori app.py
    script_path = os.path.join(os.getcwd(), "app.py")

    if not os.path.exists(script_path):
        print("❌ File app.py tidak ditemukan di direktori ini.")
        sys.exit(1)

    try:
        print("🚀 Menjalankan dashboard...")
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard dihentikan oleh user.")

if __name__ == "__main__":
    start_dashboard()
