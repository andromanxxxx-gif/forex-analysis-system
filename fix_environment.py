import os
import sys
import subprocess

def run_cmd(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Gagal menjalankan: {cmd}")
        sys.exit(1)

def main():
    print("🔧 Memperbaiki environment...")

    # Cek versi Python
    major, minor = sys.version_info[:2]
    print(f"ℹ️ Python versi terdeteksi: {major}.{minor}")

    if major < 3 or (major == 3 and minor < 10):
        print("⚠️ Versi Python Anda terlalu lama. Disarankan pakai Python 3.10+.")
    elif major == 3 and minor >= 12:
        print("⚠️ Python 3.12+ terdeteksi. Pastikan setuptools terbaru dipakai.")

    # Upgrade pip, setuptools, dan wheel
    run_cmd("python -m pip install --upgrade pip setuptools wheel")

    # Paksa upgrade setuptools ke versi terbaru (fix ImpImporter error)
    run_cmd("pip install --upgrade setuptools==70.0.0")

    # Install paket utama
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn"
    ]
    run_cmd(f"pip install {' '.join(packages)}")

    # Install requirements.txt kalau ada
    if os.path.exists("requirements.txt"):
        run_cmd("pip install -r requirements.txt")

    print("\n✅ Environment sudah diperbaiki dan siap dipakai.")

if __name__ == "__main__":
    main()
