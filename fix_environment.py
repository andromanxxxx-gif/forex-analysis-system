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

    # Upgrade pip, setuptools, dan wheel
    run_cmd("python -m pip install --upgrade pip setuptools wheel")

    # Install paket dasar data science
    packages = [
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn"
    ]
    run_cmd(f"pip install {' '.join(packages)}")

    print("\n✅ Environment sudah diperbaiki dan siap dipakai.")

if __name__ == "__main__":
    main()
