import socket
import subprocess

def get_free_port(start=8501, end=8600):
    """Cari port kosong antara 8501-8600"""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("❌ Tidak ada port kosong di range!")

if __name__ == "__main__":
    free_port = get_free_port()
    print(f"🚀 Menjalankan dashboard di port {free_port} ...")

    # Jalankan streamlit di port terpilih
    subprocess.run([
        "streamlit", "run", "dashboard/dashboard.py",
        f"--server.port={free_port}",
        "--server.headless=true"
    ])
