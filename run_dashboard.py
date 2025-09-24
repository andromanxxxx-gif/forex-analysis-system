import socket
import subprocess
import json
import os

def get_free_port(start=8501, end=8600):
    """Cari port kosong dalam range tertentu"""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("❌ Tidak ada port kosong di range!")

def load_config():
    """Membaca config.json"""
    config_path = os.path.join("config.json")
    if not os.path.exists(config_path):
        return {"port_range": {"start": 8501, "end": 8600}}
    with open(config_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # Baca config.json
    config = load_config()
    start_port = config["port_range"]["start"]
    end_port = config["port_range"]["end"]

    free_port = get_free_port(start_port, end_port)
    print(f"🚀 Menjalankan dashboard di port {free_port} ...")

    # Jalankan Streamlit
    subprocess.run([
        "streamlit", "run", "dashboard/dashboard.py",
        f"--server.port={free_port}",
        "--server.headless=true"
    ])
