import os
import sys
import webbrowser
import time
import threading

def launch_dashboard():
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard app.py tidak ditemukan di: {dashboard_path}")
        return
    def open_browser():
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:5000")
    threading.Thread(target=open_browser, daemon=True).start()
    os.chdir(os.path.dirname(dashboard_path))
    os.system(f"{sys.executable} app.py")

if __name__ == "__main__":
    launch_dashboard()
