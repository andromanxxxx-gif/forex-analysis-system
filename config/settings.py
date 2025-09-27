"""
Konfigurasi global untuk Forex Analysis System
"""

import os
from pathlib import Path

# Root project
BASE_DIR = Path(__file__).resolve().parent.parent

# Google API credentials
GOOGLE_CREDENTIALS_FILE = BASE_DIR / "config" / "client_secret.json"

# Data directories
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DATA_DUMMY = BASE_DIR / "data" / "dummy"

# Dashboard config
DASHBOARD_PORT = 8501
DASHBOARD_ADDRESS = "0.0.0.0"

# Fallback settings
NEWS_LANGUAGE = "en"
