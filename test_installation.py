# test_installation.py
import sys

def check_module(module_name):
    try:
        __import__(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - MISSING: {e}")
        return False

# Dependencies utama
modules = [
    'streamlit', 'pandas', 'numpy', 'plotly',
    'requests', 'bs4', 'lxml', 'gdown',
    'googleapiclient', 'google.auth'
]

print("Checking dependencies...")
results = [check_module(module) for module in modules]

if all(results):
    print("\n🎉 All dependencies installed successfully!")
else:
    print("\n⚠️  Some dependencies are missing.")
