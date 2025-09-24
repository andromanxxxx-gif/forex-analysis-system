# test_simple.py di root folder
import sys
sys.path.insert(0, 'src')
try:
    import data_collection
    print("Success!")
except ImportError as e:
    print(f"Error: {e}")
