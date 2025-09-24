# test_data_collection.py - Simpan di root folder
import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print("Testing data_collection.py...")

try:
    from src.data_collection import DataCollector
    print("✅ DataCollector imported successfully!")
    
    # Test instantiation
    collector = DataCollector()
    print("✅ DataCollector instantiated successfully!")
    
    # Test method
    try:
        result = collector.save_to_drive(None, "test.txt")
        print("✅ save_to_drive method works!")
    except Exception as e:
        print(f"❌ save_to_drive method error: {e}")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("Test completed!")
