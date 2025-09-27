# simple_test.py
import os

print("📁 STRUCTURE CHECK:")
print("=" * 50)

# Check if essential files exist
files_to_check = [
    'dashboard/app.py',
    'dashboard/templates/index.html',
    'dashboard/requirements.txt'
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"✅ {file} - EXISTS")
        try:
            size = os.path.getsize(file)
            print(f"   Size: {size} bytes")
            
            # Read first few lines for content check
            with open(file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                print(f"   First line: {first_line[:100]}...")
                
        except Exception as e:
            print(f"   Error reading: {e}")
    else:
        print(f"❌ {file} - MISSING")

print("\n🐍 PYTHON INFO:")
import sys
print(f"Python: {sys.version}")

print("\n📊 FOLDER STRUCTURE:")
for root, dirs, files in os.walk('.'):
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files per folder
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files) - 5} more files")
