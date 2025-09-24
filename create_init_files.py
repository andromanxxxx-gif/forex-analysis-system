# create_init_files.py - Simpan di root folder dan jalankan
import os

folders = ['src', 'config', 'dashboard', 'scripts']

for folder in folders:
    init_file = os.path.join(folder, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('# Package initialization file\n')
        print(f"Created {init_file}")
    else:
        print(f"{init_file} already exists")

print("All __init__.py files created successfully!")
