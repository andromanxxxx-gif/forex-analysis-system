def fix_dashboard_imports():
    """Perbaiki import di dashboard"""
    dashboard_dir = Path(__file__).parent / "dashboard"
    app_file = dashboard_dir / "app.py"
    
    if app_file.exists():
        # Backup original file
        shutil.copy2(app_file, app_file.with_suffix('.py.backup'))
        
        # Baca content dengan encoding utf-8
        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Jika utf-8 gagal, coba latin-1
            with open(app_file, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Ganti import yang problematic
        new_content = content.replace(
            "from src.signal_generator import SignalGenerator", 
            "try:\n    from src.signal_generator import SignalGenerator\nexcept ImportError:\n    SignalGenerator = None"
        )
        
        # Tulis kembali dengan encoding utf-8
        with open(app_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
