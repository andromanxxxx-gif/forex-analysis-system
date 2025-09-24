# scripts/setup_auth_path.py
import json
from pathlib import Path

def setup_auth_config():
    """Setup auth path configuration across the project"""
    
    auth_path = Path("C:/hp/Json/google-auth.json")
    
    if not auth_path.exists():
        print("Warning: Auth file not found at specified path")
        print(f"Expected location: {auth_path}")
        return False
    
    # Update config files if needed
    config = {
        "google_auth_path": str(auth_path),
        "auth_file_exists": auth_path.exists()
    }
    
    # Save to a config file
    config_path = Path(__file__).parent.parent / "config" / "auth_config.json"
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Auth path configuration updated: {auth_path}")
    return True

if __name__ == "__main__":
    setup_auth_config()
