import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_historical_data():
    """Buat sample data historis untuk testing"""
    os.makedirs('data/historical', exist_ok=True)
    
    # Buat data 100 hari terakhir untuk GBPJPY 1D
    dates = [datetime.now() - timedelta(days=x) for x in range(100, 0, -1)]
    
    # Buat price data yang realistis
    base_price = 180.0
    np.random.seed(42)  # Untuk konsistensi
    variations = np.random.normal(0, 0.8, 100)
    prices = base_price + np.cumsum(variations)
    
    data = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Open': prices,
        'High': prices + np.abs(np.random.normal(0, 0.5, 100)),
        'Low': prices - np.abs(np.random.normal(0, 0.5, 100)),
        'Close': prices + np.random.normal(0, 0.2, 100),
        'Volume': np.random.randint(10000, 50000, 100)
    })
    
    filepath = 'data/historical/GBPJPY_1D.csv'
    data.to_csv(filepath, index=False)
    print(f"Sample data created: {filepath}")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

if __name__ == "__main__":
    create_sample_historical_data()
