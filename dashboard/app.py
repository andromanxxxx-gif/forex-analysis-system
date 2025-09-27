import os
import pandas as pd
from datetime import datetime

def load_historical_data_from_file(pair, timeframe):
    """Load historical data from local files"""
    data_dir = 'data/historical'
    filename = f"{pair}_{timeframe}.csv"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        try:
            # Load data from CSV
            data = pd.read_csv(filepath)
            
            # Convert datetime column
            data['datetime'] = pd.to_datetime(data['datetime'])
            data.set_index('datetime', inplace=True)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close']
            if all(col in data.columns for col in required_columns):
                print(f"Loaded historical data from {filepath}: {len(data)} rows")
                return data
            else:
                print(f"Missing required columns in {filepath}")
                return None
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    else:
        print(f"Historical file not found: {filepath}")
        return None

def get_data_with_fallback(pair, timeframe, period):
    """Get data with fallback to historical files"""
    try:
        # Try to get real-time data first
        yf_symbol = pair_mapping[pair]
        yf_timeframe = timeframe_mapping[timeframe]
        
        data = yf.download(yf_symbol, period=period, interval=yf_timeframe)
        
        if not data.empty and len(data) > 20:
            print(f"Using real-time data for {pair}: {len(data)} rows")
            return data
        else:
            # Fallback to historical data
            print("Real-time data insufficient, trying historical files...")
            historical_data = load_historical_data_from_file(pair, timeframe)
            return historical_data
            
    except Exception as e:
        print(f"Error getting real-time data: {e}")
        # Fallback to historical data
        historical_data = load_historical_data_from_file(pair, timeframe)
        return historical_data
