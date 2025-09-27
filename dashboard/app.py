import pandas as pd
import sqlite3
import os
from datetime import datetime
import yfinance as yf

# Required columns untuk data OHLCV
REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

def fix_historical_data(file_path):
    """Memperbaiki file historical data yang kolomnya tidak lengkap"""
    try:
        # Cek jika file exists
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Check missing columns
            missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
            
            if missing_cols:
                print(f"Memperbaiki kolom yang hilang: {missing_cols}")
                
                # Add missing columns dengan nilai default
                for col in missing_cols:
                    if col == 'Volume':
                        df[col] = 0  # Default volume
                    elif col == 'Date':
                        # Jika kolom Date hilang, buat dari index
                        if 'Datetime' in df.columns:
                            df['Date'] = df['Datetime']
                        else:
                            df['Date'] = df.index.astype(str)
                    else:
                        # Untuk OHLC, gunakan Close sebagai default
                        df[col] = df.get('Close', 100)  # Fallback value
                
                # Pastikan kolom sesuai urutan yang benar
                df = df[REQUIRED_COLUMNS]
                
                # Backup file lama dan simpan yang baru
                backup_path = file_path + '.backup'
                os.rename(file_path, backup_path)
                df.to_csv(file_path, index=False)
                print(f"File historical diperbaiki. Backup disimpan di: {backup_path}")
            
            return df
        else:
            # Buat file baru dengan kolom yang required
            print(f"File tidak ditemukan, membuat yang baru: {file_path}")
            df = pd.DataFrame(columns=REQUIRED_COLUMNS)
            df.to_csv(file_path, index=False)
            return df
            
    except Exception as e:
        print(f"Error memperbaiki historical data: {e}")
        # Buat dataframe kosong sebagai fallback
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

def safe_database_insert(cursor, query, params):
    """Memastikan parameter yang diinsert ke database adalah scalar values, bukan Series"""
    safe_params = []
    
    for param in params:
        # Jika parameter adalah pandas Series, ambil nilai pertama
        if isinstance(param, pd.Series):
            if len(param) > 0:
                safe_params.append(param.iloc[0])
            else:
                safe_params.append(None)
        # Jika parameter adalah numpy array atau list, ambil nilai pertama
        elif hasattr(param, '__len__') and not isinstance(param, (str, int, float)):
            if len(param) > 0:
                safe_params.append(param[0])
            else:
                safe_params.append(None)
        else:
            safe_params.append(param)
    
    cursor.execute(query, safe_params)

def get_real_time_data(pair, timeframe):
    """Mendapatkan real-time data dari yfinance dengan error handling"""
    try:
        # Format pair untuk yfinance (GBPJPY -> GBPJPY=X)
        yf_symbol = f"{pair}=X"
        print(f"Fetching real-time data for {yf_symbol} ({timeframe})...")
        
        # Download data
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="1y", interval=timeframe.lower())
        
        if hist.empty:
            print(f"Data real-time kosong untuk {yf_symbol}")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        
        # Reset index dan rename kolom
        hist = hist.reset_index()
        hist.columns = ['Date' if 'Date' in str(col) else col for col in hist.columns]
        
        # Pastikan kolom yang diperlukan ada
        for col in REQUIRED_COLUMNS:
            if col not in hist.columns:
                if col == 'Volume':
                    hist[col] = 0
                else:
                    hist[col] = hist.get('Close', 100)
        
        # Select hanya kolom yang diperlukan
        hist = hist[REQUIRED_COLUMNS]
        print(f"Using real-time data for {pair}: {len(hist)} rows")
        
        return hist
        
    except Exception as e:
        print(f"Error getting real-time data: {e}")
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

def save_to_database(pair, timeframe, data):
    """Menyimpan data ke database dengan aman"""
    try:
        conn = sqlite3.connect('trading_data.db')
        cursor = conn.cursor()
        
        # Create table jika belum ada
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                timeframe TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert data terbaru saja
        if len(data) > 0:
            latest = data.iloc[-1]
            
            # Gunakan fungsi safe insert
            safe_database_insert(cursor, '''
                INSERT INTO price_data (pair, timeframe, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pair, timeframe, latest['Date'], latest['Open'], 
                  latest['High'], latest['Low'], latest['Close'], latest['Volume']))
        
        conn.commit()
        conn.close()
        print(f"Data berhasil disimpan ke database untuk {pair} {timeframe}")
        
    except Exception as e:
        print(f"Error saving to database: {e}")

def get_analysis(pair, timeframe):
    """Fungsi utama untuk analysis yang sudah diperbaiki"""
    print(f"Processing analysis for {pair} {timeframe}")
    
    # 1. Perbaiki historical data terlebih dahulu
    historical_file = f"data/historical/{pair}_{timeframe}.csv"
    os.makedirs(os.path.dirname(historical_file), exist_ok=True)
    
    historical_df = fix_historical_data(historical_file)
    
    # 2. Dapatkan real-time data
    realtime_df = get_real_time_data(pair, timeframe)
    
    # 3. Gabungkan data jika diperlukan
    if not historical_df.empty and not realtime_df.empty:
        # Combine data, hapus duplicates berdasarkan Date
        combined_df = pd.concat([historical_df, realtime_df]).drop_duplicates('Date')
        combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    elif not realtime_df.empty:
        combined_df = realtime_df
    else:
        combined_df = historical_df
    
    # 4. Simpan data yang sudah digabungkan
    if not combined_df.empty:
        combined_df.to_csv(historical_file, index=False)
        
        # 5. Simpan ke database
        save_to_database(pair, timeframe, combined_df)
        
        # 6. Lakukan analysis di sini (tambahkan logic analysis Anda)
        analysis_result = perform_technical_analysis(combined_df)
        return analysis_result
    else:
        return {"error": "No data available for analysis"}

def perform_technical_analysis(df):
    """Contoh function untuk technical analysis"""
    if df.empty:
        return {"error": "No data for analysis"}
    
    # Contoh sederhana analysis
    latest = df.iloc[-1]
    
    return {
        "pair": "GBPJPY",
        "timeframe": "1D", 
        "price": float(latest['Close']),
        "trend": "bullish" if latest['Close'] > latest['Open'] else "bearish",
        "analysis": "Technical analysis completed successfully",
        "timestamp": datetime.now().isoformat()
    }

# Test function
if __name__ == "__main__":
    # Test perbaikan
    result = get_analysis("GBPJPY", "1D")
    print("Result:", result)
