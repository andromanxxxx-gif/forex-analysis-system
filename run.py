def load_historical_data(self, timeframe):
    """Load data historis dari CSV dengan perbaikan data validation"""
    try:
        filename = f"data/XAUUSD_{timeframe}.csv"
        if not os.path.exists(filename):
            print(f"File {filename} not found, generating sample data...")
            return self.generate_sample_data(timeframe)
            
        df = pd.read_csv(filename)
        print(f"Loaded {len(df)} rows from {filename}")
        
        # Clean the data
        df = self.clean_dataframe(df, timeframe)
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return self.generate_sample_data(timeframe)

def clean_dataframe(self, df, timeframe):
    """Bersihkan dan validasi dataframe"""
    # Handle berbagai format kolom
    datetime_col = None
    for col in ['datetime', 'date', 'time', 'Timestamp', 'timestamp']:
        if col in df.columns:
            datetime_col = col
            break
    
    if datetime_col:
        df['datetime'] = pd.to_datetime(df[datetime_col], errors='coerce')
    else:
        # Jika tidak ada kolom datetime, buat dari index
        freq = self.get_freq(timeframe)
        df['datetime'] = pd.date_range(end=datetime.now(), periods=len(df), freq=freq)
    
    # Remove rows with invalid datetime
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')
    
    # Ensure required columns exist dengan nilai yang valid
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            print(f"Column {col} not found, generating...")
            if col == 'open':
                df['open'] = df['close'] * 0.999 if 'close' in df.columns else 1800.0
            elif col == 'high':
                df['high'] = (df['close'] * 1.002) if 'close' in df.columns else 1800.0
            elif col == 'low':
                df['low'] = (df['close'] * 0.998) if 'close' in df.columns else 1800.0
            elif col == 'close':
                df['close'] = 1800.0  # Default value
    
    # Convert to numeric and handle invalid values
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill NaN values with forward/backward fill
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(1800.0)
    
    if 'volume' not in df.columns:
        df['volume'] = np.random.randint(1000, 10000, len(df))
        
    print(f"Data cleaned: {len(df)} rows, columns: {list(df.columns)}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    
    return df
