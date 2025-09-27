from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
from src.trading_signals import calculate_indicators, generate_signal
from src.drive_integration import authenticate_drive, download_file
import threading, time, os

app = Flask(__name__)

pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY']
data_files = {
    'EUR/USD': 'data/raw/EURUSD.csv',
    'GBP/USD': 'data/raw/GBPUSD.csv',
    'USD/JPY': 'data/raw/USDJPY.csv'
}
# Google Drive file IDs
drive_file_ids = {
    'EUR/USD': 'YOUR_FILE_ID_EURUSD',
    'GBP/USD': 'YOUR_FILE_ID_GBPUSD',
    'USD/JPY': 'YOUR_FILE_ID_USDJPY'
}

# Fungsi auto-update data dari Drive
def update_data():
    service = authenticate_drive()
    while True:
        for pair, file_id in drive_file_ids.items():
            download_file(service, file_id, data_files[pair])
        time.sleep(10)  # update tiap 10 detik

# Jalankan thread background untuk update data
threading.Thread(target=update_data, daemon=True).start()

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_pair = request.form.get('pair') or 'EUR/USD'
    file_path = data_files[selected_pair]
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = calculate_indicators(df)
        signal, tp, sl = generate_signal(df)
    else:
        df = pd.DataFrame()
        signal = tp = sl = None

    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candlestick'
        ))
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA200'], mode='lines', name='EMA200'))

    graphJSON = fig.to_json()

    return render_template('index.html', pairs=pairs, selected_pair=selected_pair,
                           signal=signal, tp=tp, sl=sl, graphJSON=graphJSON)

if __name__ == "__main__":
    app.run(debug=True)
