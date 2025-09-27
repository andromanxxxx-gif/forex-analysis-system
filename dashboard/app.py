import sys
from pathlib import Path

# Tambahkan root project ke sys.path agar Python bisa menemukan modul src
root_path = Path(__file__).parent.parent.resolve()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from src.trading_signals import calculate_indicators, generate_signal

app = Flask(__name__)

# Dummy data generator
def get_dummy_data(pair):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
    data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.rand(100) * 100,
        'Close': np.random.rand(100) * 100,
        'High': np.random.rand(100) * 100,
        'Low': np.random.rand(100) * 100,
        'Volume': np.random.randint(100, 1000, size=100)
    })
    data = calculate_indicators(data)
    return data

@app.route("/", methods=["GET", "POST"])
def index():
    pair = request.form.get("pair", "EUR/USD")
    data = get_dummy_data(pair)
    signal = generate_signal(data)
    return render_template("index.html", pair=pair, signal=signal, tables=[data.to_html(classes='data')], titles=data.columns.values)

if __name__ == "__main__":
    app.run(debug=True)
