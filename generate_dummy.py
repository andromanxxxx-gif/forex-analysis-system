import os
import json
import random
from datetime import datetime, timedelta

# Folder penyimpanan data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

LAST_SIGNAL_FILE = os.path.join(DATA_DIR, "last_signal.json")

# Pasangan forex
PAIRS = ["GBPJPY", "USDJPY", "EURJPY", "CHFJPY"]

# Timeframe yang kita dukung
TIMEFRAMES = {
    "4H": 4,
    "1D": 24,
}

N_BARS = 100  # jumlah candle per timeframe

def generate_dummy_signals():
    signals = {}

    for pair in PAIRS:
        signals[pair] = {}

        for tf, hours in TIMEFRAMES.items():
            data = []
            time = datetime.utcnow() - timedelta(hours=N_BARS * hours)

            price = random.uniform(100, 200)

            for _ in range(N_BARS):
                signal = random.choice(["BUY", "SELL"])
                prob_up = round(random.uniform(0.4, 0.8), 2)

                price += random.uniform(-1, 1)

                row = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "signal": signal,
                    "price": round(price, 3),
                    "stop_loss": round(price - random.uniform(0.5, 1.5), 3),
                    "take_profit": round(price + random.uniform(0.5, 1.5), 3),
                    "prob_up": prob_up,
                    "news_compound": round(random.uniform(-1, 1), 2),
                    "news": random.choice([
                        "BOE mempertahankan suku bunga.",
                        "Fed memberi sinyal dovish.",
                        "Yen menguat karena risk-off sentiment.",
                        "Inflasi Eropa lebih rendah dari perkiraan."
                    ])
                }

                data.append(row)
                time += timedelta(hours=hours)

            signals[pair][tf] = data

    return signals

if __name__ == "__main__":
    signals = generate_dummy_signals()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_path = os.path.join(DATA_DIR, f"signals_dummy_{ts}.json")

    # Simpan file snapshot
    with open(file_path, "w") as f:
        json.dump(signals, f, indent=2)

    # Simpan juga last_signal.json
    with open(LAST_SIGNAL_FILE, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"✅ Dummy signals saved to:\n  {file_path}\n  {LAST_SIGNAL_FILE}")
