import os
import json
import random
from datetime import datetime, timedelta

# Folder penyimpanan data
DATA_DIR = os.path.join("data")
os.makedirs(DATA_DIR, exist_ok=True)

# Pasangan forex yang dianalisis
PAIRS = ["GBPJPY", "CHFJPY", "EURJPY", "USDJPY"]

def generate_signals(n=50):
    """
    Membuat data dummy signals berupa list of dict untuk setiap pair.
    """
    signals = {}
    now = datetime.utcnow()

    for pair in PAIRS:
        pair_signals = []
        base_price = random.uniform(150, 200)  # harga dasar acak per pair

        for i in range(n):
            ts = (now - timedelta(hours=4 * (n - i))).strftime("%Y-%m-%d %H:%M:%S")
            price = base_price + random.uniform(-2, 2)
            ema200 = base_price + random.uniform(-1, 1)
            macd = random.uniform(-1, 1)
            signal_type = random.choice(["BUY", "SELL"])
            stop_loss = round(price - random.uniform(0.5, 1.5), 2)
            take_profit = round(price + random.uniform(0.5, 1.5), 2)

            pair_signals.append({
                "time": ts,
                "price": round(price, 2),
                "ema200": round(ema200, 2),
                "macd": round(macd, 2),
                "signal": signal_type,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            })

        signals[pair] = pair_signals

    return signals


if __name__ == "__main__":
    signals = generate_signals(n=50)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_path = os.path.join(DATA_DIR, f"signals_dummy_{ts}.json")
    last_file_path = os.path.join(DATA_DIR, "last_signal.json")

    # Simpan file dummy penuh
    with open(file_path, "w") as f:
        json.dump(signals, f, indent=2)

    # Simpan file "last_signal.json" untuk dashboard (tetap format list)
    with open(last_file_path, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"Dummy signals saved to: {file_path} and {last_file_path}")
