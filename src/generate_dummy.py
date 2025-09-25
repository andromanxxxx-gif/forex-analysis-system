import os
import json
import random
from datetime import datetime, timedelta

# Folder data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_dummy_signals(pair: str, n: int = 300):
    """Generate dummy OHLC + signals untuk pair forex"""
    now = datetime.utcnow()
    start_price = random.uniform(100, 170)  # harga awal random

    signals = []
    for i in range(n):
        # waktu candle H4
        ts = now - timedelta(hours=4 * (n - i))

        # OHLC simulasi
        open_price = start_price + random.uniform(-1, 1)
        close_price = open_price + random.uniform(-0.8, 0.8)
        high_price = max(open_price, close_price) + random.uniform(0, 0.5)
        low_price = min(open_price, close_price) - random.uniform(0, 0.5)

        # prediksi sinyal
        signal = "BUY" if close_price > open_price else "SELL"

        # TP/SL dummy (0.3% – 0.6%)
        if signal == "BUY":
            take_profit = round(close_price * (1 + random.uniform(0.003, 0.006)), 3)
            stop_loss = round(close_price * (1 - random.uniform(0.003, 0.006)), 3)
        else:
            take_profit = round(close_price * (1 - random.uniform(0.003, 0.006)), 3)
            stop_loss = round(close_price * (1 + random.uniform(0.003, 0.006)), 3)

        prob_up = round(random.uniform(0.3, 0.7), 2)
        news_compound = round(random.uniform(-1, 1), 2)

        signals.append({
            "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(open_price, 3),
            "high": round(high_price, 3),
            "low": round(low_price, 3),
            "price": round(close_price, 3),   # close
            "signal": signal,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "prob_up": prob_up,
            "news_compound": news_compound
        })

        start_price = close_price

    return signals

def main():
    pairs = ["GBPJPY", "USDJPY", "EURJPY", "CHFJPY"]
    all_data = {}

    for pair in pairs:
        all_data[pair] = generate_dummy_signals(pair, n=400)

    # Simpan ke file
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_path = os.path.join(DATA_DIR, f"signals_dummy_{ts}.json")
    last_file = os.path.join(DATA_DIR, "last_signal.json")

    with open(file_path, "w") as f:
        json.dump(all_data, f, indent=2)

    with open(last_file, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Dummy signals saved to: {file_path}")
    print(f"✅ Latest dummy saved to: {last_file}")

if __name__ == "__main__":
    main()
