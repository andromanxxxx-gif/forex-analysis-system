import os
import json
import random
from datetime import datetime, timedelta

# === Konfigurasi Dummy ===
PAIRS = ["GBPJPY", "CHFJPY", "EURJPY", "USDJPY"]
N_BARS = 200  # jumlah candle dummy
TIMEFRAME = "4h"
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

def generate_signals():
    signals = {}
    now = datetime.utcnow()
    start_time = now - timedelta(hours=N_BARS * 4)

    for pair in PAIRS:
        pair_data = []
        price = random.uniform(150, 170)  # harga awal acak

        for i in range(N_BARS):
            time = start_time + timedelta(hours=i * 4)

            # Simulasi harga naik turun
            price += random.uniform(-0.5, 0.5)
            signal = random.choice(["BUY", "SELL"])
            stop_loss = price - random.uniform(0.3, 0.8) if signal == "BUY" else price + random.uniform(0.3, 0.8)
            take_profit = price + random.uniform(0.3, 0.8) if signal == "BUY" else price - random.uniform(0.3, 0.8)

            # Probabilitas & sentimen
            prob_up = round(random.uniform(0.3, 0.8), 2)
            news_compound = round(random.uniform(-1, 1), 2)

            # Prediksi harga candle berikutnya
            pred_price_next = round(price * (1 + random.uniform(-0.002, 0.002)), 3)

            pair_data.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "signal": signal,
                "price": round(price, 3),
                "stop_loss": round(stop_loss, 3),
                "take_profit": round(take_profit, 3),
                "prob_up": prob_up,
                "news_compound": news_compound,
                "pred_price_next": pred_price_next
            })

        signals[pair] = pair_data

    return signals


if __name__ == "__main__":
    signals = generate_signals()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dummy_file = os.path.join(DATA_DIR, f"signals_dummy_{ts}.json")
    last_file = os.path.join(DATA_DIR, "last_signal.json")

    with open(dummy_file, "w") as f:
        json.dump(signals, f, indent=2)

    with open(last_file, "w") as f:
        json.dump(signals, f, indent=2)
