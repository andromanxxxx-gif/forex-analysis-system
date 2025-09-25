import os
import json
import random
from datetime import datetime, timedelta
import numpy as np

# === Konfigurasi ===
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

PAIRS = ["GBPJPY", "CHFJPY", "EURJPY", "USDJPY"]
N_BARS = 300  # jumlah bar (4 jam per bar = 50 hari data)
TIMEFRAME_HOURS = 4


def generate_dummy_signals():
    signals = {}
    base_time = datetime.utcnow() - timedelta(hours=N_BARS * TIMEFRAME_HOURS)

    for pair in PAIRS:
        pair_data = []

        price = random.uniform(100, 200)  # harga awal random
        for i in range(N_BARS):
            # waktu candle
            time = base_time + timedelta(hours=i * TIMEFRAME_HOURS)

            # harga bergerak random + trend kecil
            price += random.uniform(-0.5, 0.5) + np.sin(i / 10) * 0.2

            # probabilitas naik
            prob_up = round(random.uniform(0.4, 0.7), 2)
            prob_down = round(1 - prob_up, 2)

            # sinyal BUY/SELL
            signal = "BUY" if prob_up > 0.5 else "SELL"

            # stop loss / take profit (dummy)
            sl = round(price - random.uniform(0.3, 0.6), 3)
            tp = round(price + random.uniform(0.3, 0.6), 3)

            # prediksi harga candle berikutnya
            pred_price_next = round(price * (1 + random.uniform(-0.002, 0.002)), 3)

            # news sentiment dummy
            news_compound = round(random.uniform(-1, 1), 2)

            pair_data.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "price": round(price, 3),
                "signal": signal,
                "stop_loss": sl,
                "take_profit": tp,
                "prob_up": prob_up,
                "prob_down": prob_down,
                "news_compound": news_compound,
                "pred_price_next": pred_price_next
            })

        signals[pair] = pair_data

    return signals


if __name__ == "__main__":
    signals = generate_dummy_signals()

    # filename berdasarkan timestamp
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_path = os.path.join(DATA_DIR, f"signals_dummy_{ts}.json")

    with open(file_path, "w") as f:
        json.dump(signals, f, indent=2)

    with open(os.path.join(DATA_DIR, "last_signal.json"), "w") as f:
        json.dump(signals, f, indent=2)

    print(f"✅ Dummy signals saved to:\n  {file_path}\n  {os.path.join(DATA_DIR, 'last_signal.json')}")
