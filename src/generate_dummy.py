import os
import json
import random
import pandas as pd
from datetime import datetime, timedelta

# === Konfigurasi ===
PAIRS = ["GBPJPY", "CHFJPY", "EURJPY", "USDJPY"]
OUTPUT_DIR = os.path.join("data")

N_PERIODS = 50  # jumlah data dummy
TIMEFRAME_HOURS = 4  # time frame H4 (4 jam)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dummy_signals():
    base_time = datetime.utcnow() - timedelta(hours=N_PERIODS * TIMEFRAME_HOURS)

    signals = {"pairs": {}}

    for pair in PAIRS:
        pair_data = []
        price = random.uniform(140, 160)  # harga awal random
        for i in range(N_PERIODS):
            ts = base_time + timedelta(hours=i * TIMEFRAME_HOURS)
            direction = random.choice(["BUY", "SELL"])
            prob_up = round(random.uniform(0.4, 0.9), 2)
            news_compound = round(random.uniform(-1, 1), 2)
            macd = round(random.uniform(-2, 2), 2)
            ema200 = price + random.uniform(-1, 1)

            if direction == "BUY":
                stop_loss = round(price - random.uniform(0.5, 1.0), 3)
                take_profit = round(price + random.uniform(0.5, 1.0), 3)
            else:
                stop_loss = round(price + random.uniform(0.5, 1.0), 3)
                take_profit = round(price - random.uniform(0.5, 1.0), 3)

            pair_data.append({
                "time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "signal": direction,
                "price": round(price, 3),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "prob_up": prob_up,
                "news_compound": news_compound,
                "macd": macd,
                "ema200": round(ema200, 3)
            })

            price += random.uniform(-0.5, 0.5)  # update harga acak

        signals["pairs"][pair] = pair_data

    return signals


if __name__ == "__main__":
    dummy = generate_dummy_signals()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = os.path.join(OUTPUT_DIR, f"signals_dummy_{ts}.json")
    last_file = os.path.join(OUTPUT_DIR, "last_signal.json")

    with open(out_file, "w") as f:
        json.dump(dummy, f, indent=2)

    with open(last_file, "w") as f:
        json.dump(dummy, f, indent=2)

    print(f"✅ Dummy signals saved to: {out_file} and {last_file}")
