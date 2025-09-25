import os
import json
import random
from datetime import datetime, timedelta
import numpy as np

# === Konfigurasi ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

PAIRS = ["GBPJPY", "CHFJPY", "EURJPY", "USDJPY"]
TIMEFRAMES = {"4h": 4, "1d": 24}
N_BARS = 200

NEWS_LIST = [
    "Bank of Japan mempertahankan suku bunga.",
    "The Fed memberi sinyal kenaikan suku bunga.",
    "ECB mengumumkan stimulus baru.",
    "Data inflasi Inggris di bawah ekspektasi.",
    "Pasar global melemah akibat ketidakpastian geopolitik."
]

def generate_dummy_signals():
    signals = {}
    base_time = datetime.utcnow() - timedelta(hours=N_BARS * 4)

    for pair in PAIRS:
        signals[pair] = {}
        for tf, hours in TIMEFRAMES.items():
            pair_data = []
            price = random.uniform(100, 200)

            for i in range(N_BARS):
                time = base_time + timedelta(hours=i * hours)
                price += random.uniform(-0.5, 0.5) + np.sin(i / 10) * 0.2

                prob_up = round(random.uniform(0.4, 0.7), 2)
                signal = "BUY" if prob_up > 0.5 else "SELL"

                sl = round(price - random.uniform(0.3, 0.6), 3)
                tp = round(price + random.uniform(0.3, 0.6), 3)
                pred_price_next = round(price * (1 + random.uniform(-0.002, 0.002)), 3)

                news_headline = random.choice(NEWS_LIST)
                news_compound = round(random.uniform(-1, 1), 2)

                pair_data.append({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "price": round(price, 3),
                    "signal": signal,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "prob_up": prob_up,
                    "prob_down": round(1 - prob_up, 2),
                    "news_headline": news_headline,
                    "news_compound": news_compound,
                    "pred_price_next": pred_price_next
                })

            signals[pair][tf] = pair_data

    return signals


if __name__ == "__main__":
    signals = generate_dummy_signals()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    file_path = os.path.join(DATA_DIR, f"signals_dummy_{ts}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)

    with open(os.path.join(DATA_DIR, "last_signal.json"), "w", encoding="utf-8") as f:
        json.dump(signals, f, indent=2, ensure_ascii=False)

    print(f"✅ Dummy signals saved:\n  {file_path}\n  {os.path.join(DATA_DIR, 'last_signal.json')}")
