# src/generate_dummy.py
import os
import json
import random
from datetime import datetime

# Folder tujuan simpan data dummy
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

PAIRS = ["GBPJPY", "CHFJPY", "EURJPY", "USDJPY"]

def gen_signal(symbol):
    prob = round(random.uniform(0.2, 0.8), 3)   # probabilitas dummy
    direction = "long" if prob > 0.55 else "short" if prob < 0.45 else "neutral"
    last_price = round(random.uniform(100, 200), 3)

    if direction == "long":
        tp = round(last_price * (1 + 0.02), 3)
        sl = round(last_price * (1 - 0.01), 3)
    elif direction == "short":
        tp = round(last_price * (1 - 0.02), 3)
        sl = round(last_price * (1 + 0.01), 3)
    else:
        tp, sl = None, None

    return {
        "method": "dummy",
        "prob_up": prob,
        "direction": direction,
        "tp_price": tp,
        "sl_price": sl,
        "last_price": last_price,
        "news_compound": round(random.uniform(-0.5, 0.5), 3)
    }

def main():
    results = {"generated_at": datetime.utcnow().isoformat(), "pairs": {}}

    for s in PAIRS:
        results["pairs"][s] = {
            "symbol": s,
            "timestamp": datetime.utcnow().isoformat(),
            "result": gen_signal(s)
        }

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = os.path.join(DATA_DIR, f"signals_dummy_{ts}.json")
    last_file = os.path.join(DATA_DIR, "last_signal.json")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    with open(last_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Dummy signals saved to:", out_file, "and", last_file)

if __name__ == "__main__":
    main()
