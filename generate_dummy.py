import os
import json
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ==============================
# Konfigurasi
# ==============================
PAIRS = ["GBPJPY", "USDJPY", "EURJPY", "CHFJPY"]
N_BARS = 500  # jumlah candle (H4)
OUTPUT_DIR = os.path.join("data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Fungsi Generator
# ==============================
def generate_pair_data(pair, start_price):
    data = []
    time = datetime.utcnow() - timedelta(hours=N_BARS * 4)

    # nilai awal
    last_price = start_price
    last_sent = random.uniform(-0.2, 0.2)  # sentimen berita awal
    last_prob = 0.5  # netral

    prices = []

    for i in range(N_BARS):
        # simulasi harga OHLC
        open_price = last_price
        change = random.uniform(-0.8, 0.8)
        close_price = max(50, open_price + change)
        high_price = max(open_price, close_price) + random.uniform(0, 0.4)
        low_price = min(open_price, close_price) - random.uniform(0, 0.4)

        prices.append(close_price)

        # EMA200
        if len(prices) >= 200:
            ema200 = pd.Series(prices).ewm(span=200).mean().iloc[-1]
        else:
            ema200 = close_price

        # MACD sederhana (12 vs 26 EMA)
        ema12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
        macd = ema12 - ema26

        # Tentukan sinyal
        if close_price > ema200 and macd > 0:
            signal = "BUY"
        elif close_price < ema200 and macd < 0:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Probabilitas naik (berdasar arah harga)
        if change >= 0:
            last_prob = min(1, last_prob + random.uniform(0.01, 0.05))
        else:
            last_prob = max(0, last_prob - random.uniform(0.01, 0.05))

        # Sentimen berita (random walk)
        last_sent += random.uniform(-0.05, 0.05)
        last_sent = max(-1, min(1, last_sent))

        # Stop Loss / Take Profit (sederhana)
        if signal == "BUY":
            sl = close_price - random.uniform(0.5, 1.0)
            tp = close_price + random.uniform(0.5, 1.0)
        elif signal == "SELL":
            sl = close_price + random.uniform(0.5, 1.0)
            tp = close_price - random.uniform(0.5, 1.0)
        else:
            sl, tp = None, None

        # Simpan record
        data.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "open": round(open_price, 3),
            "high": round(high_price, 3),
            "low": round(low_price, 3),
            "price": round(close_price, 3),
            "signal": signal,
            "stop_loss": round(sl, 3) if sl else None,
            "take_profit": round(tp, 3) if tp else None,
            "prob_up": round(last_prob, 2),
            "news_compound": round(last_sent, 2)
        })

        # Update
        last_price = close_price
        time += timedelta(hours=4)

    return data


# ==============================
# Main Generator
# ==============================
def main():
    signals = {}
    for pair in PAIRS:
        start_price = random.uniform(130, 160) if "JPY" in pair else random.uniform(1.0, 2.0)
        signals[pair] = generate_pair_data(pair, start_price)

    # simpan ke file dengan timestamp
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_file = os.path.join(OUTPUT_DIR, f"signals_dummy_{ts}.json")
    last_file = os.path.join(OUTPUT_DIR, "last_signal.json")

    with open(out_file, "w") as f:
        json.dump(signals, f, indent=2)

    with open(last_file, "w") as f:
        json.dump(signals, f, indent=2)

    print(f"✅ Dummy signals saved to:\n  {out_file}\n  {last_file}")


if __name__ == "__main__":
    main()
