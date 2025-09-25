import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ======================
# Utility: buat dummy data jika tidak ada dataset
# ======================
def generate_dummy_data(n=1000):
    np.random.seed(42)
    df = pd.DataFrame({
        "price": np.linspace(150, 200, n) + np.random.randn(n) * 2,
        "macd": np.random.randn(n),
        "ema200": 180 + np.random.randn(n) * 5,
        "news_compound": np.random.uniform(-1, 1, n),
        "signal": np.random.choice([0, 1], size=n)  # 0 = Sell, 1 = Buy
    })
    return df

# ======================
# Train Model
# ======================
def train_and_save_model(data: pd.DataFrame, model_path="models/predictor.pkl"):
    # Pastikan kolom yang dibutuhkan ada
    required_cols = ["price", "macd", "ema200", "news_compound", "signal"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"❌ Kolom {col} tidak ditemukan di dataset.")

    X = data[["price", "macd", "ema200", "news_compound"]]
    y = data["signal"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model XGBoost
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"✅ Model trained, akurasi test = {acc:.2f}")

    # Simpan model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model disimpan ke {model_path}")

if __name__ == "__main__":
    # Buat data dummy dulu (nanti bisa diganti dengan data real dari scraping/ohlcv)
    df = generate_dummy_data(1000)
    train_and_save_model(df)
