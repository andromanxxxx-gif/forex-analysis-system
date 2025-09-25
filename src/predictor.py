import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class Predictor:
    def __init__(self, model_path="models/predictor.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_or_create_model()

    def _load_or_create_model(self):
        """Load model jika ada, kalau tidak buat dummy model dan simpan."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"✅ Model berhasil dimuat dari {self.model_path}")
            except Exception as e:
                print(f"⚠️ Gagal load model, buat dummy model baru. ({e})")
                self._create_dummy_model()
        else:
            print("⚠️ Model tidak ditemukan, membuat dummy model baru...")
            self._create_dummy_model()

    def _create_dummy_model(self):
        """Buat dummy model RandomForestClassifier dengan data sintetis."""
        # Data sintetis sederhana
        np.random.seed(42)
        X = np.random.rand(500, 4)  # fitur random
        y = np.random.choice([0, 1], size=500)  # 0 = Sell, 1 = Buy

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Simpan model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        self.model = model
        print(f"✅ Dummy model disimpan ke {self.model_path}")

    def predict(self, df: pd.DataFrame):
        """
        Prediksi berdasarkan DataFrame input.
        - Jika ada kolom price, gunakan price untuk fitur.
        - Kalau tidak, gunakan fitur random dummy.
        Return:
            - predicted: list Buy/Sell
            - prob_up: probabilitas harga naik
        """
        if df.empty:
            return [], []

        # Fitur sederhana: gunakan price + random noise
        if "price" in df.columns:
            X = np.column_stack([
                df["price"].values,
                np.random.rand(len(df)),
                np.random.rand(len(df)),
                np.random.rand(len(df))
            ])
        else:
            X = np.random.rand(len(df), 4)

        probs = self.model.predict_proba(X)
        pred_labels = ["Buy" if p[1] > 0.5 else "Sell" for p in probs]
        prob_up = [float(p[1]) for p in probs]

        return pred_labels, prob_up


if __name__ == "__main__":
    # Test cepat
    dummy_data = pd.DataFrame({
        "price": np.linspace(150, 160, 10)
    })
    predictor = Predictor()
    preds, probs = predictor.predict(dummy_data)
    print("Predictions:", preds)
    print("Probabilities:", probs)
