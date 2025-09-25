import pandas as pd
import numpy as np
import joblib
import os

class Predictor:
    def __init__(self, horizon_hours=4, model_path="../models/predictor_model.pkl"):
        self.horizon = horizon_hours
        self.model_path = model_path
        self.model = None

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("✅ Loaded trained model from", self.model_path)
        else:
            print("⚠️ Model not found, fallback ke prediksi sederhana.")

    def predict_next(self, df: pd.DataFrame):
        if "price" not in df.columns:
            raise ValueError("DataFrame harus ada kolom 'price'.")

        last_price = df["price"].iloc[-1]
        mean_price = df["price"].tail(10).mean()

        # Default prediksi sederhana
        predicted_price = mean_price + (last_price - mean_price) * 0.3

        signal = "BUY" if predicted_price > last_price else "SELL"

        if self.model:
            # Feature untuk model
            ema_200 = df["price"].ewm(span=200).mean().iloc[-1]
            macd = df["price"].ewm(span=12).mean().iloc[-1] - df["price"].ewm(span=26).mean().iloc[-1]
            ret = df["price"].pct_change().iloc[-1]

            X_new = pd.DataFrame([[ret, ema_200, macd]], columns=["return", "ema_200", "macd"])
            pred_class = self.model.predict(X_new)[0]

            signal = "BUY" if pred_class == 1 else "SELL"
            predicted_price = last_price * (1.002 if pred_class == 1 else 0.998)

        # Hitung TP/SL
        if signal == "BUY":
            take_profit = round(predicted_price * 1.005, 3)
            stop_loss = round(predicted_price * 0.997, 3)
        else:
            take_profit = round(predicted_price * 0.995, 3)
            stop_loss = round(predicted_price * 1.003, 3)

        return {
            "Predicted Signal": signal,
            "Predicted Price": round(predicted_price, 3),
            "Take Profit": take_profit,
            "Stop Loss": stop_loss,
            "Next Time": df["time"].iloc[-1] + pd.Timedelta(hours=self.horizon)
        }
