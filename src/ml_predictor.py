import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os

class MLForexPredictor:
    def __init__(self, lookback=60, forecast_horizon=5):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.scalers = {}
    
    def load_model(self, pair, model_path, scaler_path):
        """Memuat model dan scaler untuk pair tertentu"""
        try:
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path)
            
            self.models[pair] = model
            self.scalers[pair] = scaler
            return True
        except Exception as e:
            print(f"Error loading model for {pair}: {e}")
            return False
    
    def prepare_prediction_data(self, data, pair):
        """Mempersiapkan data untuk prediksi"""
        if pair not in self.scalers:
            print(f"No scaler found for {pair}")
            return None
        
        # Pilih fitur yang sesuai dengan yang digunakan saat training
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA200', 'MACD', 'MACD_Signal', 'ATR']
        available_features = [f for f in features if f in data.columns]
        
        # Ambil data terbaru
        recent_data = data[available_features].tail(self.lookback).values
        
        # Scale data
        scaled_data = self.scalers[pair].transform(recent_data)
        
        # Reshape untuk LSTM
        X = np.array([scaled_data])
        
        return X
    
    def predict(self, data, pair):
        """Membuat prediksi menggunakan model yang sudah dilatih"""
        if pair not in self.models or pair not in self.scalers:
            print(f"Model or scaler not loaded for {pair}")
            return None
        
        # Siapkan data untuk prediksi
        X = self.prepare_prediction_data(data, pair)
        if X is None:
            return None
        
        # Buat prediksi
        prediction_scaled = self.models[pair].predict(X)
        
        # Inverse transform untuk mendapatkan nilai sebenarnya
        # Buat dummy array dengan bentuk yang sesuai
        dummy_array = np.zeros((prediction_scaled.shape[0], prediction_scaled.shape[1], len(self.scalers[pair].feature_names_in_)))
        dummy_array[:, :, 3] = prediction_scaled  # Hanya kolom 'Close'
        
        prediction = self.scalers[pair].inverse_transform(dummy_array[0])[:, 3]
        
        return prediction
    
    def calculate_confidence(self, data, prediction):
        """Menghitung confidence level berdasarkan recent accuracy"""
        if data is None or prediction is None:
            return 0.5
        
        # Bandingkan prediksi sebelumnya dengan nilai aktual
        recent_actual = data['Close'].values[-5:]
        
        if len(recent_actual) < 5:
            return 0.5
        
        # Hitung RMSE recent
        rmse = np.sqrt(np.mean((recent_actual - prediction[:len(recent_actual)])**2))
        
        # Normalize RMSE to confidence score (0-1)
        price_range = np.max(recent_actual) - np.min(recent_actual)
        if price_range > 0:
            normalized_rmse = rmse / price_range
            confidence = max(0.1, 1 - normalized_rmse)
        else:
            confidence = 0.5
        
        return confidence
