import requests
import json

class AIPredictor:
    def __init__(self, api_key="sk-73d83584fd614656926e1d8860eae9ca"):
        self.api_key = api_key
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
    
    def get_prediction(self, pair, technical_analysis, news):
        """Dapatkan prediksi dari AI"""
        prompt = f"""
        Berikan analisis dan prediksi untuk pair {pair} berdasarkan analisis teknikal dan berita berikut:

        Analisis Teknikal:
        {technical_analysis}

        Berita Terkini:
        {news}

        Berikan rekomendasi trading (BUY/SELL), level Take Profit dan Stop Loss, serta prediksi pergerakan harga untuk beberapa waktu ke depan.
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Anda adalah analis forex profesional."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"
