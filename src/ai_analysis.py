# src/ai_analysis.py
import requests
from config.settings import DEEPSEEK_API_KEY

def analyze_news(pair):
    """
    Mengirim request ke DeepSeek AI untuk analisis berita real-time
    """
    url = "https://api.deepseek.ai/analyze"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"text": f"Latest news sentiment for {pair}"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        return result.get("recommendation", "HOLD")
    except Exception as e:
        print("❌ DeepSeek API error:", e)
        return "HOLD"
