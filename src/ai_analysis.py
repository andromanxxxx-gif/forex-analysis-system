import requests
from config.settings import DEEPSEEK_API_KEY

def get_ai_recommendation(summary_text):
    """Integrasi DeepSeek AI"""
    url = "https://api.deepseek.ai/analyze"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"text": summary_text}
    try:
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        return result.get("recommendation", "Neutral")
    except:
        return "Neutral"
