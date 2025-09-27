import requests
from config.settings import DEEPSEEK_API_KEY

def analyze_news(text):
    url = "https://api.deepseek.ai/v1/analyze"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}
    payload = {"text": text}

    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        return data.get("recommendation", "No recommendation")
    except Exception as e:
        return f"Error: {e}"
