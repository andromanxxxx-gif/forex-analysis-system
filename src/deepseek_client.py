# src/deepseek_client.py
import requests
from config import settings

def analyze_market(technical_summary: str, news_summary: str = ""):
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"text": f"Technical Analysis:\n{technical_summary}\nNews Analysis:\n{news_summary}"}
    
    try:
        resp = requests.post(settings.DEEPSEEK_API_ENDPOINT, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
