import requests

class NewsAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.endpoint = "https://api.deepseek.ai/analyze"

    def analyze(self, pair):
        """
        Ambil berita terbaru dan berikan ringkasan analisis AI
        Menggunakan DeepSeek API
        """
        try:
            payload = {
                "api_key": self.api_key,
                "query": f"Forex {pair} latest news analysis"
            }
            response = requests.post(self.endpoint, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get("summary", "No summary available")
            else:
                return f"Error from AI API: {response.status_code}"
        except Exception as e:
            return f"Exception during AI analysis: {str(e)}"
