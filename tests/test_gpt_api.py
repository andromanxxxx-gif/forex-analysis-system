"""
test_gpt_api.py
Uji koneksi ke OpenAI GPT-4o dan NewsAPI untuk sistem analisis forex.
"""

import os
import requests
from openai import OpenAI
from config import OPENAI_API_KEY, NEWS_API_KEY, check_config

# ==========================
# 🔍 Cek Konfigurasi Dasar
# ==========================
print("=== TEST CONFIG ===")
check_config()

# ==========================
# ⚙️ Tes Koneksi ke OpenAI
# ==========================
print("🔹 Menguji koneksi ke GPT-4o...")

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Kamu adalah asisten analisis forex profesional."},
            {"role": "user", "content": "Buatkan analisis singkat EUR/USD berdasarkan kondisi pasar global terkini."}
        ],
        max_tokens=150,
    )
    print("✅ Koneksi GPT-4o BERHASIL\n")
    print("📊 Contoh respons GPT-4o:")
    print(response.choices[0].message.content)
except Exception as e:
    print("❌ Gagal konek ke GPT-4o:")
    print(e)

# ==========================
# ⚙️ Tes Koneksi ke NewsAPI
# ==========================
print("\n🔹 Menguji koneksi ke NewsAPI...")

try:
    url = f"https://newsapi.org/v2/top-headlines?q=forex&language=en&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        print("✅ Koneksi NewsAPI BERHASIL")
        print(f"📑 Ditemukan {len(data.get('articles', []))} artikel terkait forex.")
        if data.get("articles"):
            first = data["articles"][0]
            print(f"📰 Judul contoh: {first['title']}")
            print(f"🔗 Sumber: {first['source']['name']}")
    else:
        print(f"❌ Gagal NewsAPI (status: {r.status_code})")
except Exception as e:
    print("❌ Error saat akses NewsAPI:")
    print(e)
