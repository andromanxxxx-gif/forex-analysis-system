from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from config import settings

# Download resource NLTK (jika belum ada)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class NewsAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def analyze_sentiment(self, news_items):
        """Menganalisis sentimen dari kumpulan berita"""
        if not news_items:
            return 0, []
        
        total_sentiment = 0
        analyzed_news = []
        
        for news in news_items:
            title = news.get('title', '')
            
            # Analisis sentimen dengan TextBlob
            blob = TextBlob(title)
            polarity = blob.sentiment.polarity
            
            # Analisis sentimen dengan VADER
            vader_scores = self.sia.polarity_scores(title)
            compound = vader_scores['compound']
            
            # Gabungkan kedua skor
            combined_score = (polarity + compound) / 2
            
            # Kategorikan sentimen
            if combined_score > 0.1:
                sentiment = "Positive"
            elif combined_score < -0.1:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            analyzed_news.append({
                'title': title,
                'link': news.get('link', '#'),
                'time': news.get('time', 'Unknown'),
                'source': news.get('source', 'Unknown'),
                'sentiment': sentiment,
                'score': combined_score
            })
            
            total_sentiment += combined_score
        
        # Rata-rata sentimen
        avg_sentiment = total_sentiment / len(news_items) if news_items else 0
        
        return avg_sentiment, analyzed_news
