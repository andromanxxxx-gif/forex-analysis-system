from textblob import TextBlob

class NewsAnalyzer:
    def __init__(self):
        pass
    
    def analyze_sentiment(self, headline):
        """Kembalikan Sentimen: Positive / Neutral / Negative"""
        analysis = TextBlob(headline)
        if analysis.sentiment.polarity > 0.1:
            return "Positive"
        elif analysis.sentiment.polarity < -0.1:
            return "Negative"
        else:
            return "Neutral"
