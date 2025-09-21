import pandas as pd
import numpy as np
from datetime import datetime
from src.data_collection import DataCollector
from src.technical_analysis import TechnicalAnalyzer
from src.news_analyzer import NewsAnalyzer
from src.signal_generator import SignalGenerator
from src.utils import save_to_csv, load_pairs_list
from config import settings

class ForexAnalyzer:
    def __init__(self):
        self.data_collector = DataCollector()
        self.technical_analyzer = TechnicalAnalyzer()
        self.news_analyzer = NewsAnalyzer()
        self.signal_generator = SignalGenerator()
        self.results = {}
    
    def analyze_all_pairs(self, pairs=None):
        """Menganalisis semua pasangan forex"""
        if pairs is None:
            pairs = load_pairs_list()
        
        print("Mengambil data untuk semua pasangan...")
        all_data = self.data_collector.fetch_all_data(pairs)
        
        print("Menganalisis teknikal...")
        analyzed_data = {}
        for pair, data in all_data.items():
            analyzed_data[pair] = self.technical_analyzer.analyze(data)
        
        print("Mengambil dan menganalisis berita...")
        news_items = self.data_collector.scrape_news()
        avg_sentiment, analyzed_news = self.news_analyzer.analyze_sentiment(news_items)
        
        print("Menghasilkan sinyal trading...")
        signals = {}
        for pair, data in analyzed_data.items():
            signals[pair] = self.signal_generator.generate_signals(data, avg_sentiment)
        
        # Simpan hasil
        self.results = {
            'analyzed_data': analyzed_data,
            'signals': signals,
            'news_sentiment': avg_sentiment,
            'analyzed_news': analyzed_news
        }
        
        return self.results
    
    def generate_report(self):
        """Membuat laporan analisis"""
        if not self.results:
            print("Tidak ada hasil analisis. Jalankan analyze_all_pairs() terlebih dahulu.")
            return
        
        signals = self.results['signals']
        analyzed_news = self.results['analyzed_news']
        
        # Buat DataFrame untuk sinyal
        signals_list = []
        for pair, signal in signals.items():
            if signal:
                signals_list.append({
                    'Pair': pair.replace('=X', ''),
                    'Signal': signal['signal'],
                    'Entry': signal['entry'],
                    'Stop Loss': signal['stop_loss'],
                    'Take Profit': signal['take_profit'],
                    'Confidence': f"{signal['confidence']*100:.2f}%",
                    'News Sentiment': f"{signal['news_sentiment']:.4f}"
                })
        
        signals_df = pd.DataFrame(signals_list)
        
        # Buat DataFrame untuk berita
        news_df = pd.DataFrame(analyzed_news)
        
        return signals_df, news_df
    
    def print_results(self):
        """Mencetak hasil analisis ke konsol"""
        if not self.results:
            print("Tidak ada hasil analisis. Jalankan analyze_all_pairs() terlebih dahulu.")
            return
        
        signals = self.results['signals']
        avg_sentiment = self.results['news_sentiment']
        analyzed_news = self.results['analyzed_news']
        
        print("\n" + "="*80)
        print("FOREX ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nNews Sentiment Average: {avg_sentiment:.4f}")
        if avg_sentiment > 0.1:
            print("Overall News Sentiment: POSITIVE")
        elif avg_sentiment < -0.1:
            print("Overall News Sentiment: NEGATIVE")
        else:
            print("Overall News Sentiment: NEUTRAL")
        
        print("\nTRADING SIGNALS:")
        print("-" * 80)
        
        for pair, signal in signals.items():
            if signal:
                pair_name = pair.replace('=X', '')
                color = settings.COLORS.get(signal['signal'].lower(), settings.COLORS['neutral'])
                
                print(f"{pair_name}:")
                print(f"  Signal: {color}{signal['signal']}{settings.COLORS['end']}")
                print(f"  Entry: {signal['entry']:.5f}")
                print(f"  Stop Loss: {signal['stop_loss']:.5f}")
                print(f"  Take Profit: {signal['take_profit']:.5f}")
                print(f"  Confidence: {signal['confidence']*100:.2f}%")
                print(f"  News Sentiment: {signal['news_sentiment']:.4f}")
                print()
        
        print("\nTOP NEWS:")
        print("-" * 80)
        for i, news in enumerate(analyzed_news[:5], 1):
            sentiment_color = settings.COLORS.get(news['sentiment'].lower(), settings.COLORS['neutral'])
            print(f"{i}. {news['title']}")
            print(f"   Source: {news['source']} | Sentiment: {sentiment_color}{news['sentiment']}{settings.COLORS['end']} ({news['score']:.4f})")
            print()
