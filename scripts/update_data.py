#!/usr/bin/env python3
"""
Script to update forex data regularly
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.data_collection import DataCollector
from src.technical_analysis import TechnicalAnalyzer
from src.news_analyzer import NewsAnalyzer
import os
import json

def update_forex_data():
    """Update forex data and save to file"""
    print(f"{datetime.now()}: Starting data update...")
    
    # Initialize components
    data_collector = DataCollector()
    technical_analyzer = TechnicalAnalyzer()
    news_analyzer = NewsAnalyzer()
    
    # Load pairs list
    pairs = []
    with open('config/pairs_list.txt', 'r') as f:
        pairs = [line.strip() for line in f if line.strip()]
    
    # Fetch and process data
    all_data = data_collector.fetch_all_data(pairs)
    analyzed_data = {}
    
    for pair, data in all_data.items():
        analyzed_data[pair] = technical_analyzer.analyze(data)
    
    # Get news
    news_items = data_collector.scrape_news()
    avg_sentiment, analyzed_news = news_analyzer.analyze_sentiment(news_items)
    
    # Save data
    data_dir = 'data'
    os.makedirs(f'{data_dir}/historical', exist_ok=True)
    os.makedirs(f'{data_dir}/processed', exist_ok=True)
    os.makedirs(f'{data_dir}/news', exist_ok=True)
    
    # Save processed data
    for pair, data in analyzed_data.items():
        if data is not None:
            filename = f"{pair.replace('=', '').lower()}_processed.csv"
            data.to_csv(f'{data_dir}/processed/{filename}')
    
    # Save news
    news_data = {
        'timestamp': datetime.now().isoformat(),
        'avg_sentiment': avg_sentiment,
        'news_items': analyzed_news
    }
    
    with open(f'{data_dir}/news/latest_news.json', 'w') as f:
        json.dump(news_data, f)
    
    print(f"{datetime.now()}: Data update completed!")

if __name__ == "__main__":
    update_forex_data()
