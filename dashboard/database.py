# database.py
import sqlite3
import json
from datetime import datetime
import os

class Database:
    def __init__(self, db_path='forex_analysis.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                current_price REAL,
                price_change REAL,
                technical_indicators TEXT,
                ai_analysis TEXT,
                chart_data TEXT
            )
        ''')
        
        # News table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                headline TEXT,
                timestamp TEXT,
                url TEXT,
                saved_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT,
                timeframe TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, analysis_data):
        """Save analysis results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_results 
                (pair, timeframe, timestamp, current_price, price_change, technical_indicators, ai_analysis, chart_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_data['pair'],
                analysis_data['timeframe'],
                analysis_data['timestamp'],
                analysis_data['current_price'],
                analysis_data['price_change'],
                json.dumps(analysis_data['technical_indicators']),
                json.dumps(analysis_data['ai_analysis']),
                json.dumps(analysis_data.get('chart_data', {}))
            ))
            
            conn.commit()
            conn.close()
            print(f"Analysis saved for {analysis_data['pair']}")
            
        except Exception as e:
            print(f"Error saving analysis: {e}")
    
    def save_news(self, news_items):
        """Save news items to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for news in news_items:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_items (source, headline, timestamp, url)
                    VALUES (?, ?, ?, ?)
                ''', (news['source'], news['headline'], news['timestamp'], news['url']))
            
            conn.commit()
            conn.close()
            print(f"News items saved: {len(news_items)}")
            
        except Exception as e:
            print(f"Error saving news: {e}")
    
    def save_price_data(self, pair, timeframe, data):
        """Save price data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if len(data) > 0:
                latest = data.iloc[-1]
                
                # Convert index to date string if it's a timestamp
                if hasattr(data.index, 'strftime'):
                    date_str = data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    date_str = str(data.index[-1])
                
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data 
                    (pair, timeframe, date, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pair, timeframe, date_str,
                    float(latest['Open']), float(latest['High']),
                    float(latest['Low']), float(latest['Close']),
                    float(latest.get('Volume', 0))
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving price data: {e}")

# Global database instance
db = Database()
