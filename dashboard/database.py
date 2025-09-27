# database.py
import sqlite3
import pandas as pd
from datetime import datetime
import json

class ForexDatabase:
    def __init__(self, db_path='forex_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table untuk analisis trading
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                current_price REAL,
                signal TEXT,
                confidence REAL,
                entry_price REAL,
                take_profit_1 REAL,
                take_profit_2 REAL,
                stop_loss REAL,
                risk_reward TEXT,
                analysis_summary TEXT,
                technical_indicators TEXT
            )
        ''')
        
        # Table untuk news
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                headline TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                url TEXT
            )
        ''')
        
        # Table untuk price data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, analysis_data):
        """Save analysis data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_analysis 
            (pair, timeframe, current_price, signal, confidence, entry_price, 
             take_profit_1, take_profit_2, stop_loss, risk_reward, analysis_summary, technical_indicators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_data['pair'],
            analysis_data['timeframe'],
            analysis_data['current_price'],
            analysis_data['ai_analysis']['SIGNAL'],
            analysis_data['ai_analysis']['CONFIDENCE_LEVEL'],
            analysis_data['ai_analysis']['ENTRY_PRICE'],
            analysis_data['ai_analysis']['TAKE_PROFIT_1'],
            analysis_data['ai_analysis']['TAKE_PROFIT_2'],
            analysis_data['ai_analysis']['STOP_LOSS'],
            analysis_data['ai_analysis']['RISK_REWARD_RATIO'],
            analysis_data['ai_analysis']['ANALYSIS_SUMMARY'],
            json.dumps(analysis_data['technical_indicators'])
        ))
        
        conn.commit()
        conn.close()
    
    def save_news(self, news_items):
        """Save news data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for news in news_items:
            cursor.execute('''
                INSERT OR IGNORE INTO news_data (source, headline, url)
                VALUES (?, ?, ?)
            ''', (news['source'], news['headline'], news.get('url', '')))
        
        conn.commit()
        conn.close()
    
    def get_recent_analysis(self, pair, timeframe, limit=10):
        """Get recent analysis for a pair"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trading_analysis 
            WHERE pair = ? AND timeframe = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (pair, timeframe, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_historical_prices(self, pair, days=30):
        """Get historical prices for chart"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            WHERE pair = ? AND timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(pair,))
        conn.close()
        
        return df

# Global database instance
db = ForexDatabase()
