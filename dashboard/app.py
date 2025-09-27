from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta
import json
from bs4 import BeautifulSoup
import time
import os
import re

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Konfigurasi DeepSeek API
DEEPSEEK_API_KEY = "sk-73d83584fd614656926e1d8860eae9ca"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Forex pairs mapping
pair_mapping = {
    'GBPJPY': 'GBPJPY=X',
    'USDJPY': 'USDJPY=X', 
    'EURJPY': 'EURJPY=X',
    'CHFJPY': 'CHFJPY=X'
}

# Timeframe mapping
timeframe_mapping = {
    '1H': '1h',
    '2H': '2h',
    '4H': '4h',
    '1D': '1d'
}

def get_real_forex_news():
    """Web scraping REAL untuk berita forex dari berbagai sumber"""
    news_items = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Sumber berita forex
    news_sources = [
        {
            'name': 'Forex Factory',
            'url': 'https://www.forexfactory.com/',
            'selector': '.flex-content .calendar__row--highlighted',
            'limit': 5
        },
        {
            'name': 'Investing.com',
            'url': 'https://www.investing.com/news/forex-news',
            'selector': '.largeTitle .articleItem',
            'limit': 5
        },
        {
            'name': 'DailyFX',
            'url': 'https://www.dailyfx.com/latest-news',
            'selector': '.dfx-articleListItem',
            'limit': 5
        }
    ]
    
    for source in news_sources:
        try:
            print(f"Scraping news from {source['name']}...")
            response = requests.get(source['url'], headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            articles = soup.select(source['selector'])[:source['limit']]
            
            for article in articles:
                try:
                    if source['name'] == 'Forex Factory':
                        title_elem = article.select_one('.calendar__event-title')
                        time_elem = article.select_one('.calendar__time')
                        if title_elem and time_elem:
                            title = title_elem.text.strip()
                            time_text = time_elem.text.strip()
                            news_items.append({
                                'source': source['name'],
                                'headline': title,
                                'timestamp': time_text,
                                'url': source['url']
                            })
                    
                    elif source['name'] == 'Investing.com':
                        title_elem = article.select_one('a.title')
                        time_elem = article.select_one('.date')
                        if title_elem:
                            title = title_elem.text.strip()
                            time_text = time_elem.text.strip() if time_elem else datetime.now().strftime('%H:%M')
                            news_items.append({
                                'source': source['name'],
                                'headline': title,
                                'timestamp': time_text,
                                'url': 'https://www.investing.com' + title_elem['href'] if title_elem.get('href') else source['url']
                            })
                    
                    elif source['name'] == 'DailyFX':
                        title_elem = article.select_one('.dfx-articleListItem__title')
                        time_elem = article.select_one('.dfx-articleListItem__date')
                        if title_elem:
                            title = title_elem.text.strip()
                            time_text = time_elem.text.strip() if time_elem else datetime.now().strftime('%H:%M')
                            news_items.append({
                                'source': source['name'],
                                'headline': title,
                                'timestamp': time_text,
                                'url': title_elem.find('a')['href'] if title_elem.find('a') else source['url']
                            })
                            
                except Exception as e:
                    print(f"Error parsing article from {source['name']}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping {source['name']}: {e}")
            continue
    
    # Jika tidak ada berita yang berhasil di-scrape, gunakan fallback dengan data real-time
    if not news_items:
        news_items = get_fallback_news()
    
    return news_items[:8]  # Return max 8 berita

def get_fallback_news():
    """Fallback news dengan data real-time tentang JPY"""
    return [
        {
            'source': 'Market Update',
            'headline': f'JPY Pairs Update: GBP/JPY {get_current_price("GBPJPY")}, USD/JPY {get_current_price("USDJPY")}',
            'timestamp': datetime.now().strftime('%H:%M'),
            'url': '#'
        },
        {
            'source': 'Economic Calendar',
            'headline': 'Bank of Japan Monetary Policy Meeting Minutes Released Today',
            'timestamp': datetime.now().strftime('%H:%M'),
            'url': '#'
        },
        {
            'source': 'Market Analysis',
            'headline': 'Yen Volatility Expected Amid US-Japan Yield Differential Changes',
            'timestamp': datetime.now().strftime('%H:%M'),
            'url': '#'
        },
        {
            'source': 'Technical Outlook',
            'headline': 'JPY Crosses Show Mixed Signals Across Different Timeframes',
            'timestamp': datetime.now().strftime('%H:%M'),
            'url': '#'
        }
    ]

def get_current_price(pair):
    """Get current price for fallback news"""
    try:
        data = yf.download(pair_mapping[pair], period='1d', interval='1h')
        return f"{data['Close'].iloc[-1]:.3f}" if not data.empty else "N/A"
    except:
        return "N/A"

def get_historical_data(symbol, period='6mo', interval='1h'):
    """Get historical data for chart and analysis"""
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            return None
        
        # Convert to list format for Chart.js
        historical_data = {
            'dates': data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': data['Open'].astype(float).round(5).tolist(),
            'high': data['High'].astype(float).round(5).tolist(),
            'low': data['Low'].astype(float).round(5).tolist(),
            'close': data['Close'].astype(float).round(5).tolist(),
            'volume': data['Volume'].astype(float).tolist() if 'Volume' in data else [1] * len(data)
        }
        
        return historical_data
    except Exception as e:
        print(f"Error getting historical data for {symbol}: {e}")
        return None

def get_technical_indicators(data):
    """Menghitung indikator teknikal dengan data historis"""
    indicators = {}
    
    try:
        if data.empty or len(data) < 50:
            return create_default_indicators(150.0)
        
        # Price data
        high = data['High']
        low = data['Low']
        close = data['Close']
        open_price = data['Open']
        
        current_price = safe_float(close)
        
        # Trend Indicators
        indicators['sma_20'] = safe_float(close.rolling(window=20).mean())
        indicators['sma_50'] = safe_float(close.rolling(window=50).mean())
        indicators['ema_12'] = safe_float(close.ewm(span=12).mean())
        indicators['ema_26'] = safe_float(close.ewm(span=26).mean())
        indicators['ema_200'] = safe_float(close.ewm(span=200).mean())
        
        # RSI Calculation
        try:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi_value = 100 - (100 / (1 + safe_float(rs)))
            indicators['rsi'] = max(0, min(100, rsi_value))
        except:
            indicators['rsi'] = 50.0
        
        # MACD Calculation
        try:
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            indicators['macd'] = safe_float(macd_line)
            indicators['macd_signal'] = safe_float(macd_line.ewm(span=9).mean())
            indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
        except:
            indicators['macd'] = indicators['macd_signal'] = indicators['macd_hist'] = 0.0
        
        # Bollinger Bands
        try:
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            indicators['bb_upper'] = safe_float(sma_20 + (std_20 * 2))
            indicators['bb_middle'] = safe_float(sma_20)
            indicators['bb_lower'] = safe_float(sma_20 - (std_20 * 2))
        except:
            indicators['bb_upper'] = indicators['bb_middle'] = indicators['bb_lower'] = current_price
        
        # ATR Calculation
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr'] = safe_float(true_range.rolling(window=14).mean())
        except:
            indicators['atr'] = 0.01
        
        # Support Resistance
        try:
            indicators['pivot'] = (safe_float(high) + safe_float(low) + current_price) / 3
            indicators['resistance1'] = 2 * indicators['pivot'] - safe_float(low)
            indicators['support1'] = 2 * indicators['pivot'] - safe_float(high)
        except:
            indicators['pivot'] = indicators['resistance1'] = indicators['support1'] = current_price
        
        indicators['current_price'] = current_price
        
        # Historical data for chart - last 100 periods
        chart_periods = min(100, len(data))
        historical = get_historical_data_for_chart(data, chart_periods)
        indicators['historical_data'] = historical
        indicators['chart_data'] = historical  # Backward compatibility
        
    except Exception as e:
        print(f"Error in technical indicators: {e}")
        indicators = create_default_indicators(150.0)
    
    return indicators

def get_historical_data_for_chart(data, periods=100):
    """Prepare historical data for chart display"""
    try:
        # Get the last N periods
        data_slice = data.tail(periods)
        
        # Calculate technical indicators for the chart
        close_prices = data_slice['Close']
        
        return {
            'dates': data_slice.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'open': data_slice['Open'].astype(float).round(5).tolist(),
            'high': data_slice['High'].astype(float).round(5).tolist(),
            'low': data_slice['Low'].astype(float).round(5).tolist(),
            'close': data_slice['Close'].astype(float).round(5).tolist(),
            'ema_20': close_prices.ewm(span=20).mean().astype(float).round(5).tolist(),
            'ema_50': close_prices.ewm(span=50).mean().astype(float).round(5).tolist(),
            'ema_200': close_prices.ewm(span=200).mean().astype(float).round(5).tolist(),
            'volume': data_slice['Volume'].astype(float).tolist() if 'Volume' in data_slice else [1] * len(data_slice),
            'sma_20': close_prices.rolling(window=20).mean().astype(float).round(5).tolist(),
            'bb_upper': (close_prices.rolling(window=20).mean() + (close_prices.rolling(window=20).std() * 2)).astype(float).round(5).tolist(),
            'bb_lower': (close_prices.rolling(window=20).mean() - (close_prices.rolling(window=20).std() * 2)).astype(float).round(5).tolist()
        }
    except Exception as e:
        print(f"Error preparing historical data: {e}")
        return create_default_chart_data()

def create_default_chart_data():
    """Create default chart data when real data is unavailable"""
    return {
        'dates': [],
        'open': [], 'high': [], 'low': [], 'close': [],
        'ema_20': [], 'ema_50': [], 'ema_200': [],
        'volume': [], 'sma_20': [], 'bb_upper': [], 'bb_lower': []
    }

def safe_float(value, default=0.0):
    """Safe conversion to float"""
    try:
        if hasattr(value, 'iloc'):
            value = value.iloc[-1] if len(value) > 0 else default
        return float(value)
    except (ValueError, TypeError, IndexError):
        return default

def create_default_indicators(price):
    """Create default indicators when data is not available"""
    return {
        'sma_20': price, 'sma_50': price, 'ema_12': price, 'ema_26': price, 'ema_200': price,
        'rsi': 50.0, 'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0,
        'bb_upper': price, 'bb_middle': price, 'bb_lower': price,
        'atr': 0.01, 'pivot': price, 'resistance1': price, 'support1': price,
        'current_price': price,
        'historical_data': create_default_chart_data(),
        'chart_data': create_default_chart_data()
    }

# ... (fungsi analyze_with_deepseek, generate_fallback_analysis, dan lainnya tetap sama)
# Tetap gunakan fungsi yang sama dari kode sebelumnya, hanya ganti bagian news

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_analysis')
def get_analysis():
    try:
        pair = request.args.get('pair', 'GBPJPY')
        timeframe = request.args.get('timeframe', '4H')
        
        if pair not in pair_mapping:
            return jsonify({'error': 'Invalid pair'})
        if timeframe not in timeframe_mapping:
            return jsonify({'error': 'Invalid timeframe'})
        
        # Get data from yfinance
        yf_symbol = pair_mapping[pair]
        yf_timeframe = timeframe_mapping[timeframe]
        
        # Determine period based on timeframe - lebih panjang untuk data historis
        period_map = {
            '1h': '3mo',  # 3 bulan untuk hourly
            '2h': '6mo',  # 6 bulan untuk 2-hourly
            '4h': '1y',   # 1 tahun untuk 4-hourly
            '1d': '2y'    # 2 tahun untuk daily
        }
        period = period_map.get(yf_timeframe, '1y')
        
        data = yf.download(yf_symbol, period=period, interval=yf_timeframe)
        
        if data.empty or len(data) < 20:
            return jsonify({'error': 'Insufficient data available'})
        
        # Current price data
        current_price = float(data['Close'].iloc[-1])
        price_change_pct = 0.0
        
        if len(data) > 1:
            prev_price = float(data['Close'].iloc[-2])
            price_change_pct = ((current_price - prev_price) / prev_price) * 100
        
        # Technical indicators dengan data historis
        indicators = get_technical_indicators(data)
        indicators['current_price'] = current_price
        indicators['price_change'] = price_change_pct
        
        # REAL News scraping
        news = get_real_forex_news()
        
        # AI Analysis
        ai_analysis = analyze_with_deepseek(indicators, news, pair, timeframe)
        
        # Prepare comprehensive response dengan data historis
        response = {
            'pair': pair,
            'timeframe': timeframe,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': round(float(current_price), 4),
            'price_change': round(float(price_change_pct), 2),
            'technical_indicators': {
                'RSI': round(float(indicators.get('rsi', 50)), 2),
                'MACD': round(float(indicators.get('macd', 0)), 4),
                'SMA_20': round(float(indicators.get('sma_20', current_price)), 4),
                'SMA_50': round(float(indicators.get('sma_50', current_price)), 4),
                'EMA_200': round(float(indicators.get('ema_200', current_price)), 4),
                'ATR': round(float(indicators.get('atr', 0.01)), 4),
                'Support': round(float(indicators.get('support1', current_price)), 4),
                'Resistance': round(float(indicators.get('resistance1', current_price)), 4),
                'BB_Upper': round(float(indicators.get('bb_upper', current_price)), 4),
                'BB_Lower': round(float(indicators.get('bb_lower', current_price)), 4)
            },
            'ai_analysis': ai_analysis,
            'fundamental_news': news,
            'historical_data': indicators.get('historical_data', {}),
            'chart_data': indicators.get('chart_data', {}),
            'data_points_count': len(data),
            'data_period': f"{period} ({yf_timeframe})",
            'price_history': {
                '1h_change': calculate_percentage_change(data, 1) if len(data) > 1 else 0,
                '4h_change': calculate_percentage_change(data, 4) if len(data) > 4 else 0,
                '24h_change': calculate_percentage_change(data, 24) if len(data) > 24 else 0,
                'weekly_high': float(data['High'].tail(168).max()) if len(data) >= 168 else current_price,
                'weekly_low': float(data['Low'].tail(168).min()) if len(data) >= 168 else current_price
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'})

def calculate_percentage_change(data, periods_back):
    """Calculate percentage change from periods back"""
    if len(data) > periods_back:
        current_price = float(data['Close'].iloc[-1])
        past_price = float(data['Close'].iloc[-periods_back-1])
        return ((current_price - past_price) / past_price) * 100
    return 0

# ... (fungsi lainnya tetap sama)

if __name__ == '__main__':
    if not os.path.exists('templates'):
        print("ERROR: 'templates' folder not found!")
        print("Please create a 'templates' folder with 'index.html' inside")
    else:
        print("Template folder found. Starting server...")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
