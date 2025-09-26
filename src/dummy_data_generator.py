import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class ForexDataGenerator:
    def __init__(self):
        self.pairs = ['GBPJPY', 'USDJPY', 'CHFJPY', 'EURJPY', 'EURNZD']
        self.base_prices = {
            'GBPJPY': 185.50,
            'USDJPY': 150.80,
            'CHFJPY': 170.25,
            'EURJPY': 160.40,
            'EURNZD': 1.7650
        }
        self.volatilities = {
            'GBPJPY': 0.008,  # High volatility
            'USDJPY': 0.005,  # Medium volatility
            'CHFJPY': 0.006,
            'EURJPY': 0.007,
            'EURNZD': 0.009   # Highest volatility
        }
    
    def generate_realistic_forex_data(self, pair, periods=1000, start_date=None):
        """Generate data forex yang realistis dengan noise dan trend"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=periods//24)
        
        dates = pd.date_range(start=start_date, periods=periods, freq='H')
        base_price = self.base_prices[pair]
        volatility = self.volatilities[pair]
        
        # Generate realistic price movement
        prices = [base_price]
        opens = [base_price]
        highs = [base_price * 1.001]
        lows = [base_price * 0.999]
        volumes = [np.random.randint(10000, 50000)]
        
        for i in range(1, periods):
            # Random walk dengan drift dan volatility yang realistic
            drift = np.random.normal(0.0001, 0.0005)  # Small drift
            shock = np.random.normal(0, volatility)
            
            new_price = prices[-1] * np.exp(drift + shock)
            prices.append(new_price)
            
            # Generate realistic OHLC
            open_price = prices[-1]
            high_price = max(open_price, open_price * (1 + abs(np.random.normal(0, 0.002))))
            low_price = min(open_price, open_price * (1 - abs(np.random.normal(0, 0.002))))
            close_price = open_price * (1 + np.random.normal(0, 0.001))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            volumes.append(np.random.randint(8000, 60000))
        
        data = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return data
    
    def add_technical_indicators(self, data):
        """Menambahkan indikator teknikal ke data dummy"""
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_200'] = data['Close'].ewm(span=200).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volume-based indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        return data
    
    def generate_trend_scenario(self, pair, trend_type='bullish', periods=500):
        """Generate data dengan trend tertentu untuk testing"""
        base_price = self.base_prices[pair]
        
        if trend_type == 'bullish':
            trend = 0.0003  # Upward trend
        elif trend_type == 'bearish':
            trend = -0.0003  # Downward trend
        else:
            trend = 0  # Sideways
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='H')
        prices = [base_price]
        
        for i in range(1, periods):
            # Trend + noise
            change = trend + np.random.normal(0, self.volatilities[pair])
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.0015))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.0015))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(10000, 50000, periods)
        }, index=dates)
        
        return self.add_technical_indicators(data)
    
    def generate_multiple_scenarios(self):
        """Generate berbagai scenario untuk testing comprehensive"""
        scenarios = {}
        
        for pair in self.pairs:
            scenarios[f'{pair}_bullish'] = self.generate_trend_scenario(pair, 'bullish')
            scenarios[f'{pair}_bearish'] = self.generate_trend_scenario(pair, 'bearish')
            scenarios[f'{pair}_sideways'] = self.generate_trend_scenario(pair, 'sideways')
            scenarios[f'{pair}_volatile'] = self.generate_realistic_forex_data(pair, 300)
        
        return scenarios
    
    def save_dummy_data(self, output_dir='data/dummy/'):
        """Save semua data dummy ke folder"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate normal data
        for pair in self.pairs:
            data = self.generate_realistic_forex_data(pair, 1000)
            data_with_indicators = self.add_technical_indicators(data)
            file_path = os.path.join(output_dir, f'{pair}_dummy_data.csv')
            data_with_indicators.to_csv(file_path)
            print(f"✅ Dummy data saved: {file_path}")
        
        # Generate scenario data
        scenarios = self.generate_multiple_scenarios()
        for scenario_name, data in scenarios.items():
            file_path = os.path.join(output_dir, f'scenario_{scenario_name}.csv')
            data.to_csv(file_path)
            print(f"✅ Scenario data saved: {file_path}")
        
        print("🎉 All dummy data generated successfully!")

# Utility function untuk testing cepat
def quick_test():
    """Generate data cepat untuk testing"""
    generator = ForexDataGenerator()
    
    # Test dengan GBPJPY
    test_data = generator.generate_realistic_forex_data('GBPJPY', 100)
    test_data_with_indicators = generator.add_technical_indicators(test_data)
    
    print("📊 Sample generated data:")
    print(test_data_with_indicators.tail())
    print(f"📈 Data shape: {test_data_with_indicators.shape}")
    
    return test_data_with_indicators

if __name__ == "__main__":
    # Generate semua data dummy
    generator = ForexDataGenerator()
    generator.save_dummy_data()
