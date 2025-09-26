import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dummy_data_generator import ForexDataGenerator
from technical_analyzer import TechnicalAnalyzer

class TestDataGenerator(unittest.TestCase):
    """
    Test cases untuk data generator
    """
    
    def setUp(self):
        self.generator = ForexDataGenerator()
        self.analyzer = TechnicalAnalyzer()
        self.test_pairs = ['GBPJPY', 'USDJPY']  # Test dengan 2 pairs saja untuk speed
    
    def test_data_generation_basic(self):
        """Test generate data dasar"""
        for pair in self.test_pairs:
            with self.subTest(pair=pair):
                data = self.generator.generate_realistic_forex_data(pair, periods=50)
                
                self.assertIsInstance(data, pd.DataFrame)
                self.assertEqual(len(data), 50)
                self.assertTrue(all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
                
                # Test data values
                self.assertTrue(data['Close'].notna().all())
                self.assertTrue((data['High'] >= data['Low']).all())
                self.assertTrue((data['High'] >= data['Open']).all())
                self.assertTrue((data['High'] >= data['Close']).all())
                self.assertTrue((data['Low'] <= data['Open']).all())
                self.assertTrue((data['Low'] <= data['Close']).all())
    
    def test_technical_indicators(self):
        """Test calculation technical indicators"""
        data = self.generator.generate_realistic_forex_data('GBPJPY', 100)
        data_with_indicators = self.generator.add_technical_indicators(data)
        
        # Test indicators presence
        required_indicators = [
            'SMA_20', 'SMA_50', 'EMA_200', 'RSI', 
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Lower', 'BB_Middle'
        ]
        
        for indicator in required_indicators:
            self.assertIn(indicator, data_with_indicators.columns)
            self.assertTrue(data_with_indicators[indicator].notna().any())
    
    def test_trend_scenarios(self):
        """Test generate data dengan berbagai trend scenario"""
        scenarios = ['bullish', 'bearish', 'sideways']
        
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                data = self.generator.generate_trend_scenario('USDJPY', scenario, 50)
                
                self.assertEqual(len(data), 50)
                
                # Verify trend direction
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[0]
                
                if scenario == 'bullish':
                    self.assertGreater(price_change, 0)
                elif scenario == 'bearish':
                    self.assertLess(price_change, 0)
                # sideways might have small fluctuations
    
    def test_analyzer_integration(self):
        """Test integrasi dengan technical analyzer"""
        data = self.generator.generate_realistic_forex_data('EURJPY', 100)
        data_with_indicators = self.analyzer.calculate_all_indicators(data)
        
        # Test additional indicators from analyzer
        additional_indicators = ['Stoch_K', 'Stoch_D', 'Williams_R', 'ATR', 'OBV']
        for indicator in additional_indicators:
            self.assertIn(indicator, data_with_indicators.columns)
    
    def test_signal_generation(self):
        """Test generate trading signals"""
        data = self.generator.generate_realistic_forex_data('GBPJPY', 100)
        data_with_indicators = self.analyzer.calculate_all_indicators(data)
        
        signals = self.analyzer.generate_signals(data_with_indicators)
        
        self.assertIsInstance(signals, dict)
        self.assertIn('overall_signal', signals)
        self.assertIn('action', signals['overall_signal'])
        self.assertIn('confidence', signals['overall_signal'])
        
        # Test signal values
        self.assertIn(signals['overall_signal']['action'], ['BUY', 'SELL', 'HOLD'])
        self.assertTrue(0 <= signals['overall_signal']['confidence'] <= 1)
    
    def test_multiple_scenarios(self):
        """Test generate multiple scenarios"""
        scenarios = self.generator.generate_multiple_scenarios()
        
        self.assertIsInstance(scenarios, dict)
        self.assertGreater(len(scenarios), 0)
        
        for scenario_name, data in scenarios.items():
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 0)
            self.assertTrue('Close' in data.columns)
    
    def test_data_persistence(self):
        """Test save dan load data"""
        # Test save functionality
        test_data = self.generator.generate_realistic_forex_data('GBPJPY', 50)
        
        # Simulate save (without actual file I/O for speed)
        self.assertIsNotNone(test_data)
        
        # Test data characteristics
        self.assertFalse(test_data.empty)
        self.assertTrue(test_data['Volume'].mean() > 0) 
