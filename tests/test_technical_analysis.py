import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from dummy_data_generator import ForexDataGenerator
from backtester import ForexBacktester

class TestForexSystem(unittest.TestCase):
    def setUp(self):
        self.generator = ForexDataGenerator()
        self.backtester = ForexBacktester()
    
    def test_data_generation(self):
        """Test data generation functionality"""
        data = self.generator.generate_realistic_forex_data('GBPJPY', 100)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertTrue('Open' in data.columns)
        self.assertTrue('High' in data.columns)
        self.assertTrue('Low' in data.columns)
        self.assertTrue('Close' in data.columns)
        self.assertTrue('Volume' in data.columns)
        
        print("✅ Data generation test passed")
    
    def test_technical_indicators(self):
        """Test technical indicators calculation"""
        data = self.generator.generate_realistic_forex_data('USDJPY', 50)
        data_with_indicators = self.generator.add_technical_indicators(data)
        
        required_indicators = ['SMA_20', 'SMA_50', 'EMA_200', 'RSI', 'MACD']
        for indicator in required_indicators:
            self.assertTrue(indicator in data_with_indicators.columns)
        
        print("✅ Technical indicators test passed")
    
    def test_backtest_functionality(self):
        """Test backtesting functionality"""
        data = self.generator.generate_realistic_forex_data('EURJPY', 100)
        data_with_indicators = self.generator.add_technical_indicators(data)
        
        results = self.backtester.run_comprehensive_backtest(data_with_indicators, 'EURJPY_Test')
        
        self.assertIsInstance(results, dict)
        self.assertTrue('MA_Crossover' in results)
        self.assertTrue('RSI_Based' in results)
        
        print("✅ Backtest functionality test passed")
    
    def test_scenario_generation(self):
        """Test scenario-based data generation"""
        scenarios = self.generator.generate_multiple_scenarios()
        
        self.assertIsInstance(scenarios, dict)
        self.assertTrue(len(scenarios) > 0)
        
        for scenario_name, data in scenarios.items():
            self.assertIsInstance(data, pd.DataFrame)
            self.assertTrue(len(data) > 0)
        
        print("✅ Scenario generation test passed")

def run_all_tests():
    """Run semua tests"""
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestForexSystem)
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running Forex System Tests...")
    success = run_all_tests()
    
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
