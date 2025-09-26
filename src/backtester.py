import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ForexBacktester:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.results = {}
    
    def simple_strategy_backtest(self, data, strategy_func):
        """Backtest strategy sederhana"""
        balance = self.initial_balance
        position = None
        trades = []
        equity_curve = []
        
        for i in range(1, len(data)):
            current_data = data.iloc[:i]
            signal = strategy_func(current_data)
            
            current_price = data['Close'].iloc[i]
            equity_curve.append(balance)
            
            if signal == 'BUY' and position != 'LONG':
                if position == 'SHORT':
                    # Close short position
                    balance += (entry_price - current_price) * 10000
                
                # Open long position
                position = 'LONG'
                entry_price = current_price
                trades.append({
                    'type': 'LONG',
                    'entry_price': entry_price,
                    'entry_time': data.index[i],
                    'size': 10000
                })
            
            elif signal == 'SELL' and position != 'SHORT':
                if position == 'LONG':
                    # Close long position
                    balance += (current_price - entry_price) * 10000
                
                # Open short position
                position = 'SHORT'
                entry_price = current_price
                trades.append({
                    'type': 'SHORT',
                    'entry_price': entry_price,
                    'entry_time': data.index[i],
                    'size': 10000
                })
            
            elif signal == 'HOLD':
                # Hold current position
                pass
        
        # Close final position
        if position == 'LONG':
            balance += (data['Close'].iloc[-1] - entry_price) * 10000
        elif position == 'SHORT':
            balance += (entry_price - data['Close'].iloc[-1]) * 10000
        
        return {
            'final_balance': balance,
            'total_return': (balance - self.initial_balance) / self.initial_balance * 100,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def moving_average_crossover_strategy(self, data):
        """Strategy: EMA 20 vs EMA 50 crossover"""
        if len(data) < 50:
            return 'HOLD'
        
        ema_20 = data['Close'].ewm(span=20).mean().iloc[-1]
        ema_50 = data['Close'].ewm(span=50).mean().iloc[-1]
        prev_ema_20 = data['Close'].ewm(span=20).mean().iloc[-2]
        prev_ema_50 = data['Close'].ewm(span=50).mean().iloc[-2]
        
        # Golden cross: EMA 20 crosses above EMA 50
        if prev_ema_20 <= prev_ema_50 and ema_20 > ema_50:
            return 'BUY'
        # Death cross: EMA 20 crosses below EMA 50
        elif prev_ema_20 >= prev_ema_50 and ema_20 < ema_50:
            return 'SELL'
        else:
            return 'HOLD'
    
    def rsi_based_strategy(self, data):
        """Strategy berdasarkan RSI"""
        if len(data) < 15:
            return 'HOLD'
        
        rsi = data['RSI'].iloc[-1] if 'RSI' in data.columns else 50
        
        if rsi < 30:  # Oversold
            return 'BUY'
        elif rsi > 70:  # Overbought
            return 'SELL'
        else:
            return 'HOLD'
    
    def run_comprehensive_backtest(self, data, pair_name):
        """Jalankan multiple strategy backtest"""
        strategies = {
            'MA_Crossover': self.moving_average_crossover_strategy,
            'RSI_Based': self.rsi_based_strategy
        }
        
        results = {}
        for strategy_name, strategy_func in strategies.items():
            result = self.simple_strategy_backtest(data, strategy_func)
            results[strategy_name] = result
        
        # Simpan results
        self.results[pair_name] = results
        return results
    
    def generate_report(self, output_dir='data/backtest_results/'):
        """Generate laporan backtesting lengkap"""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report_data = []
        
        for pair, strategies in self.results.items():
            for strategy_name, result in strategies.items():
                report_data.append({
                    'Pair': pair,
                    'Strategy': strategy_name,
                    'Initial_Balance': self.initial_balance,
                    'Final_Balance': result['final_balance'],
                    'Return_Percent': result['total_return'],
                    'Number_of_Trades': len(result['trades'])
                })
        
        report_df = pd.DataFrame(report_data)
        
        # Save report
        report_path = os.path.join(output_dir, f'backtest_report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv')
        report_df.to_csv(report_path, index=False)
        
        # Generate visualization
        self.plot_results(report_df, output_dir)
        
        return report_df
    
    def plot_results(self, report_df, output_dir):
        """Plot hasil backtesting"""
        plt.figure(figsize=(12, 8))
        
        # Plot returns by strategy
        plt.subplot(2, 2, 1)
        sns.barplot(data=report_df, x='Strategy', y='Return_Percent')
        plt.title('Returns by Strategy')
        plt.xticks(rotation=45)
        
        # Plot returns by pair
        plt.subplot(2, 2, 2)
        sns.barplot(data=report_df, x='Pair', y='Return_Percent')
        plt.title('Returns by Pair')
        plt.xticks(rotation=45)
        
        # Plot number of trades
        plt.subplot(2, 2, 3)
        sns.barplot(data=report_df, x='Strategy', y='Number_of_Trades')
        plt.title('Number of Trades by Strategy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'backtest_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

# Utility untuk testing cepat
def quick_backtest_test():
    """Quick test backtester dengan data dummy"""
    from dummy_data_generator import ForexDataGenerator
    
    # Generate dummy data
    generator = ForexDataGenerator()
    test_data = generator.generate_realistic_forex_data('GBPJPY', 200)
    test_data = generator.add_technical_indicators(test_data)
    
    # Run backtest
    backtester = ForexBacktester(initial_balance=10000)
    results = backtester.run_comprehensive_backtest(test_data, 'GBPJPY_Test')
    
    print("📊 Backtest Results:")
    for strategy, result in results.items():
        print(f"Strategy {strategy}: {result['total_return']:.2f}% return")
    
    return results

if __name__ == "__main__":
    results = quick_backtest_test()
