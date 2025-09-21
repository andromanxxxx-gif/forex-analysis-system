#!/usr/bin/env python3
"""
Script to run forex analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
from src.forex_analyzer import ForexAnalyzer
from src.ml_predictor import MLForexPredictor
import os
import argparse

def run_analysis(use_ml=True):
    """Run complete forex analysis"""
    print(f"{datetime.now()}: Starting analysis...")
    
    # Initialize analyzer
    analyzer = ForexAnalyzer()
    ml_predictor = MLForexPredictor()
    
    # Load ML models if requested
    if use_ml:
        pairs = ['GBPJPY=X', 'CHFJPY=X', 'USDJPY=X', 'EURJPY=X']
        for pair in pairs:
            model_path = f"models/saved_models/{pair.replace('=', '').lower()}_model.h5"
            scaler_path = f"models/saved_models/{pair.replace('=', '').lower()}_scaler.joblib"
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                ml_predictor.load_model(pair, model_path, scaler_path)
                print(f"Loaded model for {pair}")
    
    # Run analysis
    results = analyzer.analyze_all_pairs()
    
    # Generate report
    signals_df, news_df = analyzer.generate_report()
    
    # Save results
    results_dir = 'data/results'
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    signals_df.to_csv(f'{results_dir}/signals_{timestamp}.csv', index=False)
    news_df.to_csv(f'{results_dir}/news_{timestamp}.csv', index=False)
    
    # Print results
    analyzer.print_results()
    
    print(f"{datetime.now()}: Analysis completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run forex analysis')
    parser.add_argument('--no-ml', action='store_true', help='Run without machine learning predictions')
    args = parser.parse_args()
    
    run_analysis(use_ml=not args.no_ml)
