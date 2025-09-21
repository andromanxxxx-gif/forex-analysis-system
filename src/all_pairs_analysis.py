#!/usr/bin/env python3
"""
Script untuk menganalisis semua pasangan forex
"""

import argparse
from src.forex_analyzer import ForexAnalyzer
from src.utils import save_to_csv, load_pairs_list
from config import settings

def main():
    parser = argparse.ArgumentParser(description='Analisis semua pasangan forex')
    parser.add_argument('--pairs', type=str, help='File yang berisi daftar pasangan (opsional)')
    parser.add_argument('--output', type=str, default='forex_analysis_report', help='Nama file output (tanpa ekstensi)')
    parser.add_argument('--quiet', action='store_true', help='Mode quiet (hanya output file)')
    
    args = parser.parse_args()
    
    # Muat daftar pasangan
    if args.pairs:
        pairs = load_pairs_list(args.pairs)
    else:
        pairs = settings.FOREX_PAIRS
    
    if not args.quiet:
        print(f"Menganalisis {len(pairs)} pasangan forex...")
    
    # Analisis semua pasangan
    analyzer = ForexAnalyzer()
    results = analyzer.analyze_all_pairs(pairs)
    
    # Hasilkan laporan
    signals_df, news_df = analyzer.generate_report()
    
    # Simpan hasil ke CSV
    if signals_df is not None and not signals_df.empty:
        signals_csv = f"{args.output}_signals.csv"
        save_to_csv(signals_df, signals_csv)
        
        if not args.quiet:
            print(f"Sinyal trading disimpan ke data/results/{signals_csv}")
    
    if news_df is not None and not news_df.empty:
        news_csv = f"{args.output}_news.csv"
        save_to_csv(news_df, news_csv)
        
        if not args.quiet:
            print(f"Analisis berita disimpan ke data/results/{news_csv}")
    
    # Tampilkan hasil di konsol
    if not args.quiet:
        analyzer.print_results()

if __name__ == "__main__":
    main()
