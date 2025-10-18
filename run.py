🔧 Integrated APIs:
  • Twelve Data → Real-time Prices
  • DeepSeek AI → Market Analysis
  • NewsAPI → Fundamental News
============================================================
Starting server...
 * Debugger is active!
 * Debugger PIN: 611-129-219
127.0.0.1 - - [18/Oct/2025 23:34:31] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [18/Oct/2025 23:34:31] "GET /api/debug HTTP/1.1" 200 -
127.0.0.1 - - [18/Oct/2025 23:34:31] "GET /api/debug HTTP/1.1" 200 -
Processing analysis for 1H
Loading from data/XAUUSD_1H.csv
Columns in CSV: ['datetime', 'open', 'high', 'low', 'close', 'volume']
Successfully loaded 1000 records from data/XAUUSD_1H.csv
Loaded 200 records for 1H
Calculating indicators for 200 records...
EMAs calculated
MACD calculated
RSI calculated
Bollinger Bands calculated
Stochastic calculated
All technical indicators calculated successfully
Indicators calculated
Real-time XAUUSD price from Twelve Data: $4237.99
Current price: $4237.99
No articles found from NewsAPI
Available columns in df_with_indicators: ['datetime', 'open', 'high', 'low', 'close', 'volume', 'ema_12', 'ema_26', 'ema_50', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d']
Last row data sample:
  ema_12: nan (type: <class 'numpy.float64'>)
  ema_26: nan (type: <class 'numpy.float64'>)
  ema_50: nan (type: <class 'numpy.float64'>)
  rsi: nan (type: <class 'numpy.float64'>)
  macd: nan (type: <class 'numpy.float64'>)
  macd_signal: nan (type: <class 'numpy.float64'>)
  macd_hist: nan (type: <class 'numpy.float64'>)
  stoch_k: nan (type: <class 'numpy.float64'>)
  stoch_d: nan (type: <class 'numpy.float64'>)
  bb_upper: nan (type: <class 'numpy.float64'>)
  bb_lower: nan (type: <class 'numpy.float64'>)
Warning: ema_12 is None or NaN
Warning: ema_26 is None or NaN
Warning: ema_50 is None or NaN
Warning: rsi is None or NaN
Warning: macd is None or NaN
Warning: macd_signal is None or NaN
Warning: macd_hist is None or NaN
Warning: stoch_k is None or NaN
Warning: stoch_d is None or NaN
Warning: bb_upper is None or NaN
Warning: bb_lower is None or NaN
Warning: bb_middle is None or NaN
Prepared 12 indicators for API response
Indicators data: {'ema_12': None, 'ema_26': None, 'ema_50': None, 'rsi': None, 'macd': None, 'macd_signal': None, 'macd_hist': None, 'stoch_k': None, 'stoch_d': None, 'bb_upper': None, 'bb_lower': None, 'bb_middle': None}
DeepSeek AI analysis generated successfully
Last chart point indicators:
  ema_12: None
  ema_26: None
  ema_50: None
  macd: None
  macd_signal: None
  macd_hist: None
  rsi: None
  bb_upper: None
  bb_lower: None
  bb_middle: None
  stoch_k: None
  stoch_d: None
Analysis completed for 1H. Sent 100 data points with 12 indicators.
127.0.0.1 - - [18/Oct/2025 23:35:14] "GET /api/analysis/1H?t=1760805275215 HTTP/1.1" 200 -
