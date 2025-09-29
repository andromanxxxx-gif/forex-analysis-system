# Tambahkan fungsi ini di bagian TECHNICAL ANALYSIS

def calculate_ema_series(series, period):
    """Calculate EMA for entire series"""
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd_series(series):
    """Calculate MACD components for entire series"""
    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

# Modifikasi bagian get_analysis route untuk menyertakan data EMA200 dan MACD
@app.route("/get_analysis")
def get_analysis():
    pair = request.args.get("pair","USDJPY").upper()
    timeframe = request.args.get("timeframe","1H").upper()
    use_history = request.args.get("use_history","0")=="1"

    try:
        cp = get_price_twelvedata(pair)
        if cp is None and pair in HISTORICAL and timeframe in HISTORICAL[pair]:
            cp = float(HISTORICAL[pair][timeframe].tail(1)["close"].iloc[0])
        elif cp is None:
            cp = 150 + random.uniform(-1, 1)

        if use_history:
            if pair in HISTORICAL and timeframe in HISTORICAL[pair]:
                df = HISTORICAL[pair][timeframe].tail(200)
                closes = df["close"].tolist()
                volumes = df["vol."].fillna(0).tolist() if "vol." in df.columns else None
                dates = df["date"].dt.strftime("%Y-%m-%d %H:%M").tolist()
                
                # Calculate EMA200 series for chart
                close_series = pd.Series(closes)
                ema200_series = calculate_ema_series(close_series, 200)
                ema200_values = ema200_series.tolist()
                
                # Calculate MACD series for chart
                macd_line, macd_signal, macd_histogram = calculate_macd_series(close_series)
                macd_line_values = macd_line.tolist()
                macd_signal_values = macd_signal.tolist()
                macd_histogram_values = macd_histogram.tolist()
                
            else:
                return jsonify({"error": f"Historical data for {pair}-{timeframe} not found."}), 400
        else:
            # Generate synthetic data
            closes = [cp+random.uniform(-0.1,0.1) for _ in range(100)]+[cp]
            volumes = None
            dates = [(datetime.now()-timedelta(hours=i)).strftime("%Y-%m-%d %H:%M") for i in range(100, -1, -1)]
            
            # Calculate EMA200 and MACD for synthetic data
            close_series = pd.Series(closes)
            ema200_series = calculate_ema_series(close_series, 200)
            ema200_values = ema200_series.tolist()
            
            macd_line, macd_signal, macd_histogram = calculate_macd_series(close_series)
            macd_line_values = macd_line.tolist()
            macd_signal_values = macd_signal.tolist()
            macd_histogram_values = macd_histogram.tolist()

        tech = calc_indicators(closes, volumes)
        fundamentals = get_fundamental_news(pair)
        ai = ai_deepseek_analysis(pair, tech, fundamentals)

        return jsonify({
            "pair": pair,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "current_price": cp,
            "technical_indicators": tech,
            "ai_analysis": ai,
            "fundamental_news": fundamentals,
            "chart_data": {
                "dates": dates,
                "close": closes,
                "volume": volumes if volumes else [0] * len(closes),
                "ema200": ema200_values,
                "macd_line": macd_line_values,
                "macd_signal": macd_signal_values,
                "macd_histogram": macd_histogram_values
            },
            "data_source": "Twelve Data + Historical CSV + DeepSeek + NewsAPI"
        })
    except Exception as e:
        print("Backend error:", e)
        traceback.print_exc()
        return jsonify({"error":str(e)}),500
