<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Forex AI Dashboard</title>

  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js"></script>

  <style>
    body {
      background: #f8fafc;
      color: #111827;
      font-family: "Inter", "Segoe UI", sans-serif;
      padding: 18px;
    }
    .topbar {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }
    .card-soft {
      background: #fff;
      border-radius: 12px;
      padding: 14px;
      box-shadow: 0 6px 16px rgba(0,0,0,0.05);
      margin-bottom: 14px;
    }
    #chartRoot, #macdRoot {
      border-radius: 8px;
      background: #fff;
    }
    #chartRoot { height: 400px; }
    #macdRoot { height: 120px; margin-top: 10px; }
    .muted { color: #6b7280; font-size: .95rem; }
    .ai-signal-buy { color: #0f5132; background: #d1fae5; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .ai-signal-sell { color: #6b0f1a; background: #ffe5e5; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .ai-signal-hold { color: #6a5b00; background: #fff7cc; padding: 3px 8px; border-radius: 6px; font-weight: 600; }
    .small-muted { color: #6b7280; font-size: .9rem; }
    .status-indicator {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 5px;
    }
    .status-live { background-color: #10b981; }
    .status-offline { background-color: #ef4444; }
  </style>
</head>
<body>

<div class="topbar">
  <h4 class="m-0">Forex AI Dashboard</h4>
  <label class="small-muted mb-0 ms-3">Pair</label>
  <select id="pairSelect" class="form-select form-select-sm" style="width:120px;">
    <option>USDJPY</option>
    <option>EURUSD</option>
    <option>GBPJPY</option>
    <option>CHFJPY</option>
  </select>
  <label class="small-muted mb-0">Timeframe</label>
  <select id="timeframeSelect" class="form-select form-select-sm" style="width:100px;">
    <option>1D</option>
    <option>4H</option>
    <option>1H</option>
  </select>
  <button id="analyzeBtn" class="btn btn-primary btn-sm ms-2">Analyze</button>
  <button id="runBacktestBtn" class="btn btn-warning btn-sm">Backtest</button>
  <div id="pairTicker" class="ms-3" style="font-weight:600; font-size:1.05rem;">
    USDJPY – <span id="livePrice">0.0000</span> (<span id="pairChange">0.00%</span>)
  </div>
</div>

<div class="row gx-3">
  <div class="col-lg-8">
    <div class="card-soft">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 10px;">
        <div>
          <strong>Price Chart - <span id="currentPair">USDJPY</span></strong>
          <div class="muted" id="chartSubtitle">Candlestick · EMA26 · MACD</div>
        </div>
        <div style="text-align:right">
          <div class="small-muted">Live Price</div>
          <div style="font-weight:700; font-size:1.2rem" id="currentPrice">-</div>
        </div>
      </div>
      <div id="chartRoot"></div>
      <div id="macdRoot"></div>
    </div>
  </div>

  <div class="col-lg-4">
    <div class="card-soft">
      <h5>AI Analysis</h5>
      <div id="aiSummary" class="muted small">No analysis yet. Click Analyze to start.</div>
      <div class="mt-2"><strong>Signal:</strong> <span id="aiSignalBadge">-</span></div>
      <div class="row mt-2">
        <div class="col-6"><small class="small-muted">Confidence</small><div id="aiConfidence">-</div></div>
        <div class="col-6"><small class="small-muted">Risk</small><div id="aiRisk">-</div></div>
      </div>
      <div class="row mt-2">
        <div class="col-6"><small class="small-muted">Entry</small><div id="aiEntry">-</div></div>
        <div class="col-6"><small class="small-muted">SL</small><div id="aiSL">-</div></div>
      </div>
      <div class="row mt-1">
        <div class="col-6"><small class="small-muted">TP1</small><div id="aiTP1">-</div></div>
        <div class="col-6"><small class="small-muted">TP2</small><div id="aiTP2">-</div></div>
      </div>
    </div>

    <div class="card-soft">
      <h6>Technical Overview</h6>
      <div class="row">
        <div class="col-6">
          <div class="small-muted">RSI</div>
          <div id="rsiVal">-</div>
        </div>
        <div class="col-6">
          <div class="small-muted">MACD</div>
          <div id="macdVal">-</div>
        </div>
      </div>
      <div class="row mt-2">
        <div class="col-6">
          <div class="small-muted">ADX</div>
          <div id="adxVal">-</div>
        </div>
        <div class="col-6">
          <div class="small-muted">ATR</div>
          <div id="atrVal">-</div>
        </div>
      </div>
    </div>

    <div class="card-soft">
      <h6>News & Risk</h6>
      <div id="fundamentalNews" class="small muted">No news available</div>
      <div class="mt-2">
        <div class="small-muted">Risk Assessment</div>
        <div id="riskAssessment">-</div>
      </div>
    </div>

    <div class="card-soft">
      <h6>Backtest Results</h6>
      <div id="backtestResults">
        <div class="text-center muted small">Run backtest to see results</div>
      </div>
    </div>
  </div>
</div>

<!-- Simple Backtest Modal -->
<div class="modal fade" id="backtestModal" tabindex="-1">
  <div class="modal-dialog modal-sm">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Run Backtest</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body">
        <div class="mb-3">
          <label class="form-label">Days to test</label>
          <input type="number" class="form-control" id="backtestDays" value="30" min="7" max="90">
        </div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
        <button type="button" class="btn btn-primary" id="startBacktestBtn">Run</button>
      </div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
// Simple initialization function
function initApp() {
  console.log('Initializing Forex Dashboard...');
  
  // Initialize charts with error handling
  try {
    const chart = LightweightCharts.createChart(document.getElementById('chartRoot'), {
      layout: { 
        backgroundColor: '#fff', 
        textColor: '#111' 
      },
      grid: {
        vertLines: { color: '#f3f4f6' },
        horzLines: { color: '#f3f4f6' }
      },
      width: document.getElementById('chartRoot').clientWidth,
      height: 400
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#16a34a',
      downColor: '#dc2626',
      borderVisible: false,
      wickUpColor: '#16a34a',
      wickDownColor: '#dc2626'
    });

    const emaSeries = chart.addLineSeries({
      color: '#2563eb',
      lineWidth: 1
    });

    const macdChart = LightweightCharts.createChart(document.getElementById('macdRoot'), {
      layout: { backgroundColor: '#fff', textColor: '#111' },
      grid: {
        vertLines: { color: '#f3f4f6' },
        horzLines: { color: '#f3f4f6' }
      },
      width: document.getElementById('macdRoot').clientWidth,
      height: 120
    });

    const macdLine = macdChart.addLineSeries({ color: '#0ea5e9', lineWidth: 1 });
    const macdSignal = macdChart.addLineSeries({ color: '#f59e0b', lineWidth: 1 });
    const macdHistogram = macdChart.addHistogramSeries({ color: '#a78bfa' });

    // Store chart objects globally
    window.charts = {
      main: chart,
      candle: candleSeries,
      ema: emaSeries,
      macdChart: macdChart,
      macdLine: macdLine,
      macdSignal: macdSignal,
      macdHistogram: macdHistogram
    };

    // Set initial empty data
    candleSeries.setData([]);
    emaSeries.setData([]);
    macdLine.setData([]);
    macdSignal.setData([]);
    macdHistogram.setData([]);

  } catch (error) {
    console.error('Chart initialization error:', error);
    document.getElementById('chartRoot').innerHTML = '<div class="alert alert-warning">Charts not available</div>';
    document.getElementById('macdRoot').innerHTML = '<div class="alert alert-warning">MACD not available</div>';
  }

  // Set up event listeners
  document.getElementById('analyzeBtn').addEventListener('click', analyze);
  document.getElementById('runBacktestBtn').addEventListener('click', showBacktestModal);
  document.getElementById('startBacktestBtn').addEventListener('click', runBacktest);

  // Initial analysis
  analyze();
}

// Main analysis function
async function analyze() {
  const pair = document.getElementById('pairSelect').value;
  const timeframe = document.getElementById('timeframeSelect').value;
  
  try {
    console.log('Fetching analysis for', pair, timeframe);
    
    const response = await fetch(`/api/analyze?pair=${pair}&timeframe=${timeframe}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const data = await response.json();
    console.log('Analysis data received:', data);
    
    updateUI(data);
    
  } catch (error) {
    console.error('Analysis error:', error);
    document.getElementById('aiSummary').textContent = 'Error loading analysis';
    document.getElementById('aiSummary').style.color = '#ef4444';
  }
}

// Update UI with data
function updateUI(data) {
  try {
    // Update basic info
    document.getElementById('currentPair').textContent = data.pair || 'USDJPY';
    document.getElementById('currentPrice').textContent = formatPrice(data.price_data?.current);
    
    // Update price ticker
    const change = data.price_data?.change_pct || 0;
    document.getElementById('pairChange').textContent = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
    document.getElementById('pairChange').style.color = change >= 0 ? '#16a34a' : '#ef4444';
    
    // Update AI analysis
    const ai = data.ai_analysis || {};
    document.getElementById('aiSummary').textContent = ai.analysis_summary || 'No analysis available';
    updateSignal(ai.signal);
    document.getElementById('aiConfidence').textContent = ai.confidence ? ai.confidence + '%' : '-';
    document.getElementById('aiRisk').textContent = ai.risk_level || '-';
    document.getElementById('aiEntry').textContent = ai.entry_price || '-';
    document.getElementById('aiSL').textContent = ai.stop_loss || '-';
    document.getElementById('aiTP1').textContent = ai.take_profit_1 || '-';
    document.getElementById('aiTP2').textContent = ai.take_profit_2 || '-';
    
    // Update technical data
    const tech = data.technical_analysis || {};
    document.getElementById('rsiVal').textContent = tech.momentum?.rsi?.toFixed(2) || '-';
    document.getElementById('macdVal').textContent = tech.momentum?.macd?.toFixed(4) || '-';
    document.getElementById('adxVal').textContent = tech.trend?.adx?.toFixed(2) || '-';
    document.getElementById('atrVal').textContent = tech.volatility?.atr?.toFixed(4) || '-';
    
    // Update news
    document.getElementById('fundamentalNews').textContent = data.fundamental_analysis || 'No news available';
    
    // Update risk assessment
    const risk = data.risk_assessment || {};
    updateRiskAssessment(risk);
    
    // Update charts if data available
    if (data.price_series && data.price_series.length > 0) {
      updateCharts(data.price_series);
    }
    
  } catch (error) {
    console.error('UI update error:', error);
  }
}

// Update signal display
function updateSignal(signal) {
  const badge = document.getElementById('aiSignalBadge');
  if (!signal) {
    badge.textContent = '-';
    badge.className = '';
    return;
  }
  
  badge.textContent = signal;
  badge.className = signal === 'BUY' ? 'ai-signal-buy' : 
                   signal === 'SELL' ? 'ai-signal-sell' : 'ai-signal-hold';
}

// Update risk assessment
function updateRiskAssessment(risk) {
  const element = document.getElementById('riskAssessment');
  if (!risk.approved) {
    element.innerHTML = '<span style="color: #ef4444;">✗ Not Approved</span>';
    return;
  }
  
  let html = '<span style="color: #10b981;">✓ Approved</span>';
  if (risk.risk_score !== undefined) {
    html += `<div class="small muted">Risk Score: ${risk.risk_score}/10</div>`;
  }
  element.innerHTML = html;
}

// Update charts with price data
function updateCharts(priceSeries) {
  try {
    if (!window.charts) return;
    
    const chartData = priceSeries.map(item => ({
      time: item.date.split('T')[0],
      open: parseFloat(item.open),
      high: parseFloat(item.high),
      low: parseFloat(item.low),
      close: parseFloat(item.close)
    }));
    
    // Update main chart
    window.charts.candle.setData(chartData);
    
    // Calculate and update EMA
    const closes = chartData.map(d => d.close);
    const emaData = calculateEMA(closes, 26).map((value, index) => ({
      time: chartData[index].time,
      value: value
    }));
    window.charts.ema.setData(emaData);
    
    // Calculate and update MACD
    const macdData = calculateMACD(closes);
    const macdLineData = macdData.macd.map((value, index) => ({
      time: chartData[index].time,
      value: value
    }));
    const signalLineData = macdData.signal.map((value, index) => ({
      time: chartData[index].time,
      value: value
    }));
    const histogramData = macdData.histogram.map((value, index) => ({
      time: chartData[index].time,
      value: value
    }));
    
    window.charts.macdLine.setData(macdLineData);
    window.charts.macdSignal.setData(signalLineData);
    window.charts.macdHistogram.setData(histogramData);
    
  } catch (error) {
    console.error('Chart update error:', error);
  }
}

// Technical indicator calculations
function calculateEMA(data, period) {
  const k = 2 / (period + 1);
  const ema = [data[0]];
  for (let i = 1; i < data.length; i++) {
    ema.push(data[i] * k + ema[i - 1] * (1 - k));
  }
  return ema;
}

function calculateMACD(data, fast = 12, slow = 26, signal = 9) {
  const fastEMA = calculateEMA(data, fast);
  const slowEMA = calculateEMA(data, slow);
  const macd = fastEMA.map((fast, i) => fast - slowEMA[i]);
  const signalLine = calculateEMA(macd, signal);
  const histogram = macd.map((macdVal, i) => macdVal - signalLine[i]);
  
  return { macd, signal: signalLine, histogram };
}

// Utility functions
function formatPrice(price) {
  if (!price) return '-';
  return parseFloat(price).toFixed(4);
}

// Backtest functions
function showBacktestModal() {
  const modal = new bootstrap.Modal(document.getElementById('backtestModal'));
  modal.show();
}

async function runBacktest() {
  const pair = document.getElementById('pairSelect').value;
  const days = document.getElementById('backtestDays').value;
  
  const modal = bootstrap.Modal.getInstance(document.getElementById('backtestModal'));
  modal.hide();
  
  document.getElementById('backtestResults').innerHTML = '<div class="text-center"><div class="spinner-border spinner-border-sm"></div> Running backtest...</div>';
  
  try {
    const response = await fetch('/api/backtest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pair: pair,
        days: parseInt(days),
        initial_balance: 10000
      })
    });
    
    if (!response.ok) throw new Error('Backtest failed');
    
    const result = await response.json();
    displayBacktestResults(result);
    
  } catch (error) {
    console.error('Backtest error:', error);
    document.getElementById('backtestResults').innerHTML = '<div class="alert alert-danger">Backtest failed</div>';
  }
}

function displayBacktestResults(result) {
  const summary = result.summary || {};
  let html = '';
  
  if (result.status === 'no_trades') {
    html = '<div class="alert alert-warning">No trades executed</div>';
  } else {
    html = `
      <div class="row">
        <div class="col-6">
          <div class="small-muted">Total Trades</div>
          <div>${summary.total_trades || 0}</div>
        </div>
        <div class="col-6">
          <div class="small-muted">Win Rate</div>
          <div>${(summary.win_rate || 0).toFixed(1)}%</div>
        </div>
      </div>
      <div class="row mt-2">
        <div class="col-6">
          <div class="small-muted">Profit</div>
          <div style="color: ${(summary.total_profit || 0) >= 0 ? '#10b981' : '#ef4444'}">
            $${(summary.total_profit || 0).toFixed(2)}
          </div>
        </div>
        <div class="col-6">
          <div class="small-muted">Return</div>
          <div style="color: ${(summary.return_percentage || 0) >= 0 ? '#10b981' : '#ef4444'}">
            ${(summary.return_percentage || 0).toFixed(2)}%
          </div>
        </div>
      </div>
    `;
  }
  
  document.getElementById('backtestResults').innerHTML = html;
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', initApp);
</script>
</body>
</html>
