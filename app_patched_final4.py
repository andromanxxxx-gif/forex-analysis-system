<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Forex Analysis Dashboard</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      margin: 0;
      padding: 0;
    }
    header {
      text-align: center;
      padding: 20px;
      background: #1e293b;
      font-size: 22px;
      font-weight: bold;
    }
    .container {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      padding: 20px;
    }
    .card {
      background: #1e293b;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 0 10px rgba(255, 255, 255, 0.05);
    }
    .card h2 {
      margin-top: 0;
      color: #38bdf8;
    }
    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #0f172a;
      padding: 10px;
      border-radius: 8px;
    }
    button {
      background: #38bdf8;
      border: none;
      padding: 10px 15px;
      border-radius: 6px;
      color: #0f172a;
      font-weight: bold;
      cursor: pointer;
    }
    button:hover {
      background: #0ea5e9;
    }
  </style>
</head>
<body>
  <header>📈 Forex AI Analysis Dashboard</header>
  <div class="container">
    <div class="card">
      <h2>Market Overview</h2>
      <button onclick="loadMarketOverview()">Refresh Market Overview</button>
      <pre id="market"></pre>
    </div>
    <div class="card">
      <h2>Pair Analysis</h2>
      <select id="pair">
        <option value="USDJPY">USDJPY</option>
        <option value="EURUSD">EURUSD</option>
        <option value="GBPUSD">GBPUSD</option>
        <option value="CHFJPY">CHFJPY</option>
      </select>
      <select id="timeframe">
        <option value="1D">1D</option>
        <option value="4H">4H</option>
        <option value="1H">1H</option>
        <option value="M30">M30</option>
      </select>
      <button onclick="analyzePair()">Analyze</button>
      <pre id="analysis"></pre>
    </div>
    <div class="card">
      <h2>System Status</h2>
      <button onclick="loadSystemStatus()">Check System</button>
      <pre id="status"></pre>
    </div>
    <div class="card">
      <h2>Backtest</h2>
      <input id="backtestPair" value="USDJPY" />
      <input id="backtestDays" type="number" value="30" />
      <button onclick="runBacktest()">Run Backtest</button>
      <pre id="backtest"></pre>
    </div>
  </div>

  <script>
    const apiBase = window.location.origin;

    async function safeFetch(url, options) {
      try {
        let res = await fetch(url, options);
        if (res.status === 405 && options.method === "POST") {
          console.warn("POST not allowed, retrying with GET...");
          res = await fetch(url, { method: "GET" });
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (err) {
        console.error("Request failed:", err);
        return { error: err.message };
      }
    }

    async function loadMarketOverview() {
      const data = await safeFetch(`${apiBase}/api/market_overview`, { method: "GET" });
      document.getElementById("market").textContent = JSON.stringify(data, null, 2);
    }

    async function analyzePair() {
      const pair = document.getElementById("pair").value;
      const timeframe = document.getElementById("timeframe").value;
      const data = await safeFetch(`${apiBase}/api/analyze?pair=${pair}&timeframe=${timeframe}`, { method: "GET" });
      document.getElementById("analysis").textContent = JSON.stringify(data, null, 2);
    }

    async function loadSystemStatus() {
      const data = await safeFetch(`${apiBase}/api/system_status`, { method: "GET" });
      document.getElementById("status").textContent = JSON.stringify(data, null, 2);
    }

    async function runBacktest() {
      const pair = document.getElementById("backtestPair").value;
      const days = parseInt(document.getElementById("backtestDays").value) || 30;
      const payload = { pair, timeframe: "4H", days, initial_balance: 10000 };

      const data = await safeFetch(`${apiBase}/api/backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      document.getElementById("backtest").textContent = JSON.stringify(data, null, 2);
    }

    // Auto-load on startup
    loadMarketOverview();
    loadSystemStatus();
  </script>
</body>
</html>
