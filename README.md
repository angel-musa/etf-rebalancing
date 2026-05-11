# ETF Intelligence Dashboard

A Streamlit-based portfolio analytics tool for wealth managers and quantitative analysts. It fetches live market data, computes risk-adjusted performance metrics, generates rule-based rebalancing signals, and lets you interactively adjust portfolio weights in real time.

## Problem

Portfolio managers constantly monitor ETF performance relative to benchmarks, but this workflow is manual and error-prone: pulling data from multiple sources, computing Sharpe ratios and drawdowns in spreadsheets, and making allocation decisions without a unified view. This dashboard automates that entire process in a single interface.

## Features

- **AI Portfolio Intelligence** — health score (0–100), market regime detection (low/normal/high vol), and rule-based insights surfaced automatically
- **Interactive Holdings Tab** — view current positions with live metrics (return, vol, Sharpe, beta, max drawdown); adjust weights via sliders and see the portfolio update in real time
- **Rebalancing Signals** — rule-based signals (Increase / Hold / Review) based on 60-day return vs. benchmark and relative volatility; one-click "Apply Suggested Weights"
- **Performance Charts** — cumulative returns, rolling 60-day volatility, cross-asset correlation heatmap, weight distribution
- **CSV Upload** — bring your own portfolio (`Ticker`, `Weight` columns); falls back to a built-in test portfolio
- **Exportable Reports** — download signals and trailing metrics as CSV

## Signal Logic

| Signal | Condition |
|--------|-----------|
| **Increase** | 60D return > benchmark AND volatility < benchmark |
| **Review** | 60D return lags benchmark by >3% AND volatility > benchmark |
| **Hold** | Everything else |

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd etfs

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Portfolio CSV Format

Upload a CSV with two columns to use your own holdings:

```csv
Ticker,Weight
AAPL,0.20
MSFT,0.20
GOOGL,0.15
AMZN,0.15
META,0.10
```

Weights are automatically normalized to sum to 1.0.

## Example Tickers

| Category | Tickers |
|----------|---------|
| US Broad Market | SPY, QQQ, IWM |
| International | EFA, EEM |
| Fixed Income | AGG, TLT |
| Sectors | XLF, XLK, XLV |

## Stack

- **Frontend**: Streamlit 1.57, Plotly
- **Data**: yfinance (live prices, 1hr cache)
- **Analytics**: pandas, NumPy, scikit-learn
- **Theme**: custom dark UI (`#0a0e1a` background, `#00D4FF` accent)
