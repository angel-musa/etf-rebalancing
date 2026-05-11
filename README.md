# ETF Portfolio Analytics & Rebalancing Engine

## Project Overview
This project is a sophisticated, clean, and professional web-based ETF analytics and rebalancing tool built with Python and Streamlit. It allows users to benchmark a selection of ETFs and a constructed portfolio against a chosen market index (like SPY). It automatically calculates key performance metrics, visualizes rolling statistics, and generates actionable rebalancing signals based on predefined quantitative logic.

## Why it Matters in Wealth Management
In wealth management and asset allocation, understanding relative performance and risk is crucial. This dashboard automates the workflow of:
- Tracking portfolio metrics (Sharpe, Beta, Max Drawdown).
- Identifying underperforming assets (high relative volatility and lagging returns).
- Flagging outperforming assets for potential increases in allocation.
By providing a clear, unbiased "signal," it supports data-driven portfolio rebalancing decisions, eliminating emotional biases.

## Features
- **Dynamic Asset Selection via CSV**: By default, the application runs using a generated test portfolio (`test_portfolio.csv`). You can override this by uploading your own CSV file with current portfolio positions (`Ticker` and `Weight`).
- **KPI Metrics**: Real-time calculation of Annualized Return, Volatility, Sharpe Ratio, Beta, and Max Drawdown.
- **Interactive Visualizations**: Cumulative return charts, rolling 60-day volatility, and rolling 60-day correlation using Plotly.
- **Automated Signals**: 
  - *Review*: If 60-day return lags benchmark by >3% and volatility is higher than benchmark.
  - *Increase*: If asset outperforms benchmark over 60 days with lower volatility.
  - *Hold*: Otherwise.
- **Exportable Reports**: One-click CSV download for signals and current trailing metrics.

## Installation Steps
1. Clone the repository and navigate to the project directory:
   ```bash
   cd c:\projects\etfs
   ```
2. Create and activate a Python virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Example Tickers
- **Equities**: SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000)
- **International**: EFA (EAFE), EEM (Emerging Markets)
- **Fixed Income**: AGG (US Aggregate Bond), TLT (20+ Year Treasury)
- **Sectors**: XLF (Financials), XLK (Technology), XLV (Health Care)
