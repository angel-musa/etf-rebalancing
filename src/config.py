# src/config.py

# Default configuration settings
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG"]
DEFAULT_BENCHMARK = "SPY"
DEFAULT_START_DATE_OFFSET_YEARS = 3
TRADING_DAYS_PER_YEAR = 252

# Styling Constants (if any needed outside of Streamlit defaults)
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "neutral": "#7f7f7f",
    "benchmark": "#2ca02c",
    "portfolio": "#d62728",
    "volatility": "#9467bd"
}
