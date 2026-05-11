# src/data_loader.py
import yfinance as yf
import pandas as pd
import streamlit as st
import datetime

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start_date, end_date):
    """
    Downloads adjusted close prices for a list of tickers.
    
    Args:
        tickers (list): List of ticker symbols as strings.
        start_date (datetime.date): Start date.
        end_date (datetime.date): End date.
        
    Returns:
        pd.DataFrame: DataFrame with tickers as columns and dates as index.
    """
    if not tickers:
        return pd.DataFrame()
        
    try:
        # yfinance download
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        
        # If single ticker, yf returns Series for 'Adj Close'
        # If multiple tickers, yf returns DataFrame with MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            price_col = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
            prices = data[price_col]
        else:
            price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
            prices = data[[price_col]].rename(columns={price_col: tickers[0]})
                 
        prices.dropna(how='all', inplace=True)
        prices = prices.ffill().bfill()
        
        return prices
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return pd.DataFrame()

def calculate_returns(prices):
    """
    Calculates daily returns from prices.
    """
    return prices.pct_change().dropna(how='all')
