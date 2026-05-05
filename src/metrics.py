# src/metrics.py
import numpy as np
import pandas as pd
from .config import TRADING_DAYS_PER_YEAR

def cumulative_returns(returns):
    """Calculates cumulative returns."""
    return (1 + returns).cumprod() - 1

def annualized_return(returns):
    """Calculates annualized return given daily returns."""
    cum_ret = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    return cum_ret ** (1 / n_years) - 1 if n_years > 0 else 0.0

def annualized_volatility(returns):
    """Calculates annualized volatility."""
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculates annualized Sharpe ratio."""
    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol

def calculate_beta(asset_returns, benchmark_returns):
    """Calculates beta of asset relative to benchmark."""
    # Align data
    df = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1]
    var_bench = np.var(df.iloc[:, 1], ddof=1)
    if var_bench == 0:
        return np.nan
    return cov / var_bench

def max_drawdown(returns):
    """Calculates the maximum drawdown of a return series."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def rolling_volatility(returns, window=60):
    """Calculates rolling annualized volatility."""
    return returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

def rolling_correlation(asset_returns, benchmark_returns, window=60):
    """Calculates rolling correlation between asset and benchmark."""
    return asset_returns.rolling(window=window).corr(benchmark_returns)

def rolling_return(returns, window=60):
    """Calculates rolling cumulative return over a given window."""
    return (1 + returns).rolling(window=window).apply(np.prod, raw=True) - 1
