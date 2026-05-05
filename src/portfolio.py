# src/portfolio.py
import pandas as pd

def calculate_portfolio_returns(returns_df, weights):
    """
    Calculates daily portfolio returns based on asset returns and weights.
    
    Args:
        returns_df (pd.DataFrame): Daily returns for each asset.
        weights (dict): Dictionary mapping ticker to weight (e.g., {'SPY': 0.6, 'TLT': 0.4}).
        
    Returns:
        pd.Series: Daily portfolio returns.
    """
    # Ensure weights match columns and sum to 1.0 ideally, but we normalize them here just in case.
    valid_tickers = [t for t in weights.keys() if t in returns_df.columns]
    
    if not valid_tickers:
        return pd.Series(0, index=returns_df.index)
        
    weight_series = pd.Series({t: weights[t] for t in valid_tickers})
    weight_series = weight_series / weight_series.sum()
    
    # Calculate portfolio return as sum product
    port_returns = returns_df[valid_tickers].dot(weight_series)
    port_returns.name = "Portfolio"
    
    return port_returns
