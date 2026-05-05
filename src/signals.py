# src/signals.py
import pandas as pd

def generate_signals(asset_returns, benchmark_returns):
    """
    Generates rebalancing/trading signals based on predefined logic.
    
    Logic:
    - Review: if 60D return lags benchmark by >3% AND 60D volatility > benchmark volatility.
    - Increase: if asset outperforms benchmark over 60D WITH lower volatility.
    - Hold: otherwise.
    
    Returns:
        pd.DataFrame containing signal for each asset.
    """
    # Assuming inputs are dataframes/series of daily returns
    # We will use the most recent 60 trading days
    if len(asset_returns) < 60:
        return pd.Series("Insufficient Data", index=asset_returns.columns)
        
    recent_asset_ret = asset_returns.tail(60)
    recent_bench_ret = benchmark_returns.tail(60)
    
    # Calculate 60D cumulative return
    asset_60d_ret = (1 + recent_asset_ret).prod() - 1
    bench_60d_ret = (1 + recent_bench_ret).prod() - 1
    
    # Calculate 60D volatility
    asset_60d_vol = recent_asset_ret.std()
    bench_60d_vol = recent_bench_ret.std() # Scalar
    if isinstance(bench_60d_vol, pd.Series):
        bench_60d_vol = bench_60d_vol.iloc[0]
        
    if isinstance(bench_60d_ret, pd.Series):
        bench_60d_ret = bench_60d_ret.iloc[0]

    signals = {}
    for col in asset_returns.columns:
        a_ret = asset_60d_ret[col]
        a_vol = asset_60d_vol[col]
        
        # Rule 1: Review
        if (bench_60d_ret - a_ret > 0.03) and (a_vol > bench_60d_vol):
            signals[col] = "Review"
        # Rule 2: Increase
        elif (a_ret > bench_60d_ret) and (a_vol < bench_60d_vol):
            signals[col] = "Increase"
        # Rule 3: Hold
        else:
            signals[col] = "Hold"
            
    return pd.Series(signals, name="Action")
