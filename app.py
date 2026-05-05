import streamlit as st
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.config import DEFAULT_TICKERS, DEFAULT_BENCHMARK, DEFAULT_START_DATE_OFFSET_YEARS, COLORS
from src.data_loader import load_data, calculate_returns
from src.metrics import (
    annualized_return, annualized_volatility, sharpe_ratio, calculate_beta,
    max_drawdown, rolling_volatility, rolling_correlation, cumulative_returns
)
from src.signals import generate_signals
from src.portfolio import calculate_portfolio_returns

# --- Page Config ---
st.set_page_config(page_title="ETF Portfolio Analytics", layout="wide")

# --- Custom CSS for Minimal/Professional Design ---
st.markdown("""
    <style>
        .reportview-container {
            background-color: #fafafa;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1f77b4;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #7f7f7f;
            text-transform: uppercase;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.title("Configuration")

st.sidebar.subheader("Upload Positions")
uploaded_file = st.sidebar.file_uploader("CSV with 'Ticker' and 'Weight' cols", type=["csv"])

import os
default_csv_path = "test_portfolio.csv"

if uploaded_file is not None:
    csv_to_read = uploaded_file
else:
    st.sidebar.info(f"Using default test portfolio ({default_csv_path}). Upload a new CSV to override.")
    csv_to_read = default_csv_path

try:
    if os.path.exists(default_csv_path) or uploaded_file is not None:
        portfolio_df = pd.read_csv(csv_to_read)
        if 'Ticker' not in portfolio_df.columns or 'Weight' not in portfolio_df.columns:
            st.sidebar.error("CSV must contain 'Ticker' and 'Weight' columns.")
            st.stop()
        
        tickers = portfolio_df['Ticker'].astype(str).str.upper().str.strip().tolist()
        weights = dict(zip(tickers, portfolio_df['Weight'].astype(float)))
        if uploaded_file is not None:
            st.sidebar.success(f"Loaded {len(tickers)} assets from upload.")
    else:
        st.sidebar.error(f"Default file {default_csv_path} not found.")
        st.stop()
except Exception as e:
    st.sidebar.error(f"Error parsing CSV: {e}")
    st.stop()

benchmark_input = st.sidebar.text_input("Benchmark", value=DEFAULT_BENCHMARK)
benchmark = benchmark_input.strip().upper()

if benchmark not in tickers:
    tickers_to_download = tickers + [benchmark]
else:
    tickers_to_download = tickers

end_date_default = datetime.date.today()
start_date_default = end_date_default - datetime.timedelta(days=DEFAULT_START_DATE_OFFSET_YEARS * 365)

start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

st.sidebar.subheader("Portfolio Weights")
for ticker in tickers:
    if ticker != benchmark:
         st.sidebar.text(f"{ticker}: {weights.get(ticker, 0):.2%}")
# Ensure they sum to 1 by normalizing later if needed, but for the UI let the user input.

# --- Data Loading ---
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

with st.spinner("Downloading Data..."):
    prices = load_data(tickers_to_download, start_date, end_date)

if prices.empty:
    st.error("Failed to load data. Please check the tickers and date range.")
    st.stop()

returns = calculate_returns(prices)

# Separate asset returns and benchmark returns
if benchmark in returns.columns:
    bench_returns = returns[[benchmark]]
    asset_returns = returns[[t for t in tickers if t != benchmark]]
else:
    st.error(f"Benchmark {benchmark} data not found.")
    st.stop()

# Calculate Portfolio returns
port_returns = calculate_portfolio_returns(asset_returns, weights)
port_returns_df = port_returns.to_frame("Portfolio")

# Combine all for easier plotting
all_returns = pd.concat([asset_returns, port_returns_df, bench_returns], axis=1)

# --- Main Dashboard ---
st.title("ETF Portfolio Analytics & Rebalancing Engine")
st.markdown("Benchmarking ETF holdings against sector/market indices.")

# 1. KPI Cards for Portfolio vs Benchmark
st.subheader("Performance Overview")

def create_kpi_card(label, port_val, bench_val, help_text, is_percentage=True):
    format_str = "{:.2%}" if is_percentage else "{:.2f}"
    port_str = format_str.format(port_val)
    bench_str = format_str.format(bench_val)
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label" title="{help_text}" style="cursor: help;">{label} &#9432;</div>
            <div class="metric-value">{port_str}</div>
            <div style="font-size: 0.8rem; color: #7f7f7f;">Bench: {bench_str}</div>
        </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    create_kpi_card("Ann. Return", annualized_return(port_returns), annualized_return(bench_returns[benchmark]), "Annualized compound rate of return")
with col2:
    create_kpi_card("Ann. Volatility", annualized_volatility(port_returns), annualized_volatility(bench_returns[benchmark]), "Annualized standard deviation of daily returns")
with col3:
    create_kpi_card("Sharpe Ratio", sharpe_ratio(port_returns), sharpe_ratio(bench_returns[benchmark]), "Risk-adjusted return (Return / Volatility)", is_percentage=False)
with col4:
    beta_val = calculate_beta(port_returns, bench_returns)
    create_kpi_card("Beta vs Bench", beta_val, 1.0, "Measure of volatility relative to the benchmark", is_percentage=False)
with col5:
    create_kpi_card("Max Drawdown", max_drawdown(port_returns), max_drawdown(bench_returns[benchmark]), "Largest peak-to-trough drop in portfolio value")

st.markdown("---")

# 2. Interactive Charts
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Cumulative Returns")
    cum_ret = cumulative_returns(all_returns)
    fig_cum = px.line(cum_ret, x=cum_ret.index, y=cum_ret.columns, labels={'value': 'Cumulative Return', 'index': 'Date'})
    fig_cum.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text='')
    st.plotly_chart(fig_cum, use_container_width=True)

with col_chart2:
    st.subheader("Rolling 60-Day Volatility")
    roll_vol = rolling_volatility(all_returns, window=60)
    fig_vol = px.line(roll_vol, x=roll_vol.index, y=roll_vol.columns, labels={'value': 'Annualized Volatility', 'index': 'Date'})
    fig_vol.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text='')
    st.plotly_chart(fig_vol, use_container_width=True)

# Rolling Correlation Chart
st.subheader(f"Rolling 60-Day Correlation vs {benchmark}")
corr_df = pd.DataFrame()
for col in asset_returns.columns:
    corr_df[col] = rolling_correlation(asset_returns[col], bench_returns[benchmark], window=60)
corr_df['Portfolio'] = rolling_correlation(port_returns, bench_returns[benchmark], window=60)

fig_corr = px.line(corr_df.dropna(), x=corr_df.dropna().index, y=corr_df.columns, labels={'value': 'Correlation', 'index': 'Date'})
fig_corr.update_layout(margin=dict(l=0, r=0, t=30, b=0), legend_title_text='')
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# 3. Signal Table
st.subheader("Rebalancing Signals")
st.markdown("Signals based on trailing 60-day relative performance and volatility.")

signals = generate_signals(asset_returns, bench_returns)

# Create a detailed table for the signals
signal_details = []
for col in asset_returns.columns:
    a_ret = (1 + asset_returns[col].tail(60)).prod() - 1
    b_ret = (1 + bench_returns[benchmark].tail(60)).prod() - 1
    a_vol = asset_returns[col].tail(60).std() * np.sqrt(252)
    b_vol = bench_returns[benchmark].tail(60).std() * np.sqrt(252)
    
    signal_details.append({
        "Asset": col,
        "Signal": signals[col],
        "60D Return": f"{a_ret:.2%}",
        "Bench 60D Return": f"{b_ret:.2%}",
        "60D Ann. Vol": f"{a_vol:.2%}",
        "Bench 60D Vol": f"{b_vol:.2%}"
    })

signal_df = pd.DataFrame(signal_details).set_index("Asset")

# Style the dataframe
def highlight_signals(val):
    if val == "Review":
        color = 'lightcoral'
    elif val == "Increase":
        color = 'lightgreen'
    else:
        color = 'lightgrey'
    return f'background-color: {color}'

styled_signal_df = signal_df.style.map(highlight_signals, subset=['Signal'])
st.dataframe(styled_signal_df, use_container_width=True)

# 4. Download Report
csv = signal_df.to_csv().encode('utf-8')
st.download_button(
    label="Download Signals CSV",
    data=csv,
    file_name='etf_signals.csv',
    mime='text/csv',
)
