import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    path = hf_hub_download(
        repo_id=config.DATA_REPO,
        filename="master_data.parquet",
        repo_type="dataset",
        token=config.HF_TOKEN
    )
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_returns_matrix(df, universe_tickers):
    """Return ETF log returns (only) as DataFrame."""
    returns = pd.DataFrame(index=df.index)
    for ticker in universe_tickers:
        if ticker in df.columns:
            price = df[ticker]
            if not price.isna().all():
                returns[ticker] = np.log(price / price.shift(1))
    returns = returns.dropna(how='all')
    return returns

def prepare_combined_matrix(df, universe_tickers):
    """
    Return combined DataFrame: ETF log returns + macro levels.
    Macro columns are taken as levels (we assume stationary, but we can difference if needed).
    """
    # ETF returns
    etf_returns = prepare_returns_matrix(df, universe_tickers)
    # Macro levels (same index)
    macro = df[config.MACRO_COLUMNS].copy() if config.INCLUDE_MACRO else pd.DataFrame()
    combined = pd.concat([etf_returns, macro], axis=1).dropna()
    return combined
