"""
Daily training: on rolling 252‑day window, fit DFM via EM, forecast next‑day returns,
score each ETF by factor‑explained component, rank and output top 3 per universe.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from kalman_dfm import KalmanDFM

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        # Prepare returns (ETFs only, no macro for this engine – but we could include macro as extra series)
        returns = data_manager.prepare_returns_matrix(df, tickers)
        if returns.empty or len(returns) < config.ROLLING_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Use last ROLLING_WINDOW days for training
        train_returns = returns.iloc[-config.ROLLING_WINDOW:].values   # T x n
        # Fit DFM
        n_assets = train_returns.shape[1]
        dfm = KalmanDFM(n_assets=n_assets, k_factors=config.K_FACTORS, em_iter=config.EM_ITERATIONS)
        dfm.fit(train_returns)

        # Forecast next-day returns for the last day of training (i.e., tomorrow relative to training end)
        # We need to predict using the latest observation: the last row of train_returns
        last_obs = train_returns[-1:].reshape(1, -1)
        pred_returns = dfm.forecast_returns(last_obs, horizon=1)   # shape (n,)

        # Create list of (ticker, pred_return)
        assets = returns.columns.tolist()
        pred_dict = {ticker: pred_returns[i] for i, ticker in enumerate(assets)}
        # Sort descending
        sorted_etfs = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [{"ticker": ticker, "pred_return": float(ret)} for ticker, ret in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]} (pred returns: {[e['pred_return'] for e in top_etfs]})")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "run_date": today
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/kalman_dfm_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Kalman Smoother Dynamic Factor Model complete ===")

if __name__ == "__main__":
    main()
