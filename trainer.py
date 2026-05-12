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
        # Prepare combined data (ETF returns + macro)
        combined = data_manager.prepare_combined_matrix(df, tickers)
        if combined.empty or len(combined) < config.ROLLING_WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Use last ROLLING_WINDOW days for training
        train_data = combined.iloc[-config.ROLLING_WINDOW:].values   # T x (n_etfs + n_macro)
        n_assets = train_data.shape[1]
        
        # Fit DFM on all observed series (ETFs + macro)
        dfm = KalmanDFM(n_assets=n_assets, k_factors=config.K_FACTORS, em_iter=config.EM_ITERATIONS)
        dfm.fit(train_data)

        # Forecast next-day values for ALL series (ETFs + macro)
        last_obs = train_data[-1:].reshape(1, -1)
        pred_all = dfm.forecast_returns(last_obs, horizon=1)   # shape (n_assets,)
        
        # Extract only ETF returns (first len(tickers) entries)
        n_etfs = len(tickers)
        pred_returns = pred_all[:n_etfs]   # only the ETF part
        
        # Build dictionary with tickers and predicted returns
        assets = tickers   # etf names in order (same as columns in combined)
        pred_dict = {ticker: pred_returns[i] for i, ticker in enumerate(assets)}
        sorted_etfs = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        top_etfs = [{"ticker": ticker, "pred_return": float(ret)} for ticker, ret in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]} (pred returns: {[e['pred_return'] for e in top_etfs]})")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "run_date": today,
            "factor_loadings": dfm.get_factor_loadings().tolist()  # includes macro as well
        }

    # Save results
    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/kalman_dfm_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Kalman DFM with Macro Factors complete ===")

if __name__ == "__main__":
    main()
