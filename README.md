# Kalman Smoother Dynamic Factor Model

State‑space dynamic factor model with time‑varying loadings.  
Uses EM + Kalman smoother on a rolling 252‑day window to forecast next‑day ETF returns.

- **Latent factors:** 3 (AR(1))
- **Output:** Top 3 ETFs per universe (FI_COMMODITIES, EQUITY_SECTORS, COMBINED)
- **Daily retraining** via GitHub Actions
- **Results stored** on Hugging Face Hub

## Run locally
```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
