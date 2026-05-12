import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

# Page config
st.set_page_config(
    page_title="Kalman DFM Engine",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .universe-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .etf-card {
        background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .etf-card:hover {
        transform: translateY(-5px);
    }
    .etf-ticker {
        font-size: 1.3rem;
        font-weight: bold;
    }
    .etf-return {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .positive {
        color: #00cc96;
    }
    .negative {
        color: #ef553b;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔄 Kalman Smoother Dynamic Factor Model</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">State‑space factor model with time‑varying loadings | EM estimation | Next‑day return forecast</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🔄 Kalman DFM")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown("**Method:** Dynamic Factor Model (Kalman filter + RTS smoother)")
st.sidebar.markdown(f"**Latent factors:** {config.K_FACTORS}")
st.sidebar.markdown("**Rolling window:** 252 days")
st.sidebar.markdown("---")
st.sidebar.caption("Data: [P2SAMAPA/fi-etf-macro-signal-master-data](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)")

# Explanation
with st.expander("📖 How does the model work?"):
    st.markdown("""
    <div class="info-box">
    The Dynamic Factor Model assumes that asset returns are driven by a small number of <strong>latent factors</strong> (here k=3) with <strong>time‑varying loadings</strong>:
    <br><br>
    <code>r_t = Λ_t · f_t + ε_t</code><br>
    <code>f_t = A · f_{t-1} + η_t</code>
    <br><br>
    - <strong>Λ_t</strong> : factor loadings (estimated via EM)<br>
    - <strong>f_t</strong> : latent factors (AR(1) process)<br>
    - <strong>ε_t, η_t</strong> : Gaussian noise<br>
    <br>
    The model is re‑estimated daily on the last 252 days using the Expectation‑Maximisation algorithm (Kalman smoother). The next‑day return forecast is <code>Λ_t · E[f_{t+1}]</code>.
    </div>
    """, unsafe_allow_html=True)

# Load data
OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'kalman_dfm_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run `trainer.py` first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("📈 Top ETFs by Predicted Return")
st.markdown("*Forecasted next‑day return from the dynamic factor model.*")

# Display each universe
for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            pred = etf["pred_return"]
            color_class = "positive" if pred > 0 else "negative"
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-return">pred return <span class="{color_class}">{pred:.2%}</span></div>
            </div>
            """, unsafe_allow_html=True)
    # Optional expander with factor loadings
    if "factor_loadings" in uni_data:
        with st.expander("📊 Factor Loadings (3 factors)"):
            loadings = np.array(uni_data["factor_loadings"])
            # Get ETF names from top_etfs (or we could store full list, but for brevity we show only top ETFs loadings)
            etf_names = [e["ticker"] for e in top_etfs]
            if loadings.shape[0] >= len(etf_names):
                df_load = pd.DataFrame(loadings[:len(etf_names)], index=etf_names, columns=[f"Factor {i+1}" for i in range(config.K_FACTORS)])
                st.dataframe(df_load, use_container_width=True)
    st.divider()

# Historical prediction trend (if multiple files)
st.header("📉 Historical Top ETF Predicted Return")
with st.spinner("Loading historical data..."):
    json_files = [f for f in files if f.endswith('.json') and 'kalman_dfm_' in f]
    json_files.sort(reverse=True)
    history = []
    for fname in json_files[:30]:
        try:
            fs = HfFileSystem(token=HF_TOKEN)
            with fs.open(f"datasets/{OUTPUT_REPO}/{fname}", "r") as f:
                hist = json.load(f)
                run_date = hist['run_date']
                for uni, val in hist['universes'].items():
                    if 'top_etfs' in val and val['top_etfs']:
                        top_pred = val['top_etfs'][0]['pred_return']
                        history.append({"date": run_date, "universe": uni, "top_pred_return": top_pred})
        except:
            pass
    if history:
        df_hist = pd.DataFrame(history)
        fig = px.line(df_hist, x="date", y="top_pred_return", color="universe",
                      title="Top ETF Predicted Return Over Time",
                      labels={"top_pred_return": "Predicted Return", "date": "Run Date"})
        fig.update_layout(height=400, legend_title="Universe")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough historical data to plot trend.")

st.caption("The model is retrained daily on a rolling 252‑day window. Positive predicted return → long signal.")
