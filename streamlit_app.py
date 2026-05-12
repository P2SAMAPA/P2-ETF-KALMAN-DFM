import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Kalman DFM", layout="wide")
st.title("🔄 Kalman Smoother Dynamic Factor Model")
st.caption("State‑space model with time‑varying loadings | Forecasts next‑day returns via latent factors")

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
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** Dynamic Factor Model (Kalman filter + EM)")
st.sidebar.write(f"**Factors:** {config.K_FACTORS}")

universes = data["universes"]
for universe_name, uni_data in universes.items():
    st.subheader(f"🌍 {universe_name}")
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        st.info("No predictions")
        continue
    cols = st.columns(3)
    for i, etf in enumerate(top_etfs):
        with cols[i]:
            st.metric(f"#{i+1} {etf['ticker']}", f"pred return = {etf['pred_return']:.4%}")
    st.divider()

st.caption("Predicted next‑day return from the dynamic factor model. Higher = stronger long signal.")
