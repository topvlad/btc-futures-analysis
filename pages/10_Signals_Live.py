# pages/10_Signals_Live.py
import asyncio
import pandas as pd
import streamlit as st
from datetime import timedelta

from core.data_sources import fetch_klines, fetch_open_interest, fetch_taker_longshort_ratio, fetch_premium_index_klines, ws_stream
from core.features import join_oi, join_taker, join_premium
from core.signals import signal_impulse_participation, signal_funding_meanrev

st.set_page_config(page_title="Signals LIVE", page_icon="⚡", layout="wide")

st.title("⚡ LIVE Signals (1–5–15m)")
symbol = st.text_input("Symbol", value="BTCUSDT").upper()
colA, colB, colC = st.columns(3)
interval = colA.selectbox("Interval", ["1m","5m","15m"], index=0)
setup   = colB.selectbox("Setup", ["A: Impulse+Participation","B: Funding→MeanRev"], index=0)
live    = colC.toggle("Start WebSocket", value=False)

with st.sidebar:
    st.subheader("Parameters (quick)")
    d_oi_min = st.slider("ΔOI% min (A)", 0.1, 2.0, 0.5, 0.1)
    taker_min = st.slider("Taker ratio min (A)", 1.00, 1.50, 1.10, 0.01)
    prem_z = st.slider("Premium z-threshold (B)", 1.0, 4.0, 2.0, 0.1)

@st.cache_data(ttl=20)
def _load_baseline(symbol: str, interval: str):
    kl = fetch_klines(symbol, interval, limit=500)
    oi = fetch_open_interest(symbol, "1m", limit=500)
    tk = fetch_taker_longshort_ratio(symbol, "5m", limit=200)
    pr = fetch_premium_index_klines(symbol, "1m", limit=500)

    df = join_oi(kl, oi)
    df = join_taker(df, tk)
    df = join_premium(df, pr)
    return df

placeholder = st.empty()
log_ph = st.container()

def _render(df: pd.DataFrame):
    if "A" in setup:
        enriched = signal_impulse_participation(df, params={"d_oi_min": d_oi_min/100, "taker_ratio_min": taker_min})
        sigs = enriched[["t","c","long_sig","short_sig","sl_long","tp_long","sl_short","tp_short"]].tail(50)
    else:
        enriched = signal_funding_meanrev(df, win_z=60, z_thr=prem_z)
        sigs = enriched[["t","c","long_sig","short_sig","sl_long","tp_long","sl_short","tp_short","prem_z"]].tail(50)

    with placeholder.container():
        st.line_chart(enriched.set_index("t")[["c"]].tail(500))
        st.dataframe(sigs, use_container_width=True)

df0 = _load_baseline(symbol, interval)
_render(df0)

async def _run_ws():
    async for ev in ws_stream(symbol, interval):
        try:
            stream = ev.get("stream","")
            data   = ev.get("data",{})
            if "kline" in stream:
                k = data.get("k", {})
                if k.get("x"):  # only on kline close
                    # оновимо базу (легко і стабільно)
                    df = _load_baseline(symbol, interval)
                    _render(df)
            elif "aggTrade" in stream:
                # агр. трейди можна використати для майбутнього обрахунку taker-imbalance у реальному часі
                pass
        except Exception as e:
            st.toast(f"WS error: {e}", icon="⚠️")

if live:
    st.info("WebSocket running… (оновлення по закриттю свічі)")
    asyncio.run(_run_ws())
