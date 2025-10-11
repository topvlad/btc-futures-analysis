# pages/20_Backtest_WalkForward.py
import streamlit as st
import pandas as pd
from datetime import datetime
from core.data_sources import fetch_klines, fetch_open_interest, fetch_taker_longshort_ratio, fetch_premium_index_klines
from core.features import join_oi, join_taker, join_premium
from core.signals import signal_impulse_participation, signal_funding_meanrev
from core.backtest import backtest_signals, BTConfig

st.set_page_config(page_title="Backtest Walk-Forward", page_icon="🧪", layout="wide")
st.title("🧪 Walk-Forward Backtest")

symbol = st.text_input("Symbol", value="BTCUSDT").upper()
interval = st.selectbox("Interval", ["1m","5m","15m"], index=1)
setup = st.selectbox("Setup", ["A","B"], index=0)

col1,col2,col3 = st.columns(3)
fee = col1.number_input("Fee rate", value=0.0004, step=0.0001, format="%.4f")
slip = col2.number_input("Slippage (USDT)", value=0.5, step=0.1)
risk = col3.number_input("Risk per trade (USDT)", value=40.0, step=5.0)

@st.cache_data(ttl=60*30)
def _load(symbol: str, interval: str):
    kl = fetch_klines(symbol, interval, limit=1000)
    oi = fetch_open_interest(symbol, "1m", limit=1000)
    tk = fetch_taker_longshort_ratio(symbol, "5m", limit=1000)
    pr = fetch_premium_index_klines(symbol, "1m", limit=1000)
    df = join_oi(kl, oi)
    df = join_taker(df, tk)
    df = join_premium(df, pr)
    return df

df = _load(symbol, interval)

if setup == "A":
    sig = signal_impulse_participation(df)
else:
    sig = signal_funding_meanrev(df)

cfg = BTConfig(fee_rate=fee, slip=slip, risk_usdt=risk)
res = backtest_signals(sig, setup=setup, cfg=cfg)

st.subheader("Результати")
st.write(res)
st.line_chart(sig.set_index("t")[["c"]].tail(1000))
st.dataframe(sig.tail(200), use_container_width=True)
