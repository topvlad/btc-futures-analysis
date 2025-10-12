# pages/20_Backtest_WalkForward.py — v1.0
import math, numpy as np, pandas as pd, streamlit as st
from datetime import datetime, timezone
from core.data_sources import fetch_klines, fetch_taker_longshort_ratio

st.set_page_config(page_title="🧪 Walk-Forward Backtest", layout="wide")

st.title("🧪 Walk-Forward Backtest")
symbol = st.selectbox("Symbol", ["BTCUSDT","ETHUSDT"], index=0)
interval = st.selectbox("Interval", ["1m","5m","15m"], index=1)
setup = st.radio("Setup", ["A","B"], index=0, horizontal=True)
fee = st.number_input("Fee rate", min_value=0.0, max_value=0.01, value=0.0004, step=0.0001, format="%.4f")
slip = st.number_input("Slippage (USDT)", min_value=0.0, max_value=10.0, value=0.50, step=0.1, format="%.2f")
risk_usdt = st.number_input("Risk per trade (USDT)", min_value=5.0, max_value=200.0, value=40.0, step=5.0, format="%.2f")

@st.cache_data(ttl=60, show_spinner=False)
def _load(sym, tf):
    rows = fetch_klines(sym, tf, limit=1500)
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["ret"] = df["close"].pct_change()
    df["hl"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr"] = (df["hl"]).rolling(14).mean().replace(0,np.nan).fillna(method="bfill")
    return df

df = _load(symbol, interval)
if df.empty:
    st.warning("Price feed unavailable — try again later.")
    st.stop()

# --- synthetic taker series (very light): use endpoint once per 50 bars with fallback in data_sources ---
def _taker_series(sym, tf, n):
    vals = []
    for i in range(n):
        if i % 50 == 0:
            r, _, _ = fetch_taker_longshort_ratio(sym, {"1m":"5m","5m":"5m","15m":"15m"}.get(tf,"5m"), limit=20)
            vals.append(1.0 if r is None else float(r))
        else:
            vals.append(vals[-1])
    return pd.Series(vals, index=df.index[-n:])

# --- strategy rules ---
def backtest_A(df: pd.DataFrame):
    win = 40
    dd = df.copy().tail(800).reset_index(drop=True)
    dd["hh"] = dd["high"].rolling(win).max()
    dd["ll"] = dd["low"].rolling(win).min()
    taker = _taker_series(symbol, interval, len(dd))
    pos = 0
    entry=stop=tp=None
    equity = 0.0
    trades=[]
    for i in range(win+1, len(dd)):
        px = dd.loc[i,"close"]; atr = dd.loc[i,"atr"]
        if pos==0:
            # LONG breakout filter
            if dd.loc[i-1,"close"] > dd.loc[i-1,"hh"] and taker.iloc[i] >= 1.1:
                entry = px + slip
                stop  = entry - 1.5*max(atr, 1e-6)
                tp    = entry + 3.0*max(atr, 1e-6)
                qty   = risk_usdt / (entry - stop)
                pos   = +1
                trades.append(dict(side="LONG",i=i,entry=entry,qty=qty))
            # SHORT breakdown filter
            elif dd.loc[i-1,"close"] < dd.loc[i-1,"ll"] and taker.iloc[i] <= (1/1.1):
                entry = px - slip
                stop  = entry + 1.5*max(atr, 1e-6)
                tp    = entry - 3.0*max(atr, 1e-6)
                qty   = risk_usdt / (stop - entry)
                pos   = -1
                trades.append(dict(side="SHORT",i=i,entry=entry,qty=qty))
        else:
            # manage
            if pos==+1:
                if px >= tp or px <= stop:
                    pnl = (min(max(px, stop), tp) - entry)*trades[-1]["qty"] - fee*entry*trades[-1]["qty"]
                    equity += pnl
                    pos=0
            else:
                if px <= tp or px >= stop:
                    pnl = (entry - max(min(px, stop), tp))*trades[-1]["qty"] - fee*entry*trades[-1]["qty"]
                    equity += pnl
                    pos=0
    return equity, trades, dd

def backtest_B(df: pd.DataFrame):
    dd = df.copy().tail(800).reset_index(drop=True)
    prem_z_win = 60
    # simple mean-reversion proxy: use price z-score of returns (since live premium may be flaky)
    z = (dd["ret"] - dd["ret"].rolling(prem_z_win).mean()) / (dd["ret"].rolling(prem_z_win).std() + 1e-12)
    pos=0; entry=stop=tp=None; equity=0.0; trades=[]
    for i in range(prem_z_win+1, len(dd)):
        px = dd.loc[i,"close"]; atr = dd.loc[i,"atr"]
        if pos==0:
            if z.iloc[i] >= 2.0:  # overbought → SHORT
                entry = px - slip; stop = entry + 1.5*max(atr,1e-6); tp = entry - 3.0*max(atr,1e-6)
                qty = risk_usdt / (stop - entry); pos=-1; trades.append(dict(side="SHORT",i=i,entry=entry,qty=qty))
            elif z.iloc[i] <= -2.0:  # oversold → LONG
                entry = px + slip; stop = entry - 1.5*max(atr,1e-6); tp = entry + 3.0*max(atr,1e-6)
                qty = risk_usdt / (entry - stop); pos=+1; trades.append(dict(side="LONG",i=i,entry=entry,qty=qty))
        else:
            if pos==+1:
                if px >= tp or px <= stop:
                    pnl = (min(max(px, stop), tp) - entry)*trades[-1]["qty"] - fee*entry*trades[-1]["qty"]
                    equity += pnl; pos=0
            else:
                if px <= tp or px >= stop:
                    pnl = (entry - max(min(px, stop), tp))*trades[-1]["qty"] - fee*entry*trades[-1]["qty"]
                    equity += pnl; pos=0
    return equity, trades, dd

equity, trades, dd = (backtest_A(df) if setup=="A" else backtest_B(df))

# --- summary ---
wins = [t for t in trades if ("exit_pnl" not in t)]  # placeholder if you log exits
n = len(trades)
st.subheader("Results")
st.write(f"Trades: **{n}**  · Final PnL (USDT): **{equity:.2f}**")
if n:
    st.line_chart(dd.set_index(dd.index)["close"], height=180, use_container_width=True)
    st.caption("Price (sample window used in test)")
else:
    st.info("No trades generated with current parameters and sample.")
