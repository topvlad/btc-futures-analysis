# pages/20_Backtest_WalkForward.py — v1.3
import numpy as np, pandas as pd, streamlit as st
from core.data_sources import fetch_klines, fetch_taker_longshort_ratio

st.set_page_config(page_title="🧪 Walk-Forward Backtest", layout="wide")
st.title("🧪 Walk-Forward Backtest")

# ---- Controls (with help) ----
symbol = st.selectbox("Symbol", ["BTCUSDT","ETHUSDT"], index=0,
                      help="Perp on Binance.")
interval = st.selectbox("Interval", ["1m","5m","15m"], index=1,
                        help="Backtest timeframe.")
setup = st.radio("Setup", ["A","B"], index=0, horizontal=True,
                 help="A: breakout+order-flow bias (uses sparse taker). B: price z-score mean reversion.")
fee = st.number_input("Fee rate", 0.0, 0.01, 0.0004, 0.0001, format="%.4f",
                      help="Proportional fee applied on entry.")
slip = st.number_input("Slippage (USDT)", 0.0, 10.0, 0.50, 0.1, format="%.2f",
                       help="Added to long entry / subtracted from short entry.")
risk_usdt = st.number_input("Risk per trade (USDT)", 5.0, 500.0, 40.0, 5.0, format="%.2f",
                            help="Position size = risk / stop distance.")

# Strategy knobs
c1, c2, c3 = st.columns(3)
lookback_break = c1.number_input("Breakout lookback (A)", 10, 100, 20, 1)
taker_min = c2.number_input("Taker ratio min (A)", 1.00, 3.00, 1.10, 0.05)
z_window = c3.number_input("Price z-window (B)", 20, 200, 60, 5)

@st.cache_data(ttl=90, show_spinner=False)
def _load(sym, tf):
    rows = fetch_klines(sym, tf, limit=1800)
    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df.sort_values("ts").reset_index(drop=True)
    # returns & ATR proxy
    df["ret"] = df["close"].pct_change()
    tr = (df["high"]-df["low"]).abs()
    tr2= (df["high"]-df["close"].shift()).abs()
    tr3= (df["low"] -df["close"].shift()).abs()
    df["atr"] = pd.concat([tr,tr2,tr3], axis=1).max(axis=1).rolling(14).mean().bfill()
    return df

df = _load(symbol, interval)
if df.empty:
    st.warning("Price feed unavailable — try again later.")
    st.stop()

def _taker_series(sym, tf, n, every=50):
    vals=[]; last=1.0
    period = {"1m":"5m","5m":"5m","15m":"15m"}.get(tf,"5m")
    for i in range(n):
        if i % every == 0:
            r,_,_ = fetch_taker_longshort_ratio(sym, period, limit=20)
            last = 1.0 if r is None else float(r)
        vals.append(last)
    return pd.Series(vals)

# ---- Setup A ----
def backtest_A(df: pd.DataFrame):
    dd = df.tail(1200).reset_index(drop=True)
    if len(dd) < lookback_break + 5:
        return 0.0, [], dd
    dd["hh"] = dd["high"].rolling(lookback_break).max()
    dd["ll"] = dd["low"].rolling(lookback_break).min()
    dd["mom3"] = dd["close"].pct_change(3)
    taker = _taker_series(symbol, interval, len(dd))

    pos=0; entry=stop=tp=qty=None; equity=0.0; trades=[]
    for i in range(lookback_break+1, len(dd)):
        px = dd.loc[i,"close"]; atr=max(dd.loc[i,"atr"],1e-6)
        if pos==0:
            long_bias  = taker.iloc[i] >= taker_min or dd.loc[i,"mom3"] > 0.001
            short_bias = taker.iloc[i] <= (1.0/taker_min) or dd.loc[i,"mom3"] < -0.001
            if dd.loc[i-1,"close"] > dd.loc[i-1,"hh"] and long_bias:
                entry = px + slip; stop = entry - 1.5*atr; tp = entry + 3.0*atr
                qty = risk_usdt / max(entry - stop, 1e-9); pos=+1
                trades.append(dict(side="LONG", ts=dd.loc[i,"ts"], entry=entry, stop=stop, tp=tp))
            elif dd.loc[i-1,"close"] < dd.loc[i-1,"ll"] and short_bias:
                entry = px - slip; stop = entry + 1.5*atr; tp = entry - 3.0*atr
                qty = risk_usdt / max(stop - entry, 1e-9); pos=-1
                trades.append(dict(side="SHORT", ts=dd.loc[i,"ts"], entry=entry, stop=stop, tp=tp))
        else:
            if pos==+1:
                hit_tp = px >= tp; hit_st = px <= stop
                if hit_tp or hit_st:
                    exit_px = tp if hit_tp else stop
                    pnl = (exit_px - entry)*qty - fee*entry*qty
                    trades[-1].update(exit_ts=dd.loc[i,"ts"], exit=exit_px, pnl=pnl)
                    equity += pnl; pos=0
            else:
                hit_tp = px <= tp; hit_st = px >= stop
                if hit_tp or hit_st:
                    exit_px = tp if hit_tp else stop
                    pnl = (entry - exit_px)*qty - fee*entry*qty
                    trades[-1].update(exit_ts=dd.loc[i,"ts"], exit=exit_px, pnl=pnl)
                    equity += pnl; pos=0
    return equity, trades, dd

# ---- Setup B ----
def backtest_B(df: pd.DataFrame):
    dd = df.tail(1200).reset_index(drop=True)
    if len(dd) < z_window + 5: return 0.0, [], dd
    z = (dd["ret"] - dd["ret"].rolling(z_window).mean()) / (dd["ret"].rolling(z_window).std() + 1e-12)
    pos=0; entry=stop=tp=qty=None; equity=0.0; trades=[]
    for i in range(z_window+1, len(dd)):
        px = dd.loc[i,"close"]; atr=max(dd.loc[i,"atr"],1e-6)
        if pos==0:
            if z.iloc[i] >= 2.0:
                entry = px - slip; stop=entry + 1.5*atr; tp=entry - 3.0*atr
                qty=risk_usdt/max(stop-entry,1e-9); pos=-1
                trades.append(dict(side="SHORT", ts=dd.loc[i,"ts"], entry=entry, stop=stop, tp=tp))
            elif z.iloc[i] <= -2.0:
                entry = px + slip; stop=entry - 1.5*atr; tp=entry + 3.0*atr
                qty=risk_usdt/max(entry-stop,1e-9); pos=+1
                trades.append(dict(side="LONG", ts=dd.loc[i,"ts"], entry=entry, stop=stop, tp=tp))
        else:
            if pos==+1:
                hit_tp = px >= tp; hit_st = px <= stop
                if hit_tp or hit_st:
                    exit_px = tp if hit_tp else stop
                    pnl=(exit_px-entry)*qty - fee*entry*qty
                    trades[-1].update(exit_ts=dd.loc[i,"ts"], exit=exit_px, pnl=pnl)
                    equity += pnl; pos=0
            else:
                hit_tp = px <= tp; hit_st = px >= stop
                if hit_tp or hit_st:
                    exit_px = tp if hit_tp else stop
                    pnl=(entry-exit_px)*qty - fee*entry*qty
                    trades[-1].update(exit_ts=dd.loc[i,"ts"], exit=exit_px, pnl=pnl)
                    equity += pnl; pos=0
    return equity, trades, dd

equity, trades, dd = (backtest_A(df) if setup=="A" else backtest_B(df))

st.subheader("Results")
st.write(f"Trades: **{len(trades)}** · Final PnL (USDT): **{equity:.2f}**")
st.line_chart(dd.set_index("ts")["close"], height=180, use_container_width=True)
if trades:
    tdf = pd.DataFrame(trades)
    tdf["ts"] = pd.to_datetime(tdf["ts"], utc=True)
    if "exit_ts" in tdf: tdf["exit_ts"] = pd.to_datetime(tdf["exit_ts"], utc=True, errors="coerce")
    st.dataframe(
        tdf[["ts","side","entry","stop","tp","exit_ts","exit","pnl"]]
        .rename(columns={"ts":"Entry time","exit_ts":"Exit time","tp":"TP","pnl":"PnL"}),
        use_container_width=True,
    )
else:
    st.info("No trades with current parameters. Try a longer interval or lower thresholds.")
