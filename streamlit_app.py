# streamlit_app.py
# A faster, usable BTC dashboard with helicopter view, signal matrix, vol, funding & OI
import os
import json
import math
import time
import requests
import pandas as pd
import numpy as np

import streamlit as st
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from urllib.parse import quote

try:
    import plotly.graph_objects as go
    PLOTLY = True
except Exception:
    PLOTLY = False

# ======== CONFIG ========
REPORT_URL_DEFAULT = "https://topvlad.github.io/btc-futures-analysis/report.json"
SYMBOLS = ["BTCUSDT", "BTCUSDC"]
TFS = ["15m", "1h", "4h", "1d"]
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
SPOT_BASES = ["https://api.binance.com","https://api1.binance.com","https://api2.binance.com","https://api3.binance.com","https://api4.binance.com","https://api5.binance.com"]
FAPI_BASES = ["https://fapi.binance.com","https://fapi1.binance.com","https://fapi2.binance.com","https://fapi3.binance.com","https://fapi4.binance.com","https://fapi5.binance.com"]

st.set_page_config(page_title="BTC Futures — Trend & Regime", layout="wide")

# ======== SIDEBAR ========
st.sidebar.header("Data Sources")
report_url = st.sidebar.text_input("GitHub Pages report.json", value=REPORT_URL_DEFAULT)
cf_worker = st.sidebar.text_input("Cloudflare Worker (optional)", help="If set, requests proxy via ?u=<upstream> on this worker.")
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
if auto_refresh:
    st_autorefresh = st.experimental_memo.clear  # noop alias
    st.experimental_rerun  # will rerun after scripts; Streamlit Cloud uses built-in timer below
st.sidebar.caption("Spot klines → fast charts; report.json → backup & comparison.")

# ======== UTILS ========
def build_worker_url(worker_base: str, full_upstream_url: str) -> str:
    if not worker_base:
        return full_upstream_url
    return f"{worker_base.rstrip('/')}/?u={quote(full_upstream_url, safe='')}"

def http_json(url: str, params=None, timeout=8, allow_worker=False):
    params = params or {}
    bases = []
    if allow_worker and cf_worker:
        # encode the "primary" base into worker
        bases.append(("WORKER", build_worker_url(cf_worker, url + ("?" + requests.compat.urlencode(params) if params else "")), {}))
    # direct
    bases.append(("DIRECT", url, params))
    last_e = None
    for label, base, p in bases:
        try:
            r = requests.get(base, params=None if label=="WORKER" else p, timeout=timeout)
            if r.status_code != 200: last_e = f"status {r.status_code}"; continue
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" not in ct and not base.endswith("/klines"):
                last_e = "non_json"; continue
            return r.json()
        except Exception as e:
            last_e = str(e)
            continue
    raise RuntimeError(f"http_json failed: {last_e}")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_report(url: str):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    payload = json.loads(r.content.decode("utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Bad report.json schema.")
    return payload

@st.cache_data(ttl=60, show_spinner=False)
def fetch_spot_klines(symbol: str, interval: str, limit: int = 500):
    # Try via worker (fast & resilient), else direct
    for base in SPOT_BASES:
        full = f"{base}/api/v3/klines"
        try:
            if cf_worker:
                upstream = f"{SPOT_BASES[0]}/api/v3/klines?symbol={symbol}&interval={interval}&limit={min(1000,limit)}"
                data = http_json(upstream, allow_worker=True)
            else:
                data = http_json(full, params={"symbol": symbol, "interval": interval, "limit": min(1000,limit)})
            if isinstance(data, list) and data:
                return data
        except Exception:
            continue
    raise RuntimeError("All spot kline bases failed.")

@st.cache_data(ttl=120, show_spinner=False)
def fetch_funding(symbol: str, limit: int = 48):
    # /fapi/v1/fundingRate — latest records, ascending order
    url = f"{FAPI_BASES[0]}/fapi/v1/fundingRate"
    try:
        if cf_worker:
            upstream = f"{FAPI_BASES[0]}/fapi/v1/fundingRate?symbol={symbol}&limit={min(1000,limit)}"
            data = http_json(upstream, allow_worker=True)
        else:
            data = http_json(url, params={"symbol": symbol, "limit": min(1000,limit)})
        return data if isinstance(data, list) else []
    except Exception:
        return []

@st.cache_data(ttl=60, show_spinner=False)
def fetch_open_interest(symbol: str):
    url = f"{FAPI_BASES[0]}/fapi/v1/openInterest"
    try:
        if cf_worker:
            upstream = f"{FAPI_BASES[0]}/fapi/v1/openInterest?symbol={symbol}"
            data = http_json(upstream, allow_worker=True)
        else:
            data = http_json(url, params={"symbol": symbol})
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def to_df_klines(raw):
    cols = ["openTime","open","high","low","close","volume","closeTime","qav","numTrades","takerBase","takerQuote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["closeTime"].astype("int64"), unit="ms", utc=True)
    return df.dropna().sort_values("ts").reset_index(drop=True)

def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100/(1+rs))
def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

def adx(df, n=14):
    # Wilder’s ADX needs high/low/close
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = (high.diff()).where(lambda x: x >  low.diff().abs(), 0.0).clip(lower=0)
    minus_dm = (low.diff().abs()).where(lambda x: x > high.diff(), 0.0).clip(lower=0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    tr_n = tr.rolling(n).sum()
    plus_di  = 100 * (plus_dm.rolling(n).sum() / tr_n)
    minus_di = 100 * (minus_dm.rolling(n).sum() / tr_n)
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx_val = dx.rolling(n).mean()
    return adx_val

def tf_open_of_current(interval: str, now_utc=None):
    now_utc = now_utc or datetime.now(timezone.utc)
    step = TF_SECONDS.get(interval, 3600)
    epoch = int(now_utc.timestamp())
    start = epoch - (epoch % step)
    return pd.to_datetime(start, unit="s", utc=True)

def drop_unclosed(df, interval):
    cutoff = tf_open_of_current(interval)
    out = df[df["ts"] < cutoff]
    return out if not out.empty else df

def trend_flags(df):
    s = df["close"]
    ema20 = ema(s, 20); ema50 = ema(s, 50); ema200 = ema(s, 200)
    macd_line, macd_sig, macd_hist = macd(s)
    # ADX from OHLC
    adx_val = adx(df, 14)
    out = {
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "ema200": float(ema200.iloc[-1]),
        "rsi14": float(rsi(s,14).iloc[-1]),
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(macd_sig.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "adx14": float(adx_val.iloc[-1]),
        "price_vs_ema200": "above" if s.iloc[-1] > ema200.iloc[-1] else "below",
        "ema20_cross_50": "bull" if ema20.iloc[-1] > ema50.iloc[-1] else "bear",
        "macd_cross": "bull" if macd_line.iloc[-1] > macd_sig.iloc[-1] else "bear",
        "ema50_slope": float((ema50.iloc[-1] - ema50.iloc[-3]) / max(1e-9, ema50.iloc[-3]))
    }
    return out

def realized_vol(s, win=20):
    # dailyized (for arbitrary TF): stdev of returns * sqrt(periods_per_year)
    ret = s.pct_change().dropna()
    stdev = ret.rolling(win).std().iloc[-1]
    # annualization factor: 365d * (24h*3600/TF_SECONDS) if TF < 1d; else 365
    step = int((s.index[-1] - s.index[-2]).total_seconds())
    per_day = int(86400/step) if step else 1
    per_year = max(1, 365*per_day)
    return float(stdev * math.sqrt(per_year))

def regime_score(sig):
    # +1 each if uptrend feature aligns; -1 if not; ADX>25 boosts confidence
    score = 0
    score += 1 if sig["price_vs_ema200"] == "above" else -1
    score += 1 if sig["ema20_cross_50"] == "bull" else -1
    score += 1 if sig["macd_cross"] == "bull" else -1
    conf = "high" if sig.get("adx14",0) >= 25 else ("medium" if sig.get("adx14",0) >= 20 else "low")
    label = "Trend ↑" if score >= 2 else ("Sideways" if -1 <= score <= 1 else "Trend ↓")
    return score, conf, label

def kpretty(x):
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

# ======== HEADER ========
st.title("BTC — Regime, Signals & Futures Context")
sub = st.empty()

# ======== LOAD report.json (for comparison & table) ========
payload = None
if report_url:
    try:
        payload = fetch_report(report_url)
    except Exception as e:
        st.warning(f"report.json load failed: {e}")

# ======== MAIN CONTROLS ========
c1, c2, c3 = st.columns([1.2,1.2,1])
symbol = c1.selectbox("Symbol", SYMBOLS, index=0)
tf = c2.selectbox("Timeframe", TFS, index=TFS.index("1h"))
limit = c3.slider("Bars (chart calc)", min_value=200, max_value=1000, value=500, step=100)

# ======== FETCH FRESH KLINES FOR CHARTS ========
try:
    raw = fetch_spot_klines(symbol, tf, limit=limit+5)
    df = to_df_klines(raw)
    df = drop_unclosed(df, tf)
    last_ts = df["ts"].iloc[-1]
    last_close = float(df["close"].iloc[-1])
except Exception as e:
    st.error(f"Spot klines failed: {e}")
    st.stop()

# ======== SIGNALS & HELICOPTER VIEW ========
sig = trend_flags(df)
sc, conf, label = regime_score(sig)
rv20 = realized_vol(df.set_index("ts")["close"], 20)
rv60 = realized_vol(df.set_index("ts")["close"], 60)

cA, cB, cC, cD = st.columns([1.2, 1.2, 1.2, 1.2])
cA.metric("Regime", f"{label}", f"conf: {conf}")
cB.metric("Last Close", kpretty(last_close), last_ts.strftime("%Y-%m-%d %H:%M UTC"))
cC.metric("Realized Vol (20)", f"{rv20*100:0.1f}%")
cD.metric("Realized Vol (60)", f"{rv60*100:0.1f}%")

# ======== SIGNAL MATRIX (multi-TF quick view) ========
def tf_row(tframe):
    raw2 = fetch_spot_klines(symbol, tframe, limit=400)
    df2  = drop_unclosed(to_df_klines(raw2), tframe)
    s2   = trend_flags(df2)
    score, conf2, lab2 = regime_score(s2)
    return {
        "tf": tframe,
        "close": df2["close"].iloc[-1],
        "ema200_dir": "↑" if s2["price_vs_ema200"]=="above" else "↓",
        "ema20>50": "✓" if s2["ema20_cross_50"]=="bull" else "×",
        "macd": "✓" if s2["macd_cross"]=="bull" else "×",
        "adx": s2["adx14"],
        "score": score,
        "conf": conf2,
        "label": lab2
    }

matrix_rows = [tf_row(t) for t in TFS]
mat = pd.DataFrame(matrix_rows).set_index("tf")
def color_yesno(val, yes="✓", no="×"):
    if val == yes: return "background-color:#d1ffd6"
    if val == no:  return "background-color:#ffd1d1"
    return ""
st.subheader("Signal matrix (multi-TF)")
st.dataframe(
    mat.style.applymap(lambda v: color_yesno(v)).format({"close":"{:.2f}","adx":"{:.1f}"}),
    use_container_width=True
)

# ======== PRICE CHART ========
with st.expander("Chart", expanded=True):
    s = df.set_index("ts")["close"]
    ema20 = ema(s,20); ema50=ema(s,50); ema200=ema(s,200)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC", opacity=0.8
    ))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema20, name="EMA20", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema50, name="EMA50", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema200, name="EMA200", line=dict(width=1.5)))
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ======== FUNDING & OI ========
with st.expander("Funding & Open Interest (Futures context)"):
    fcol, ocol = st.columns(2)
    # Funding history (try)
    try:
        fr = fetch_funding(symbol, limit=48)
        if fr:
            fdf = pd.DataFrame(fr)
            fdf["fundingRate"] = pd.to_numeric(fdf["fundingRate"], errors="coerce")
            fdf["fundingTime"] = pd.to_datetime(fdf["fundingTime"], unit="ms", utc=True)
            last_f = fdf["fundingRate"].iloc[-1]*100
            fcol.metric("Last funding", f"{last_f:.4f}% / 8h")
            fcol.line_chart(fdf.set_index("fundingTime")["fundingRate"]*100, height=160)
        else:
            fcol.info("No funding data.")
    except Exception as e:
        fcol.warning(f"Funding error: {e}")

    # Open Interest (current)
    try:
        oi = fetch_open_interest(symbol)
        if oi:
            oi_qty = float(oi.get("openInterest", 0.0))
            ocol.metric("Open interest (contracts)", f"{oi_qty:,.0f}")
        else:
            ocol.info("No OI data.")
    except Exception as e:
        ocol.warning(f"OI error: {e}")

# ======== REPORT.JSON RAW / PIVOT for comparison ========
if payload:
    if payload.get("stale"):
        st.warning("report.json is marked **STALE**; charts use live spot klines.")
    rows = payload.get("data", [])
    dfj = pd.json_normalize(rows)
    with st.expander("report.json — raw & pivot"):
        if not dfj.empty:
            st.dataframe(dfj, use_container_width=True)
            ok = dfj[dfj["error"].isna()] if "error" in dfj.columns else dfj
            if not ok.empty:
                pivot = ok.pivot_table(
                    index=["symbol","tf"],
                    values=["last_close","ema20","ema50","ema200","rsi14","macd","macd_signal","macd_hist"],
                    aggfunc="first"
                ).sort_index()
                st.dataframe(pivot, use_container_width=True)
        else:
            st.info("Empty data in report.json.")
