# streamlit_app.py — v1.12.0
# New: Universe Index (Top-N composite) + added 1w timeframe
# Clean mid-layout (Narrative left, Matrix right), notes for each metric, closed-bar & vol line under Prices.

import os, json, math, time, re, requests, pandas as pd, numpy as np
import streamlit as st
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
from email.utils import parsedate_to_datetime

try:
    import plotly.graph_objects as go
    PLOTLY = True
except Exception:
    PLOTLY = False

# ========= CONFIG =========
REPORT_URL_DEFAULT = "https://topvlad.github.io/btc-futures-analysis/report.json"

# Added weekly timeframe
TFS = ["15m", "1h", "4h", "1d", "1w"]
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}

SPOT_BASES = ["https://api.binance.com","https://api1.binance.com","https://api2.binance.com","https://api3.binance.com","https://api4.binance.com","https://api5.binance.com"]
FAPI_BASES = ["https://fapi.binance.com","https://fapi1.binance.com","https://fapi2.binance.com","https://fapi3.binance.com","https://fapi4.binance.com","https://fapi5.binance.com"]

CF_WORKER_DEFAULT = os.getenv("CF_WORKER_URL", "https://binance-proxy.brokerof-net.workers.dev")

NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=crypto&hl=en-US&gl=US&ceid=US:en",
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://cointelegraph.com/rss",
]
NEWS_LOOKBACK_HOURS = 24

COIN_NAME = {
    "BTC":"Bitcoin","ETH":"Ethereum","BNB":"BNB","SOL":"Solana","XRP":"XRP",
    "ADA":"Cardano","DOGE":"Dogecoin","TON":"Toncoin","TRX":"TRON","LINK":"Chainlink",
    "APT":"Aptos","ARB":"Arbitrum","SUI":"Sui","MATIC":"Polygon","PEPE":"Pepe",
    "LTC":"Litecoin","BCH":"Bitcoin Cash","DOT":"Polkadot","AVAX":"Avalanche",
}

st.set_page_config(page_title="Top-N — Regime, Signals & Futures Context", layout="wide")

# ========= UTILITIES =========
def _base_from_symbol(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("USDT") or s.endswith("USDC"): return s[:-4]
    return s

def _news_term(symbol: str) -> str:
    base = _base_from_symbol(symbol)
    return COIN_NAME.get(base, base)

def build_worker_url(worker_base: str, full_upstream_url: str) -> str:
    if not worker_base: return full_upstream_url
    return f"{worker_base.rstrip('/')}/?u={quote(full_upstream_url, safe='')}"

def http_json(url: str, params=None, timeout=8, allow_worker=False):
    params = params or {}
    attempts = []
    if allow_worker and cf_worker:
        attempts.append(("WORKER", build_worker_url(cf_worker, url + ("?" + requests.compat.urlencode(params) if params else "")), {}))
    attempts.append(("DIRECT", url, params))
    last_e = None
    for label, full, p in attempts:
        try:
            r = requests.get(full, params=None if label=="WORKER" else p, timeout=timeout,
                             headers={"Accept":"application/json","User-Agent":"binfapp/1.12.0"})
            if r.status_code != 200: last_e = f"status {r.status_code}"; continue
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" not in ct and not ("/klines" in full or "/candles" in full):
                last_e = "non_json"; continue
            return r.json()
        except Exception as e:
            last_e = str(e); continue
    raise RuntimeError(f"http_json failed: {last_e}")

def http_text(url: str, timeout=8):
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"binfapp/1.12.0"}); r.raise_for_status()
    return r.text

# ========= DYNAMIC UNIVERSE =========
TOPN_DEFAULT = 10

# New: pull CoinGecko Top-N with caps/vols (for weighting)
@st.cache_data(ttl=3600, show_spinner=False)
def _coingecko_topn_with_caps(per_page=100, page=1):
    """
    Returns list of dicts: {base, symbol, market_cap, total_volume}
    """
    try:
        j = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency":"usd","order":"market_cap_desc","per_page":per_page,"page":page},
            timeout=8, headers={"Accept":"application/json","User-Agent":"binfapp/1.12.0"}
        ).json()
        out = []
        for it in j:
            sym = (it.get("symbol") or "").upper().strip()
            if sym in ("USDT","USDC","FDUSD","TUSD","DAI","USDD","USDE","PYUSD"):
                continue
            sym = {"XBT":"BTC","WETH":"ETH"}.get(sym, sym)
            out.append({
                "base": sym,
                "symbol": sym,
                "market_cap": float(it.get("market_cap") or 0.0),
                "total_volume": float(it.get("total_volume") or 0.0)
            })
        # De-duplicate by base, preserve order
        seen, dedup = set(), []
        for it in out:
            if it["base"] in seen: continue
            seen.add(it["base"]); dedup.append(it)
        return dedup
    except Exception:
        # safe baseline
        bases = ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","TON","TRX","LINK"]
        return [{"base":b,"symbol":b,"market_cap":1.0,"total_volume":1.0} for b in bases]

@st.cache_data(ttl=3600, show_spinner=False)
def _binance_futures_universe_from_bases(candidates: list[str], _seed=None):
    try:
        ex = http_json(f"{FAPI_BASES[0]}/fapi/v1/exchangeInfo", timeout=10, allow_worker=bool(CF_WORKER_DEFAULT))
        syms = ex.get("symbols", []) if isinstance(ex, dict) else []
        tradeables = {s["symbol"] for s in syms
                      if s.get("quoteAsset") == "USDT"
                      and s.get("status") == "TRADING"
                      and s.get("contractType","PERPETUAL").upper() in ("PERPETUAL","CURRENT_QUARTER","NEXT_QUARTER")}
        out = []
        base_map = {"IOTA":"IOTA","WBTC":"BTC","BCH":"BCH","TON":"TON","SHIB":"SHIB",
                    "MATIC":"MATIC","PEPE":"PEPE","SUI":"SUI","APT":"APT","ARB":"ARB",
                    "AVAX":"AVAX","LTC":"LTC","DOT":"DOT","LINK":"LINK"}
        for base in candidates:
            b = base_map.get(base, base)
            cand = f"{b}USDT"
            if cand in tradeables:
                out.append(cand)
        return out
    except Exception:
        return [f"{b}USDT" for b in ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","TON","TRX","LINK"]]

# ========= SIDEBAR =========
st.sidebar.header("Data & Options")

topn = st.sidebar.number_input("Universe size (Top-N by market cap)", min_value=5, max_value=30,
                               value=TOPN_DEFAULT, step=1,
                               help="Coins from CoinGecko top market caps, filtered to Binance USDT perps.")
if "universe_seed" not in st.session_state:
    st.session_state.universe_seed = 0
if st.sidebar.button("↻ Refresh universe now", help="Force refresh of the Top-N list (bypass 1h cache)."):
    st.session_state.universe_seed = int(time.time())

# Weighting scheme for Universe Index
weighting_scheme = st.sidebar.selectbox(
    "Universe Index weighting",
    options=["Market cap (default)","Liquidity (24h volume)","Equal weight"],
    index=0,
    help="Applies to the composite in the Universe Index tab."
)

# Universe discovery with caps/vols
cg = _coingecko_topn_with_caps()
bases_ranked = [it["base"] for it in cg]
bases_trimmed = bases_ranked[:int(topn)]
SYMBOLS = _binance_futures_universe_from_bases(bases_trimmed, _seed=st.session_state.universe_seed)
if not SYMBOLS:
    SYMBOLS = ["BTCUSDT","ETHUSDT"]
if "BTCUSDT" in SYMBOLS:
    SYMBOLS.remove("BTCUSDT"); SYMBOLS.insert(0, "BTCUSDT")

st.sidebar.caption(f"Universe: {', '.join([_base_from_symbol(s) for s in SYMBOLS[:10]])}{'…' if len(SYMBOLS)>10 else ''}")

symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=0)
report_url = st.sidebar.text_input("GitHub Pages report.json (debug)", value=REPORT_URL_DEFAULT)
cf_worker = st.sidebar.text_input("Cloudflare Worker (optional)", value=CF_WORKER_DEFAULT)
overlay_plan = st.sidebar.checkbox("Draw Plan A/B on chart (1h)", value=True)
extra_emas = st.sidebar.multiselect("Extra EMAs (on chart)", options=[100, 400, 800], default=[400])
auto_refresh = st.sidebar.checkbox("Auto-refresh UI every 60s (uses last CLOSED bar; no repaint)", value=False)

refresh_seed = None
if auto_refresh:
    refresh_seed = int(time.time() // 60)
    st.markdown("<script>setTimeout(function(){location.reload();}, 60000);</script>", unsafe_allow_html=True)

# ========= DATA HELPERS =========
def _to_df_binance(raw):
    cols = ["openTime","open","high","low","close","volume","closeTime","qav","numTrades","takerBase","takerQuote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ("open","high","low","close","volume"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["closeTime"].astype("int64"), unit="ms", utc=True)
    return df[["ts","open","high","low","close"]].dropna().sort_values("ts").reset_index(drop=True)

def _to_df_okx(raw):
    df = pd.DataFrame(raw, columns=["ts_ms","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])
    for c in ("open","high","low","close"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts_ms"].astype("int64"), unit="ms", utc=True)
    return df[["ts","open","high","low","close"]].dropna().sort_values("ts").reset_index(drop=True)

def _to_df_coinbase(raw):
    df = pd.DataFrame(raw, columns=["t","low","high","open","close","vol"])
    for c in ("open","high","low","close"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_datetime(df["t"].astype("int64"), unit="s", utc=True)
    return df[["ts","open","high","low","close"]].dropna().sort_values("ts").reset_index(drop=True)

def _binance_spot_klines(symbol: str, interval: str, limit: int, worker_url: str | None):
    if worker_url:
        upstream = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={min(2000,limit)}"
        data = http_json(upstream, allow_worker=True)
        return _to_df_binance(data), "binance(worker)"
    for base in SPOT_BASES:
        data = http_json(f"{base}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": min(2000,limit)}, timeout=8)
        return _to_df_binance(data), f"binance({base})"

def _okx_klines(symbol: str, interval: str, limit: int):
    base = _base_from_symbol(symbol)
    inst = f"{base}-USDT"
    # Added 1w mapping
    bar_map = {"15m":"15m","1h":"1H","4h":"4H","1d":"1D","1w":"1W"}
    bar = bar_map.get(interval, "1H")
    j = http_json("https://www.okx.com/api/v5/market/candles",
                  params={"instId": inst, "bar": bar, "limit": min(500,limit)}, timeout=8)
    data = j.get("data") or []
    if not data: raise RuntimeError("OKX empty")
    return _to_df_okx(data), "okx"

def _coinbase_klines(symbol: str, interval: str, limit: int):
    base = _base_from_symbol(symbol)
    product_map = {"BTC":"BTC-USD","ETH":"ETH-USD"}
    product = product_map.get(base, "BTC-USD")
    # Coinbase has fixed granularities; no native 1w → fall back to 1d for weekly requests
    gran_map = {"15m":900,"1h":3600,"4h":14400,"1d":86400,"1w":86400}
    gran = gran_map.get(interval, 3600)
    data = http_json(f"https://api.exchange.coinbase.com/products/{product}/candles", params={"granularity": gran, "limit": min(300,limit)}, timeout=8)
    if not isinstance(data, list) or not data: raise RuntimeError("Coinbase empty")
    return _to_df_coinbase(data), "coinbase"

@st.cache_data(ttl=60, show_spinner=False)
def get_klines_df(symbol: str, interval: str, limit: int = 1000, _seed=None):
    errs=[]
    try: df, src = _binance_spot_klines(symbol, interval, limit, cf_worker); return df, src
    except Exception as e: errs.append(f"binance:{e}")
    try: df, src = _okx_klines(symbol, interval, min(500,limit)); return df, src
    except Exception as e: errs.append(f"okx:{e}")
    try: df, src = _coinbase_klines(symbol, interval, min(300,limit)); return df, src
    except Exception as e: errs.append(f"coinbase:{e}")
    raise RuntimeError("All providers failed → " + " | ".join(errs))

# ========= Indicators =========
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi_series(s, n=14):
    d = s.diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100/(1+rs))

def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow); sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

def adx(df, n=14):
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(high - low),(high - close.shift()).abs(),(low - close.shift()).abs()], axis=1).max(axis=1)
    tr_n = tr.rolling(n).sum()
    plus_di  = 100 * (pd.Series(plus_dm, index=df.index).rolling(n).sum() / tr_n)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(n).sum() / tr_n)
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    return dx.rolling(n).mean()

def tf_open_of_current(interval: str):
    now_utc = datetime.now(timezone.utc); step = TF_SECONDS.get(interval, 3600)
    epoch = int(now_utc.timestamp()); start = epoch - (epoch % step)
    return pd.to_datetime(start, unit="s", utc=True)

def drop_unclosed(df, interval):
    cutoff = tf_open_of_current(interval)
    out = df[df["ts"] < cutoff]
    return out if not out.empty else df

def realized_vol_series(s, win=20):
    ret = s.pct_change().dropna(); stdev = ret.rolling(win).std()
    if len(s) >= 2:
        step = int((s.index[-1] - s.index[-2]).total_seconds())
    else:
        step = TF_SECONDS["1h"]
    per_day = int(86400/max(step,1)); per_year = max(1, 365*per_day)
    return stdev * math.sqrt(per_year)

def trend_flags(df):
    s = df["close"]
    ema20 = ema(s,20); ema50 = ema(s,50); ema200 = ema(s,200)
    rsi = rsi_series(s,14); rsi_ma = rsi.rolling(5).mean()
    macd_line, macd_sig, macd_hist = macd(s)
    adx_val = adx(df,14)
    return {
        "ema20": float(ema20.iloc[-1]), "ema50": float(ema50.iloc[-1]), "ema200": float(ema200.iloc[-1]),
        "rsi14": float(rsi.iloc[-1]), "rsi_ma5": float(rsi_ma.iloc[-1]),
        "rsi_cross": "✓" if rsi.iloc[-1] > rsi_ma.iloc[-1] else "×",
        "macd": float(macd_line.iloc[-1]), "macd_signal": float(macd_sig.iloc[-1]), "macd_hist": float(macd_hist.iloc[-1]),
        "adx14": float(adx_val.iloc[-1]),
        "price_vs_ema200": "above" if s.iloc[-1] > ema200.iloc[-1] else "below",
        "ema20_cross_50": "bull" if ema20.iloc[-1] > ema50.iloc[-1] else "bear",
        "macd_cross": "bull" if macd_line.iloc[-1] > macd_sig.iloc[-1] else "bear",
        "ema50_slope": float((ema50.iloc[-1] - ema50.iloc[-3]) / max(1e-9, ema50.iloc[-3])),
    }

def regime_score(sig):
    adx_v = sig.get("adx14",0)
    conf = "high" if adx_v >= 28 else ("medium" if adx_v >= 22 else "low")
    score = (1 if sig["price_vs_ema200"]=="above" else -1) + (1 if sig["ema20_cross_50"]=="bull" else -1) + (1 if sig["macd_cross"]=="bull" else -1)
    label = "Trend ↑" if score >= 2 else ("Sideways" if -1 <= score <= 1 else "Trend ↓")
    return score, conf, label

def funding_tilt(last_rate: float):
    if last_rate is None: return "unknown", "n/a", "n/a"
    mag = abs(last_rate)
    level = "neutral" if mag < 0.0001 else ("elevated" if mag < 0.0005 else "extreme")
    side = "Longs → Shorts" if last_rate > 0 else ("Shorts → Longs" if last_rate < 0 else "Flat")
    bps = last_rate * 10000.0
    return f"{bps:+.2f} bps / 8h", level, side

def per_tf_recommendation(sig):
    up = (sig["price_vs_ema200"]=="above"); align = (sig["ema20_cross_50"]=="bull"); mom = (sig["macd_cross"]=="bull"); adx_ok = sig["adx14"] >= 22
    if up and align and mom and adx_ok: return "Buy dips"
    if (not up) and (sig["ema20_cross_50"]=="bear") and (sig["macd_cross"]=="bear") and adx_ok: return "Sell rips"
    return "Range trade"

def kpretty(x):
    try: return f"{x:,.2f}"
    except Exception: return str(x)

# === RV one-liner ===
def rv_one_liner(rv20: float, rv60: float) -> str:
    ratio = (rv20 / max(rv60, 1e-9))
    if ratio >= 1.15:
        skew = "short-term volatility is rising vs medium-term (momentum bias; use wider stops)."
    elif ratio <= 0.85:
        skew = "short-term volatility is cooling vs medium-term (mean-reversion bias; tighter stops ok)."
    else:
        skew = "short- and medium-term volatility are aligned (no strong vol edge)."
    hi = max(rv20, rv60)
    if   hi >= 100: level = "very high"
    elif hi >= 70:  level = "high"
    elif hi >= 40:  level = "moderate"
    else:           level = "calm"
    return f"Volatility is {level}; {skew}"

# ========= CORE LOAD (multi-TF for selected symbol) =========
@st.cache_data(ttl=60, show_spinner=False)
def get_all_tf_data(symbol: str, tfs: list[str], limit_each: int = 2000, _seed=None):
    out = {}
    for tf in tfs:
        df, src = get_klines_df(symbol, tf, limit=limit_each, _seed=_seed)
        df = drop_unclosed(df, tf)
        out[tf] = (df, src)
    return out

try:
    tf_data = get_all_tf_data(symbol, TFS, limit_each=1000, _seed=(refresh_seed if auto_refresh else None))
except Exception as e:
    st.error(f"Klines failed: {e}"); st.stop()

# Use 1h for token chart & plan
df_1h, src_chart = tf_data["1h"]
last_closed_bar_time = df_1h["ts"].iloc[-1]
last_close = float(df_1h["close"].iloc[-1])

# Header captions
st.caption(f"Prices: **{src_chart}** · Worker: {'ON' if cf_worker else 'OFF'}")
s_1h = df_1h.set_index("ts")["close"]
rv20_1h = realized_vol_series(s_1h, 20).iloc[-1] * 100
rv60_1h = realized_vol_series(s_1h, 60).iloc[-1] * 100
st.caption("Using last CLOSED 1h bar (no repaint). Current 1h bar is excluded until it closes.")
st.caption(rv_one_liner(rv20_1h, rv60_1h))

# Aggregate regime from TFs
signals = {tf: trend_flags(tf_data[tf][0]) for tf in TFS}
def aggregate_regime(signals: dict):
    ups = downs = sides = 0; strong = 0
    for tf, s in signals.items():
        sc, conf, lab = regime_score(s)
        if conf in ("medium","high"): strong += 1
        if lab == "Trend ↑": ups += 1
        elif lab == "Trend ↓": downs += 1
        else: sides += 1
    label = "Trend ↑" if ups >= max(downs,sides) and ups >= 2 else ("Trend ↓" if downs >= max(ups,sides) and downs >= 2 else "Sideways")
    conf = "high" if strong >= 3 else ("medium" if strong == 2 else "low")
    return {"label": label, "conf": conf, "ups": ups, "downs": downs, "sides": sides}
agg = aggregate_regime(signals)

# ========= Universe snapshot (for Universe tab) =========
@st.cache_data(ttl=180, show_spinner=False)
def _symbol_vitals(symbol: str):
    try:
        df1h, _ = get_klines_df(symbol, "1h", limit=600)
        df1h = drop_unclosed(df1h, "1h")
        s = df1h.set_index("ts")["close"]
        rv20 = float(realized_vol_series(s, 20).iloc[-1] * 100)
        rv60 = float(realized_vol_series(s, 60).iloc[-1] * 100)
        sig = trend_flags(df1h)
        _, _, lab = regime_score(sig)
        above200 = 1 if sig["price_vs_ema200"] == "above" else 0
        macd_bull = 1 if sig["macd_cross"] == "bull" else 0
        adx14 = float(sig["adx14"])
        rsi14 = float(sig["rsi14"])
        return {"ok": True, "rv20": rv20, "rv60": rv60, "label": lab,
                "above200": above200, "macd_bull": macd_bull, "adx14": adx14, "rsi14": rsi14}
    except Exception:
        return {"ok": False}

def _level_note_rv(x: float) -> str:
    if x >= 100: return "very high"
    if x >= 70:  return "high"
    if x >= 40:  return "moderate"
    return "calm"

def _note_adx(x: float) -> str:
    if x >= 28: return "strong trend"
    if x >= 22: return "building trend"
    return "low trend strength"

def _note_rsi(x: float) -> str:
    if x >= 70: return "overbought-ish"
    if x <= 30: return "oversold-ish"
    return "neutral"

def _universe_guide(avg20, avg60, breadth_pct, ups, downs, mom_breadth):
    level = _level_note_rv(max(avg20, avg60))
    bias = "up" if ups > downs else ("down" if downs > ups else "mixed")
    skew_ratio = avg20 / max(avg60, 1e-9)
    if skew_ratio >= 1.15:
        skew = "vol expanding short-term → momentum entries hold better"
    elif skew_ratio <= 0.85:
        skew = "vol cooling short-term → mean-reversion entries work better"
    else:
        skew = "vol profile balanced"
    if breadth_pct >= 60 and mom_breadth >= 55 and bias == "up":
        action = "prefer LONGs on pullbacks"
    elif breadth_pct <= 40 and mom_breadth <= 45 and bias == "down":
        action = "prefer SHORTs on bounces"
    else:
        action = "trade both sides; keep size moderate"
    return f"Market level: {level}; bias: {bias}; {skew}; {action}"

@st.cache_data(ttl=180, show_spinner=False)
def universe_snapshot(symbols: list[str], max_symbols: int = 12, _seed=None):
    syms = symbols[:max_symbols]
    rows = []
    for s in syms:
        v = _symbol_vitals(s)
        if v.get("ok"): rows.append((s, v))
    if not rows: return None

    rv20s = [v["rv20"] for _, v in rows]
    rv60s = [v["rv60"] for _, v in rows]
    labels = [v["label"] for _, v in rows]
    above = sum(v["above200"] for _, v in rows)
    mom_bull = sum(v["macd_bull"] for _, v in rows)
    adx_vals = [v["adx14"] for _, v in rows]
    rsi_vals = [v["rsi14"] for _, v in rows]

    avg20 = float(np.mean(rv20s)); avg60 = float(np.mean(rv60s))
    med20 = float(np.median(rv20s)); med60 = float(np.median(rv60s))
    ups = sum(1 for x in labels if x == "Trend ↑")
    downs = sum(1 for x in labels if x == "Trend ↓")
    sides = sum(1 for x in labels if x == "Sideways")
    breadth = 100.0 * above / max(1, len(rows))
    mom_breadth = 100.0 * mom_bull / max(1, len(rows))
    adx_avg = float(np.mean(adx_vals))
    rsi_avg = float(np.mean(rsi_vals))
    guide = _universe_guide(avg20, avg60, breadth, ups, downs, mom_breadth)

    return {
        "n": len(rows),
        "avg20": avg20, "avg60": avg60, "med20": med20, "med60": med60,
        "ups": ups, "downs": downs, "sides": sides,
        "breadth": breadth, "mom_breadth": mom_breadth,
        "adx_avg": adx_avg, "rsi_avg": rsi_avg,
        "guide": guide
    }

snap = universe_snapshot(SYMBOLS, max_symbols=topn, _seed=(refresh_seed if auto_refresh else None))

# ========= Funding, OI & News helpers =========
def narrative_story(agg, signals, funding_level, funding_side, fund_mean_24h, news_risk):
    s1h, s4h, sd = signals["1h"], signals["4h"], signals["1d"]
    lines = []
    if agg["label"] == "Trend ↑": lines.append("Tone: buyers have the ball; most frames lean up.")
    elif agg["label"] == "Trend ↓": lines.append("Tone: sellers in control; bounces meet supply.")
    else: lines.append("Tone: balanced — chop until a clean break sticks.")
    lines.append("Daily vs EMA200: " + ("above (long bias)" if sd["price_vs_ema200"]=="above" else "below (short bias)"))
    lines.append("4h alignment: " + ("EMA20>EMA50 (trend aligned)" if signals["4h"]["ema20_cross_50"]=="bull" else "EMA20<EMA50 (not aligned)"))
    lines.append("1h momentum: " + ("MACD>signal (up)" if s1h["macd_cross"]=="bull" else "MACD<signal (down)"))
    if s1h["rsi14"] >= 70 or s4h["rsi14"] >= 70: lines.append("RSI hot on intraday → prefer entries after a pause.")
    if s1h["rsi14"] <= 30 or s4h["rsi14"] <= 30: lines.append("RSI cold → avoid chasing breakdowns; wait for a bounce.")
    if funding_level in ("elevated","extreme"):
        lines.append(f"Funding {funding_level}: {funding_side.lower()} — better to enter on pullbacks/pops, not chases.")
    if fund_mean_24h is not None:
        tilt = "longs pay" if fund_mean_24h>0 else ("shorts pay" if fund_mean_24h<0 else "flat")
        lines.append(f"24h mean funding: {fund_mean_24h*100:.3f}%/8h ({tilt}).")
    if news_risk == "elevated":
        lines.append("Headline risk up — keep size smaller or wait 30–60m after news.")
    return lines

@st.cache_data(ttl=120, show_spinner=False)
def fetch_funding(symbol: str, limit: int = 48, _seed=None):
    try:
        base = f"{FAPI_BASES[0]}/fapi/v1/fundingRate"
        data = http_json(base, params={"symbol": symbol, "limit": min(1000,limit)}, allow_worker=bool(cf_worker))
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df = df.dropna().sort_values("fundingTime")
            return df, "binance"
    except Exception:
        pass
    try:
        inst = f"{_base_from_symbol(symbol)}-USDT-SWAP"
        j = http_json("https://www.okx.com/api/v5/public/funding-rate-history",
                      params={"instId": inst, "limit": min(100, limit)}, timeout=8)
        arr = j.get("data") or []
        if not arr: return None, "okx_empty"
        df = pd.DataFrame(arr)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df["fundingTime"] = pd.to_datetime(pd.to_numeric(df["fundingTime"], errors="coerce"), unit="ms", utc=True)
        df = df.dropna().sort_values("fundingTime")
        return df, "okx"
    except Exception:
        return None, "none"

@st.cache_data(ttl=60, show_spinner=False)
def fetch_open_interest(symbol: str, _seed=None):
    try:
        j = http_json(f"{FAPI_BASES[0]}/fapi/v1/openInterest", params={"symbol": symbol}, allow_worker=bool(cf_worker))
        if isinstance(j, dict) and "openInterest" in j:
            val = float(j.get("openInterest") or 0.0)
            ts = datetime.now(timezone.utc)
            return pd.DataFrame({"ts":[ts], "oi":[val]}), "binance"
    except Exception:
        pass
    try:
        uly = f"{_base_from_symbol(symbol)}-USDT"
        j = http_json("https://www.okx.com/api/v5/public/open-interest", params={"instType":"SWAP","uly":uly}, timeout=8)
        arr = j.get("data") or []
        if not arr: return None, "okx_empty"
        oi = float(arr[0].get("oi") or 0.0)
        ts = datetime.now(timezone.utc)
        return pd.DataFrame({"ts":[ts], "oi":[oi]}), "okx"
    except Exception:
        return None, "none"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(feeds: list[str], lookback_hours: int = 24, max_items: int = 8, term: str = "Bitcoin", _seed=None):
    out = []; cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    patt = re.compile(fr"\b({re.escape(term)}|{re.escape(term.split()[0])})\b", re.I)
    for u in feeds:
        try:
            xml = http_text(u, timeout=8)
            for item in re.findall(r"<item>(.*?)</item>", xml, flags=re.S|re.I):
                title_match = re.search(r"<title>(.*?)</title>", item, flags=re.S|re.I)
                link_match  = re.search(r"<link>(.*?)</link>", item, flags=re.S|re.I)
                date_match  = re.search(r"<pubDate>(.*?)</pubDate>", item, flags=re.S|re.I)
                title = re.sub(r"<.*?>", "", (title_match.group(1).strip() if title_match else ""))
                link  = re.sub(r"<.*?>", "", (link_match.group(1).strip() if link_match else ""))
                pub   = (date_match.group(1).strip() if date_match else None)
                if not title or not link: continue
                if not patt.search(title): continue
                try:
                    dt = parsedate_to_datetime(pub)
                    if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
                    dt = dt.astimezone(timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)
                if dt >= cutoff: out.append({"title": title, "link": link, "ts": dt, "src": u})
        except Exception:
            continue
    seen, dedup = set(), []
    for it in sorted(out, key=lambda x: x["ts"], reverse=True):
        if it["link"] in seen: continue
        seen.add(it["link"]); dedup.append(it)
    return dedup[:max_items]

fdf, fsrc = fetch_funding(symbol, limit=48, _seed=(refresh_seed if auto_refresh else None))
last_funding = float(fdf["fundingRate"].iloc[-1]) if isinstance(fdf, pd.DataFrame) and not fdf.empty else None
funding_str, funding_level, funding_side = (funding_tilt(last_funding) if last_funding is not None else ("n/a","neutral","Flat"))
fund_mean_24h = float(fdf["fundingRate"].tail(3).mean()) if isinstance(fdf, pd.DataFrame) and len(fdf) >= 3 else None

news_items = fetch_news(NEWS_FEEDS, NEWS_LOOKBACK_HOURS, 8, term=_news_term(symbol),
                        _seed=(refresh_seed if auto_refresh else None))
news_risk = "elevated" if news_items and any(re.search(r"(hack|exploit|liquidat|halt|delay|lawsuit|ban|shutdown|outage|security|CPI|rate)", n["title"], flags=re.I) for n in news_items[:5]) else "normal"

# ========= PLAN A/B =========
def _atr_like(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _plan_bias_1h(signals: dict, df_1h: pd.DataFrame) -> bool:
    """
    Return True if we should favor LONGs based on 1h; False if SHORTs.
    1h rules:
      - must be above EMA200 AND (EMA20>EMA50 OR MACD>signal)
    4h strong-down veto:
      - if 4h is below EMA200 AND MACD bear AND ADX>=28, flip to SHORT bias
    """
    s1 = signals["1h"]; s4 = signals["4h"]

    one_hour_up = (s1["price_vs_ema200"] == "above") and (
        (s1["ema20_cross_50"] == "bull") or (s1["macd_cross"] == "bull")
    )

    strong_4h_down = (
        (s4["price_vs_ema200"] == "below") and
        (s4["macd_cross"] == "bear") and
        (s4.get("adx14", 0) >= 28)
    )

    if one_hour_up and not strong_4h_down:
        return True

    if not one_hour_up:
        return False

    return True

def swing_levels(df, lookback=30):
    highs = df["high"].tail(lookback); lows = df["low"].tail(lookback)
    return float(highs.max()), float(lows.min())

def plan_levels(df_1h: pd.DataFrame, favor_long: bool):
    s = df_1h["close"]
    ema20v, ema50v, ema200v = float(ema(s,20).iloc[-1]), float(ema(s,50).iloc[-1]), float(ema(s,200).iloc[-1])
    swing_hi = float(df_1h["high"].tail(40).max())
    swing_lo = float(df_1h["low"].tail(40).min())
    atr = float(_atr_like(df_1h, 14).iloc[-1] or 0)

    entry = (ema20v + ema50v) / 2.0
    buf = max(0.0015 * entry, 0.35 * atr)

    if favor_long and s.iloc[-1] < ema200v:
        favor_long = False  # safety flip

    if favor_long:
        stop  = min(swing_lo, entry) - buf
        risk  = max(1e-9, entry - stop)
        t1    = entry + risk
        t2    = max(entry + 2*risk, swing_hi)
        note  = ("Plan A (long): buy dip toward EMA20/EMA50; stop below last swing-low "
                 "with volatility buffer; T1=1R, T2=2R/prev high.")
        alt   = ("Plan B (flip): if 1h loses EMA200 and fails to reclaim, switch to short "
                 "on a weak bounce into EMAs.")
    else:
        stop  = max(swing_hi, entry) + buf
        risk  = max(1e-9, stop - entry)
        t1    = entry - risk
        t2    = min(entry - 2*risk, swing_lo)
        note  = ("Plan A (short): sell bounce into EMA20/EMA50; stop above last swing-high "
                 "with volatility buffer; T1=1R, T2=2R/prev low.")
        alt   = ("Plan B (flip): if 1h reclaims EMA200 and holds, prefer longs on pullbacks.")

    return {"entry":entry, "stop":stop, "t1":t1, "t2":t2,
            "note":note, "alt":alt,
            "ema200": ema200v, "atr": atr}

favor_long = _plan_bias_1h(signals, df_1h)
plan = plan_levels(df_1h.copy(), favor_long)

# ========= MATRIX (multi-TF) =========
def tf_row(tframe):
    df2, _ = tf_data[tframe]
    s2 = signals[tframe]
    score, conf2, lab2 = regime_score(s2)
    rv20 = realized_vol_series(df2.set_index("ts")["close"], 20).iloc[-1] * 100
    rec = per_tf_recommendation(s2)
    return {
        "tf": tframe, "close": df2["close"].iloc[-1],
        "ema200_dir": "↑" if s2["price_vs_ema200"]=="above" else "↓",
        "ema20>50": "✓" if s2["ema20_cross_50"]=="bull" else "×",
        "macd": "✓" if s2["macd_cross"]=="bull" else "×",
        "rsi": s2["rsi14"], "rsi_ma5": s2["rsi_ma5"], "rsi>ma5": s2["rsi_cross"],
        "adx": s2["adx14"], "rv20%": rv20,
        "score": score, "conf": conf2, "label": lab2, "recommend": rec
    }

matrix_rows = [tf_row(t) for t in TFS]
mat = pd.DataFrame(matrix_rows).set_index("tf")

def color_yesno(val, yes="✓", no="×"):
    if val == yes: return "background-color:#d1ffd6"
    if val == no:  return "background-color:#ffd1d1"
    return ""
def color_updown(val):
    return "background-color:#d1ffd6" if val=="↑" else ("background-color:#ffd1d1" if val=="↓" else "")
def color_rsi(val):
    try:
        v=float(val); 
        if v>=70: return "background-color:#ffd1d1"
        if v<=30: return "background-color:#d1ffd6"
        return ""
    except: return ""
def color_adx(val):
    try:
        v=float(val)
        if v>=28: return "background-color:#d1ffd6"
        if v>=22: return "background-color:#fff2cc"
        return "background-color:#ffd1d1"
    except: return ""
def color_rec(val):
    if val=="Buy dips": return "background-color:#d1ffd6"
    if val=="Sell rips": return "background-color:#ffd1d1"
    return "background-color:#fff2cc"

# ========= UNIVERSE INDEX (MVP) =========
# Build weights once per day (UTC): reconstitute Top-N & weights
def _weights_from_cg(bases: list[str], scheme: str):
    # lookup cg row by base
    by_base = {it["base"]: it for it in cg}
    rows = [by_base.get(b, {"market_cap":1.0,"total_volume":1.0}) for b in bases]
    if scheme.startswith("Market cap"):
        raw = np.array([r["market_cap"] or 0.0 for r in rows], dtype=float)
    elif scheme.startswith("Liquidity"):
        raw = np.array([r["total_volume"] or 0.0 for r in rows], dtype=float)
    else:
        raw = np.ones(len(rows), dtype=float)
    raw = np.where(np.isfinite(raw) & (raw>0), raw, 0.0)
    if raw.sum() <= 0:
        w = np.ones_like(raw) / max(1,len(raw))
    else:
        w = raw / raw.sum()
    return w

@st.cache_data(ttl=24*3600, show_spinner=False)
def universe_daily_spec(symbols: list[str], bases_for_symbols: list[str], scheme: str, day_key: str):
    """
    Returns dict with 'symbols', 'weights' arrays for the day (freeze basket/weights for 24h).
    """
    weights = _weights_from_cg(bases_for_symbols, scheme)
    return {"symbols": symbols, "bases": bases_for_symbols, "weights": weights.tolist(), "scheme": scheme, "day": day_key}

def _today_key():
    # UTC date as key
    return datetime.utcnow().strftime("%Y-%m-%d")

def _symbols_and_bases_for_universe(symbols: list[str]):
    bases = [_base_from_symbol(s) for s in symbols]
    return symbols, bases

# Composite builder (close-only, normalized weighted sum)
@st.cache_data(ttl=300, show_spinner=False)
def composite_series(symbols: list[str], weights: list[float], interval: str, limit_each: int = 1500, method: str = "arith"):
    """
    method: 'arith' (normalized weighted sum) or 'geom' (weighted geometric)
    Returns DataFrame: ts, composite (float)
    """
    dfs = []
    for s in symbols:
        df, _ = get_klines_df(s, interval, limit=limit_each)
        df = drop_unclosed(df, interval)
        dfs.append(df[["ts","close"]].rename(columns={"close":_base_from_symbol(s)}))
    if not dfs: raise RuntimeError("No data for composite")
    # inner-join on timestamps to ensure aligned bars
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on="ts", how="inner")
    if merged.empty or merged.shape[1] < 2:
        raise RuntimeError("Composite merge resulted in empty set")
    merged = merged.sort_values("ts").reset_index(drop=True)
    cols = [c for c in merged.columns if c!="ts"]
    X = merged[cols].astype(float)

    # normalize each asset to 1.0 at start
    norm = X / X.iloc[0]
    w = np.array(weights, dtype=float)
    w = w[:len(cols)]
    if w.sum() <= 0: w = np.ones(len(cols))/len(cols)

    if method == "geom":
        # weighted log-sum → exp back
        lognorm = np.log(norm.replace(0, np.nan)).fillna(method="ffill").fillna(0.0)
        comp = np.exp(np.matmul(lognorm.values, w.reshape(-1,1)).ravel())
    else:
        comp = np.matmul(norm.values, w.reshape(-1,1)).ravel()

    out = pd.DataFrame({"ts": merged["ts"], "composite": comp})
    return out

# ========= TABS LAYOUT =========
tab_token, tab_universe = st.tabs(["Token", "Universe Index"])

with tab_token:
    # ---- LEFT: Narrative & Decision ----
    left, spacer, right = st.columns([1.0, 0.02, 1.4])

    with left:
        st.subheader("Token Vitals (multi-TF)")
        st.markdown(
            f"Aggregate Regime: **{agg['label']}** (confidence: *{agg['conf']}*)<br>"
            f"<span style='color:#6b7280'>TF votes — ↑:{agg['ups']} / ↓:{agg['downs']} / ↔:{agg['sides']}</span>",
            unsafe_allow_html=True,
        )
        tale_lines = narrative_story(agg, signals, funding_level, funding_side, fund_mean_24h, news_risk)
        st.markdown("\n".join([f"- {ln}" for ln in tale_lines]))

        st.markdown(
            (
                f"**{plan['note']}**  \n"
                f"*Alternative:* {plan['alt']}  \n"
                f"<span style='color:#6b7280'>Plan bias source: 1h ("
                f"{'above' if signals['1h']['price_vs_ema200']=='above' else 'below'} EMA200; "
                f"{'EMA20>EMA50' if signals['1h']['ema20_cross_50']=='bull' else 'EMA20<EMA50'}; "
                f"{'MACD>signal' if signals['1h']['macd_cross']=='bull' else 'MACD<signal'}). "
                "4h veto applies only if strong downtrend (below EMA200 & MACD bear & ADX≥28).</span>"
            ),
            unsafe_allow_html=True
        )

    with right:
        st.subheader("Signal matrix (multi-TF)")
        styled = (
            mat.style
              .applymap(color_updown, subset=["ema200_dir"])
              .applymap(color_yesno, subset=["ema20>50","macd","rsi>ma5"])
              .applymap(color_rsi, subset=["rsi","rsi_ma5"])
              .applymap(color_adx, subset=["adx"])
              .applymap(color_rec, subset=["recommend"])
              .format({"close":"{:.2f}","rsi":"{:.1f}","rsi_ma5":"{:.1f}","adx":"{:.1f}","rv20%":"{:.1f}%"})
        )
        st.dataframe(styled, use_container_width=True)

    # ========= CHART =========
    with st.expander("Chart (1h)", expanded=True):
        df = df_1h.copy()
        x_ts = df["ts"]; close_s = df["close"]
        ema20s, ema50s, ema200s = ema(close_s,20), ema(close_s,50), ema(close_s,200)
        if not PLOTLY:
            st.line_chart(close_s.set_axis(x_ts), use_container_width=True)
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=x_ts, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                name="OHLC", opacity=0.85
            ))
            fig.add_trace(go.Scatter(x=x_ts, y=ema20s,  name="EMA20",  line=dict(width=1.5)))
            fig.add_trace(go.Scatter(x=x_ts, y=ema50s,  name="EMA50",  line=dict(width=1.5)))
            fig.add_trace(go.Scatter(x=x_ts, y=ema200s, name="EMA200", line=dict(width=1.5)))
            for n in sorted(set(int(x) for x in extra_emas if isinstance(x,(int,float)))):
                try:
                    fig.add_trace(go.Scatter(x=x_ts, y=ema(close_s, n), name=f"EMA{n}", line=dict(width=1)))
                except Exception:
                    pass
            if overlay_plan:
                for y, name, dash in [
                    (plan["entry"],"Entry","dot"),
                    (plan["stop"],"Stop","dash"),
                    (plan["t1"],"T1","dash"),
                    (plan["t2"],"T2","dash")
                ]:
                    fig.add_hline(y=y, line_dash=dash, annotation_text=f"{name}: {y:,.2f}", annotation_position="right")
            fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # ========= Funding & News =========
    with st.expander("Funding & Open Interest (Futures context)"):
        try:
            if isinstance(fdf, pd.DataFrame) and not fdf.empty:
                st.metric(f"Last funding ({fsrc})", f"{fdf['fundingRate'].iloc[-1]*100:.4f}% / 8h")
                if fund_mean_24h is not None:
                    st.caption(f"24h mean: {fund_mean_24h*100:.4f}% / 8h  ·  bias: {'longs pay' if fund_mean_24h>0 else ('shorts pay' if fund_mean_24h<0 else 'flat')}")
                st.line_chart(fdf.set_index("fundingTime")["fundingRate"]*100, height=160)
            else:
                st.info("No funding data.")
        except Exception as e:
            st.warning(f"Funding error: {e}")

    with st.expander("News (last 24h)", expanded=False):
        if news_items:
            for n in news_items:
                ts = n["ts"].strftime("%Y-%m-%d %H:%M UTC")
                st.markdown(f"- [{n['title']}]({n['link']})  —  *{ts}*")
        else:
            st.info(f"No fresh {_news_term(symbol)} headlines found in the last 24h.")

# ========= UNIVERSE TAB =========
with tab_universe:
    st.subheader("Universe Index (Top-N Composite)")

    # Daily basket/weights freeze
    syms_for_universe, bases_for_universe = _symbols_and_bases_for_universe(SYMBOLS[:topn])
    spec = universe_daily_spec(syms_for_universe, bases_for_universe, weighting_scheme, _today_key())

    # Build composite on selected timeframe (use 1h by default for UI responsiveness; allow change)
    comp_tf = st.selectbox("Composite timeframe", options=["15m","1h","4h","1d"], index=1,
                           help="Composite built from close-only series; candles require heavy OHLC sync.")
    comp_method = st.selectbox("Index combine method", options=["Arithmetic (normalized sum)","Geometric (log-sum)"], index=0)
    method_key = "geom" if comp_method.startswith("Geometric") else "arith"

    try:
        comp_df = composite_series(spec["symbols"], spec["weights"], comp_tf, limit_each=1200, method=method_key)
        comp_df = comp_df.dropna().sort_values("ts")
        comp_df = comp_df.iloc[-1000:]  # keep it snappy
        scomp = comp_df.set_index("ts")["composite"]
        c_ema20, c_ema50, c_ema200 = ema(scomp,20), ema(scomp,50), ema(scomp,200)
        c_rv20 = float(realized_vol_series(scomp,20).iloc[-1] * 100)
        c_rv60 = float(realized_vol_series(scomp,60).iloc[-1] * 100)
        c_macd_line, c_macd_sig, c_macd_hist = macd(scomp)

        # Headline
        h1, h2, h3, h4 = st.columns([1.2,1.2,1.2,1.2])
        h1.metric("Composite RV20%", f"{c_rv20:.1f}%")
        h2.metric("Composite RV60%", f"{c_rv60:.1f}%")
        if snap:
            h3.metric("Breadth (above 1h EMA200)", f"{snap['breadth']:.0f}%")
            h4.metric("Momentum breadth (MACD>signal)", f"{snap['mom_breadth']:.0f}%")
        st.caption(rv_one_liner(c_rv20, c_rv60))

        # Chart
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comp_df["ts"], y=comp_df["composite"], name="Composite", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=c_ema20.index, y=c_ema20.values, name="EMA20", line=dict(width=1.3)))
            fig.add_trace(go.Scatter(x=c_ema50.index, y=c_ema50.values, name="EMA50", line=dict(width=1.3)))
            fig.add_trace(go.Scatter(x=c_ema200.index, y=c_ema200.values, name="EMA200", line=dict(width=1.3)))
            fig.update_layout(height=460, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(comp_df.set_index("ts")["composite"], height=300, use_container_width=True)

        # Top-N snapshot table (moved here from Token tab)
        if snap:
            now_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
            rows = [
                ("Avg RV20", f"{snap['avg20']:.1f}%", _level_note_rv(snap['avg20'])),
                ("Avg RV60", f"{snap['avg60']:.1f}%", _level_note_rv(snap['avg60'])),
                ("Median RV20", f"{snap['med20']:.1f}%", ""),
                ("Median RV60", f"{snap['med60']:.1f}%", ""),
                ("Breadth (above 1h EMA200)", f"{snap['breadth']:.0f}%", "higher = broader uptrend"),
                ("Momentum breadth (MACD>signal)", f"{snap['mom_breadth']:.0f}%", "higher = more coins with bullish momentum"),
                ("Regime votes", f"↑ {snap['ups']} / ↓ {snap['downs']} / ↔ {snap['sides']}", "bias from up vs down counts"),
                ("Avg ADX14 (trend strength)", f"{snap['adx_avg']:.1f}", _note_adx(snap['adx_avg'])),
                ("Avg RSI14", f"{snap['rsi_avg']:.1f}", _note_rsi(snap['rsi_avg'])),
                ("Universe guide", "", snap['guide'])
            ]
            table_html = [
                f"""
<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:12px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <div style="font-weight:600;">Top-N snapshot (N={snap['n']})</div>
    <div style="font-size:12px;color:#6b7280;">{now_utc}</div>
  </div>
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr>
        <th style="text-align:left;border-bottom:1px solid #e5e7eb;padding:6px 4px;color:#6b7280;width:36%;">Parameter</th>
        <th style="text-align:left;border-bottom:1px solid #e5e7eb;padding:6px 4px;color:#6b7280;width:20%;">Value</th>
        <th style="text-align:left;border-bottom:1px solid #e5e7eb;padding:6px 4px;color:#6b7280;width:44%;">Note / dynamics</th>
      </tr>
    </thead>
    <tbody>
"""
            ]
            for p, v, n in rows:
                table_html.append(
                    f"""<tr>
<td style="padding:6px 4px;border-bottom:1px solid #f1f5f9;">{p}</td>
<td style="padding:6px 4px;border-bottom:1px solid #f1f5f9;">{v}</td>
<td style="padding:6px 4px;border-bottom:1px solid #f1f5f9;color:#374151;">{n}</td>
</tr>"""
                )
            table_html.append("</tbody></table></div>")
            st.markdown("\n".join(table_html), unsafe_allow_html=True)
        else:
            st.info("Top-N snapshot not available (providers busy).")

        # Helper text
        st.caption(
            "Notes: Basket & weights reconstituted once per UTC day (frozen for the session). "
            "Composite built from close-only, normalized per-asset at window start; "
            "choose Arithmetic or Geometric combine method."
        )
    except Exception as e:
        st.error(f"Composite build failed: {e}")

# ========= report.json (debug) =========
payload = None
@st.cache_data(ttl=300, show_spinner=False)
def fetch_report(url: str, _seed=None):
    r = requests.get(url, timeout=15); r.raise_for_status()
    payload = json.loads(r.content.decode("utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Bad report.json schema.")
    return payload

if report_url:
    try: payload = fetch_report(report_url, _seed=(refresh_seed if auto_refresh else None))
    except Exception: payload = None

if payload:
    with st.expander("report.json — raw & pivot (debug)", expanded=False):
        if payload.get("stale"): st.warning("report.json is STALE; charts use live klines.")
        rows = payload.get("data", []); dfj = pd.json_normalize(rows)
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
