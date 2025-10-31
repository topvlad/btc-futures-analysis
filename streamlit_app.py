# streamlit_app.py — v1.13.1
# Changes vs 1.13.0 (user-driven):
# - Lightweight chart OFF by default (can still be toggled on).
# - Plan A/B added for 4h (with own bias & levels).
# - "Playbook" blocks: condensed, human-meaningful use-case of signals
#   (what to do now, when invalidated, if/then) on both tabs.
# - Chart draws plan lines on 1h and on 4h (when respective TF selected).
# - Minor text/UX tweaks; no breaking changes.

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

TFS = ["15m", "1h", "4h", "1d", "1w"]
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
TF_LIMITS  = {"15m": 1200, "1h": 900, "4h": 700, "1d": 420, "1w": 240}

SPOT_BASES = ["https://api.binance.com","https://api1.binance.com","https://api2.binance.com",
              "https://api3.binance.com","https://api4.binance.com","https://api5.binance.com"]
FAPI_BASES = ["https://fapi.binance.com","https://fapi1.binance.com","https://fapi2.binance.com",
              "https://fapi3.binance.com","https://fapi4.binance.com","https://fapi5.binance.com"]

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

# ========= CSS =========
st.markdown("""
<style>
.block-container{padding-top:1.1rem; padding-bottom:2rem;}
.scroll-wrap{overflow:auto; -webkit-overflow-scrolling:touch; max-height: 480px;}
.scroll-wrap-sm{overflow:auto; -webkit-overflow-scrolling:touch; max-height: 360px;}
.tile-table td, .tile-table th { padding: 8px 8px; }
@media (max-width: 640px){
  .tile-table td, .tile-table th { padding: 6px 6px !important; font-size: 13px !important; }
}
.small-note{ color:#6b7280; font-size:12px; }
.badge{display:inline-block; padding:2px 8px; border-radius:9999px; font-size:12px; margin-right:6px; background:#eef2ff;}
.code-like{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; background:#f5f7fa; border:1px solid #e5e7eb; padding:6px 8px; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

# ========= UTIL =========
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

def http_json(url: str, params=None, timeout=7, allow_worker=False):
    params = params or {}
    attempts = []
    if allow_worker and cf_worker:
        attempts.append(("WORKER", build_worker_url(cf_worker, url + ("?" + requests.compat.urlencode(params) if params else "")), {}))
    attempts.append(("DIRECT", url, params))
    last_e = None
    for label, full, p in attempts:
        try:
            r = requests.get(full, params=None if label=="WORKER" else p, timeout=timeout,
                             headers={"Accept":"application/json","User-Agent":"binfapp/1.13.1"})
            if r.status_code != 200: last_e = f"status {r.status_code}"; continue
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" not in ct and not ("/klines" in full or "/candles" in full):
                last_e = "non_json"; continue
            return r.json()
        except Exception as e:
            last_e = str(e); continue
    raise RuntimeError(f"http_json failed: {last_e}")

def http_text(url: str, timeout=7):
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"binfapp/1.13.1"}); r.raise_for_status()
    return r.text

# ========= DYNAMIC UNIVERSE =========
TOPN_DEFAULT = 10

@st.cache_data(ttl=3600, show_spinner=False)
def _coingecko_topn_with_caps(per_page=100, page=1):
    try:
        j = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency":"usd","order":"market_cap_desc","per_page":per_page,"page":page},
            timeout=7, headers={"Accept":"application/json","User-Agent":"binfapp/1.13.1"}
        ).json()
        out = []
        for it in j:
            sym = (it.get("symbol") or "").upper().strip()
            if sym in ("USDT","USDC","FDUSD","TUSD","DAI","USDD","USDE","PYUSD"): continue
            sym = {"XBT":"BTC","WETH":"ETH"}.get(sym, sym)
            out.append({"base": sym, "market_cap": float(it.get("market_cap") or 0.0),
                        "total_volume": float(it.get("total_volume") or 0.0)})
        seen, dedup = set(), []
        for it in out:
            if it["base"] in seen: continue
            seen.add(it["base"]); dedup.append(it)
        return dedup
    except Exception:
        bases = ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","TON","TRX","LINK"]
        return [{"base":b,"market_cap":1.0,"total_volume":1.0} for b in bases]

@st.cache_data(ttl=3600, show_spinner=False)
def _binance_futures_universe_from_bases(candidates: list[str], _seed=None):
    try:
        ex = http_json(f"{FAPI_BASES[0]}/fapi/v1/exchangeInfo", timeout=8, allow_worker=bool(CF_WORKER_DEFAULT))
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
            if cand in tradeables: out.append(cand)
        return out
    except Exception:
        return [f"{b}USDT" for b in ["BTC","ETH","BNB","SOL","XRP","ADA","DOGE","TON","TRX","LINK"]]

# ========= SIDEBAR =========
st.sidebar.header("Data & Options")

topn = st.sidebar.number_input("Universe size (Top-N by market cap)", min_value=5, max_value=30,
                               value=TOPN_DEFAULT, step=1,
                               help="Coins from CoinGecko top caps, filtered to Binance USDT perps.")
if "universe_seed" not in st.session_state:
    st.session_state.universe_seed = 0
if st.sidebar.button("↻ Refresh universe list", help="Force refresh of the Top-N list (bypass 1h cache)."):
    st.session_state.universe_seed = int(time.time())

weighting_scheme = st.sidebar.selectbox(
    "Universe Index weighting",
    options=["Market cap (default)","Liquidity (24h volume)","Equal weight"],
    index=0,
    help="Applies to the composite in the Universe Index tab."
)

cg = _coingecko_topn_with_caps()
bases_ranked = [it["base"] for it in cg]
bases_trimmed = bases_ranked[:int(topn)]
SYMBOLS = _binance_futures_universe_from_bases(bases_trimmed, _seed=st.session_state.universe_seed)
if not SYMBOLS: SYMBOLS = ["BTCUSDT","ETHUSDT"]
if "BTCUSDT" in SYMBOLS:
    SYMBOLS.remove("BTCUSDT"); SYMBOLS.insert(0, "BTCUSDT")
st.sidebar.caption(f"Universe: {', '.join([_base_from_symbol(s) for s in SYMBOLS[:10]])}{'…' if len(SYMBOLS)>10 else ''}")

symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=0)

cf_worker  = st.sidebar.text_input("Cloudflare Worker (optional)", value=CF_WORKER_DEFAULT)
report_url = st.sidebar.text_input("GitHub Pages report.json (debug URL)", value=REPORT_URL_DEFAULT)

# Performance toggles
col_perf1, col_perf2 = st.sidebar.columns(2)
auto_refresh_on = col_perf1.checkbox("Auto-refresh", value=False, help="Refresh every ~12 min.")
light_mode_chart = col_perf2.checkbox("Lightweight chart", value=False, help="Line chart (faster); OFF by default.")

if auto_refresh_on:
    st.autorefresh(interval=12*60*1000, key="auto_refresh_12m")

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
        data = http_json(f"{base}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": min(2000,limit)}, timeout=7)
        return _to_df_binance(data), f"binance({base})"

def _okx_klines(symbol: str, interval: str, limit: int):
    base = _base_from_symbol(symbol); inst = f"{base}-USDT"
    bar_map = {"15m":"15m","1h":"1H","4h":"4H","1d":"1D","1w":"1W"}
    bar = bar_map.get(interval, "1H")
    j = http_json("https://www.okx.com/api/v5/market/candles",
                  params={"instId": inst, "bar": bar, "limit": min(500,limit)}, timeout=7)
    data = j.get("data") or []
    if not data: raise RuntimeError("OKX empty")
    return _to_df_okx(data), "okx"

def _coinbase_klines(symbol: str, interval: str, limit: int):
    base = _base_from_symbol(symbol); product_map = {"BTC":"BTC-USD","ETH":"ETH-USD"}
    product = product_map.get(base, "BTC-USD")
    gran_map = {"15m":900,"1h":3600,"4h":14400,"1d":86400,"1w":86400}
    gran = gran_map.get(interval, 3600)
    data = http_json(f"https://api.exchange.coinbase.com/products/{product}/candles",
                     params={"granularity": gran, "limit": min(300,limit)}, timeout=7)
    if not isinstance(data, list) or not data: raise RuntimeError("Coinbase empty")
    return _to_df_coinbase(data), "coinbase"

@st.cache_data(ttl=120, show_spinner=False)
def get_klines_df(symbol: str, interval: str, limit: int | None = None, _seed=None):
    limit = int(limit or TF_LIMITS.get(interval, 900))
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
    ema20v, ema50v, ema200v = ema(s,20), ema(s,50), ema(s,200)
    rsi14v = rsi_series(s,14); rsi_ma = rsi14v.rolling(5).mean()
    macd_line, macd_sig, macd_hist = macd(s)
    adx_val = adx(df,14)
    return {
        "ema20": float(ema20v.iloc[-1]), "ema50": float(ema50v.iloc[-1]), "ema200": float(ema200v.iloc[-1]),
        "rsi14": float(rsi14v.iloc[-1]), "rsi_ma5": float(rsi_ma.iloc[-1]),
        "rsi_cross": "✓" if rsi14v.iloc[-1] > rsi_ma.iloc[-1] else "×",
        "macd": float(macd_line.iloc[-1]), "macd_signal": float(macd_sig.iloc[-1]), "macd_hist": float(macd_hist.iloc[-1]),
        "adx14": float(adx_val.iloc[-1]),
        "price_vs_ema200": "above" if s.iloc[-1] > ema200v.iloc[-1] else "below",
        "ema20_cross_50": "bull" if ema20v.iloc[-1] > ema50v.iloc[-1] else "bear",
        "macd_cross": "bull" if macd_line.iloc[-1] > macd_sig.iloc[-1] else "bear",
        "ema50_slope": float((ema50v.iloc[-1] - ema50v.iloc[-3]) / max(1e-9, ema50v.iloc[-3])),
    }

def regime_score(sig):
    adx_v = sig.get("adx14",0)
    conf = "high" if adx_v >= 28 else ("medium" if adx_v >= 22 else "low")
    score = (1 if sig["price_vs_ema200"]=="above" else -1) + (1 if sig["ema20_cross_50"]=="bull" else -1) + (1 if sig["macd_cross"]=="bull" else -1)
    label = "Trend ↑" if score >= 2 else ("Sideways" if -1 <= score <= 1 else "Trend ↓")
    return score, conf, label

def funding_tilt(last_rate: float):
    if last_rate is None:
        return "unknown", "n/a", "n/a"
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

def rv_one_liner(rv20: float, rv60: float) -> str:
    ratio = (rv20 / max(rv60, 1e-9))
    if ratio >= 1.15:  skew = "short-term vol rising vs medium-term (momentum bias; wider stops)."
    elif ratio <= 0.85: skew = "short-term vol cooling (mean-reversion bias; tighter stops)."
    else:               skew = "short- and medium-term vol aligned."
    hi = max(rv20, rv60)
    level = "very high" if hi>=100 else ("high" if hi>=70 else ("moderate" if hi>=40 else "calm"))
    return f"Volatility is {level}; {skew}"

# ========= CORE LOAD =========
@st.cache_data(ttl=120, show_spinner=False)
def get_all_tf_data(symbol: str, tfs: list[str], _seed=None):
    out = {}
    for tf in tfs:
        df, src = get_klines_df(symbol, tf, limit=TF_LIMITS.get(tf))
        df = drop_unclosed(df, tf)
        out[tf] = (df, src)
    return out

try:
    tf_data = get_all_tf_data(symbol, TFS, _seed=None)
except Exception as e:
    st.error(f"Klines failed: {e}"); st.stop()

# ========= FUNDING / NEWS =========
@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding(symbol: str, limit: int = 48, _seed=None):
    try:
        data = http_json(f"{FAPI_BASES[0]}/fapi/v1/fundingRate",
                         params={"symbol": symbol, "limit": min(1000,limit)}, allow_worker=bool(cf_worker))
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
                      params={"instId": inst, "limit": min(100, limit)}, timeout=7)
        arr = j.get("data") or []
        if not arr: return None, "okx_empty"
        df = pd.DataFrame(arr)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        df["fundingTime"] = pd.to_datetime(pd.to_numeric(df["fundingTime"], errors="coerce"), unit="ms", utc=True)
        df = df.dropna().sort_values("fundingTime")
        return df, "okx"
    except Exception:
        return None, "none"

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(feeds: list[str], lookback_hours: int = 24, max_items: int = 8, term: str = "Bitcoin", _seed=None):
    out = []; cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    first = term.split()[0] if term else "Bitcoin"
    patt = re.compile(fr"\b({re.escape(term)}|{re.escape(first)})\b", re.I)
    for u in feeds:
        try:
            xml = http_text(u, timeout=7)
            for item in re.findall(r"<item>(.*?)</item>", xml, flags=re.S|re.I):
                title_match = re.search(r"<title>(.*?)</title>", item, flags=re.S|re.I)
                link_match  = re.search(r"<link>(.*?)</link>", item, flags=re.S|re.I)
                date_match  = re.search(r"<pubDate>(.*?)</pubDate>", item, flags=re.S|re.I)
                title = re.sub(r"<.*?>", "", (title_match.group(1).strip() if title_match else ""))
                link  = re.sub(r"<.*?>", "", (link_match.group(1).strip() if link_match else ""))
                pub   = (date_match.group(1).strip() if date_match else None)
                if not title or not link: continue
                if term and not patt.search(title): continue
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

# ========= UNIVERSE HELPERS =========
def _weights_from_cg(bases: list[str], scheme: str):
    by_base = {it["base"]: it for it in cg}
    rows = [by_base.get(b, {"market_cap":1.0,"total_volume":1.0}) for b in bases]
    if scheme.startswith("Market cap"): raw = np.array([r["market_cap"] or 0.0 for r in rows], dtype=float)
    elif scheme.startswith("Liquidity"): raw = np.array([r["total_volume"] or 0.0 for r in rows], dtype=float)
    else: raw = np.ones(len(rows), dtype=float)
    raw = np.where(np.isfinite(raw) & (raw>0), raw, 0.0)
    return (raw/raw.sum()) if raw.sum()>0 else np.ones_like(raw)/max(1,len(raw))

@st.cache_data(ttl=24*3600, show_spinner=False)
def universe_daily_spec(symbols: list[str], bases_for_symbols: list[str], scheme: str, day_key: str):
    weights = _weights_from_cg(bases_for_symbols, scheme)
    return {"symbols": symbols, "bases": bases_for_symbols, "weights": weights.tolist(), "scheme": scheme, "day": day_key}

def _today_key(): return datetime.utcnow().strftime("%Y-%m-%d")

@st.cache_data(ttl=300, show_spinner=False)
def composite_series(symbols: list[str], weights: list[float], interval: str, method: str = "arith"):
    dfs = []
    for s in symbols:
        df, _ = get_klines_df(s, interval, limit=TF_LIMITS.get(interval))
        df = drop_unclosed(df, interval)
        dfs.append(df[["ts","close"]].rename(columns={"close":_base_from_symbol(s)}))
    if not dfs: raise RuntimeError("No data for composite")
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on="ts", how="inner")
    if merged.empty or merged.shape[1] < 2: raise RuntimeError("Composite merge empty")
    merged = merged.sort_values("ts").reset_index(drop=True)
    cols = [c for c in merged.columns if c!="ts"]
    X = merged[cols].astype(float)
    norm = X / X.iloc[0]
    w = np.array(weights, dtype=float)[:len(cols)]
    w = (w / w.sum()) if w.sum()>0 else np.ones(len(cols))/len(cols)
    if method == "geom":
        lognorm = np.log(norm.replace(0, np.nan)).fillna(method="ffill").fillna(0.0)
        comp = np.exp(np.matmul(lognorm.values, w.reshape(-1,1)).ravel())
    else:
        comp = np.matmul(norm.values, w.reshape(-1,1)).ravel()
    return pd.DataFrame({"ts": merged["ts"], "composite": comp})

@st.cache_data(ttl=240, show_spinner=False)
def _symbol_vitals_tf(s: str, tf: str):
    try:
        df, _ = get_klines_df(s, tf, limit=TF_LIMITS.get(tf))
        df = drop_unclosed(df, tf)
        srs = df.set_index("ts")["close"]
        rv20 = float(realized_vol_series(srs, 20).iloc[-1] * 100)
        rv60 = float(realized_vol_series(srs, 60).iloc[-1] * 100)
        sig = trend_flags(df)
        _, _, lab = regime_score(sig)
        above200 = 1 if sig["price_vs_ema200"] == "above" else 0
        macd_bull = 1 if sig["macd_cross"] == "bull" else 0
        adx14 = float(sig["adx14"]); rsi14 = float(sig["rsi14"])
        return {"ok": True, "rv20": rv20, "rv60": rv60, "label": lab,
                "above200": above200, "macd_bull": macd_bull, "adx14": adx14, "rsi14": rsi14}
    except Exception:
        return {"ok": False}

# ====== Buckets & Notes (unchanged) ======
def _bucket(value: float, edges: list[float], labels: list[str]):
    if value is None or not np.isfinite(value): return "n/a", 2
    idx = 0
    for i, e in enumerate(edges):
        if value < e:
            idx = i
            break
    else:
        idx = 4
    return labels[idx], idx

def _note_rv(value: float):
    labels = ["low","med-low","average","med-high","high"]
    lab, idx = _bucket(value, [30, 45, 60, 85], labels)
    text = {
        0: "very calm; breakouts often fade; tight stops ok",
        1: "calm; mean-reversion favoured",
        2: "balanced; no strong vol edge",
        3: "elevated; momentum entries hold better",
        4: "high; wide swings; size down / wider stops",
    }[idx]
    return f"{lab}", text, idx

def _note_adx(value: float):
    labels = ["low","med-low","average","med-high","high"]
    lab, idx = _bucket(value, [17, 22, 28, 38], labels)
    text = {
        0: "very weak trend; ranges dominate",
        1: "weak trend; breakouts unreliable",
        2: "trend building; breakouts improving",
        3: "strong trend; breakouts more likely to hold",
        4: "very strong trend; pullbacks preferred to chasing",
    }[idx]
    return f"{lab}", text, idx

def _note_rsi(value: float):
    labels = ["low","med-low","average","med-high","high"]
    lab, idx = _bucket(value, [25.01, 40.01, 60.01, 75.01], labels)
    text = {
        0: "oversold zone",
        1: "tilt to oversold",
        2: "neutral",
        3: "tilt to overbought",
        4: "overbought zone",
    }[idx]
    return f"{lab}", text, idx

def _note_pct(value: float, kind: str):
    labels = ["low","med-low","average","med-high","high"]
    lab, idx = _bucket(value, [30, 45, 55, 70], labels)
    if kind == "breadth":
        extra = {
            0: "narrow advance; leadership thin",
            1: "modestly narrow participation",
            2: "balanced participation",
            3: "broadening participation",
            4: "very broad uptrend participation",
        }[idx]
    else:
        extra = {
            0: "few coins with bullish momentum",
            1: "limited momentum leadership",
            2: "mixed momentum",
            3: "solid momentum participation",
            4: "widespread momentum",
        }[idx]
    return f"{lab}", extra, idx

def _cross_effects_text(avg20, avg60, breadth, mom_breadth, adx_avg):
    skew_ratio = avg20 / max(avg60, 1e-9)
    if skew_ratio >= 1.15:
        skew_note = "short-term vol expanding → momentum bias"
    elif skew_ratio <= 0.85:
        skew_note = "short-term vol cooling → mean-reversion bias"
    else:
        skew_note = "vol horizons aligned"

    _, _, adx_idx = _note_adx(adx_avg)
    _, _, br_idx  = _note_pct(breadth, "breadth")
    if adx_idx >= 3 and br_idx <= 1:
        trend_breadth = "strong trend but narrow participation (fragile leadership)"
    elif adx_idx >= 3 and br_idx >= 3:
        trend_breadth = "strong trend with broad participation (healthier)"
    elif adx_idx <= 1 and br_idx >= 3:
        trend_breadth = "broad participation but weak trend (likely choppy ups)"
    else:
        trend_breadth = "trend strength and participation balanced"

    _, _, mom_idx = _note_pct(mom_breadth, "mom")
    if br_idx >= 3 and mom_idx <= 1:
        mom_cross = "breadth good but momentum weak → pullback risk / rotation"
    elif br_idx <= 1 and mom_idx >= 3:
        mom_cross = "momentum pockets exist despite narrow breadth → selective longs"
    else:
        mom_cross = "breadth and momentum breadth consistent"

    return f"{skew_note}; {trend_breadth}; {mom_cross}"

def _universe_guide(avg20, avg60, breadth, ups, downs, mom_breadth, adx_avg):
    bias = "up" if ups > downs else ("down" if downs > ups else "mixed")
    _, _, i20 = _note_rv(avg20); _, _, i60 = _note_rv(avg60)
    level = max(i20, i60)
    level_str = ["very low","low","moderate","elevated","high"][level]
    cross = _cross_effects_text(avg20, avg60, breadth, mom_breadth, adx_avg)
    if bias == "up" and breadth >= 60 and mom_breadth >= 55:
        action = "Prefer LONGs on pullbacks"
    elif bias == "down" and breadth <= 40 and mom_breadth <= 45:
        action = "Prefer SHORTs on bounces"
    else:
        action = "Trade both sides; keep size moderate"
    return f"{action} · Mode: {level_str} · {cross}"

# ========= APP LAYOUT =========
tab_token, tab_universe = st.tabs(["Token", "Universe Index"])

# ===== Helpers for Plan on multiple TFs =====
def _atr_like(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l),(h - c.shift()).abs(),(l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _favor_long_from_signal(sig_tf: dict, sig_higher: dict | None = None) -> bool:
    # Generic bias: within-TF alignment; if higher TF strongly opposite → flip to conservative False.
    up = (sig_tf["price_vs_ema200"] == "above")
    align = (sig_tf["ema20_cross_50"] == "bull")
    mom = (sig_tf["macd_cross"] == "bull")
    base_yes = (up and (align or mom))
    if not sig_higher:
        return base_yes
    # If higher TF is strong opposite (below EMA200 & MACD bear & ADX strong), veto longs.
    strong_down_higher = (
        (sig_higher["price_vs_ema200"] == "below")
        and (sig_higher["macd_cross"] == "bear")
        and (sig_higher.get("adx14", 0) >= 28)
    )
    if base_yes and strong_down_higher:
        return False
    return base_yes

def plan_levels(df: pd.DataFrame, favor_long: bool):
    s = df["close"]
    ema20v, ema50v, ema200v = float(ema(s,20).iloc[-1]), float(ema(s,50).iloc[-1]), float(ema(s,200).iloc[-1])
    swing_hi = float(df["high"].tail(40).max()); swing_lo = float(df["low"].tail(40).min())
    atr = float(_atr_like(df, 14).iloc[-1] or 0)
    entry = (ema20v + ema50v) / 2.0; buf = max(0.0015 * entry, 0.35 * atr)
    if favor_long and s.iloc[-1] < ema200v: favor_long = False
    if favor_long:
        stop = min(swing_lo, entry) - buf; risk = max(1e-9, entry - stop)
        t1 = entry + risk; t2 = max(entry + 2*risk, swing_hi)
        note = "Plan A (long): buy dip to EMA20/EMA50; SL under swing-low+buffer; T1=1R, T2=2R/prev high."
        alt  = "Plan B: if loses EMA200 and fails to reclaim, flip to shorts on weak bounce."
        side = "LONG"
    else:
        stop = max(swing_hi, entry) + buf; risk = max(1e-9, stop - entry)
        t1 = entry - risk; t2 = min(entry - 2*risk, swing_lo)
        note = "Plan A (short): sell bounce to EMA20/EMA50; SL above swing-high+buffer; T1=1R, T2=2R/prev low."
        alt  = "Plan B: if reclaims EMA200 and holds, prefer longs on pullbacks."
        side = "SHORT"
    return {"entry":entry, "stop":stop, "t1":t1, "t2":t2, "note":note, "alt":alt, "side": side}

# ================= TOKEN TAB =================
with tab_token:
    vitals_tf = st.selectbox("Vitals / Chart TF", options=TFS, index=TFS.index("1h"))

    # DFs
    df_1h,  src_1h  = tf_data["1h"]
    df_4h,  src_4h  = tf_data["4h"]
    df_vtf, src_vtf = tf_data[vitals_tf]

    # Header captions (TF-aware)
    st.caption(f"Prices source: **{src_vtf}** · Worker: {'ON' if cf_worker else 'OFF'}")
    s_vtf = df_vtf.set_index("ts")["close"]
    rv20_vtf = realized_vol_series(s_vtf, 20).iloc[-1] * 100
    rv60_vtf = realized_vol_series(s_vtf, 60).iloc[-1] * 100
    st.caption(f"Using last CLOSED bar for **{vitals_tf}** (no repaint). • {rv_one_liner(rv20_vtf, rv60_vtf)}")

    # Signals all frames
    signals = {tf: trend_flags(tf_data[tf][0]) for tf in TFS}
    s1h, s4h, s1d = signals["1h"], signals["4h"], signals["1d"]

    # Aggregate regime
    def aggregate_regime(signals: dict):
        ups = downs = sides = 0; strong = 0
        for _, s in signals.items():
            _, conf, lab = regime_score(s)
            if conf in ("medium","high"): strong += 1
            if lab == "Trend ↑": ups += 1
            elif lab == "Trend ↓": downs += 1
            else: sides += 1
        label = "Trend ↑" if ups >= max(downs,sides) and ups >= 2 else ("Trend ↓" if downs >= max(ups,sides) and downs >= 2 else "Sideways")
        conf = "high" if strong >= 3 else ("medium" if strong == 2 else "low")
        return {"label": label, "conf": conf, "ups": ups, "downs": downs, "sides": sides}
    agg = aggregate_regime(signals)

    # ===== PLAYBOOK (condensed, human) =====
    favor_long_1h = _favor_long_from_signal(s1h, s4h)
    favor_long_4h = _favor_long_from_signal(s4h, s1d)

    plan_1h = plan_levels(df_1h.copy(), favor_long_1h)
    plan_4h = plan_levels(df_4h.copy(), favor_long_4h)

    st.subheader("Playbook")
    cA, cB = st.columns([1,1])
    with cA:
        st.markdown(f"**1h bias:** `{plan_1h['side']}`  \n"
                    f"**Plan A:** Entry `{plan_1h['entry']:.2f}` • SL `{plan_1h['stop']:.2f}` • T1 `{plan_1h['t1']:.2f}` • T2 `{plan_1h['t2']:.2f}`  \n"
                    f"<span class='small-note'>Alt: {plan_1h['alt']}</span>", unsafe_allow_html=True)
    with cB:
        st.markdown(f"**4h bias:** `{plan_4h['side']}`  \n"
                    f"**Plan A:** Entry `{plan_4h['entry']:.2f}` • SL `{plan_4h['stop']:.2f}` • T1 `{plan_4h['t1']:.2f}` • T2 `{plan_4h['t2']:.2f}`  \n"
                    f"<span class='small-note'>Alt: {plan_4h['alt']}</span>", unsafe_allow_html=True)

    st.markdown(
        f"<span class='badge'>Now: <b>{agg['label']}</b> / conf {agg['conf']}</span>"
        f"<span class='small-note'>  TF votes — ↑:{agg['ups']} / ↓:{agg['downs']} / ↔:{agg['sides']}</span>",
        unsafe_allow_html=True
    )

    # Quick invalidation hints (plain english)
    invos = []
    if plan_1h["side"]=="LONG": invos.append("1h closes under EMA200 and fails to reclaim → stand down on longs.")
    else: invos.append("1h reclaims and holds above EMA200 → avoid chasing shorts.")
    if plan_4h["side"]=="LONG": invos.append("4h trend breaks (EMA20<EMA50) + MACD bear on 4h → scale down long exposure.")
    else: invos.append("4h regains alignment (EMA20>EMA50) + MACD bull → reduce short bias.")
    st.markdown("- " + "\n- ".join(invos))

    # ===== Chart =====
    with st.expander(f"Chart ({vitals_tf})", expanded=True):
        df = df_vtf.copy()
        x_ts = df["ts"]; close_s = df["close"]
        if not PLOTLY or light_mode_chart:
            st.line_chart(close_s.set_axis(x_ts), use_container_width=True, height=420)
            st.caption("Line chart (fast). Turn OFF 'Lightweight chart' for candlesticks.")
        else:
            ema20s, ema50s, ema200s = ema(close_s,20), ema(close_s,50), ema(close_s,200)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=x_ts, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                         name="OHLC", opacity=0.9))
            fig.add_trace(go.Scatter(x=x_ts, y=ema20s,  name="EMA20",  line=dict(width=1.4)))
            fig.add_trace(go.Scatter(x=x_ts, y=ema50s,  name="EMA50",  line=dict(width=1.4)))
            fig.add_trace(go.Scatter(x=x_ts, y=ema200s, name="EMA200", line=dict(width=1.4)))

            # Plan lines on 1h & 4h depending on selected TF
            if vitals_tf == "1h":
                P = plan_1h
                for y, name, dash in [(P["entry"],"Entry","dot"),(P["stop"],"Stop","dash"),
                                      (P["t1"],"T1","dash"),(P["t2"],"T2","dash")]:
                    fig.add_hline(y=y, line_dash=dash, annotation_text=f"{name}: {y:,.2f}", annotation_position="right")
            elif vitals_tf == "4h":
                P = plan_4h
                for y, name, dash in [(P["entry"],"Entry","dot"),(P["stop"],"Stop","dash"),
                                      (P["t1"],"T1","dash"),(P["t2"],"T2","dash")]:
                    fig.add_hline(y=y, line_dash=dash, annotation_text=f"{name}: {y:,.2f}", annotation_position="right")
            else:
                st.caption("Plan lines drawn on **1h** and **4h** (select respective TF).")

            fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # ===== Signal matrix (kept, but purpose-line added) =====
    def tf_row(tframe):
        df2, _ = tf_data[tframe]
        s2 = signals[tframe]
        score, conf2, lab2 = regime_score(s2)
        rv20 = realized_vol_series(df2.set_index("ts")["close"], 20).iloc[-1] * 100
        rec = per_tf_recommendation(s2)
        return {"tf": tframe, "close": df2["close"].iloc[-1],
                "ema200_dir": "↑" if s2["price_vs_ema200"]=="above" else "↓",
                "ema20>50": "✓" if s2["ema20_cross_50"]=="bull" else "×",
                "macd": "✓" if s2["macd_cross"]=="bull" else "×",
                "rsi": s2["rsi14"], "rsi_ma5": s2["rsi_ma5"], "rsi>ma5": s2["rsi_cross"],
                "adx": s2["adx14"], "rv20%": rv20,
                "score": score, "conf": conf2, "label": lab2, "recommend": rec}
    mat = pd.DataFrame([tf_row(t) for t in TFS]).set_index("tf")

    # One-liner purpose (why look here)
    st.caption("Purpose: quick **bias check per TF** and a **simple action hint** (Buy dips / Sell rips / Range trade).")

    def color_yesno(val, yes="✓", no="×"):
        if val == yes: return "background-color:#d1ffd6"
        if val == no:  return "background-color:#ffd1d1"
        return ""
    def color_updown(val):
        return "background-color:#d1ffd6" if val=="↑" else ("background-color:#ffd1d1" if val=="↓" else "")
    def color_rsi(val):
        try:
            v=float(val)
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

    styled = (
        mat.style
          .applymap(color_updown, subset=["ema200_dir"])
          .applymap(color_yesno, subset=["ema20>50","macd","rsi>ma5"])
          .applymap(color_rsi, subset=["rsi","rsi_ma5"])
          .applymap(color_adx, subset=["adx"])
          .applymap(color_rec, subset=["recommend"])
          .format({"close":"{:.2f}","rsi":"{:.1f}","rsi_ma5":"{:.1f}","adx":"{:.1f}","rv20%":"{:.1f}%"})
    )
    st.dataframe(styled, use_container_width=True, height=260)

    # ===== Funding & News (unchanged UI) =====
    st.subheader("Funding & News")
    c1, c2 = st.columns([1,2])
    with c1:
        df_f, src_f = fetch_funding(symbol, limit=48)
        if df_f is not None and not df_f.empty:
            last = df_f.iloc[-1]
            tilt, lvl, side = funding_tilt(float(last["fundingRate"]))
            st.markdown(f"**Latest funding:** {tilt}  \n*Level:* {lvl} · *Flow:* {side}")
            st.line_chart(df_f.set_index("fundingTime")["fundingRate"], height=140, use_container_width=True)
            st.caption(f"Source: {src_f}")
        else:
            st.info("Funding history unavailable from providers.")
    with c2:
        term = _news_term(symbol)
        items = fetch_news(NEWS_FEEDS, lookback_hours=NEWS_LOOKBACK_HOURS, max_items=8, term=term)
        if items:
            html = ['<div class="scroll-wrap">']
            for it in items:
                ts = it["ts"].strftime('%Y-%m-%d %H:%M UTC')
                html.append(f"""
<div style="padding:8px 0; border-top:1px solid #e5e7eb;">
  <div><a href="{it['link']}" target="_blank">{it['title']}</a></div>
  <div class="small-note">{ts}</div>
</div>""")
            html.append("</div>")
            st.markdown("\n".join(html), unsafe_allow_html=True)
        else:
            st.info("No fresh headlines matched the token in the last 24h.")

# ================= UNIVERSE TAB =================
with tab_universe:
    st.subheader("Universe Index (Top-N Composite)")
    syms_for_universe, bases_for_universe = SYMBOLS[:topn], [_base_from_symbol(s) for s in SYMBOLS[:topn]]
    spec = universe_daily_spec(syms_for_universe, bases_for_universe, weighting_scheme, _today_key())

    comp_tf = st.selectbox("Composite timeframe", options=["15m","1h","4h","1d"], index=1)
    comp_method = st.selectbox("Index combine method", options=["Arithmetic (normalized sum)","Geometric (log-sum)"], index=0)
    method_key = "geom" if comp_method.startswith("Geometric") else "arith"

    try:
        comp_df = composite_series(spec["symbols"], spec["weights"], comp_tf, method=method_key).dropna().sort_values("ts")
        comp_df = comp_df.iloc[-min(len(comp_df), 1000):]
        scomp = comp_df.set_index("ts")["composite"]
        c_ema20, c_ema50, c_ema200 = ema(scomp,20), ema(scomp,50), ema(scomp,200)
        c_rv20 = float(realized_vol_series(scomp,20).iloc[-1] * 100)
        c_rv60 = float(realized_vol_series(scomp,60).iloc[-1] * 100)

        # ===== Universe Playbook (condensed) =====
        snap = None
        try:
            from statistics import mean
        except Exception:
            pass
        from math import isnan

        # Snapshot for guidance
        from numpy import mean as npmean
        # reuse function
        @st.cache_data(ttl=240, show_spinner=False)
        def _universe_snapshot_tf_local(symbols: list[str], tf: str, max_symbols: int = 12):
            syms = symbols[:max_symbols]; rows = []
            for s in syms:
                v = _symbol_vitals_tf(s, tf)
                if v.get("ok"): rows.append((s, v))
            if not rows: return None
            rv20s = [v["rv20"] for _, v in rows]; rv60s = [v["rv60"] for _, v in rows]
            labels = [v["label"] for _, v in rows]
            above = sum(v["above200"] for _, v in rows); mom_bull = sum(v["macd_bull"] for _, v in rows)
            adx_vals = [v["adx14"] for _, v in rows]; rsi_vals = [v["rsi14"] for _, v in rows]
            avg20 = float(np.mean(rv20s)); avg60 = float(np.mean(rv60s))
            ups = sum(1 for x in labels if x == "Trend ↑"); downs = sum(1 for x in labels if x == "Trend ↓"); sides = len(labels)-ups-downs
            breadth = 100.0 * above / max(1, len(rows)); mom_breadth = 100.0 * mom_bull / max(1, len(rows))
            adx_avg = float(np.mean(adx_vals)); rsi_avg = float(np.mean(rsi_vals))
            guide = _universe_guide(avg20, avg60, breadth, ups, downs, mom_breadth, adx_avg)
            return {"n": len(rows), "avg20": avg20, "avg60": avg60,
                    "ups": ups, "downs": downs, "sides": sides, "breadth": breadth,
                    "mom_breadth": mom_breadth, "adx_avg": adx_avg, "rsi_avg": rsi_avg, "guide": guide}

        snap = _universe_snapshot_tf_local(SYMBOLS, tf=comp_tf, max_symbols=topn)
        if snap:
            st.markdown(
                f"**Playbook:** {snap['guide']}  \n"
                f"<span class='small-note'>Breadth: {snap['breadth']:.0f}% · Momentum breadth: {snap['mom_breadth']:.0f}% · ADX(avg): {snap['adx_avg']:.1f}</span>",
                unsafe_allow_html=True
            )

        st.caption(rv_one_liner(c_rv20, c_rv60))

        # Chart
        if PLOTLY and not light_mode_chart:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comp_df["ts"], y=comp_df["composite"], name="Composite", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=c_ema20.index, y=c_ema20.values, name="EMA20", line=dict(width=1.3)))
            fig.add_trace(go.Scatter(x=c_ema50.index, y=c_ema50.values, name="EMA50", line=dict(width=1.3)))
            fig.add_trace(go.Scatter(x=c_ema200.index, y=c_ema200.values, name="EMA200", line=dict(width=1.3)))
            fig.update_layout(height=460, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(comp_df.set_index("ts")["composite"], height=320, use_container_width=True)

        # Purpose line
        st.caption("Purpose: universe = **market weather** for Top-N basket. Use it to size risk, choose trend vs mean-reversion, and pick sides.")

        # Details table (kept)
        if snap:
            now_utc = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
            rows = [
                ("Regime votes", f"↑ {snap['ups']} / ↓ {snap['downs']} / ↔ {snap['sides']}", "bias from up vs down counts"),
                ("Breadth (above EMA200)", f"{snap['breadth']:.0f}%", "participation across basket"),
                ("Momentum breadth (MACD>signal)", f"{snap['mom_breadth']:.0f}%", "how many with bullish momentum"),
                ("Avg ADX14", f"{snap['adx_avg']:.1f}", "trend strength"),
                ("Avg RSI14", f"{snap['rsi_avg']:.1f}", "overbought/oversold tilt"),
                ("Avg RV20 / RV60", f"{snap['avg20']:.1f}% / {snap['avg60']:.1f}%", "volatility level & skew"),
            ]
            html = ["""
<div class="scroll-wrap">
  <table class="tile-table" style="width:100%; border-collapse:collapse;">
    <thead><tr><th style="text-align:left;">Metric</th><th style="text-align:left;">Value</th><th style="text-align:left;">Note</th></tr></thead>
    <tbody>
"""]
            for p, v, n in rows:
                html.append(f"<tr><td style='border-top:1px solid #e5e7eb;'>{p}</td><td style='border-top:1px solid #e5e7eb;'>{v}</td><td style='border-top:1px solid #e5e7eb;'>{n}</td></tr>")
            html.append(f"</tbody></table><div class='small-note' style='margin-top:6px;'>{now_utc}</div></div>")
            st.markdown("\n".join(html), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Composite build failed: {e}")

# ========= report.json (debug) =========
@st.cache_data(ttl=600, show_spinner=False)
def fetch_report(url: str, _seed=None):
    r = requests.get(url, timeout=12); r.raise_for_status()
    payload = json.loads(r.content.decode("utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Bad report.json schema.")
    return payload

with st.expander("report.json — raw & pivot (debug)", expanded=False):
    payload = None
    try:
        payload = fetch_report(report_url, _seed=None)
    except Exception:
        payload = None
        st.caption("Unable to load report.json (debug).")
    if payload:
        if payload.get("stale"): st.warning("report.json is STALE; charts use live klines.")
        rows = payload.get("data", []); dfj = pd.json_normalize(rows)
        if not dfj.empty:
            st.dataframe(dfj, use_container_width=True, height=240)
            ok = dfj[dfj["error"].isna()] if "error" in dfj.columns else dfj
            if not ok.empty:
                pivot = ok.pivot_table(
                    index=["symbol","tf"],
                    values=["last_close","ema20","ema50","ema200","rsi14","macd","macd_signal","macd_hist"],
                    aggfunc="first"
                ).sort_index()
                st.dataframe(pivot, use_container_width=True, height=220)
