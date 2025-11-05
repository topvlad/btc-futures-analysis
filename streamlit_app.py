# streamlit_app.py — v1.13.5-hotfix1
# Hotfix:
# - Replace segmented_control(selection=...) → segmented_control(default=...)
# Other:
# - Human-readable Playbook; sliding triplet; 1M timeframe; Main/Universe; Universe Index.

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

# --- add monthly timeframe (1M) ---
if "1M" not in TFS:
    TFS.append("1M")
TF_SECONDS.update({"1M": 30*86400})
TF_LIMITS.update({"1M": 240})

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
.tile-table td, .tile-table th { padding: 8px 8px; }
@media (max-width: 640px){
  .tile-table td, .tile-table th { padding: 6px 6px !important; font-size: 13px !important; }
}
.small-note{ color:#6b7280; font-size:12px; }
.badge{display:inline-block; padding:2px 8px; border-radius:9999px; font-size:12px; margin-right:6px; background:#eef2ff;}
hr{margin: 0.6rem 0;}
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
                             headers={"Accept":"application/json","User-Agent":"binfapp/1.13.5-hotfix1"})
            if r.status_code != 200: last_e = f"status {r.status_code}"; continue
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" not in ct and not ("/klines" in full or "/candles" in full):
                last_e = "non_json"; continue
            return r.json()
        except Exception as e:
            last_e = str(e); continue
    raise RuntimeError(f"http_json failed: {last_e}")

def http_text(url: str, timeout=7):
    r = requests.get(url, timeout=timeout, headers={"User-Agent":"binfapp/1.13.5-hotfix1"}); r.raise_for_status()
    return r.text

# ========= UNIVERSE (sources) =========
TOPN_DEFAULT = 10

@st.cache_data(ttl=3600, show_spinner=False)
def _coingecko_topn_with_caps(per_page=100, page=1):
    try:
        j = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency":"usd","order":"market_cap_desc","per_page":per_page,"page":page},
            timeout=7, headers={"Accept":"application/json","User-Agent":"binfapp/1.13.5-hotfix1"}
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
    help="Applies to the composite in the Universe section."
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
    inst = f"{_base_from_symbol(symbol)}-USDT"
    bar_map = {"15m":"15m","1h":"1H","4h":"4H","1d":"1D","1w":"1W","1M":"1M"}
    bar = bar_map.get(interval, "1H")
    j = http_json("https://www.okx.com/api/v5/market/candles",
                  params={"instId": inst, "bar": bar, "limit": min(500,limit)}, timeout=7)
    data = j.get("data") or []
    if not data: raise RuntimeError("OKX empty")
    return _to_df_okx(data), "okx"

def _coinbase_klines(symbol: str, interval: str, limit: int):
    base = _base_from_symbol(symbol); product_map = {"BTC":"BTC-USD","ETH":"ETH-USD"}
    product = product_map.get(base, "BTC-USD")
    gran_map = {"15m":900,"1h":3600,"4h":14400,"1d":86400,"1w":86400,"1M":86400}
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

# ===== Helpers for Universe snapshot / guide =====
def _bucket(value: float, edges: list[float], labels: list[str]):
    if value is None or not np.isfinite(value): return "n/a", 2
    for i, e in enumerate(edges):
        if value < e:
            return labels[i], i
    return labels[-1], 4

def _note_rv(value: float):
    labels = ["low","med-low","average","med-high","high"]
    lab, idx = _bucket(value, [30, 45, 60, 85], labels)
    txt = {0:"very calm; breakouts fade", 1:"calm; mean-reversion favoured",
           2:"balanced", 3:"elevated; momentum holds", 4:"high; size down / wider stops"}[idx]
    return lab, txt, idx

def _note_adx(value: float):
    labels = ["low","med-low","average","med-high","high"]
    lab, idx = _bucket(value, [17, 22, 28, 38], labels)
    txt = {0:"very weak trend", 1:"weak trend", 2:"trend building",
           3:"strong trend", 4:"very strong trend"}[idx]
    return lab, txt, idx

def _note_pct(value: float, kind: str):
    labels = ["low","med-low","average","med-high","high"]
    lab, idx = _bucket(value, [30, 45, 55, 70], labels)
    if kind == "breadth":
        extra = {0:"narrow advance",1:"modestly narrow",2:"balanced",
                 3:"broadening",4:"very broad"}[idx]
    else:
        extra = {0:"few with momentum",1:"limited momentum",2:"mixed",
                 3:"solid momentum",4:"widespread momentum"}[idx]
    return lab, extra, idx

def _cross_effects_text(avg20, avg60, breadth, mom_breadth, adx_avg):
    skew_ratio = avg20 / max(avg60, 1e-9)
    if skew_ratio >= 1.15: skew_note = "short-term vol expanding → momentum bias"
    elif skew_ratio <= 0.85: skew_note = "short-term vol cooling → mean-reversion bias"
    else: skew_note = "vol horizons aligned"
    _, _, adx_idx = _note_adx(adx_avg)
    _, _, br_idx  = _note_pct(breadth, "breadth")
    if adx_idx >= 3 and br_idx <= 1:
        trend_breadth = "strong trend but narrow participation"
    elif adx_idx >= 3 and br_idx >= 3:
        trend_breadth = "strong trend with broad participation"
    elif adx_idx <= 1 and br_idx >= 3:
        trend_breadth = "broad participation but weak trend"
    else:
        trend_breadth = "trend strength and participation balanced"
    _, _, mom_idx = _note_pct(mom_breadth, "mom")
    if br_idx >= 3 and mom_idx <= 1:
        mom_cross = "breadth good but momentum weak → pullback risk / rotation"
    elif br_idx <= 1 and mom_idx >= 3:
        mom_cross = "momentum pockets despite narrow breadth → selective longs"
    else:
        mom_cross = "breadth & momentum breadth consistent"
    return f"{skew_note}; {trend_breadth}; {mom_cross}"

def _universe_guide(avg20, avg60, breadth, ups, downs, mom_breadth, adx_avg):
    bias = "up" if ups > downs else ("down" if downs > ups else "flat")
    rv_lab20, _, _ = _note_rv(avg20)
    rv_lab60, _, _ = _note_rv(avg60)
    adx_lab, _, _  = _note_adx(adx_avg)
    br_lab, _, _   = _note_pct(breadth, "breadth")
    mom_lab, _, _  = _note_pct(mom_breadth, "mom")
    cross = _cross_effects_text(avg20, avg60, breadth, mom_breadth, adx_avg)
    head = f"Market bias: {bias.upper()} • RV(20/60): {rv_lab20}/{rv_lab60} • ADX: {adx_lab} • Breadth: {br_lab} • Momentum breadth: {mom_lab}"
    return {"headline": head, "cross": cross}

# ========= Human Playbook =========
def human_playbook_from_sig(sig: dict) -> dict:
    up = (sig["price_vs_ema200"] == "above")
    ema_ok = (sig["ema20_cross_50"] == "bull")
    macd_ok = (sig["macd_cross"] == "bull")
    adx_val = sig.get("adx14", 0.0)
    adx_ok = adx_val >= 22
    if up and ema_ok and macd_ok and adx_ok:
        state = "Тренд вгору підтверджений"
        plan_a = "Шукати покупки на відкатах до EMA20/EMA50"
        plan_b = "Фіксувати частину прибутку на локальних екстремумах, якщо ADX починає знижуватись"
    elif (not up) and (not ema_ok) and (not macd_ok) and adx_ok:
        state = "Тренд вниз підтверджений"
        plan_a = "Розглядати продажі на підйомах до EMA20/EMA50"
        plan_b = "Фіксувати частину прибутку на падіннях, якщо ADX слабшає"
    elif adx_val < 22:
        state = "Ринку бракує сили тренду (флет/слабкий тренд)"
        plan_a = "Торгувати від рівнів: купувати від підтримок/продавати від опорів"
        plan_b = "Зменшити розмір позиції, обмежити кількість угод"
    else:
        state = "Сигнали змішані"
        if up:
            plan_a = "Легкий пріоритет покупок, чекати підтвердження MACD/EMA"
            plan_b = "Нейтрально: працювати від рівнів із короткими стопами"
        else:
            plan_a = "Легкий пріоритет продажів, чекати підтвердження MACD/EMA"
            plan_b = "Нейтрально: працювати від рівнів із короткими стопами"
    if up:
        risk = "Скасування лонг-плану, якщо ціна закріпиться нижче EMA50 і MACD піде нижче сигналу"
    else:
        risk = "Скасування шорт-плану, якщо ціна закріпиться вище EMA50 і MACD піде вище сигналу"
    return {
        "state": state, "plan_a": plan_a, "plan_b": plan_b, "risk": risk,
        "adx": adx_val,
        "ema20": sig["ema20"], "ema50": sig["ema50"], "ema200": sig["ema200"],
        "rsi14": sig["rsi14"], "macd": sig["macd"], "macd_signal": sig["macd_signal"]
    }

def tf_triplet(selected_tf: str) -> list[str]:
    mapping = {
        "15m": ["15m","1h","4h"],
        "1h":  ["1h","4h","1d"],
        "4h":  ["4h","1d","1w"],
        "1d":  ["1d","1w","1M"],
        "1w":  ["1w","1M","1M"],
        "1M":  ["1M","1M","1M"]
    }
    return mapping.get(selected_tf, ["1h","4h","1d"])

def compute_signals_for_tfs(symbol: str, tfs: list[str]) -> dict:
    out = {}
    for tf in tfs:
        try:
            df, _src = get_klines_df(symbol, tf, TF_LIMITS.get(tf, 500))
            df = drop_unclosed(df, tf)
            if len(df) < 210:
                out[tf] = None
                continue
            sig = trend_flags(df.assign(high=df["high"], low=df["low"], close=df["close"]))
            out[tf] = sig
        except Exception:
            out[tf] = None
    return out

def render_playbook_triplet(symbol: str, tf_selected: str):
    tfs = tf_triplet(tf_selected)
    sigs = compute_signals_for_tfs(symbol, tfs)
    st.subheader("Playbook (ковзне вікно з 3 ТФ)")
    cols = st.columns(len(tfs))
    for i, tf in enumerate(tfs):
        with cols[i]:
            st.markdown(f"**{tf.upper()}**")
            sig = sigs.get(tf)
            if not sig:
                st.caption("Недостатньо даних або джерело не відповідає.")
                continue
            human = human_playbook_from_sig(sig)
            st.markdown(
                f"- **Стан:** {human['state']}\n"
                f"- **План A:** {human['plan_a']}\n"
                f"- **План B:** {human['plan_b']}\n"
                f"- **Ризик / скасування:** {human['risk']}\n"
                f"<span class='small-note'>ADX: {human['adx']:.1f} • EMA20: {human['ema20']:.2f} • EMA50: {human['ema50']:.2f} • EMA200: {human['ema200']:.2f} • RSI14: {human['rsi14']:.1f}</span>",
                unsafe_allow_html=True
            )

# ========= Chart helpers =========
def render_chart(df: pd.DataFrame, tf: str, light: bool):
    if light or (not PLOTLY):
        st.line_chart(df.set_index("ts")["close"])
        return
    fig = go.Figure(data=[go.Candlestick(
        x=df["ts"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price"
    )])
    s = df["close"]
    fig.add_trace(go.Scatter(x=df["ts"], y=ema(s,20), mode="lines", name="EMA20"))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema(s,50), mode="lines", name="EMA50"))
    fig.add_trace(go.Scatter(x=df["ts"], y=ema(s,200), mode="lines", name="EMA200"))
    fig.update_layout(height=520, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig, use_container_width=True)

# ========= Universe Index =========
def _weights_from_scheme(cg_rows, scheme: str, bases: list[str]):
    dfw = pd.DataFrame(cg_rows)
    dfw = dfw[dfw["base"].isin(bases)].copy()
    if scheme.startswith("Market cap"):
        w = dfw["market_cap"].replace(0, np.nan)
    elif scheme.startswith("Liquidity"):
        w = dfw["total_volume"].replace(0, np.nan)
    else:
        w = pd.Series(1.0, index=dfw.index)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1/len(dfw), index=dfw.index)
    return dict(zip(dfw["base"], w.values))

@st.cache_data(ttl=120, show_spinner=False)
def universe_index_series(symbols: list[str], tf: str, scheme: str, cg_rows) -> pd.DataFrame:
    bases = [_base_from_symbol(s) for s in symbols]
    wmap = _weights_from_scheme(cg_rows, scheme, bases)
    closes = []
    for s in symbols:
        try:
            df,_ = get_klines_df(s, tf, TF_LIMITS.get(tf, 500))
            df = drop_unclosed(df, tf)
            if df.empty: continue
            cl = df[["ts","close"]].copy()
            base = _base_from_symbol(s)
            w = wmap.get(base, 0.0)
            cl["wclose"] = (cl["close"] / float(cl["close"].iloc[0])) * w
            closes.append(cl[["ts","wclose"]])
        except Exception:
            continue
    if not closes: return pd.DataFrame()
    allc = closes[0]
    for part in closes[1:]:
        allc = allc.merge(part, on="ts", how="outer")
    allc = allc.sort_values("ts").fillna(method="ffill")
    allc["universe"] = allc.drop(columns=["ts"]).sum(axis=1)
    return allc[["ts","universe"]]

def render_universe_tab(tf_for_index: str):
    st.subheader("Universe Index")
    series = universe_index_series(SYMBOLS[:topn], tf_for_index, weighting_scheme, cg)
    if series.empty:
        st.info("Не вдалося зібрати Universe Index для обраних параметрів.")
        return
    if light_mode_chart:
        st.line_chart(series.set_index("ts")["universe"])
    else:
        if PLOTLY:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["ts"], y=series["universe"], mode="lines", name="Universe Index"))
            fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(series.set_index("ts")["universe"])

    snap_tf = tf_for_index
    ups=downs=0; adxes=[]; mom_bulls=0; total=0
    rv20s=[]; rv60s=[]
    for s in SYMBOLS[:topn]:
        try:
            df,_ = get_klines_df(s, snap_tf, TF_LIMITS.get(snap_tf, 500))
            df = drop_unclosed(df, snap_tf)
            if len(df) < 210: continue
            sig = trend_flags(df.assign(high=df["high"], low=df["low"], close=df["close"]))
            total += 1
            if sig["price_vs_ema200"]=="above": ups+=1
            else: downs+=1
            if sig["macd_cross"]=="bull": mom_bulls+=1
            adxes.append(sig["adx14"])
            sclose = df["close"]
            rv20s.append(float(realized_vol_series(sclose,20).iloc[-1]*100))
            rv60s.append(float(realized_vol_series(sclose,60).iloc[-1]*100))
        except Exception:
            continue
    if total>0:
        breadth = 100.0*ups/total
        mom_breadth = 100.0*mom_bulls/total
        adx_avg = float(np.nanmean(adxes)) if len(adxes)>0 else np.nan
        avg20 = float(np.nanmean(rv20s)) if rv20s else np.nan
        avg60 = float(np.nanmean(rv60s)) if rv60s else np.nan
        guide = _universe_guide(avg20, avg60, breadth, ups, downs, mom_breadth, adx_avg)
        st.markdown(f"**Snapshot ({snap_tf.upper()}):** {guide['headline']}")
        st.caption(guide["cross"])

# ========= Main / Universe switch (with URL query support) =========
def get_query_params():
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def set_query_param(**kwargs):
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)

q = get_query_params()
tab_param = (q.get("tab") if isinstance(q.get("tab"), str) else (q.get("tab",[None])[0])) if q else None
tab_default = "token" if tab_param not in ("token","universe") else tab_param

if hasattr(st, "segmented_control"):
    toggle = st.segmented_control(
        "View",
        options=["Main","Universe"],
        default=("Main" if tab_default=="token" else "Universe"),
        help="Switch view. Also via ?tab=token|universe"
    )
else:
    toggle = st.radio("View", options=["Main","Universe"],
                      index=0 if tab_default=="token" else 1, horizontal=True)

current_tab = "token" if (toggle=="Main") else "universe"
if current_tab != tab_default:
    set_query_param(tab=current_tab)

# ========= Timeframe selector =========
tf_selected = st.selectbox("Timeframe", options=TFS, index=TFS.index("1h") if "1h" in TFS else 0)

# ========= MAIN TAB =========
if current_tab == "token":
    st.header(f"{symbol} — {tf_selected.upper()}")
    try:
        df_raw, src = get_klines_df(symbol, tf_selected, TF_LIMITS.get(tf_selected, 500))
        df = drop_unclosed(df_raw, tf_selected)
    except Exception as e:
        st.error(f"Failed to load klines: {e}")
        st.stop()
    render_chart(df.rename(columns={"open":"open","high":"high","low":"low","close":"close"}), tf_selected, light_mode_chart)
    st.caption(f"Source: {src}. Bars: {len(df)}")
    if tf_selected == "1d":
        with st.expander("1D Playbook — сигнали та планові рівні (EMA)"):
            s = df["close"]
            st.write(f"EMA20: {ema(s,20).iloc[-1]:.2f} • EMA50: {ema(s,50).iloc[-1]:.2f} • EMA200: {ema(s,200).iloc[-1]:.2f}")
            st.caption("Планові рівні — ковзні середні як динамічні підтримки/опори.")
    render_playbook_triplet(symbol, tf_selected)

# ========= UNIVERSE TAB =========
else:
    st.header("Universe")
    tf_universe = st.selectbox("Universe Index timeframe", options=TFS, index=TFS.index("4h") if "4h" in TFS else 0)
    render_universe_tab(tf_universe)
    with st.expander("Universe constituents"):
        st.write(", ".join([_base_from_symbol(s) for s in SYMBOLS[:topn]]))

# ========== END ==========
