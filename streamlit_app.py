# streamlit_app.py — v1.9.2
# BTC dashboard: matrix-first layout, per-TF RV20, adaptive confidence, auto Plan A/B with chart overlays
# NEW in 1.9.2:
# - Real auto-refresh every 60s (optional) + cache "seed" so loaders rerun once/minute
# - Explicit "last CLOSED bar" badge (no repaint)
# - Small help captions/clarifications

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
SYMBOLS = ["BTCUSDT", "BTCUSDC"]
TFS = ["15m", "1h", "4h", "1d"]
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

SPOT_BASES = ["https://api.binance.com","https://api1.binance.com","https://api2.binance.com","https://api3.binance.com","https://api4.binance.com","https://api5.binance.com"]
FAPI_BASES = ["https://fapi.binance.com","https://fapi1.binance.com","https://fapi2.binance.com","https://fapi3.binance.com","https://fapi4.binance.com","https://fapi5.binance.com"]

CF_WORKER_DEFAULT = os.getenv("CF_WORKER_URL", "https://binance-proxy.brokerof-net.workers.dev")

NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=Bitcoin&hl=en-US&gl=US&ceid=US:en",
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://cointelegraph.com/rss",
]
NEWS_LOOKBACK_HOURS = 24

st.set_page_config(page_title="BTC — Regime, Signals & Futures Context", layout="wide")

# ========= SIDEBAR =========
st.sidebar.header("Data & Options")
symbol = st.sidebar.selectbox("Symbol", SYMBOLS, index=0, help="Perp/spot ticker used for prices & context.")
report_url = st.sidebar.text_input("GitHub Pages report.json (debug)", value=REPORT_URL_DEFAULT, help="Optional. Shows raw/pivot from your GH Pages report for verification.")
cf_worker = st.sidebar.text_input("Cloudflare Worker (optional)", value=CF_WORKER_DEFAULT, help="If set, API calls to Binance can be proxied via this URL.")

overlay_plan = st.sidebar.checkbox("Draw Plan A/B on chart (1h)", value=True)
extra_emas = st.sidebar.multiselect("Extra EMAs (on chart)", options=[100, 400, 800], default=[400])
auto_refresh = st.sidebar.checkbox(
    "Auto-refresh UI every 60s (uses last CLOSED bar; no repaint)",
    value=False
)

# Force client-side reload every 60s + create a seed that invalidates caches once/minute
refresh_seed = None
if auto_refresh:
    refresh_seed = int(time.time() // 60)  # changes once/minute
    st.markdown("<script>setTimeout(function(){location.reload();}, 60000);</script>", unsafe_allow_html=True)

st.sidebar.caption("Prices: Binance→OKX→Coinbase. FR/OI: Binance→OKX. Worker, if set, proxies Binance.")

# ========= HTTP helpers =========
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
            r = requests.get(full, params=None if label=="WORKER" else p, timeout=timeout)
            if r.status_code != 200: last_e = f"status {r.status_code}"; continue
            ct = (r.headers.get("content-type") or "").lower()
            if "application/json" not in ct and not ("/klines" in full or "/candles" in full):
                last_e = "non_json"; continue
            return r.json()
        except Exception as e:
            last_e = str(e); continue
    raise RuntimeError(f"http_json failed: {last_e}")

def http_text(url: str, timeout=8):
    r = requests.get(url, timeout=timeout); r.raise_for_status()
    return r.text

# ========= report.json (debug) =========
@st.cache_data(ttl=300, show_spinner=False)
def fetch_report(url: str, _seed=None):
    r = requests.get(url, timeout=15); r.raise_for_status()
    payload = json.loads(r.content.decode("utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Bad report.json schema.")
    return payload

# ========= Funding & OI =========
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
    # OKX fallback
    try:
        inst = "BTC-USDT-SWAP" if symbol.endswith("USDT") else "BTC-USDC-SWAP"
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
        uly = "BTC-USDT" if symbol.endswith("USDT") else "BTC-USDC"
        j = http_json("https://www.okx.com/api/v5/public/open-interest", params={"instType":"SWAP","uly":uly}, timeout=8)
        arr = j.get("data") or []
        if not arr: return None, "okx_empty"
        oi = float(arr[0].get("oi") or 0.0)
        ts = datetime.now(timezone.utc)
        return pd.DataFrame({"ts":[ts], "oi":[oi]}), "okx"
    except Exception:
        return None, "none"

# ========= Spot klines (providers) =========
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
        upstream = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={min(1000,limit)}"
        data = http_json(upstream, allow_worker=True)
        return _to_df_binance(data), "binance(worker)"
    for base in SPOT_BASES:
        data = http_json(f"{base}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": min(1000,limit)}, timeout=8)
        return _to_df_binance(data), f"binance({base})"

def _okx_klines(symbol: str, interval: str, limit: int):
    inst = "BTC-USDT" if symbol.endswith("USDT") else "BTC-USDC"
    bar_map = {"15m":"15m","1h":"1H","4h":"4H","1d":"1D"}
    bar = bar_map.get(interval, "1H")
    j = http_json("https://www.okx.com/api/v5/market/candles", params={"instId": inst, "bar": bar, "limit": min(500,limit)}, timeout=8)
    data = j.get("data") or []
    if not data: raise RuntimeError("OKX empty")
    return _to_df_okx(data), "okx"

def _coinbase_klines(symbol: str, interval: str, limit: int):
    product = "BTC-USD"
    gran = {"15m":900,"1h":3600,"4h":14400,"1d":86400}.get(interval, 3600)
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
    ret = s.pct_change().dropna()
    stdev = ret.rolling(win).std()
    if len(s) >= 2:
        step = int((s.index[-1] - s.index[-2]).total_seconds())
    else:
        step = TF_SECONDS["1h"]
    per_day = int(86400/max(step,1))
    per_year = max(1, 365*per_day)
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

# ========= CORE LOAD =========
@st.cache_data(ttl=60, show_spinner=False)
def get_all_tf_data(symbol: str, tfs: list[str], limit_each: int = 1000, _seed=None):
    out = {}
    for tf in tfs:
        df, src = get_klines_df(symbol, tf, limit=limit_each, _seed=_seed)
        df = drop_unclosed(df, tf)
        out[tf] = (df, src)
    return out

try:
    tf_data = get_all_tf_data(symbol, TFS, limit_each=1000, _seed=refresh_seed)
except Exception as e:
    st.error(f"Klines failed: {e}"); st.stop()

# Use 1h for chart & plan
df_1h, src_chart = tf_data["1h"]
last_closed_bar_time = df_1h["ts"].iloc[-1]  # guaranteed closed
last_close = float(df_1h["close"].iloc[-1])
st.caption(f"Prices: **{src_chart}**  ·  Worker: {'ON' if cf_worker else 'OFF'}")

# ========= SIGNALS & aggregate =========
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

# Top metrics (1h)
s_1h = df_1h.set_index("ts")["close"]
rv20_1h = realized_vol_series(s_1h, 20).iloc[-1] * 100
rv60_1h = realized_vol_series(s_1h, 60).iloc[-1] * 100
cA, cB, cC, cD = st.columns([1.2,1.2,1.2,1.2])
cA.metric("Regime (aggregate)", f"{agg['label']}", f"conf: {agg['conf']}")
cB.metric("Last Close (1h)", kpretty(last_close), last_closed_bar_time.strftime("%Y-%m-%d %H:%M UTC"))
cC.metric("RV20% (1h, annualized)", f"{rv20_1h:0.1f}%")
cD.metric("RV60% (1h, annualized)", f"{rv60_1h:0.1f}%")
st.caption("Using **last CLOSED** 1h bar (no repaint). The current 1h bar is excluded until it closes.")

# Funding / OI
fdf, fsrc = fetch_funding(symbol, limit=48, _seed=refresh_seed)
odf, osrc = fetch_open_interest(symbol, _seed=refresh_seed)
last_funding = float(fdf["fundingRate"].iloc[-1]) if isinstance(fdf, pd.DataFrame) and not fdf.empty else None
funding_str, funding_level, funding_side = funding_tilt(last_funding)
fund_mean_24h = float(fdf["fundingRate"].tail(3).mean()) if isinstance(fdf, pd.DataFrame) and len(fdf) >= 3 else None

# News (24h)
@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(feeds: list[str], lookback_hours: int = 24, max_items: int = 8, _seed=None):
    out = []; cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
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
                if not re.search(r"\b(btc|bitcoin)\b", title, flags=re.I): continue
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

news_items = fetch_news(NEWS_FEEDS, NEWS_LOOKBACK_HOURS, 8, _seed=refresh_seed)
news_risk = "elevated" if news_items and any(re.search(r"(hack|exploit|liquidat|halt|delay|lawsuit|ban|shutdown|outage|security|CPI|rate)", n["title"], flags=re.I) for n in news_items[:5]) else "normal"

# Narrative
def narrative_story(agg, signals, funding_level, funding_side, fund_mean_24h, news_risk):
    s1h, s4h, sd = signals["1h"], signals["4h"], signals["1d"]
    lines = []
    if agg["label"] == "Trend ↑": lines.append("**Tone:** buyers have the ball; most frames lean up.")
    elif agg["label"] == "Trend ↓": lines.append("**Tone:** sellers in control; bounces meet supply.")
    else: lines.append("**Tone:** balanced — chop until a clean break sticks.")
    lines.append("**Daily** " + ("above" if sd["price_vs_ema200"]=="above" else "below") + " long trend (EMA200).")
    lines.append("**4h** " + ("aligned (EMA20>EMA50)" if signals["4h"]["ema20_cross_50"]=="bull" else "not aligned (EMA20<EMA50)"))
    lines.append("**1h momentum** " + ("up (MACD>signal)" if s1h["macd_cross"]=="bull" else "down (MACD<signal)"))
    if s1h["rsi14"] >= 70 or s4h["rsi14"] >= 70: lines.append("RSI is **hot** on intraday → prefer entries after a pause.")
    if s1h["rsi14"] <= 30 or s4h["rsi14"] <= 30: lines.append("RSI is **cold** → avoid chasing breakdowns; wait for a bounce.")
    if funding_level in ("elevated","extreme"):
        lines.append(f"Funding **{funding_level}**: {funding_side.lower()} — better to enter on pullbacks/pops, not chases.")
    if fund_mean_24h is not None:
        tilt = "longs pay" if fund_mean_24h>0 else ("shorts pay" if fund_mean_24h<0 else "flat")
        lines.append(f"24h mean funding: **{fund_mean_24h*100:.3f}%/8h** ({tilt}).")
    if news_risk == "elevated":
        lines.append("**Headline risk up** — keep size smaller or wait 30–60m after news.")
    return lines

tale_lines = narrative_story(agg, signals, funding_level, funding_side, fund_mean_24h, news_risk)

# Plans (1h)
def swing_levels(df, lookback=30):
    highs = df["high"].tail(lookback); lows = df["low"].tail(lookback)
    return float(highs.max()), float(lows.min())

def plan_levels(df_1h, regime_up: bool):
    s = df_1h["close"]
    ema20v, ema50v = float(ema(s,20).iloc[-1]), float(ema(s,50).iloc[-1])
    swing_hi, swing_lo = swing_levels(df_1h, 30)
    if regime_up:
        entry = (ema20v + ema50v)/2.0
        stop  = min(swing_lo, entry * 0.993)
        risk  = entry - stop
        t1    = entry + risk
        t2    = max(entry + 2*risk, swing_hi)
        note  = "Plan A (long): buy dip into EMA20/EMA50 after 1h closes back above; stop below last higher-low; T1=1R, T2=2R/prev high."
        alt   = "Plan B (flip): if 1h closes below EMA200 and fails to reclaim, look for short on a weak bounce."
    else:
        entry = (ema20v + ema50v)/2.0
        stop  = max(swing_hi, entry * 1.007)
        risk  = stop - entry
        t1    = entry - risk
        t2    = min(entry - 2*risk, swing_lo)
        note  = "Plan A (short): sell bounce into EMA20/EMA50 after rejection; stop above last lower-high; T1=1R, T2=2R/prev low."
        alt   = "Plan B (flip): if 1h reclaims EMA200 and holds, prefer longs on pullbacks."
    return {"entry":entry,"stop":stop,"t1":t1,"t2":t2,"note":note,"alt":alt}

regime_is_up = (agg["label"] == "Trend ↑")
plan = plan_levels(df_1h.copy(), regime_is_up)

st.subheader("Narrative & decision (multi-TF synthesis)")
st.markdown(
    f"**Aggregate Regime:** **{agg['label']}** (confidence: *{agg['conf']}*) · TF votes — ↑:{agg['ups']} / ↓:{agg['downs']} / ↔:{agg['sides']}  \n"
    f"**Funding tilt:** *{funding_str}* — *{funding_level}*, **{funding_side}** · **News risk:** *{news_risk}*"
)
st.markdown("\n".join([f"- {ln}" for ln in tale_lines]))
st.markdown(f"**{plan['note']}**  \n*Plan B:* {plan['alt']}")

# ========= Matrix =========
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

with st.expander("Legend", expanded=False):
    st.markdown(
        """
- **ema200_dir** — above/below EMA200 (big-picture bias).  
- **ema20>50, macd, rsi>ma5** — alignment of short-term trend, momentum, RSI direction (✓ good, × opposite).  
- **rsi / rsi_ma5** — RSI(14) and its 5-bar average (green ≤30, red ≥70).  
- **adx** — trend strength (**high ≥28**, **medium 22–28**, **low <22**).  
- **rv20%** — annualized realized volatility from last 20 bars of that TF.  
- **recommend** — per-TF action: **Buy dips / Sell rips / Range trade**.  
- Using **last CLOSED** bar on each TF to avoid repaint.
        """.strip()
    )

# ========= CHART (1h) =========
with st.expander("Chart (1h)", expanded=True):
    df = df_1h.copy()
    x_ts = df["ts"]
    close_s = df["close"]
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

# ========= Funding & OI =========
with st.expander("Funding & Open Interest (Futures context)"):
    fcol, ocol = st.columns(2)
    try:
        if isinstance(fdf, pd.DataFrame) and not fdf.empty:
            last_f = fdf["fundingRate"].iloc[-1]*100
            fcol.metric(f"Last funding ({fsrc})", f"{last_f:.4f}% / 8h")
            if fund_mean_24h is not None:
                fcol.caption(f"24h mean: {fund_mean_24h*100:.4f}% / 8h  ·  bias: {'longs pay' if fund_mean_24h>0 else ('shorts pay' if fund_mean_24h<0 else 'flat')}")
            fcol.line_chart(fdf.set_index("fundingTime")["fundingRate"]*100, height=160)
        else:
            fcol.info("No funding data.")
    except Exception as e:
        fcol.warning(f"Funding error: {e}")

    try:
        if isinstance(odf, pd.DataFrame) and not odf.empty:
            ocol.metric(f"Open interest ({osrc})", f"{float(odf['oi'].iloc[-1]):,.0f}")
            ocol.caption("Snapshot only (no public OI history in this view).")
        else:
            ocol.info("No OI data.")
    except Exception as e:
        ocol.warning(f"OI error: {e}")

# ========= NEWS =========
with st.expander("News (last 24h)", expanded=False):
    items = fetch_news(NEWS_FEEDS, NEWS_LOOKBACK_HOURS, 8, _seed=refresh_seed)
    if items:
        for n in items:
            ts = n["ts"].strftime("%Y-%m-%d %H:%M UTC")
            st.markdown(f"- [{n['title']}]({n['link']})  —  *{ts}*")
    else:
        st.info("No fresh Bitcoin headlines found in the last 24h.")

# ========= report.json (debug) =========
payload = None
if report_url:
    try: payload = fetch_report(report_url, _seed=refresh_seed)
    except Exception: payload = None

if payload:
    with st.expander("report.json — raw & pivot (debug)", expanded=False):
        if payload.get("stale"): st.warning("report.json is **STALE**; charts use live klines.")
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
