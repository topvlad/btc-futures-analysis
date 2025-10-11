# streamlit_app.py
# BTC dashboard: multi-TF regime & narrative, signal matrix (RSI/RSI-MA), vol, funding/OI, and 24h news
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

# ======== CONFIG ========
REPORT_URL_DEFAULT = "https://topvlad.github.io/btc-futures-analysis/report.json"
SYMBOLS = ["BTCUSDT", "BTCUSDC"]
TFS = ["15m", "1h", "4h", "1d"]
TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

SPOT_BASES = [
    "https://api.binance.com","https://api1.binance.com","https://api2.binance.com",
    "https://api3.binance.com","https://api4.binance.com","https://api5.binance.com"
]
FAPI_BASES = [
    "https://fapi.binance.com","https://fapi1.binance.com","https://fapi2.binance.com",
    "https://fapi3.binance.com","https://fapi4.binance.com","https://fapi5.binance.com"
]

# Default your Worker here (editable in UI)
CF_WORKER_DEFAULT = os.getenv("CF_WORKER_URL", "https://binance-proxy.brokerof-net.workers.dev")

NEWS_FEEDS = [
    "https://news.google.com/rss/search?q=Bitcoin&hl=en-US&gl=US&ceid=US:en",
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://cointelegraph.com/rss",
]
NEWS_LOOKBACK_HOURS = 24

st.set_page_config(page_title="BTC — Regime, Signals & Futures Context", layout="wide")

# ======== SIDEBAR ========
st.sidebar.header("Data Sources & Options")
report_url = st.sidebar.text_input("GitHub Pages report.json", value=REPORT_URL_DEFAULT)
cf_worker = st.sidebar.text_input("Cloudflare Worker (optional)", value=CF_WORKER_DEFAULT,
                                  help="If set, requests proxy via ?u=<upstream> on this worker.")
extra_emas = st.sidebar.multiselect("Extra EMAs (on chart)", options=[100, 400, 800], default=[400])
auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)
if auto_refresh:
    st.experimental_set_query_params(_=int(time.time()))
st.sidebar.caption("Spot klines → fast charts (via Worker if set). FR/OI via Binance (Worker) → OKX fallback.")

# ======== HTTP helpers ========
def build_worker_url(worker_base: str, full_upstream_url: str) -> str:
    if not worker_base:
        return full_upstream_url
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
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

# ======== report.json (debug table only) ========
@st.cache_data(ttl=300, show_spinner=False)
def fetch_report(url: str):
    r = requests.get(url, timeout=15); r.raise_for_status()
    payload = json.loads(r.content.decode("utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Bad report.json schema.")
    return payload

# ======== Funding & OI (Binance → OKX fallback) ========
@st.cache_data(ttl=120, show_spinner=False)
def fetch_funding(symbol: str, limit: int = 48):
    try:
        base = f"{FAPI_BASES[0]}/fapi/v1/fundingRate"
        if cf_worker:
            upstream = f"{FAPI_BASES[0]}/fapi/v1/fundingRate?symbol={symbol}&limit={min(1000,limit)}"
            data = http_json(upstream, allow_worker=True)
        else:
            data = http_json(base, params={"symbol": symbol, "limit": min(1000,limit)})
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df = df.dropna().sort_values("fundingTime")
            return df, "binance"
    except Exception:
        pass
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
def fetch_open_interest(symbol: str):
    try:
        base = f"{FAPI_BASES[0]}/fapi/v1/openInterest"
        if cf_worker:
            upstream = f"{FAPI_BASES[0]}/fapi/v1/openInterest?symbol={symbol}"
            j = http_json(upstream, allow_worker=True)
        else:
            j = http_json(base, params={"symbol": symbol})
        if isinstance(j, dict) and "openInterest" in j:
            val = float(j.get("openInterest") or 0.0)
            ts = datetime.now(timezone.utc)
            return pd.DataFrame({"ts":[ts], "oi":[val]}), "binance"
    except Exception:
        pass
    try:
        uly = "BTC-USDT" if symbol.endswith("USDT") else "BTC-USDC"
        j = http_json("https://www.okx.com/api/v5/public/open-interest",
                      params={"instType":"SWAP","uly":uly}, timeout=8)
        arr = j.get("data") or []
        if not arr: return None, "okx_empty"
        oi = float(arr[0].get("oi") or 0.0)
        ts = datetime.now(timezone.utc)
        return pd.DataFrame({"ts":[ts], "oi":[oi]}), "okx"
    except Exception:
        return None, "none"

# ======== Klines (Binance → OKX → Coinbase) ========
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
        return _to_df_binance(data), "binance_spot(worker)"
    for base in SPOT_BASES:
        data = http_json(f"{base}/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": min(1000,limit)}, timeout=8)
        return _to_df_binance(data), f"binance_spot({base})"

def _okx_klines(symbol: str, interval: str, limit: int):
    inst = "BTC-USDT" if symbol.endswith("USDT") else "BTC-USDC"
    bar_map = {"15m":"15m","1h":"1H","4h":"4H","1d":"1D"}
    bar = bar_map.get(interval, "1H")
    j = http_json("https://www.okx.com/api/v5/market/candles",
                  params={"instId": inst, "bar": bar, "limit": min(500,limit)}, timeout=8)
    data = j.get("data") or []
    if not data: raise RuntimeError("OKX empty")
    return _to_df_okx(data), "okx"

def _coinbase_klines(symbol: str, interval: str, limit: int):
    product = "BTC-USD"
    gran = {"15m":900,"1h":3600,"4h":14400,"1d":86400}.get(interval, 3600)
    data = http_json(f"https://api.exchange.coinbase.com/products/{product}/candles",
                     params={"granularity": gran, "limit": min(300,limit)}, timeout=8)
    if not isinstance(data, list) or not data: raise RuntimeError("Coinbase empty")
    return _to_df_coinbase(data), "coinbase"

@st.cache_data(ttl=60, show_spinner=False)
def get_klines_df(symbol: str, interval: str, limit: int = 500):
    errs=[]
    try: df, src = _binance_spot_klines(symbol, interval, limit, cf_worker); return df, src
    except Exception as e: errs.append(f"binance:{e}")
    try: df, src = _okx_klines(symbol, interval, limit); return df, src
    except Exception as e: errs.append(f"okx:{e}")
    try: df, src = _coinbase_klines(symbol, interval, limit); return df, src
    except Exception as e: errs.append(f"coinbase:{e}")
    raise RuntimeError("All providers failed → " + " | ".join(errs))

# ======== Indicators & helpers ========
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi_series(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0); loss = -d.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100/(1+rs))

def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
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

def trend_flags(df):
    s = df["close"]
    ema20 = ema(s,20); ema50 = ema(s,50); ema200 = ema(s,200)
    rsi = rsi_series(s,14)
    rsi_ma = rsi.rolling(5).mean()
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

def realized_vol(s, win=20):
    ret = s.pct_change().dropna(); stdev = ret.rolling(win).std().iloc[-1]
    step = int((s.index[-1] - s.index[-2]).total_seconds()) if len(s) >= 2 else TF_SECONDS["1h"]
    per_day = int(86400/step) if step else 1; per_year = max(1, 365*per_day)
    return float(stdev * math.sqrt(per_year))

def regime_score(sig):
    score = (1 if sig["price_vs_ema200"]=="above" else -1) + (1 if sig["ema20_cross_50"]=="bull" else -1) + (1 if sig["macd_cross"]=="bull" else -1)
    conf = "high" if sig.get("adx14",0) >= 25 else ("medium" if sig.get("adx14",0) >= 20 else "low")
    label = "Trend ↑" if score >= 2 else ("Sideways" if -1 <= score <= 1 else "Trend ↓")
    return score, conf, label

def funding_tilt(last_rate: float):
    if last_rate is None: return "unknown", "n/a", "n/a"
    mag = abs(last_rate)
    level = "neutral" if mag < 0.0001 else ("elevated" if mag < 0.0005 else "extreme")  # 0.01% / 0.05%
    side = "Longs → Shorts" if last_rate > 0 else ("Shorts → Longs" if last_rate < 0 else "Flat")
    bps = last_rate * 10000.0
    return f"{bps:+.2f} bps / 8h", level, side

def kpretty(x):
    try: return f"{x:,.2f}"
    except Exception: return str(x)

# ======== HEADER ========
st.title("BTC — Regime, Signals & Futures Context")

# ======== COLLAPSED INSTRUMENT (Symbol), Chart TF stays visible ========
with st.expander("Instrument & options", expanded=False):
    symbol = st.selectbox("Symbol", SYMBOLS, index=0)
    st.caption("The Signal Matrix and Narrative always use **all TFs**. The selector below affects only the chart.")
c_tf, c_bars = st.columns([1,1])
chart_tf = c_tf.selectbox("Chart timeframe", TFS, index=TFS.index("1h"))
limit = c_bars.slider("Bars (chart calc)", min_value=200, max_value=1000, value=500, step=100)

# ======== report.json (for debug table later) ========
payload = None
if report_url:
    try: payload = fetch_report(report_url)
    except Exception as e: st.warning(f"report.json load failed: {e}")

# ======== CORE: fetch ALL TFs for matrix + narrative ========
@st.cache_data(ttl=60, show_spinner=False)
def get_all_tf_data(symbol: str, tfs: list[str], limit_each: int = 500):
    out = {}
    for tf in tfs:
        df, src = get_klines_df(symbol, tf, limit=limit_each)
        df = drop_unclosed(df, tf)
        out[tf] = (df, src)
    return out

try:
    tf_data = get_all_tf_data(symbol, TFS, limit_each=500)
    df_chart, src_chart = tf_data[chart_tf]
    last_ts = df_chart["ts"].iloc[-1]; last_close = float(df_chart["close"].iloc[-1])
    st.caption(f"Prices: **{src_chart}**  ·  Worker: {'ON' if cf_worker else 'OFF'}")
except Exception as e:
    st.error(f"Klines failed: {e}"); st.stop()

# ======== SIGNALS (per TF) & aggregate ========
signals = {tf: trend_flags(tf_data[tf][0]) for tf in TFS}

def aggregate_regime(signals: dict):
    score_sum, strong = 0, 0
    ups = downs = sides = 0
    for tf, s in signals.items():
        sc, conf, lab = regime_score(s)
        score_sum += sc
        if conf in ("medium","high"): strong += 1
        if lab == "Trend ↑": ups += 1
        elif lab == "Trend ↓": downs += 1
        else: sides += 1
    label = "Trend ↑" if ups >= max(downs, sides) and ups >= 2 else ("Trend ↓" if downs >= max(ups, sides) and downs >= 2 else "Sideways")
    conf = "high" if strong >= 3 else ("medium" if strong == 2 else "low")
    return {"label": label, "conf": conf, "ups": ups, "downs": downs, "sides": sides, "score_sum": score_sum}

agg = aggregate_regime(signals)

# vol on chart TF
rv20 = realized_vol(df_chart.set_index("ts")["close"], 20)
rv60 = realized_vol(df_chart.set_index("ts")["close"], 60)

cA, cB, cC, cD = st.columns([1.2, 1.2, 1.2, 1.2])
cA.metric("Regime (aggregate)", f"{agg['label']}", f"conf: {agg['conf']}")
cB.metric("Last Close", kpretty(last_close), last_ts.strftime("%Y-%m-%d %H:%M UTC"))
cC.metric(f"Realized Vol (20, {chart_tf})", f"{rv20*100:0.1f}%")
cD.metric(f"Realized Vol (60, {chart_tf})", f"{rv60*100:0.1f}%")

# ======== FUNDING & OI (for narrative) ========
fdf, fsrc = fetch_funding(symbol, limit=48)
odf, osrc = fetch_open_interest(symbol)
last_funding = float(fdf["fundingRate"].iloc[-1]) if isinstance(fdf, pd.DataFrame) and not fdf.empty else None
funding_str, funding_level, funding_side = funding_tilt(last_funding)

# ======== NEWS (last 24h) ========
@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(feeds: list[str], lookback_hours: int = 24, max_items: int = 8):
    out = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
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
                if not re.search(r"\b(btc|bitcoin)\b", title, flags=re.I):  # bitcoin-only filter
                    continue
                try:
                    dt = parsedate_to_datetime(pub)
                    if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
                    dt = dt.astimezone(timezone.utc)
                except Exception:
                    dt = datetime.now(timezone.utc)
                if dt >= cutoff:
                    out.append({"title": title, "link": link, "ts": dt, "src": u})
        except Exception:
            continue
    seen, dedup = set(), []
    for it in sorted(out, key=lambda x: x["ts"], reverse=True):
        if it["link"] in seen: continue
        seen.add(it["link"]); dedup.append(it)
    return dedup[:max_items]

news_items = fetch_news(NEWS_FEEDS, NEWS_LOOKBACK_HOURS, max_items=8)
news_risk = "normal"
if news_items:
    bad_kw = r"(hack|exploit|liquidat|downtime|halt|delay|lawsuit|ban|shutdown|outage|security|ETF\s+delay|SEC\s+delay|CPI|rate\s+decision|halted)"
    if any(re.search(bad_kw, n["title"], flags=re.I) for n in news_items[:5]):
        news_risk = "elevated"

# ======== NARRATIVE (plain language, multi-TF + FR + news) ========
def narrative_playbook(agg, signals, funding_level, funding_side, news_risk):
    # easy helpers
    def tf_rsi(tf): return signals[tf]["rsi14"]
    def tf_above200(tf): return signals[tf]["price_vs_ema200"] == "above"

    long_notes, short_notes, manage_notes = [], [], []

    # Long ideas
    if agg["label"] == "Trend ↑" and agg["conf"] in ("medium","high"):
        long_notes.append("Look to **buy a pullback** near EMA20/EMA50 on 15m–1h after a candle closes back above the line.")
        long_notes.append("Place a **stop** a little **below the last swing low**. First targets are recent highs.")
    else:
        long_notes.append("Counter-trend long only if **1h closes back above EMA20** and forms a **higher low**. Use **smaller size**.")

    # Short ideas
    if agg["label"] == "Trend ↓" and agg["conf"] in ("medium","high"):
        short_notes.append("Look to **sell a bounce** into EMA20/EMA50 on 15m–1h after a rejection candle.")
        short_notes.append("Place a **stop** a little **above the last swing high**. First targets are recent lows.")
    else:
        short_notes.append("Counter-trend short only if **1h closes back below EMA20** and makes a **lower high**. Use **smaller size**.")

    # RSI stretch guardrails (1h + 4h)
    if tf_rsi("1h") >= 70 or tf_rsi("4h") >= 70:
        long_notes.append("RSI hot (≥70): avoid buying breakouts; wait for a dip/consolidation.")
    if tf_rsi("1h") <= 30 or tf_rsi("4h") <= 30:
        short_notes.append("RSI cold (≤30): avoid shorting breakdowns; wait for a bounce/consolidation.")

    # Funding nudge
    if funding_level in ("elevated","extreme"):
        if funding_side.startswith("Longs"):
            long_notes.append("Funding **positive**: don’t chase green candles; prefer entries **on dips**.")
        elif funding_side.startswith("Shorts"):
            short_notes.append("Funding **negative**: don’t chase red candles; prefer entries **on pops**.")

    # News nudge
    if news_risk == "elevated":
        long_notes.append("News risk high: use **smaller size** or wait 30–60 minutes after headlines.")
        short_notes.append("News risk high: use **smaller size** or wait 30–60 minutes after headlines.")

    # Management if already in
    if agg["label"] == "Trend ↑":
        manage_notes.append("**If long:** move your stop just **below EMA50 (1h)** or the latest higher-low; take some profit near prior highs.")
        manage_notes.append("**If short:** avoid adding unless price **closes below EMA200 (1d)**; consider taking risk off on strength.")
    elif agg["label"] == "Trend ↓":
        manage_notes.append("**If short:** move your stop just **above EMA50 (1h)** or the latest lower-high; take some profit near prior lows.")
        manage_notes.append("**If long:** avoid adding unless price **reclaims EMA200 (1d)**; consider taking risk off on weakness.")
    else:
        manage_notes.append("**Sideways:** keep positions smaller; add/trim at range edges; avoid fresh breakouts without follow-through.")

    return long_notes, short_notes, manage_notes

long_bul, short_bul, mgmt_bul = narrative_playbook(agg, signals, funding_level, funding_side, news_risk)

with st.container():
    st.subheader("Narrative & decision (multi-TF synthesis)")
    st.markdown(
        f"""
**Aggregate Regime:** **{agg['label']}** (confidence: *{agg['conf']}*) · TF votes — ↑:{agg['ups']} / ↓:{agg['downs']} / ↔:{agg['sides']}  
**Funding tilt:** **{funding_str}** — *{funding_level}*, **{funding_side}** · **News risk:** *{news_risk}*
"""
    )
    cL, cS, cM = st.columns(3)
    cL.markdown("**If considering LONG:**\n\n" + ("\n".join([f"- {x}" for x in long_bul]) if long_bul else "- n/a"))
    cS.markdown("**If considering SHORT:**\n\n" + ("\n".join([f"- {x}" for x in short_bul]) if short_bul else "- n/a"))
    cM.markdown("**If already IN (management):**\n\n" + ("\n".join([f"- {x}" for x in mgmt_bul]) if mgmt_bul else "- n/a"))

# ======== SIGNAL MATRIX (multi-TF) with richer columns + color coding ========
def tf_row(tframe):
    df2, _src2 = tf_data[tframe]
    s2 = signals[tframe]
    score, conf2, lab2 = regime_score(s2)
    return {
        "tf": tframe,
        "close": df2["close"].iloc[-1],
        "ema200_dir": "↑" if s2["price_vs_ema200"]=="above" else "↓",
        "ema20>50": "✓" if s2["ema20_cross_50"]=="bull" else "×",
        "macd": "✓" if s2["macd_cross"]=="bull" else "×",
        "rsi": s2["rsi14"],
        "rsi_ma5": s2["rsi_ma5"],
        "rsi>ma5": s2["rsi_cross"],
        "adx": s2["adx14"],
        "score": score,
        "conf": conf2,
        "label": lab2
    }

matrix_rows = [tf_row(t) for t in TFS]
mat = pd.DataFrame(matrix_rows).set_index("tf")

# ---- color helpers ----
def color_yesno(val, yes="✓", no="×"):
    if val == yes: return "background-color:#d1ffd6"      # green
    if val == no:  return "background-color:#ffd1d1"      # red
    return ""

def color_updown(val):
    if val == "↑": return "background-color:#d1ffd6"
    if val == "↓": return "background-color:#ffd1d1"
    return ""

def color_rsi(val):
    try:
        v = float(val)
        if v >= 70:  return "background-color:#ffd1d1"    # overbought -> red (risk for longs)
        if v <= 30:  return "background-color:#d1ffd6"    # oversold   -> green (risk for shorts)
        return ""
    except Exception:
        return ""

def color_adx(val):
    try:
        v = float(val)
        if v >= 25: return "background-color:#d1ffd6"      # strong trend
        if v >= 20: return "background-color:#fff2cc"      # building
        return "background-color:#ffd1d1"                  # weak
    except Exception:
        return ""

st.subheader("Signal matrix (multi-TF)")
styled = (
    mat.style
      .applymap(color_updown, subset=["ema200_dir"])
      .applymap(color_yesno, subset=["ema20>50","macd","rsi>ma5"])
      .applymap(color_rsi, subset=["rsi","rsi_ma5"])
      .applymap(color_adx, subset=["adx"])
      .format({"close":"{:.2f}","rsi":"{:.1f}","rsi_ma5":"{:.1f}","adx":"{:.1f}"})
)
st.dataframe(styled, use_container_width=True)

with st.expander("Legend", expanded=False):
    st.markdown(
        """
- **ema200_dir (↑/↓)** — price above/below EMA200 (big-picture bias).  
- **ema20>50 (✓/×)** — short-term trend aligned with mid-term.  
- **macd (✓/×)** — momentum bias (MACD line above/below signal).  
- **rsi / rsi_ma5** — RSI(14) and its 5-bar moving average; **rsi>ma5** ✓ when RSI is rising vs its MA.  
  - RSI **≥70** tinted red (hot), **≤30** tinted green (cold).  
- **ADX** — trend strength (**<20** weak/red, **20–25** building/yellow, **>25** trending/green).  
- **label/score/conf** — quick aggregation per TF.
        """.strip()
    )

# ======== PRICE CHART (only uses chart_tf) ========
with st.expander("Chart", expanded=True):
    s = df_chart.set_index("ts")["close"]
    ema20, ema50, ema200 = ema(s,20), ema(s,50), ema(s,200)
    if not PLOTLY:
        st.info("Plotly not installed — simple line fallback.")
        st.line_chart(s, use_container_width=True)
    else:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_chart["ts"], open=df_chart["open"], high=df_chart["high"],
                                     low=df_chart["low"], close=df_chart["close"], name="OHLC", opacity=0.8))
        fig.add_trace(go.Scatter(x=df_chart["ts"], y=ema20,  name="EMA20",  line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=df_chart["ts"], y=ema50,  name="EMA50",  line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=df_chart["ts"], y=ema200, name="EMA200", line=dict(width=1.5)))
        for n in sorted(set(int(x) for x in extra_emas if isinstance(x,(int,float)))):
            try:
                fig.add_trace(go.Scatter(x=df_chart["ts"], y=ema(s, n), name=f"EMA{n}", line=dict(width=1)))
            except Exception:
                pass
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=30,b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# ======== FUNDING & OI PANELS ========
with st.expander("Funding & Open Interest (Futures context)"):
    fcol, ocol = st.columns(2)
    try:
        if isinstance(fdf, pd.DataFrame) and not fdf.empty:
            last_f = fdf["fundingRate"].iloc[-1]*100
            fcol.metric(f"Last funding ({fsrc})", f"{last_f:.4f}% / 8h")
            fcol.line_chart(fdf.set_index("fundingTime")["fundingRate"]*100, height=160)
        else:
            fcol.info("No funding data.")
    except Exception as e:
        fcol.warning(f"Funding error: {e}")

    try:
        if isinstance(odf, pd.DataFrame) and not odf.empty:
            ocol.metric(f"Open interest ({osrc})", f"{float(odf['oi'].iloc[-1]):,.0f}")
        else:
            ocol.info("No OI data.")
    except Exception as e:
        ocol.warning(f"OI error: {e}")

# ======== NEWS (last 24h) ========
with st.expander("News (last 24h)", expanded=True if news_items else False):
    if news_items:
        for n in news_items:
            ts = n["ts"].strftime("%Y-%m-%d %H:%M UTC")
            st.markdown(f"- [{n['title']}]({n['link']})  —  *{ts}*")
    else:
        st.info("No fresh Bitcoin headlines found in the last 24h.")

# ======== report.json table (debug) ========
payload = payload or {}
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
        else:
            st.info("Empty data in report.json.")
