# analyze.py (v1.70) — spot-first, closed-candle only, fast run
import os, time, json, io, csv, zipfile, math
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode, quote

# ── Config ─────────────────────────────────────────────────────────────────────
CF_WORKER_BASE = (os.getenv("CF_WORKER_BASE") or "").strip()

FAPI_PRIMARY = "https://fapi.binance.com"
FAPI_BASES = [
    *([CF_WORKER_BASE] if CF_WORKER_BASE else []),
    FAPI_PRIMARY,
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
    "https://fapi5.binance.com",
]

API_PRIMARY = "https://api.binance.com"
SPOT_BASES = [
    *([CF_WORKER_BASE] if CF_WORKER_BASE else []),
    API_PRIMARY,
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://api5.binance.com",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "btc-futures-analysis/1.70 (+github actions)",
    "Accept": "application/json,*/*",
    "Accept-Encoding": "gzip, deflate",
})

CI = os.getenv("GITHUB_ACTIONS") == "true"
TIMEOUT_CONNECT = 4
TIMEOUT_READ    = 6
RETRIES         = 1 if CI else 3
BACKOFF         = 1.35
FETCH_DEADLINE_SEC = 12 if CI else 25  # менші дедлайни → швидше фейлимось і переходимо далі

# Vision (як останній резерв)
VISION_DAILY   = "https://data.binance.vision/data/futures/um/daily/klines"
VISION_MONTHLY = "https://data.binance.vision/data/futures/um/monthly/klines"

SYMBOLS = ["BTCUSDT", "BTCUSDC"]
TFS     = ["15m","1h","4h","1d"]
LIMIT_PER_TF = 200

_TF_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# ── HTTP helpers ───────────────────────────────────────────────────────────────
def _is_json(resp): return "application/json" in (resp.headers.get("content-type") or "").lower()
def _build_worker_url(full: str) -> str: return f"{CF_WORKER_BASE}?u={quote(full, safe='')}"

def _http_get_json_from_bases(path: str, params: dict, bases: list[str], primary_for_worker: str | None):
    started = time.monotonic()
    last = None

    import random
    arr = bases[:]
    if CF_WORKER_BASE and arr and arr[0] == CF_WORKER_BASE:
        tail = arr[1:]; random.shuffle(tail); arr = [CF_WORKER_BASE] + tail
    else:
        random.shuffle(arr)

    for base in arr:
        for i in range(RETRIES + 1):
            if time.monotonic() - started > FETCH_DEADLINE_SEC:
                raise TimeoutError(f"fetch_deadline_exceeded for {path} {params}")
            try:
                if CF_WORKER_BASE and base == CF_WORKER_BASE:
                    if not primary_for_worker:
                        last = "worker_not_supported"; break
                    upstream = f"{primary_for_worker}{path}?{urlencode(params)}"
                    url = _build_worker_url(upstream)
                    r = SESSION.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), allow_redirects=False)
                else:
                    url = f"{base}{path}"
                    r = SESSION.get(url, params=params, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), allow_redirects=False)

                if r.status_code not in (200, 203): last=f"bad_status:{r.status_code}"; time.sleep(BACKOFF**i); continue
                if (not _is_json(r)) and ((r.content or b"")[:1]==b"<" or not r.content): last="non_json_200"; time.sleep(BACKOFF**i); continue
                return r.json()
            except Exception as e:
                last = str(e); time.sleep(BACKOFF**i); continue
    raise RuntimeError(f"_get_failed_after_retries:{last}")

def _http_get_json_spot(path:str, params:dict):
    return _http_get_json_from_bases(path, params, SPOT_BASES, API_PRIMARY)

def _http_get_json_fapi(path:str, params:dict):
    return _http_get_json_from_bases(path, params, FAPI_BASES, FAPI_PRIMARY)

# ── Utilities ──────────────────────────────────────────────────────────────────
def _align_open_of_current(ts_utc: pd.Timestamp, interval: str) -> pd.Timestamp:
    # повертає відкриття поточної, ще незакритої свічки (UTC)
    if getattr(ts_utc, "tzinfo", None) is None:
        ts = ts_utc.tz_localize("UTC")
    else:
        ts = ts_utc.tz_convert("UTC")
    step = _TF_SECONDS.get(interval, 3600)
    epoch = int(ts.timestamp())
    open_cur = epoch - (epoch % step)
    return pd.to_datetime(open_cur, unit="s", utc=True)

def _validate_df(data, label: str) -> pd.DataFrame:
    if not isinstance(data, list) or not data or not isinstance(data[0], (list,tuple)):
        raise ValueError(f"Unexpected klines payload from {label}")
    df = pd.DataFrame(data, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ts"]    = pd.to_datetime(pd.to_numeric(df["closeTime"], errors="coerce"), unit="ms", utc=True)
    df = df[["ts","close"]].dropna().sort_values("ts").reset_index(drop=True)
    if df.empty: raise ValueError(f"Empty klines from {label}")
    return df

def _drop_unclosed(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    # залишаємо лише свічки, що закрились: ts < open_of_current
    open_cur = _align_open_of_current(pd.Timestamp.now(tz=timezone.utc), interval)
    df2 = df[df["ts"] < open_cur]
    # на випадок, якщо API повернув порожньо після відсічення — лишимо останню свічку без змін (краще щось ніж нічого)
    return df2 if not df2.empty else df

# ── Spot klines (основне джерело) ─────────────────────────────────────────────
def _spot_symbol(symbol: str) -> str:
    s = (symbol or "").upper()
    # Binance spot має BTCUSDT і BTCUSDC — лишаємо як є; інакше fallback на BTCUSDT
    if s.endswith("USDT") or s.endswith("USDC"):
        return s
    return "BTCUSDT"

def fetch_from_spot_klines(symbol: str, interval: str, limit: int) -> tuple[pd.DataFrame, str]:
    sp_sym = _spot_symbol(symbol)
    data = _http_get_json_spot("/api/v3/klines", {"symbol": sp_sym, "interval": interval, "limit": min(1000, limit + 5)})
    df = _validate_df(data, f"spot_klines({sp_sym})")
    df = _drop_unclosed(df, interval)
    if len(df) > limit: df = df.iloc[-limit:].reset_index(drop=True)
    return df, "spot_klines"

# ── Futures klines (резерв) ───────────────────────────────────────────────────
def _continuous_params(symbol: str) -> tuple[str,str]:
    sym=(symbol or "").upper()
    if sym in {"BTCUSDT","BTCUSDC"}: return sym,"PERPETUAL"
    if "_" in sym:
        base,suf=sym.split("_",1)
        ct={"PERP":"PERPETUAL","PERPETUAL":"PERPETUAL"}.get(suf.strip(),suf.strip())
        return base, ct
    return sym, "PERPETUAL"

def fetch_from_futures_klines(symbol: str, interval: str, limit: int) -> tuple[pd.DataFrame, str]:
    params={"interval":interval,"limit":limit}
    try:
        data=_http_get_json_fapi("/fapi/v1/klines",{**params,"symbol":symbol})
        df=_validate_df(data,"futures_klines(klines)")
        df=_drop_unclosed(df, interval)
        return (df if len(df)<=limit else df.iloc[-limit:].reset_index(drop=True)), "futures_klines"
    except Exception as e1:
        pair,ct=_continuous_params(symbol)
        data=_http_get_json_fapi("/fapi/v1/continuousKlines",{**params,"pair":pair,"contractType":ct})
        df=_validate_df(data,"futures_klines(continuous)")
        df=_drop_unclosed(df, interval)
        return (df if len(df)<=limit else df.iloc[-limit:].reset_index(drop=True)), "futures_klines"

# ── Vision (останній резерв) ──────────────────────────────────────────────────
def _vision_month_url(symbol:str, interval:str, y:int, m:int)->str:
    mm=f"{m:02d}"
    return f"{VISION_MONTHLY}/{symbol}/{interval}/{symbol}-{interval}-{y}-{mm}.zip"

def _vision_day_urls(symbol:str, interval:str, d:datetime)->list[str]:
    s=d.strftime("%Y-%m-%d")
    base=f"{VISION_DAILY}/{symbol}/{interval}/{symbol}-{interval}-{s}"
    return [base+".zip", base+".csv"]

def _csv_rows_from_bytes(b:bytes)->list[list]:
    text=b.decode("utf-8", errors="replace")
    out=[]
    for row in csv.reader(io.StringIO(text)):
        if not row: continue
        if row[0].strip().lower() in ("open_time","open time"): continue
        if len(row)<11: continue
        out.append(row[:12])
    return out

def _get_zip_rows(url:str)->list[list] | None:
    r=SESSION.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
    if r.status_code==404: return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name=next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not name: return None
        return _csv_rows_from_bytes(zf.read(name))

def fetch_from_vision(symbol:str, interval:str, limit:int)->tuple[pd.DataFrame,str]:
    # Спробуємо 1–2 місяці + кілька останніх днів — лише як резерв, щоб не витрачати час
    today=datetime.utcnow().date()
    rows=[]
    y,m=today.year,today.month
    for k in range(2):
        yy= y if m-k>0 else y-((k-(m-1))//12+1)
        mm= ((m-k-1)%12)+1
        url=_vision_month_url(symbol, interval, yy, mm)
        try:
            part=_get_zip_rows(url)
            if part: rows.extend(part)
        except Exception:
            pass
        if len(rows)>=limit: break
    # кілька останніх днів (поточний місяць може бути неповний)
    for d in range(7):
        day=today - timedelta(days=d)
        got=None
        for u in _vision_day_urls(symbol, interval, datetime.combine(day, datetime.min.time())):
            try:
                got=_get_zip_rows(u) if u.endswith(".zip") else None
                if got is None:
                    r=SESSION.get(u, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
                    if r.status_code!=404:
                        r.raise_for_status(); got=_csv_rows_from_bytes(r.content)
            except Exception:
                got=None
            if got: break
        if got: rows.extend(got)
        if len(rows)>=limit: break

    if not rows: raise RuntimeError("vision: no data")
    df=_validate_df(rows,"binance_vision")
    df=_drop_unclosed(df, interval)
    if len(df)>limit: df=df.iloc[-limit:].reset_index(drop=True)
    return df,"binance_vision"

# ── Public fetch (spot → futures → vision) ─────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int = LIMIT_PER_TF) -> tuple[pd.DataFrame, str]:
    # 1) spot (швидко і стабільно через воркер)
    try:
        return fetch_from_spot_klines(symbol, interval, limit)
    except Exception as e_spot:
        spot_err = str(e_spot)
    else:
        spot_err = None

    # 2) futures (якщо треба)
    try:
        return fetch_from_futures_klines(symbol, interval, limit)
    except Exception as e_fut:
        fut_err = str(e_fut)
    else:
        fut_err = None

    # 3) vision (останній резерв)
    try:
        return fetch_from_vision(symbol, interval, limit)
    except Exception as e_vis:
        vis_err = str(e_vis)
        raise RuntimeError(f"spot failed: {spot_err}; futures failed: {fut_err}; vision failed: {vis_err}")

# ── Indicators ────────────────────────────────────────────────────────────────
def ema(s,n):   return s.ewm(span=n, adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); gain=d.clip(lower=0); loss=-d.clip(upper=0)
    rs=gain.rolling(n).mean()/loss.rolling(n).mean()
    return 100 - (100/(1+rs))
def macd(s,fast=12,slow=26,signal=9):
    line=ema(s,fast)-ema(s,slow)
    sig=line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

# ── Compute & report ──────────────────────────────────────────────────────────
def compute_block(symbol, interval):
    df, route = fetch_klines(symbol, interval, limit=LIMIT_PER_TF)
    close = df["close"]
    out = {
        "symbol": symbol, "tf": interval,
        "last_ts": df["ts"].iloc[-1].isoformat(),
        "last_close": float(close.iloc[-1]),
        "route": route, "source": route,
    }
    for n in (20,50,200):
        out[f"ema{n}"]=float(ema(close,n).iloc[-1])
    out["rsi14"]=float(rsi(close,14).iloc[-1])
    m_line,m_sig,m_hist=macd(close)
    out["macd"]=float(m_line.iloc[-1]); out["macd_signal"]=float(m_sig.iloc[-1]); out["macd_hist"]=float(m_hist.iloc[-1])
    out["price_vs_ema200"]="above" if out["last_close"]>out["ema200"] else "below"
    out["ema20_cross_50"]="bull" if out["ema20"]>out["ema50"] else "bear"
    out["macd_cross"]="bull" if out["macd"]>out["macd_signal"] else "bear"
    return out

def main():
    blocks, stale = [], False
    for s in SYMBOLS:
        for tf in TFS:
            print(f"[analyze] fetching {s} {tf} ...", flush=True)
            try:
                b=compute_block(s, tf)
                print(f"[analyze] ok {s} {tf} via {b['route']} last_close={b['last_close']}", flush=True)
                blocks.append(b)
            except Exception as e:
                stale=True
                print(f"[analyze] FAIL {s} {tf}: {e}", flush=True)
                blocks.append({"symbol": s, "tf": tf, "error": "fetch_failed", "message": str(e)})

    report={"generated_at": datetime.now(timezone.utc).isoformat(), "stale": stale, "data": blocks}
    os.makedirs("public", exist_ok=True)
    with open("public/report.json","w",encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2)
    with open("public/index.html","w",encoding="utf-8") as f:
        f.write("""<h1>BTC Futures Reports</h1>
<p><a href="report.json">report.json</a></p>""")

if __name__=="__main__":
    main()
