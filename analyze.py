# analyze.py (v1.52) — fix order, monthly Vision, multi-source live tail
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

SPOT_BASES = [
    *([CF_WORKER_BASE] if CF_WORKER_BASE else []),
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://api5.binance.com",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "btc-futures-analysis/1.52 (+github actions)",
    "Accept": "application/json,*/*",
    "Accept-Encoding": "gzip, deflate",
})

CI = os.getenv("GITHUB_ACTIONS") == "true"
TIMEOUT_CONNECT = 4
TIMEOUT_READ    = 6
RETRIES         = 2 if CI else 4
BACKOFF         = 1.4
FETCH_DEADLINE_SEC = 18 if CI else 35

# Vision bases
VISION_DAILY   = "https://data.binance.vision/data/futures/um/daily/klines"
VISION_MONTHLY = "https://data.binance.vision/data/futures/um/monthly/klines"

SYMBOLS = ["BTCUSDT", "BTCUSDC"]
TFS     = ["15m","1h","4h","1d"]
LIMIT_PER_TF = 200

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
        for i in range(RETRIES):
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

                if r.status_code not in (200,203): last=f"bad_status:{r.status_code}"; time.sleep(BACKOFF**i); continue
                if (not _is_json(r)) and ((r.content or b"")[:1]==b"<" or not r.content): last="non_json_200"; time.sleep(BACKOFF**i); continue
                return r.json()
            except Exception as e:
                last=str(e); time.sleep(BACKOFF**i); continue
    raise RuntimeError(f"_get_failed_after_retries:{last}")

def _http_get_json_fapi(path:str, params:dict): return _http_get_json_from_bases(path, params, FAPI_BASES, FAPI_PRIMARY)
def _http_get_json_spot(path: str, params: dict):
    # тепер воркер теж використовується (primary_for_worker="https://api.binance.com")
    return _http_get_json_from_bases(path, params, SPOT_BASES, primary_for_worker="https://api.binance.com")
   
# ── REST klines ────────────────────────────────────────────────────────────────
def _continuous_params(symbol: str) -> tuple[str,str]:
    sym=(symbol or "").upper()
    if sym in {"BTCUSDT","BTCUSDC"}: return sym,"PERPETUAL"
    if "_" in sym:
        base,suf=sym.split("_",1)
        ct={"PERP":"PERPETUAL","PERPETUAL":"PERPETUAL"}.get(suf.strip(),suf.strip())
        return base, ct
    return sym, "PERPETUAL"

def _validate_df(data, label: str) -> pd.DataFrame:
    if not isinstance(data, list) or not data or not isinstance(data[0], (list,tuple)):
        raise ValueError(f"Unexpected klines payload from {label}")
    df=pd.DataFrame(data, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ])
    df["close"]=pd.to_numeric(df["close"], errors="coerce")
    df["ts"]=pd.to_datetime(pd.to_numeric(df["closeTime"], errors="coerce"), unit="ms", utc=True)
    df=df[["ts","close"]].dropna().sort_values("ts").reset_index(drop=True)  # <─ ключова правка: СОРТУЄМО
    if df.empty: raise ValueError(f"Empty klines from {label}")
    return df

def fetch_from_binance_api(symbol:str, interval:str, limit:int) -> tuple[pd.DataFrame,str]:
    params={"interval":interval,"limit":limit}
    pair,ct=_continuous_params(symbol)
    errors=[]

    try:
        data=_http_get_json_fapi("/fapi/v1/klines",{**params,"symbol":symbol})
        return _validate_df(data,"klines"),"klines"
    except Exception as e:
        errors.append(f"klines:{e}")

    try:
        data=_http_get_json_fapi("/fapi/v1/continuousKlines",{**params,"pair":pair,"contractType":ct})
        return _validate_df(data,"continuousKlines"),"continuousKlines"
    except Exception as e:
        errors.append(f"continuous:{e}")

    try:
        data=_http_get_json_fapi("/fapi/v1/markPriceKlines",{**params,"symbol":symbol})
        return _validate_df(data,"markPriceKlines"),"markPriceKlines"
    except Exception as e:
        errors.append(f"markPrice:{e}")
        raise RuntimeError("; ".join(errors))

# ── Vision monthly/daily ──────────────────────────────────────────────────────
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

def _get_vision_zip(url:str)->list[list] | None:
    r=SESSION.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
    if r.status_code==404: return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name=next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
        if not name: return None
        return _csv_rows_from_bytes(zf.read(name))

def fetch_from_binance_vision(symbol:str, interval:str, limit:int)->tuple[pd.DataFrame,str]:
    # 1) спочатку підтягуємо місячні архіви, від поточного місяця назад
    today=datetime.utcnow().date()
    rows: list[list]=[]
    bars_per_day={"15m":96,"1h":24,"4h":6,"1d":1}.get(interval,24)
    approx_per_month=bars_per_day*30
    max_months=18  # достатньо для 200 daily
    y, m=today.year, today.month

    fetched_month=False
    for k in range(max_months):
        yy= y if m-k>0 else y-((k-(m-1))//12+1)
        mm= ((m-k-1)%12)+1
        url=_vision_month_url(symbol, interval, yy, mm)
        try:
            part=_get_vision_zip(url)
            if part:
                rows.extend(part); fetched_month=True
                if len(rows)>=limit: break
        except Exception:
            continue

    # 2) на додачу — останні кілька днів (поточний місяць може бути неповним)
    extra_days = max(0, math.ceil(limit - len(rows)))
    days_needed = min(14, math.ceil(extra_days / bars_per_day) + 3)
    for d in range(days_needed):
        day=today - timedelta(days=d)
        got=None
        for u in _vision_day_urls(symbol, interval, datetime.combine(day, datetime.min.time())):
            try:
                if u.endswith(".zip"):
                    got=_get_vision_zip(u)
                else:
                    r=SESSION.get(u, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
                    if r.status_code==404: got=None
                    else:
                        r.raise_for_status(); got=_csv_rows_from_bytes(r.content)
            except Exception:
                got=None
            if got: break
        if got:
            rows.extend(got)
            if len(rows)>=limit: break

    if not rows:
        raise RuntimeError("binance vision: no data retrieved")

    df=_validate_df(rows,"binance_vision")  # тут ще раз сортуємо і чистимо
    if len(df)>limit: df=df.iloc[-limit:].reset_index(drop=True)
    return df,"binance_vision"

# ── Live tail (multi-source) ───────────────────────────────────────────────────
def _try_live_price(symbol:str) -> tuple[float,str] | None:
    # 1) Binance futures mark / last
    try:
        j=_http_get_json_fapi("/fapi/v1/premiumIndex",{"symbol":symbol}); p=float(j.get("markPrice")); return p,"live_mark"
    except Exception: pass
    try:
        j=_http_get_json_fapi("/fapi/v1/ticker/price",{"symbol":symbol}); p=float(j.get("price")); return p,"live_fapi"
    except Exception: pass
    # 2) Binance spot
    try:
        j=_http_get_json_spot("/api/v3/ticker/price",{"symbol": symbol if symbol.endswith("USDT") else "BTCUSDT"})
        p=float(j.get("price")); return p,"live_spot"
    except Exception: pass
    # 3) OKX
    try:
        r=SESSION.get("https://www.okx.com/api/v5/market/ticker", params={"instId":"BTC-USDT"}, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
        r.raise_for_status(); j=r.json()
        p=float(j["data"][0]["last"]); return p,"live_okx"
    except Exception: pass
    # 4) Bitstamp
    try:
        r=SESSION.get("https://www.bitstamp.net/api/v2/ticker/btcusdt", timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
        r.raise_for_status(); j=r.json(); p=float(j["last"]); return p,"live_bitstamp"
    except Exception: pass
    # 5) Coinbase (Advanced)
    try:
        r=SESSION.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker", timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
        r.raise_for_status(); j=r.json(); p=float(j.get("price") or j.get("last")); return p,"live_coinbase"
    except Exception: pass
    # 6) Kraken
    try:
        r=SESSION.get("https://api.kraken.com/0/public/Ticker", params={"pair":"XBTUSDT"}, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
        r.raise_for_status(); j=r.json()
        # В Kraken ключ трохи інший, але перше значення vwap/last у 'c'
        val=list(j["result"].values())[0]["c"][0]; p=float(val); return p,"live_kraken"
    except Exception: pass
    return None

_TF_SECONDS={"15m":900,"1h":3600,"4h":14400,"1d":86400}
def _align_to_tf(ts_utc:pd.Timestamp, interval:str)->pd.Timestamp:
    step=_TF_SECONDS.get(interval,3600); epoch=int(ts_utc.timestamp()); aligned=epoch - (epoch%step)
    return pd.to_datetime(aligned, unit="s", utc=True)

def _align_to_tf_boundary(ts_utc: pd.Timestamp, interval: str) -> pd.Timestamp:
    # robust: якщо раптом naive — локалізуємо, якщо aware — конвертуємо
    if getattr(ts_utc, "tz", None) is None:
        ts = ts_utc.tz_localize("UTC")
    else:
        ts = ts_utc.tz_convert("UTC")
    step = _TF_SECONDS.get(interval, 3600)
    epoch = int(ts.timestamp())
    aligned = epoch - (epoch % step)
    return pd.to_datetime(aligned, unit="s", utc=True)
   
def stitch_live_tail(df:pd.DataFrame, symbol:str, interval:str)->tuple[pd.DataFrame,str]:
    probe=_try_live_price(symbol)
    if not probe: return df,"binance_vision"  # без live
    live,src=probe
    now_aligned = _align_to_tf_boundary(pd.Timestamp.now(tz="UTC"), interval)
    last_ts=df["ts"].iloc[-1]
    if now_aligned>last_ts:
        df2=pd.concat([df, pd.DataFrame([{"ts":now_aligned,"close":float(live)}])], ignore_index=True)
    else:
        df2=df.copy(); df2.loc[df2.index[-1],"close"]=float(live)
    return df2,f"binance_vision+{src}"

# ── Public fetch ───────────────────────────────────────────────────────────────
def fetch_klines(symbol:str, interval:str, limit:int=LIMIT_PER_TF)->tuple[pd.DataFrame,str]:
    # 1) спроба через REST (якщо пощастить)
    try:
        return fetch_from_binance_api(symbol, interval, limit)
    except Exception as api_exc:
        # 2) Vision (місячні+добові) + стіб живої ціни
        try:
            df,route=fetch_from_binance_vision(symbol, interval, limit)
            df2,route2=stitch_live_tail(df, symbol, interval)
            return df2, route2
        except Exception as vis_exc:
            raise RuntimeError(f"API chain failed: {api_exc}; Vision fallback failed: {vis_exc}") from vis_exc

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
