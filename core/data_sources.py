# core/data_sources.py — v0.6 (robust GET + multi-proxy + taker period fix)
import os, time, json, random, asyncio, requests
from datetime import datetime, timezone
from urllib.parse import quote
from typing import Optional, Tuple, List

# ---- Binance endpoints + CF Worker proxies ----
FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
    "https://fapi5.binance.com",
]
# allow custom worker via env (as у тебе в головній апці)
ENV_WORKER = os.getenv("CF_WORKER_URL", "").strip()
PROXIES = [p for p in [
    ENV_WORKER or None,
    "https://binance-proxy.brokerof-net.workers.dev",
    # додатковий запасний (можеш зробити свій клон у Cloudflare і вставити тут)
    "https://binance-proxy-2.brokerof-net.workers.dev",
] if p]

DEFAULT_TIMEOUT = 8
USER_AGENT = "btc-futures-analysis/0.6 (+streamlit)"

def _compose_upstream(url: str, params: Optional[dict]) -> str:
    if params:
        return url + "?" + requests.compat.urlencode(params)
    return url

def _direct_get(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()

def _proxy_get(upstream_full_url: str, timeout: int = DEFAULT_TIMEOUT):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    last_err = None
    for px in PROXIES:
        try:
            proxied = f"{px.rstrip('/')}/?u={quote(upstream_full_url, safe='')}"
            r = requests.get(proxied, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(0.15)
            continue
    if last_err:
        raise last_err
    raise RuntimeError("No proxy configured")

def _robust_get(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT, prefer_proxy: bool = False):
    """
    Надійний GET:
      1) (опційно) пробуємо проксі спочатку
      2) далі 2-3 випадкових бази fapi
      3) в кінці — проксі знову
    З джитером і малими затримками між ретраями.
    """
    upstream_full = _compose_upstream(url, params)

    if prefer_proxy and PROXIES:
        try:
            return _proxy_get(upstream_full, timeout=timeout)
        except Exception:
            pass

    bases = FAPI_BASES.copy()
    random.shuffle(bases)
    last_err = None
    for base in bases[:3]:
        try:
            alt = url.replace("https://fapi.binance.com", base)
            return _direct_get(alt, params=params, timeout=timeout)
        except Exception as e:
            last_err = e
            time.sleep(0.2 + random.random()*0.2)

    # final fallback → proxy
    try:
        return _proxy_get(upstream_full, timeout=timeout)
    except Exception as e:
        # якщо все погано — піднімаємо останню
        raise e if e else last_err

# -------- Helpers --------
def _norm_taker_period(period: str) -> str:
    # Binance supports: 5m,15m,30m,1h,2h,4h,6h,12h,1d
    period = period.lower()
    if period in {"5m","15m","30m","1h","2h","4h","6h","12h","1d"}:
        return period
    # map common chart intervals to nearest supported
    mapping = {"1m":"5m", "3m":"5m", "5m":"5m", "15m":"15m", "30m":"30m",
               "1h":"1h","2h":"2h","4h":"4h","6h":"6h","12h":"12h","1d":"1d"}
    return mapping.get(period, "5m")

# -------- Public fetchers --------
def fetch_klines(symbol: str, interval: str, limit: int = 240):
    url = "https://fapi.binance.com/fapi/v1/klines"
    data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit}, prefer_proxy=True)
    out = []
    for x in data:
        out.append({
            "ts": datetime.utcfromtimestamp(x[0]/1000).replace(tzinfo=timezone.utc),
            "open": float(x[1]), "high": float(x[2]),
            "low": float(x[3]), "close": float(x[4]),
            "volume": float(x[5]),
        })
    return out

def fetch_open_interest(symbol: str) -> float:
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    j = _robust_get(url, {"symbol": symbol.upper()}, prefer_proxy=True)
    try:
        return float(j.get("openInterest", 0.0))
    except Exception:
        return 0.0

def fetch_taker_longshort_ratio(symbol: str, period: str = "5m", limit: int = 30) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
    norm = _norm_taker_period(period)
    arr = _robust_get(url, {"symbol": symbol.upper(), "period": norm, "limit": limit}, prefer_proxy=True)
    if not arr:
        return None, None, None
    last = arr[-1]
    try:
        buy = float(last.get("buyVol", "0"))
        sell = float(last.get("sellVol", "0"))
        ratio = float(last.get("buySellRatio", "1"))
        denom = (buy + sell) if (buy + sell) != 0 else 1.0
        imb = (buy - sell) / denom
        ts = int(last.get("timestamp", 0))
        return ratio, imb, ts
    except Exception:
        return None, None, None

def fetch_premium_index_klines(symbol: str, interval: str = "1m", limit: int = 60) -> Tuple[Optional[float], Optional[int]]:
    url = "https://fapi.binance.com/futures/data/premiumIndexKlines"
    data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit}, prefer_proxy=True)
    if not data:
        return None, None
    last = data[-1]
    try:
        premium_close = float(last[4])
        close_time = int(last[6])
        return premium_close, close_time
    except Exception:
        return None, None

# ---------- Optional WS ----------
import websockets
async def ws_stream(symbol: str = "btcusdt", interval: str = "1m", callback=None, max_msgs: int = 100):
    uri = f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_{interval}"
    async with websockets.connect(uri, ping_interval=20) as ws:
        n = 0
        while True:
            raw = await ws.recv()
            data = json.loads(raw)
            if "k" not in data: 
                continue
            k = data["k"]
            payload = {
                "ts": datetime.utcfromtimestamp(k["t"]/1000).replace(tzinfo=timezone.utc),
                "open": float(k["o"]), "high": float(k["h"]),
                "low": float(k["l"]), "close": float(k["c"]),
                "is_closed": bool(k["x"]),
            }
            if callable(callback):
                try: callback(payload)
                except Exception: pass
            n += 1
            if max_msgs and n >= max_msgs:
                break

def stream_live(symbol: str, interval: str, callback):
    try:
        asyncio.run(ws_stream(symbol, interval, callback))
    except Exception as e:
        print("WS error:", e)
