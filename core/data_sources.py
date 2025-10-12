# core/data_sources.py — v0.7 (single-proxy, soft failures, spot fallback for klines)
import os, time, json, random, asyncio, requests
from datetime import datetime, timezone
from urllib.parse import quote
from typing import Optional, Tuple, List

FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
    "https://fapi5.binance.com",
]

ENV_WORKER = os.getenv("CF_WORKER_URL", "").strip()
# якщо секрет не заданий — використаємо твій робочий дефолт
DEFAULT_WORKER = "https://binance-proxy.brokerof-net.workers.dev"
PROXY_URL = ENV_WORKER if ENV_WORKER else DEFAULT_WORKER

DEFAULT_TIMEOUT = 8
USER_AGENT = "btc-futures-analysis/0.7 (+streamlit)"

def _compose_upstream(url: str, params: Optional[dict]) -> str:
    return url + ("?" + requests.compat.urlencode(params) if params else "")

def _direct_get(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = requests.get(url, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()

def _proxy_get(upstream_full_url: str, timeout: int = DEFAULT_TIMEOUT):
    if not PROXY_URL:
        raise RuntimeError("No proxy configured")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    proxied = f"{PROXY_URL.rstrip('/')}/?u={quote(upstream_full_url, safe='')}"
    r = requests.get(proxied, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()

def _robust_get(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT, prefer_proxy: bool = False):
    upstream_full = _compose_upstream(url, params)

    # 1) proxy-first (за бажанням)
    if prefer_proxy:
        try:
            return _proxy_get(upstream_full, timeout=timeout)
        except Exception:
            pass

    # 2) кілька прямих баз
    bases = FAPI_BASES.copy()
    random.shuffle(bases)
    for base in bases[:3]:
        try:
            alt = url.replace("https://fapi.binance.com", base)
            return _direct_get(alt, params=params, timeout=timeout)
        except Exception:
            time.sleep(0.2 + random.random()*0.2)

    # 3) фінальний фолбек — через проксі
    return _proxy_get(upstream_full, timeout=timeout)

# ------------ Public fetchers (SOFT: ніколи не кидають виняток) ------------

def fetch_klines(symbol: str, interval: str, limit: int = 240):
    """
    Повертає list[dict]: ts, open, high, low, close, volume.
    Якщо futures недоступні — деградуємо на SPOT klines (ціна для UI/імпульсу достатня).
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit}, prefer_proxy=True)
    except Exception:
        # SPOT fallback
        try:
            url = "https://api.binance.com/api/v3/klines"
            data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": min(1000,limit)}, prefer_proxy=True)
        except Exception:
            return []  # повна деградація

    out = []
    for x in data:
        try:
            out.append({
                "ts": datetime.utcfromtimestamp(x[0]/1000).replace(tzinfo=timezone.utc),
                "open": float(x[1]), "high": float(x[2]),
                "low": float(x[3]), "close": float(x[4]),
                "volume": float(x[5]),
            })
        except Exception:
            continue
    return out

def fetch_open_interest(symbol: str) -> Optional[float]:
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        j = _robust_get(url, {"symbol": symbol.upper()}, prefer_proxy=True)
        return float(j.get("openInterest", 0.0))
    except Exception:
        return None

def _norm_taker_period(period: str) -> str:
    # Binance supports: 5m,15m,30m,1h,2h,4h,6h,12h,1d
    mapping = {"1m":"5m","3m":"5m","5m":"5m","15m":"15m","30m":"30m",
               "1h":"1h","2h":"2h","4h":"4h","6h":"6h","12h":"12h","1d":"1d"}
    p = (period or "5m").lower()
    return mapping.get(p, "5m")

def fetch_taker_longshort_ratio(symbol: str, period: str = "5m", limit: int = 30) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    try:
        url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
        norm = _norm_taker_period(period)
        arr = _robust_get(url, {"symbol": symbol.upper(), "period": norm, "limit": limit}, prefer_proxy=True)
        if not arr:
            return None, None, None
        last = arr[-1]
        buy = float(last.get("buyVol", "0"))
        sell = float(last.get("sellVol", "0"))
        ratio = float(last.get("buySellRatio", "1"))
        denom = (buy + sell) if (buy + sell) else 1.0
        imb = (buy - sell) / denom
        ts = int(last.get("timestamp", 0))
        return ratio, imb, ts
    except Exception:
        return None, None, None

def fetch_premium_index_klines(symbol: str, interval: str = "1m", limit: int = 60) -> Tuple[Optional[float], Optional[int]]:
    try:
        url = "https://fapi.binance.com/futures/data/premiumIndexKlines"
        data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit}, prefer_proxy=True)
        if not data:
            return None, None
        last = data[-1]
        premium_close = float(last[4])
        close_time = int(last[6])
        return premium_close, close_time
    except Exception:
        return None, None

# ---------- Optional WS (не використовується у поточному LIVE) ----------
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
