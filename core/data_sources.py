# core/data_sources.py — v0.4 (robust GET + CF Worker fallback + correct taker fields)
import asyncio
import json
import time
import random
from datetime import datetime, timezone
from urllib.parse import quote
from typing import Optional, Tuple, List

import requests

# ---- Binance endpoints + CF Worker proxy ----
FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
    "https://fapi5.binance.com",
]
CF_WORKER = "https://binance-proxy.brokerof-net.workers.dev"

DEFAULT_TIMEOUT = 8
USER_AGENT = "btc-futures-analysis/0.4 (+streamlit)"

def _compose_upstream(url: str, params: Optional[dict]) -> str:
    if params:
        return url + "?" + requests.compat.urlencode(params)
    return url

def _direct_get(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT):
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, params=params, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()

def _proxy_get(upstream_full_url: str, timeout: int = DEFAULT_TIMEOUT):
    headers = {"User-Agent": USER_AGENT}
    proxied = f"{CF_WORKER.rstrip('/')}/?u={quote(upstream_full_url, safe='')}"
    r = requests.get(proxied, timeout=timeout, headers=headers)
    r.raise_for_status()
    return r.json()

def _robust_get(url: str, params: Optional[dict] = None, timeout: int = DEFAULT_TIMEOUT, try_proxy_first: bool = False):
    """
    Robust GET with:
      - optional proxy-first
      - multiple direct bases
      - jitter + small retries
    """
    upstream_full = _compose_upstream(url, params)

    # 1) Try proxy first if asked
    if try_proxy_first:
        try:
            return _proxy_get(upstream_full, timeout=timeout)
        except Exception:
            pass

    # 2) Try direct (rotate bases a bit)
    bases = FAPI_BASES.copy()
    random.shuffle(bases)
    for base in bases[:3]:  # try a few bases, not all
        try:
            # replace host part if needed
            if url.startswith("https://fapi.binance.com"):
                alt = url.replace("https://fapi.binance.com", base)
            else:
                alt = url
            return _direct_get(alt, params=params, timeout=timeout)
        except Exception:
            time.sleep(0.2)

    # 3) Fallback to proxy
    return _proxy_get(upstream_full, timeout=timeout)

# ---------- Public fetchers (minimal schemas) ----------

def fetch_klines(symbol: str, interval: str, limit: int = 200):
    """
    Futures klines (OHLCV). Returns list[dict] with keys: ts, open, high, low, close, volume.
    """
    url = f"https://fapi.binance.com/fapi/v1/klines"
    data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit})
    out = []
    for x in data:
        out.append({
            "ts": datetime.utcfromtimestamp(x[0] / 1000).replace(tzinfo=timezone.utc),
            "open": float(x[1]), "high": float(x[2]),
            "low": float(x[3]), "close": float(x[4]),
            "volume": float(x[5]),
        })
    return out

def fetch_open_interest(symbol: str) -> float:
    """
    Snapshot OI.
    """
    url = f"https://fapi.binance.com/fapi/v1/openInterest"
    j = _robust_get(url, {"symbol": symbol.upper()})
    try:
        return float(j.get("openInterest", 0.0))
    except Exception:
        return 0.0

def fetch_taker_longshort_ratio(symbol: str, interval: str = "5m", limit: int = 30) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    takerlongshortRatio endpoint returns:
      - buyVol, sellVol, buySellRatio, timestamp
    We compute:
      - ratio = buySellRatio (float)
      - imb   = (buy - sell)/(buy + sell) for potential later use
    Returns (ratio, imbalance, timestamp_ms)
    """
    url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
    arr = _robust_get(url, {"symbol": symbol.upper(), "period": interval, "limit": limit})
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

def fetch_premium_index_klines(symbol: str, interval: str = "1m", limit: int = 30) -> Tuple[Optional[float], Optional[int]]:
    """
    premiumIndexKlines: close field = premium/basis value.
    Returns (premium_close, closeTime_ms)
    """
    url = "https://fapi.binance.com/futures/data/premiumIndexKlines"
    data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit})
    if not data:
        return None, None
    last = data[-1]
    try:
        premium_close = float(last[4])
        t_close = int(last[6])
        return premium_close, t_close
    except Exception:
        return None, None

# ---------- Lightweight WS (optional) ----------

import websockets

async def ws_stream(symbol: str = "btcusdt", interval: str = "1m", callback=None, max_msgs: int = 100):
    """
    Kline close stream. Calls `callback(dict)` when message arrives.
    """
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
                "ts": datetime.utcfromtimestamp(k["t"] / 1000).replace(tzinfo=timezone.utc),
                "open": float(k["o"]), "high": float(k["h"]),
                "low": float(k["l"]), "close": float(k["c"]),
                "is_closed": bool(k["x"]),
            }
            if callable(callback):
                try:
                    callback(payload)
                except Exception:
                    pass
            n += 1
            if max_msgs and n >= max_msgs:
                break

def stream_live(symbol: str, interval: str, callback):
    """
    Synchronous helper for Streamlit to run WS quickly.
    """
    try:
        asyncio.run(ws_stream(symbol, interval, callback))
    except Exception as e:
        print("WS error:", e)
