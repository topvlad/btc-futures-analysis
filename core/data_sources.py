# core/data_sources.py — v0.9 (robust GET, single-proxy, soft returns, multi-exchange klines + taker 24h fallback)
import os, time, json, random, asyncio, requests
from datetime import datetime, timezone
from urllib.parse import quote
from typing import Optional, Tuple, List

# ---------- Config ----------
FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
    "https://fapi5.binance.com",
]
SPOT_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://api5.binance.com",
]
PROXY_URL = (os.getenv("CF_WORKER_URL") or os.getenv("CF_WORKER_BASE") or "").strip()

DEFAULT_TIMEOUT = 8
USER_AGENT = "btc-futures-analysis/0.9 (+streamlit)"

# ---------- Helpers ----------
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

    # 1) proxy first (optional)
    if prefer_proxy and PROXY_URL:
        try:
            return _proxy_get(upstream_full, timeout=timeout)
        except Exception:
            pass

    # 2) 2–3 shuffled bases (fapi or api)
    bases = FAPI_BASES.copy() if "fapi.binance.com" in url else SPOT_BASES.copy()
    random.shuffle(bases)
    for base in bases[:3]:
        try:
            alt = url.replace("https://fapi.binance.com", base).replace("https://api.binance.com", base)
            return _direct_get(alt, params=params, timeout=timeout)
        except Exception:
            time.sleep(0.2 + random.random()*0.2)

    # 3) proxy
    if PROXY_URL:
        return _proxy_get(upstream_full, timeout=timeout)

    raise RuntimeError("All attempts failed and no proxy configured")

# ---------- KLInes (soft) ----------
def fetch_klines(symbol: str, interval: str, limit: int = 240) -> List[dict]:
    # 1) Futures
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit}, prefer_proxy=True)
        return _klines_to_rows(data)
    except Exception:
        pass
    # 2) SPOT
    try:
        url = "https://api.binance.com/api/v3/klines"
        data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": min(1000,limit)}, prefer_proxy=True)
        return _klines_to_rows(data)
    except Exception:
        pass
    # 3) OKX spot
    try:
        inst = _okx_inst(symbol); bar = _okx_bar(interval)
        j = _direct_get("https://www.okx.com/api/v5/market/candles", {"instId": inst, "bar": bar, "limit": min(500,limit)})
        rows = []
        for x in j.get("data", []):
            ts = datetime.utcfromtimestamp(int(x[0])/1000).replace(tzinfo=timezone.utc)
            rows.append({"ts": ts, "open": float(x[1]), "high": float(x[2]), "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])})
        rows.sort(key=lambda r: r["ts"])
        return rows[-limit:] if rows else []
    except Exception:
        pass
    # 4) Coinbase spot
    try:
        product = "BTC-USD" if symbol.upper().startswith("BTC") else ("ETH-USD" if symbol.upper().startswith("ETH") else "BTC-USD")
        gran = {"1m":60,"5m":300,"15m":900,"30m":1800,"1h":3600}.get(interval,300)
        data = _direct_get(f"https://api.exchange.coinbase.com/products/{product}/candles", {"granularity":gran, "limit":min(300,limit)})
        rows = []
        for x in data:  # [time, low, high, open, close, volume]
            ts = datetime.utcfromtimestamp(int(x[0])).replace(tzinfo=timezone.utc)
            rows.append({"ts": ts, "open": float(x[3]), "high": float(x[2]), "low": float(x[1]), "close": float(x[4]), "volume": float(x[5])})
        rows.sort(key=lambda r: r["ts"])
        return rows[-limit:] if rows else []
    except Exception:
        pass
    return []

def _klines_to_rows(data) -> List[dict]:
    rows = []
    for x in data:
        try:
            ts = datetime.utcfromtimestamp(int(x[0])/1000).replace(tzinfo=timezone.utc)
            rows.append({"ts": ts, "open": float(x[1]), "high": float(x[2]), "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])})
        except Exception:
            continue
    rows.sort(key=lambda r: r["ts"])
    return rows

def _okx_inst(symbol: str) -> str:
    s = symbol.upper()
    if s.startswith("BTC"): return "BTC-USDT"
    if s.startswith("ETH"): return "ETH-USDT"
    return "BTC-USDT"

def _okx_bar(interval: str) -> str:
    return {"1m":"1m","3m":"3m","5m":"5m","15m":"15m","30m":"30m","1h":"1H"}.get(interval, "5m")

# ---------- OI / Taker / Premium (soft) ----------
def fetch_open_interest(symbol: str) -> Optional[float]:
    try:
        url = "https://fapi.binance.com/fapi/v1/openInterest"
        j = _robust_get(url, {"symbol": symbol.upper()}, prefer_proxy=True)
        return float(j.get("openInterest", 0.0))
    except Exception:
        return None

def _norm_taker_period(period: str) -> str:
    mapping = {"1m":"5m","3m":"5m","5m":"5m","15m":"15m","30m":"30m","1h":"1h","2h":"2h","4h":"4h","6h":"6h","12h":"12h","1d":"1d"}
    return mapping.get((period or "5m").lower(), "5m")

def fetch_taker_longshort_ratio(symbol: str, period: str = "5m", limit: int = 30) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Primary: /futures/data/takerlongshortRatio
    Fallback: /fapi/v1/ticker/24hr → approximate ratio from takerBuyBaseAssetVolume vs total volume.
    Returns (ratio, imbalance, timestamp_ms) or (None, None, None).
    """
    # Primary endpoint
    try:
        url = "https://fapi.binance.com/futures/data/takerlongshortRatio"
        norm = _norm_taker_period(period)
        arr = _robust_get(url, {"symbol": symbol.upper(), "period": norm, "limit": limit}, prefer_proxy=True)
        if arr:
            last = arr[-1]
            buy = float(last.get("buyVol", "0"))
            sell = float(last.get("sellVol", "0"))
            ratio = float(last.get("buySellRatio", "1"))
            denom = (buy + sell) or 1.0
            imb = (buy - sell) / denom
            ts = int(last.get("timestamp", 0))
            return ratio, imb, ts
    except Exception:
        pass

    # Fallback: 24h ticker
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        j = _robust_get(url, {"symbol": symbol.upper()}, prefer_proxy=True)
        # fields: "volume", "takerBuyBaseAssetVolume"
        vol = float(j.get("volume", "0") or 0.0)
        buy = float(j.get("takerBuyBaseAssetVolume", "0") or 0.0)
        if vol > 0:
            sell = max(vol - buy, 0.0)
            ratio = (buy / max(sell, 1e-12)) if sell > 0 else 2.0  # if no sells, assume strong buy
            denom = (buy + sell) or 1.0
            imb = (buy - sell) / denom
            ts = int(datetime.now(timezone.utc).timestamp() * 1000)
            return ratio, imb, ts
    except Exception:
        pass

    return None, None, None

def fetch_premium_index_klines(symbol: str, interval: str = "1m", limit: int = 60) -> Tuple[Optional[float], Optional[int]]:
    try:
        url = "https://fapi.binance.com/futures/data/premiumIndexKlines"
        data = _robust_get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit}, prefer_proxy=True)
        if not data: return None, None
        last = data[-1]
        premium_close = float(last[4])
        close_time = int(last[6])
        return premium_close, close_time
    except Exception:
        return None, None

# ---------- Optional WS (kept for future use) ----------
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
