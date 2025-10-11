# core/data_sources.py — v0.2 (websocket + CF fallback)
import requests, asyncio, json
from urllib.parse import quote
from datetime import datetime, timezone

# Binance + CF Worker
FAPI_BASE = "https://fapi.binance.com"
CF_WORKER = "https://binance-proxy.brokerof-net.workers.dev"

def _get(url, params=None, timeout=8):
    params = params or {}
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        proxied = f"{CF_WORKER.rstrip('/')}/?u={quote(url, safe='')}?{requests.compat.urlencode(params)}"
        r2 = requests.get(proxied, timeout=timeout)
        r2.raise_for_status()
        return r2.json()

def fetch_klines(symbol: str, interval: str, limit: int = 200):
    url = f"{FAPI_BASE}/fapi/v1/klines"
    data = _get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit})
    out = []
    for x in data:
        out.append({
            "ts": datetime.utcfromtimestamp(x[0]/1000).replace(tzinfo=timezone.utc),
            "open": float(x[1]), "high": float(x[2]),
            "low": float(x[3]), "close": float(x[4]),
            "volume": float(x[5])
        })
    return out

def fetch_open_interest(symbol: str):
    url = f"{FAPI_BASE}/fapi/v1/openInterest"
    data = _get(url, {"symbol": symbol.upper()})
    return float(data.get("openInterest", 0.0))

def fetch_taker_longshort_ratio(symbol: str, interval="5m", limit=30):
    url = f"{FAPI_BASE}/futures/data/takerlongshortRatio"
    data = _get(url, {"symbol": symbol.upper(), "period": interval, "limit": limit})
    if not data: return None
    latest = data[-1]
    return float(latest["longShortRatio"]), float(latest["longAccountRatio"]), latest["timestamp"]

def fetch_premium_index_klines(symbol: str, interval="1m", limit=30):
    url = f"{FAPI_BASE}/fapi/v1/premiumIndexKlines"
    data = _get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit})
    if not data: return None
    latest = data[-1]
    return float(latest[4]), latest[0]

# --- live stream via websocket ---
import websockets

async def ws_stream(symbol="btcusdt", interval="1m", callback=None, max_msgs=50):
    uri = f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_{interval}"
    async with websockets.connect(uri, ping_interval=20) as ws:
        n = 0
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if "k" not in data: continue
            k = data["k"]
            payload = {
                "ts": datetime.utcfromtimestamp(k["t"]/1000).replace(tzinfo=timezone.utc),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "is_closed": k["x"]
            }
            if callback: callback(payload)
            n += 1
            if max_msgs and n >= max_msgs: break

# Simple helper for Streamlit async-safe loop
def stream_live(symbol, interval, callback):
    try:
        asyncio.run(ws_stream(symbol, interval, callback))
    except Exception as e:
        print("Stream error:", e)
