# core/data_sources.py
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import websockets

BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_WS   = "wss://fstream.binance.com/stream"

# --------- утиліти кешу / throttle ----------
class SimpleTTLCache:
    def __init__(self, ttl_sec: float = 5.0):
        self.ttl = ttl_sec
        self.store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str):
        now = time.time()
        if key in self.store:
            ts, val = self.store[key]
            if now - ts <= self.ttl:
                return val
        return None

    def set(self, key: str, value: Any):
        self.store[key] = (time.time(), value)

_cache = SimpleTTLCache(ttl_sec=3.0)  # дрібний кеш для REST

# --------- REST: kline / OI / ratios / funding ----------
def _get(url: str, params: Dict[str, Any]) -> Any:
    key = url + "|" + json.dumps(params, sort_keys=True)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    _cache.set(key, data)
    return data

def fetch_klines(symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fapi/v1/klines"
    data = _get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit})
    cols = ["t_open","o","h","l","c","v","t_close","q","n","t_maker","q_maker","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["t"] = pd.to_datetime(df["t_close"], unit="ms", utc=True)
    for col in ["o","h","l","c","v","q"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["t","o","h","l","c","v","q"]].sort_values("t").reset_index(drop=True)
    return df

def fetch_open_interest(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/futures/data/openInterestKlines"
    data = _get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit})
    # [openTime,open,high,low,close,volume,closeTime,...] -> беремо close як OI
    cols = ["t_open","oi_o","oi_h","oi_l","oi_c","vol","t_close","q","n","taker_b","taker_s","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["t"] = pd.to_datetime(df["t_close"], unit="ms", utc=True)
    df["oi"] = pd.to_numeric(df["oi_c"], errors="coerce")
    return df[["t","oi"]].sort_values("t").reset_index(drop=True)

def fetch_taker_longshort_ratio(symbol: str, period: str = "5m", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/futures/data/takerlongshortRatio"
    data = _get(url, {"symbol": symbol.upper(), "period": period, "limit": limit})
    # [{ "timestamp":..., "buyVol": "...", "sellVol": "...", "buySellRatio": "..."}, ...]
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["buyVol","sellVol","buySellRatio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # taker_imbalance = (buy - sell)/(buy + sell)
    denom = (df["buyVol"] + df["sellVol"]).replace(0, np.nan)
    df["taker_imb"] = (df["buyVol"] - df["sellVol"]) / denom
    return df[["t","buyVol","sellVol","buySellRatio","taker_imb"]]

def fetch_funding_history(symbol: str, limit: int = 100) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    data = _get(url, {"symbol": symbol.upper(), "limit": limit})
    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["rate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df[["t","rate"]].sort_values("t")

def fetch_premium_index_klines(symbol: str, interval: str = "1m", limit: int = 200) -> pd.DataFrame:
    url = f"{BINANCE_FAPI}/futures/data/premiumIndexKlines"
    data = _get(url, {"symbol": symbol.upper(), "interval": interval, "limit": limit})
    cols = ["t_open","o","h","l","c","v","t_close","q","n","taker_b","taker_s","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["t"] = pd.to_datetime(df["t_close"], unit="ms", utc=True)
    for col in ["o","h","l","c"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # premium = close (це "basis"/premium index) — використовуємо для z-score
    df.rename(columns={"c":"premium"}, inplace=True)
    return df[["t","premium"]]

# --------- WS: мульти-стрім (aggTrade + kline) ----------
@dataclass
class LiveBars:
    last_kline: Optional[Dict[str, Any]] = None
    last_trade: Optional[Dict[str, Any]] = None

async def ws_stream(symbol: str, interval: str = "1m"):
    """
    Повертає генератор подій: kline close + aggTrade (агр. трейди).
    Викликати з asyncio в Streamlit (через st.session_state + asyncio.run).
    """
    sym = symbol.lower()
    stream = f"{sym}@kline_{interval}/{sym}@aggTrade"
    url = f"{BINANCE_WS}?streams={stream}"

    async with websockets.connect(url, ping_interval=15, ping_timeout=20) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            yield data  # сирі івенти; сторінка сама агрегує
