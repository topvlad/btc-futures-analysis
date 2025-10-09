# analyze.py (v1.40) — Vision + live tail stitch; spot/futures live price fallback
import os, time, json, io, csv, zipfile, math
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode, quote

# =========================
# Config
# =========================
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

# Spot (на випадок, якщо futures недоступний)
SPOT_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://api5.binance.com",
]

USE_CONTINUOUS = False
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "btc-futures-analysis/1.40 (+github actions)",
    "Accept": "application/json,*/*",
    "Accept-Encoding": "gzip, deflate",
})

CI = os.getenv("GITHUB_ACTIONS") == "true"
TIMEOUT_CONNECT = 4
TIMEOUT_READ    = 6
RETRIES         = 2 if CI else 4
BACKOFF         = 1.4
FETCH_DEADLINE_SEC = 18 if CI else 35

VISION_BASE = "https://data.binance.vision/data/futures/um/daily/klines"

SYMBOLS = ["BTCUSDT", "BTCUSDC"]
TFS     = ["15m","1h","4h","1d"]
LIMI
