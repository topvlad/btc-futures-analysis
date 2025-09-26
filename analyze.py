# analyze.py (витяг ключових частин)
import os, time, json, math, requests
import pandas as pd
from datetime import datetime, timezone
# analyze.py (витяг із верхньої частини файлу)

import os, time, json, math, requests
from statistics import mean

# ✅ Правильні бази для USDⓈ-M Futures
FAPI_BASES = [
    "https://fapi.binance.com",      # основний
    "https://fapi1.binance.com",     # резерви (деякі регіони)
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

# Якщо хочеш збирати безперервний контракт (перпетуал) замість символа:
USE_CONTINUOUS = False  # або True
PAIR = "BTCUSDT"
CONTRACT_TYPE = "PERPETUAL"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "btc-futures-analysis/1.0 (+github actions)"
})
TIMEOUT = 10
RETRIES = 4
BACKOFF = 1.6

def _get(url, params):
    last_exc = None
    for base in FAPI_BASES:
        full = base + url
        for i in range(RETRIES):
            try:
                r = SESSION.get(full, params=params, timeout=TIMEOUT)
                if r.status_code == 451:
                    # інколи зустрічається на окремих хостах — пробуємо інший
                    break
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last_exc = e
                time.sleep(BACKOFF**i)
        # пробуємо наступний base
    raise last_exc

def fetch_klines(symbol: str, interval: str, limit: int = 200):
    if USE_CONTINUOUS:
        # ✅ Continuous Klines (перпетуал індекс по парі)
        # /fapi/v1/continuousKlines?pair=BTCUSDT&contractType=PERPETUAL&interval=15m&limit=200
        data = _get(
            "/fapi/v1/continuousKlines",
            {"pair": PAIR, "contractType": CONTRACT_TYPE, "interval": interval, "limit": limit}
        )
    else:
        # ✅ Звичайні USDⓈ-M Futures Klines
        # /fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=200
        data = _get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )
    closes = [float(x[4]) for x in data]
    ts = [int(x[6]) for x in data]   # closeTime мс
    return closes, ts


def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast, ema_slow = ema(series, fast), ema(series, slow)
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_line
    return line, signal_line, hist

def compute_block(symbol, interval):
    df = fetch_klines(symbol, interval, limit=200)
    close = df["close"]
    out = {
        "symbol": symbol, "tf": interval,
        "last_ts": df["ts"].iloc[-1].isoformat(),
        "last_close": float(close.iloc[-1]),
    }
    for n in (20, 50, 200):
        out[f"ema{n}"] = float(ema(close, n).iloc[-1])
    out["rsi14"] = float(rsi(close, 14).iloc[-1])
    m_line, m_sig, m_hist = macd(close)
    out["macd"] = float(m_line.iloc[-1])
    out["macd_signal"] = float(m_sig.iloc[-1])
    out["macd_hist"] = float(m_hist.iloc[-1])
    # прості сигнали
    out["price_vs_ema200"] = "above" if out["last_close"] > out["ema200"] else "below"
    out["ema20_cross_50"] = "bull" if out["ema20"] > out["ema50"] else "bear"
    out["macd_cross"] = "bull" if out["macd"] > out["macd_signal"] else "bear"
    return out

def main():
    symbols = ["BTCUSDT", "BTCUSDC"]
    tfs = ["15m","1h","4h","1d"]
    blocks = []
    stale = False
    try:
        for s in symbols:
            for tf in tfs:
                blocks.append(compute_block(s, tf))
    except Exception as e:
        # не валимо білд — маркуємо як stale
        stale = True
        blocks.append({"error": "fetch_failed", "message": str(e)})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stale": stale,
        "data": blocks,
    }
    os.makedirs("public", exist_ok=True)
    with open("public/report.json","w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    # простий HTML-індекс
    with open("public/index.html","w", encoding="utf-8") as f:
        f.write("""<h1>BTC Futures Reports</h1>
<p><a href="report.json">report.json</a></p>""")

if __name__ == "__main__":
    main()
