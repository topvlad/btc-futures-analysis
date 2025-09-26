# analyze.py
import os, time, json, math, requests
import pandas as pd
from datetime import datetime, timezone

FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

USE_CONTINUOUS = False
PAIR = "BTCUSDT"
CONTRACT_TYPE = "PERPETUAL"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "btc-futures-analysis/1.1 (+github actions)",
    "Accept": "application/json",
})
TIMEOUT = 12
RETRIES = 5
BACKOFF = 1.8

def _get(url, params):
    last_exc = None
    for base in FAPI_BASES:
        full = base + url
        for i in range(RETRIES):
            try:
                r = SESSION.get(full, params=params, timeout=TIMEOUT)
                # якщо регіональний/частотний блок або edge-помилки — ретраї/зміна хоста
                if r.status_code in (451, 403, 429, 418, 520, 521, 522, 523, 524, 525, 526):
                    time.sleep(BACKOFF**i)
                    continue
                r.raise_for_status()

                # ДЕФЕНС від HTML/порожнього тіла з кодом 200
                ctype = (r.headers.get("Content-Type") or "").lower()
                text0 = r.text[:64].strip() if r.text is not None else ""
                if ("json" not in ctype) and (text0.startswith("<") or text0 == ""):
                    # пробуємо ще раз/інший базовий хост
                    raise ValueError(f"Non-JSON 200 from {full}")

                return r.json()

            except Exception as e:
                last_exc = e
                time.sleep(BACKOFF**i)
        # наступний base
    raise RuntimeError(f"_get_failed_after_retries: {last_exc}")

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    if USE_CONTINUOUS:
        data = _get(
            "/fapi/v1/continuousKlines",
            {"pair": PAIR, "contractType": CONTRACT_TYPE, "interval": interval, "limit": limit}
        )
    else:
        data = _get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": interval, "limit": limit}
        )

    # Валідація структури
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError("Unexpected klines payload shape")

    df = pd.DataFrame(data, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ts"] = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    df = df[["ts","close"]].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty klines dataframe")
    return df

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
    out["price_vs_ema200"] = "above" if out["last_close"] > out["ema200"] else "below"
    out["ema20_cross_50"] = "bull" if out["ema20"] > out["ema50"] else "bear"
    out["macd_cross"] = "bull" if out["macd"] > out["macd_signal"] else "bear"
    return out

def main():
    symbols = ["BTCUSDT", "BTCUSDC"]
    tfs = ["15m","1h","4h","1d"]
    blocks = []
    stale = False

    for s in symbols:
        for tf in tfs:
            try:
                blocks.append(compute_block(s, tf))
            except Exception as e:
                stale = True
                blocks.append({"symbol": s, "tf": tf, "error": "fetch_failed", "message": str(e)})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stale": stale,
        "data": blocks,
    }
    os.makedirs("public", exist_ok=True)
    with open("public/report.json","w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("public/index.html","w", encoding="utf-8") as f:
        f.write("""<h1>BTC Futures Reports</h1>
<p><a href="report.json">report.json</a></p>""")

if __name__ == "__main__":
    main()
