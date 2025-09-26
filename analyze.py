# analyze.py
import os, time, json, requests
import pandas as pd
from datetime import datetime, timezone

FAPI_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
    # опційно: свій CF-Worker першим у списку
    # "https://<your-subdomain>.workers.dev",
]

USE_CONTINUOUS = False  # True -> /continuousKlines за PAIR/CONTRACT_TYPE
PAIR = "BTCUSDT"
CONTRACT_TYPE = "PERPETUAL"

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "btc-futures-analysis/1.2 (+github actions)",
    "Accept": "application/json",
})

CI = os.getenv("GITHUB_ACTIONS") == "true"
TIMEOUT_CONNECT = 4
TIMEOUT_READ    = 5
RETRIES         = 2 if CI else 4
BACKOFF         = 1.4
# жорсткий верхній ліміт часу на один фетч (усі ретраї/хости разом)
FETCH_DEADLINE_SEC = 18 if CI else 35

def _get(url, params):
    started = time.monotonic()
    last_exc = None
    for base in FAPI_BASES:
        full = base + url
        for i in range(RETRIES):
            if time.monotonic() - started > FETCH_DEADLINE_SEC:
                raise TimeoutError(f"fetch_deadline_exceeded for {url} {params}")
            try:
                r = SESSION.get(
                    full, params=params,
                    timeout=(TIMEOUT_CONNECT, TIMEOUT_READ),
                    allow_redirects=False,
                )
                sc = r.status_code
                if sc in (451, 403, 429, 418, 520, 521, 522, 523, 524, 525, 526):
                    time.sleep(BACKOFF**i)
                    continue
                r.raise_for_status()

                ctype = (r.headers.get("Content-Type") or "").lower()
                body  = r.content or b""
                # захист від HTML/порожнього 200
                if ("json" not in ctype) and (body[:1] == b"<" or len(body) == 0):
                    raise ValueError(f"Non-JSON 200 from {full}")

                return r.json()
            except Exception as e:
                last_exc = e
                time.sleep(BACKOFF**i)
    raise RuntimeError(f"_get_failed_after_retries: {last_exc}")

def fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    params = {"interval": interval, "limit": limit}
    if USE_CONTINUOUS:
        data = _get("/fapi/v1/continuousKlines", {**params, "pair": PAIR, "contractType": CONTRACT_TYPE})
    else:
        data = _get("/fapi/v1/klines", {**params, "symbol": symbol})

    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError("Unexpected klines payload shape")

    df = pd.DataFrame(data, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ts"]    = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    df = df[["ts","close"]].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty klines dataframe")
    return df

def ema(s, n):   return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100/(1+rs))
def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig  = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig

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
    out["ema20_cross_50"]  = "bull"  if out["ema20"] > out["ema50"]  else "bear"
    out["macd_cross"]      = "bull"  if out["macd"]   > out["macd_signal"] else "bear"
    return out

def main():
    symbols = ["BTCUSDT", "BTCUSDC"]
    tfs = ["15m","1h","4h","1d"]
    blocks, stale = [], False

    for s in symbols:
        for tf in tfs:
            print(f"[analyze] fetching {s} {tf} ...", flush=True)
            try:
                b = compute_block(s, tf)
                print(f"[analyze] ok {s} {tf} last_close={b['last_close']}", flush=True)
                blocks.append(b)
            except Exception as e:
                stale = True
                print(f"[analyze] FAIL {s} {tf}: {e}", flush=True)
                blocks.append({"symbol": s, "tf": tf, "error": "fetch_failed", "message": str(e)})

    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "stale": stale, "data": blocks}
    os.makedirs("public", exist_ok=True)
    with open("public/report.json","w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("public/index.html","w", encoding="utf-8") as f:
        f.write("""<h1>BTC Futures Reports</h1>
<p><a href="report.json">report.json</a></p>""")

if __name__ == "__main__":
    main()
