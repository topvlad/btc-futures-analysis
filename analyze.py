# analyze.py
import os, time, json, io, gzip, csv, math
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode, quote

# =========================
# Config / Environment
# =========================
CF_WORKER_BASE = (os.getenv("CF_WORKER_BASE") or "").strip()
# If set, we will try the Worker FIRST. The worker must accept ?u=<encoded full target URL>.

FAPI_PRIMARY = "https://fapi.binance.com"

FAPI_BASES = [
    *([CF_WORKER_BASE] if CF_WORKER_BASE else []),
    FAPI_PRIMARY,
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

USE_CONTINUOUS = False  # True => always use /continuousKlines

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "btc-futures-analysis/1.3 (+github actions)",
    "Accept": "application/json,*/*",
    "Accept-Encoding": "gzip, deflate",
})

CI = os.getenv("GITHUB_ACTIONS") == "true"

TIMEOUT_CONNECT = 4
TIMEOUT_READ    = 6
RETRIES         = 2 if CI else 4
BACKOFF         = 1.4
# Upper bound wall-time for a single logical fetch across retries/hosts
FETCH_DEADLINE_SEC = 18 if CI else 35

# Binance Vision (official data mirror)
VISION_BASE = "https://data.binance.vision/data/futures/um/daily/klines"

SYMBOLS = ["BTCUSDT", "BTCUSDC"]
TFS     = ["15m","1h","4h","1d"]
LIMIT_PER_TF = 200


# =========================
# Low-level HTTP
# =========================
def _is_json_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("content-type") or "").lower()
    return "application/json" in ct

def _build_worker_url(target_full_url: str) -> str:
    # CF worker receives the full upstream URL in ?u=
    return f"{CF_WORKER_BASE}?u={quote(target_full_url, safe='')}"

def _http_get_json(path: str, params: dict) -> dict | list:
    """
    Tries all bases with retries. If base == CF worker, passes full target URL via ?u=
    Defends against HTML 200/empty body.
    """
    started = time.monotonic()
    last_exc = None

    # Randomize non-worker bases a bit (keep worker first)
    bases = FAPI_BASES[:]
    if CF_WORKER_BASE and bases and bases[0] == CF_WORKER_BASE:
        tail = bases[1:]
        import random
        random.shuffle(tail)
        bases = [CF_WORKER_BASE] + tail
    else:
        import random
        random.shuffle(bases)

    for base in bases:
        for i in range(RETRIES):
            if time.monotonic() - started > FETCH_DEADLINE_SEC:
                raise TimeoutError(f"fetch_deadline_exceeded for {path} {params}")
            try:
                if CF_WORKER_BASE and base == CF_WORKER_BASE:
                    # Build full upstream to primary (not the rotated mirrors) to avoid nesting workers
                    upstream = f"{FAPI_PRIMARY}{path}?{urlencode(params)}"
                    url = _build_worker_url(upstream)
                    r = SESSION.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), allow_redirects=False)
                else:
                    url = f"{base}{path}"
                    r = SESSION.get(url, params=params, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), allow_redirects=False)

                sc = r.status_code
                if sc in (451, 403, 429, 418, 520, 521, 522, 523, 524, 525, 526):
                    # transient: backoff & retry
                    time.sleep(BACKOFF**i)
                    continue

                r.raise_for_status()

                # Hard guard: 200 but not JSON → treat as failure (CF/anti-bot HTML)
                body = r.content or b""
                if (not _is_json_response(r)) and (len(body) == 0 or body[:1] == b"<"):
                    raise ValueError(f"Non-JSON 200 from {base}{path}")

                return r.json()
            except Exception as e:
                last_exc = e
                time.sleep(BACKOFF**i)
                continue

    raise RuntimeError(f"_get_failed_after_retries: {last_exc}")


# =========================
# Klines helpers
# =========================
def _continuous_params_for_symbol(symbol: str) -> tuple[str, str]:
    """Return (pair, contractType) for continuousKlines."""
    if not symbol:
        raise ValueError("Symbol required")

    overrides = {
        "BTCUSDT": ("BTCUSDT", "PERPETUAL"),
        "BTCUSDC": ("BTCUSDC", "PERPETUAL"),
    }
    sym = symbol.upper()
    if sym in overrides:
        return overrides[sym]

    if "_" in sym:
        base, suffix = sym.split("_", 1)
        contract_type = {
            "PERP": "PERPETUAL",
            "PERPETUAL": "PERPETUAL",
        }.get(suffix.strip(), suffix.strip())
        if not base or not contract_type:
            raise ValueError(f"Unable to derive continuous from '{symbol}'")
        return base, contract_type

    # Default assumption for UM perpetuals
    return sym, "PERPETUAL"


def _validate_klines_payload(data, route_label: str) -> pd.DataFrame:
    if not isinstance(data, list) or not data or not isinstance(data[0], list):
        raise ValueError(f"Unexpected klines payload shape from {route_label}")

    df = pd.DataFrame(data, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ts"]    = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    df = df[["ts","close"]].dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Empty klines dataframe from {route_label}")
    return df


def fetch_from_binance_api(symbol: str, interval: str, limit: int) -> tuple[pd.DataFrame, str]:
    """Try official endpoints with fallbacks: klines → continuousKlines → markPriceKlines."""
    params = {"interval": interval, "limit": limit}
    pair, contract_type = _continuous_params_for_symbol(symbol)

    # Forced continuous
    if USE_CONTINUOUS:
        data = _http_get_json("/fapi/v1/continuousKlines", {**params, "pair": pair, "contractType": contract_type})
        return _validate_klines_payload(data, "continuousKlines (forced)"), "continuousKlines (forced)"

    # 1) klines
    try:
        data = _http_get_json("/fapi/v1/klines", {**params, "symbol": symbol})
        return _validate_klines_payload(data, "klines"), "klines"
    except Exception as e_kl:
        last_exc = e_kl

    # 2) continuousKlines
    try:
        data = _http_get_json("/fapi/v1/continuousKlines", {**params, "pair": pair, "contractType": contract_type})
        return _validate_klines_payload(data, "continuousKlines (fallback)"), "continuousKlines (fallback)"
    except Exception as e_cont:
        last_exc = e_cont

    # 3) markPriceKlines (NOTE: mark price, not last trade)
    try:
        data = _http_get_json("/fapi/v1/markPriceKlines", {**params, "symbol": symbol})
        return _validate_klines_payload(data, "markPriceKlines (fallback)"), "markPriceKlines (fallback)"
    except Exception as e_mark:
        raise RuntimeError(
            f"klines failed: {last_exc}; continuousKlines failed: {e_cont}; markPriceKlines failed: {e_mark}"
        )


# =========================
# Binance Vision (archive) fallback
# =========================
# Vision has daily CSV files per symbol & interval:
#   /data/futures/um/daily/klines/<SYMBOL>/<INTERVAL>/<SYMBOL>-<INTERVAL>-YYYY-MM-DD.csv
# We'll pull recent days until we have >= limit rows and then take the tail.

def _vision_file_url(symbol: str, interval: str, date_obj: datetime) -> str:
    d = date_obj.strftime("%Y-%m-%d")
    return f"{VISION_BASE}/{symbol}/{interval}/{symbol}-{interval}-{d}.csv"

def _download_csv(url: str) -> list[list]:
    r = SESSION.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
    if r.status_code == 404:
        raise FileNotFoundError("vision 404")
    r.raise_for_status()

    # Files may be served plain CSV; sometimes gz exists alongside .csv. We stick to .csv here.
    content = r.content
    # Decode as UTF-8
    text = content.decode("utf-8", errors="replace")
    rows = []
    for row in csv.reader(io.StringIO(text)):
        # Kline CSV columns conform to REST klines shape
        if not row or len(row) < 11:
            continue
        rows.append(row[:12])  # ensure 12 cols shape
    if not rows:
        raise ValueError("vision empty csv")
    return rows

def fetch_from_binance_vision(symbol: str, interval: str, limit: int) -> tuple[pd.DataFrame, str]:
    today_utc = datetime.utcnow().date()
    rows: list[list] = []

    # Try up to last 7 days (usually 1-3 is enough)
    for delta in range(0, 7):
        day = today_utc - timedelta(days=delta)
        url = _vision_file_url(symbol, interval, datetime.combine(day, datetime.min.time()))
        try:
            part = _download_csv(url)
            rows.extend(part)
            if len(rows) >= limit:
                break
        except FileNotFoundError:
            # No file for that day/interval yet — skip
            continue
        except Exception:
            # Network glitch — try next day
            continue

    if not rows:
        raise RuntimeError("binance vision: no data retrieved")

    # Shape rows like REST klines and reuse the same validator
    df = _validate_klines_payload(rows, "binance_vision")
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df, "binance_vision"


def fetch_klines(symbol: str, interval: str, limit: int = LIMIT_PER_TF) -> tuple[pd.DataFrame, str]:
    # Try API chain first; if it fails entirely, hit Vision archive
    try:
        return fetch_from_binance_api(symbol, interval, limit)
    except Exception as api_exc:
        # Final fallback: Vision
        try:
            return fetch_from_binance_vision(symbol, interval, limit)
        except Exception as vis_exc:
            raise RuntimeError(
                f"API chain failed: {api_exc}; Vision fallback failed: {vis_exc}"
            ) from vis_exc


# =========================
# Indicators
# =========================
def ema(s, n):   return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100 / (1 + rs))

def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig  = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


# =========================
# Compute block
# =========================
def compute_block(symbol, interval):
    df, route = fetch_klines(symbol, interval, limit=LIMIT_PER_TF)
    close = df["close"]
    out = {
        "symbol": symbol,
        "tf": interval,
        "last_ts": df["ts"].iloc[-1].isoformat(),
        "last_close": float(close.iloc[-1]),
        "route": route,
        "source": route,  # explicit for streamlit
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


# =========================
# Main
# =========================
def main():
    blocks, stale = [], False
    for s in SYMBOLS:
        for tf in TFS:
            print(f"[analyze] fetching {s} {tf} ...", flush=True)
            try:
                b = compute_block(s, tf)
                print(f"[analyze] ok {s} {tf} via {b['route']} last_close={b['last_close']}", flush=True)
                blocks.append(b)
            except Exception as e:
                stale = True
                print(f"[analyze] FAIL {s} {tf}: {e}", flush=True)
                blocks.append({"symbol": s, "tf": tf, "error": "fetch_failed", "message": str(e)})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stale": stale,
        "data": blocks
    }
    os.makedirs("public", exist_ok=True)
    with open("public/report.json","w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("public/index.html","w", encoding="utf-8") as f:
        f.write("""<h1>BTC Futures Reports</h1>
<p><a href="report.json">report.json</a></p>""")

if __name__ == "__main__":
    main()
