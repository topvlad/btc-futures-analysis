# analyze.py (fix e_cont + Vision zip support + fapi5)
import os, time, json, io, csv, zipfile
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode, quote

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

USE_CONTINUOUS = False
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "btc-futures-analysis/1.31 (+github actions)",
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
LIMIT_PER_TF = 200


def _is_json_response(resp: requests.Response) -> bool:
    ct = (resp.headers.get("content-type") or "").lower()
    return "application/json" in ct

def _build_worker_url(target_full_url: str) -> str:
    return f"{CF_WORKER_BASE}?u={quote(target_full_url, safe='')}"

def _http_get_json(path: str, params: dict) -> dict | list:
    started = time.monotonic()
    last_exc = None

    bases = FAPI_BASES[:]
    # перемішуємо, але воркер — перший
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
                    upstream = f"{FAPI_PRIMARY}{path}?{urlencode(params)}"
                    url = _build_worker_url(upstream)
                    r = SESSION.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), allow_redirects=False)
                else:
                    url = f"{base}{path}"
                    r = SESSION.get(url, params=params, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ), allow_redirects=False)

                if r.status_code not in (200, 203):
                    last_exc = f"bad_status:{r.status_code}"
                    time.sleep(BACKOFF**i); continue

                if (not _is_json_response(r)) and ((r.content or b"")[:1] == b"<" or not r.content):
                    last_exc = "non_json_200"
                    time.sleep(BACKOFF**i); continue

                return r.json()
            except Exception as e:
                last_exc = str(e)
                time.sleep(BACKOFF**i)
                continue

    raise RuntimeError(f"_get_failed_after_retries:{last_exc}")


def _continuous_params_for_symbol(symbol: str) -> tuple[str, str]:
    overrides = {"BTCUSDT": ("BTCUSDT", "PERPETUAL"), "BTCUSDC": ("BTCUSDC", "PERPETUAL")}
    sym = (symbol or "").upper()
    if sym in overrides: return overrides[sym]
    if "_" in sym:
        base, suf = sym.split("_", 1)
        ct = {"PERP":"PERPETUAL","PERPETUAL":"PERPETUAL"}.get(suf.strip(), suf.strip())
        if not base or not ct: raise ValueError(f"bad symbol {symbol}")
        return base, ct
    return sym, "PERPETUAL"


def _validate_klines_payload(data, route_label: str) -> pd.DataFrame:
    if not isinstance(data, list) or not data or not isinstance(data[0], (list, tuple)):
        raise ValueError(f"Unexpected klines payload from {route_label}")
    df = pd.DataFrame(data, columns=[
        "openTime","open","high","low","close","volume",
        "closeTime","qav","numTrades","takerBase","takerQuote","ignore"
    ])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ts"]    = pd.to_datetime(df["closeTime"], unit="ms", utc=True)
    df = df[["ts","close"]].dropna().reset_index(drop=True)
    if df.empty: raise ValueError(f"Empty klines from {route_label}")
    return df


def fetch_from_binance_api(symbol: str, interval: str, limit: int) -> tuple[pd.DataFrame, str]:
    params = {"interval": interval, "limit": limit}
    pair, contract_type = _continuous_params_for_symbol(symbol)

    errors = []  # збираємо тексти, щоб не падати на e_cont

    if USE_CONTINUOUS:
        try:
            data = _http_get_json("/fapi/v1/continuousKlines", {**params, "pair": pair, "contractType": contract_type})
            return _validate_klines_payload(data, "continuousKlines (forced)"), "continuousKlines (forced)"
        except Exception as e:
            errors.append(f"continuous(forced): {e}")

    # 1) klines
    try:
        data = _http_get_json("/fapi/v1/klines", {**params, "symbol": symbol})
        return _validate_klines_payload(data, "klines"), "klines"
    except Exception as e:
        errors.append(f"klines: {e}")

    # 2) continuousKlines
    try:
        data = _http_get_json("/fapi/v1/continuousKlines", {**params, "pair": pair, "contractType": contract_type})
        return _validate_klines_payload(data, "continuousKlines (fallback)"), "continuousKlines (fallback)"
    except Exception as e:
        errors.append(f"continuous(fallback): {e}")

    # 3) markPriceKlines
    try:
        data = _http_get_json("/fapi/v1/markPriceKlines", {**params, "symbol": symbol})
        return _validate_klines_payload(data, "markPriceKlines (fallback)"), "markPriceKlines (fallback)"
    except Exception as e:
        errors.append(f"markPrice(fallback): {e}")
        raise RuntimeError("; ".join(errors))


# ===== Binance Vision (.zip/.csv) =====
def _vision_urls(symbol: str, interval: str, date_obj: datetime) -> list[str]:
    d = date_obj.strftime("%Y-%m-%d")
    base = f"{VISION_BASE}/{symbol}/{interval}/{symbol}-{interval}-{d}"
    # спочатку .zip (частіше існує), потім .csv
    return [base + ".zip", base + ".csv"]

def _try_vision_download(url: str) -> list[list] | None:
    r = SESSION.get(url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ))
    if r.status_code == 404:
        return None
    r.raise_for_status()
    if url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            # беремо перший CSV із архіву
            name = next((n for n in zf.namelist() if n.lower().endswith(".csv")), None)
            if not name: return None
            data = zf.read(name).decode("utf-8", errors="replace")
    else:
        data = r.content.decode("utf-8", errors="replace")

    rows = []
    for row in csv.reader(io.StringIO(data)):
        if not row or len(row) < 11:
            continue
        rows.append(row[:12])
    return rows or None

def fetch_from_binance_vision(symbol: str, interval: str, limit: int) -> tuple[pd.DataFrame, str]:
    today = datetime.utcnow().date()
    all_rows: list[list] = []

    # пройдемося назад до 14 днів, поки не вистачить рядків
    for delta in range(0, 14):
        day = today - timedelta(days=delta)
        got = None
        for url in _vision_urls(symbol, interval, datetime.combine(day, datetime.min.time())):
            try:
                rows = _try_vision_download(url)
                if rows:
                    got = rows; break
            except Exception:
                continue
        if got:
            all_rows.extend(got)
            if len(all_rows) >= limit:
                break

    if not all_rows:
        raise RuntimeError("binance vision: no data retrieved")

    df = _validate_klines_payload(all_rows, "binance_vision")
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df, "binance_vision"


def fetch_klines(symbol: str, interval: str, limit: int = LIMIT_PER_TF) -> tuple[pd.DataFrame, str]:
    try:
        return fetch_from_binance_api(symbol, interval, limit)
    except Exception as api_exc:
        try:
            return fetch_from_binance_vision(symbol, interval, limit)
        except Exception as vis_exc:
            raise RuntimeError(f"API chain failed: {api_exc}; Vision fallback failed: {vis_exc}") from vis_exc


# ===== indicators =====
def ema(s, n):   return s.ewm(span=n, adjust=False).mean()
def rsi(s, n=14):
    d = s.diff(); gain = d.clip(lower=0); loss = -d.clip(upper=0)
    rs = gain.rolling(n).mean() / loss.rolling(n).mean()
    return 100 - (100 / (1 + rs))
def macd(s, fast=12, slow=26, signal=9):
    line = ema(s, fast) - ema(s, slow)
    sig  = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def compute_block(symbol, interval):
    df, route = fetch_klines(symbol, interval, limit=LIMIT_PER_TF)
    close = df["close"]
    out = {
        "symbol": symbol, "tf": interval,
        "last_ts": df["ts"].iloc[-1].isoformat(),
        "last_close": float(close.iloc[-1]),
        "route": route, "source": route,
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

    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "stale": stale, "data": blocks}
    os.makedirs("public", exist_ok=True)
    with open("public/report.json","w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("public/index.html","w", encoding="utf-8") as f:
        f.write("""<h1>BTC Futures Reports</h1>
<p><a href="report.json">report.json</a></p>""")

if __name__ == "__main__":
    main()
