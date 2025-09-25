# analyze.py
import requests
import sys, json
import pandas as pd
from datetime import datetime, timezone
import time


SYMBOLS = ["BTCUSDT", "BTCUSDC"]
INTERVALS = ["15m", "1h", "4h", "1d"]  # «яструбині» ТФ
LIMIT = 200  # достатньо для EMA200/MACD
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

def fetch_klines(symbol, interval, limit=LIMIT):
    try:
        r = requests.get(BASE_URL, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=20)
        if r.status_code == 451:
            raise RuntimeError("HTTP 451 from Binance (CI IP blocked)")
        r.raise_for_status()
        data = r.json()
        closes = [float(c[4]) for c in data]
        ts_close = int(data[-1][6]) if data else None
        return closes, ts_close
    except Exception as e:
        # пробросимо далі — обробимо у main()
        raise

def fetch_with_retry(url, params, tries=3, pause=1.5):
    for i in range(tries):
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        if i < tries-1:
            time.sleep(pause)
    r.raise_for_status()

def ema(series, n):
    return pd.Series(series).ewm(span=n, adjust=False).mean().iloc[-1]

def rsi(series, n=14):
    s = pd.Series(series)
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else float("inf")
    return 100 - (100 / (1 + rs))

def macd(series, f=12, s=26, sig=9):
    s = pd.Series(series)
    ema_f = s.ewm(span=f, adjust=False).mean()
    ema_s = s.ewm(span=s, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal = macd_line.ewm(span=sig, adjust=False).mean()
    hist = macd_line - signal
    return macd_line.iloc[-1], signal.iloc[-1], hist.iloc[-1]

def basic_signal(close, e20, e50, e200, rsi14, macd_line, signal_line, hist):
    notes = []
    trend = "neutral"

    # Trend via EMA200
    if close > e200 and e50 > e200:
        trend = "bullish"
    elif close < e200 and e50 < e200:
        trend = "bearish"

    # Momentum via EMA20/50
    if e20 > e50: notes.append("EMA20>EMA50 (up-momentum)")
    if e20 < e50: notes.append("EMA20<EMA50 (down-momentum)")

    # RSI zones
    if rsi14 >= 70: notes.append("RSI overbought")
    elif rsi14 <= 30: notes.append("RSI oversold")
    else: notes.append("RSI neutral")

    # MACD cross
    if macd_line > signal_line: notes.append("MACD>Signal")
    else: notes.append("MACD<Signal")

    # Histogram sign
    if hist > 0: notes.append("Hist +")
    else: notes.append("Hist -")

    return trend, ", ".join(notes)

def main():
    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        for sym in SYMBOLS:
            for tf in INTERVALS:
                closes, ts = fetch_klines(sym, tf)
                close = closes[-1]
                e20, e50, e200 = ema(closes,20), ema(closes,50), ema(closes,200)
                r = rsi(closes)
                m, sig, h = macd(closes)
                trend, notes = basic_signal(close, e20, e50, e200, r, m, sig, h)
                rows.append({
                    "timestamp_utc": now_iso, "symbol": sym, "tf": tf,
                    "close": round(close, 2),
                    "ema20": round(e20, 2), "ema50": round(e50, 2), "ema200": round(e200, 2),
                    "rsi14": round(r, 2), "macd": round(m,5), "signal": round(sig,5), "hist": round(h,5),
                    "trend": trend, "notes": notes, "exchange_close_time_ms": ts
                })
        df = pd.DataFrame(rows)
        df.to_csv("report.csv", index=False)
        df.to_json("report.json", orient="records")
        print("\n=== SUMMARY (last run) ===")
        print(df.to_string(index=False))
    except Exception as e:
        # створимо статус-файли, щоб Pages все одно деплоївся
        status = {"error": "fetch_failed", "message": str(e), "hint": "Likely HTTP 451 from Binance on CI runner"}
        with open("report.json", "w") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        with open("report.csv", "w") as f:
            f.write("error,message\nfetch_failed,\"{}\"\n".format(str(e).replace('"','\"')))
        print("WARN:", status)
        # ВАЖЛИВО: завершуємося кодом 0, щоб деплой Pages не відмінився
        sys.exit(0)

if __name__ == "__main__":
    main()
