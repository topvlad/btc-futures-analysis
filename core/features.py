# core/features.py
import numpy as np
import pandas as pd

def realized_vol(df: pd.DataFrame, price_col: str = "c", win: int = 30) -> pd.Series:
    r = np.log(df[price_col]).diff()
    rv = np.sqrt(r.rolling(win).apply(lambda x: np.sum(x**2), raw=True))
    return rv

def atr(df: pd.DataFrame, win: int = 14) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    return tr.rolling(win).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    pv = (df["c"] * df["v"]).cumsum()
    vv = df["v"].cumsum().replace(0, np.nan)
    return pv / vv

def pct_rank(s: pd.Series, win: int = 1440) -> pd.Series:
    # відносний перцентиль за ковзним вікном (наприклад, 30–60 днів для 1m)
    return s.rolling(win).apply(lambda x: (x.argsort().argsort()[-1])/(len(x)-1) if len(x)>1 else np.nan, raw=False)

def delta_pct(s: pd.Series) -> pd.Series:
    return s.pct_change()

def taker_imbalance(buy_vol: pd.Series, sell_vol: pd.Series) -> pd.Series:
    denom = (buy_vol + sell_vol).replace(0, np.nan)
    return (buy_vol - sell_vol) / denom

def zscore(s: pd.Series, win: int = 120) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - m) / sd

def join_oi(kl: pd.DataFrame, oi: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge_asof(kl.sort_values("t"), oi.sort_values("t"), on="t", direction="backward")
    df["d_oi_pct"] = df["oi"].pct_change()
    return df

def join_taker(kl: pd.DataFrame, lsr: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge_asof(kl.sort_values("t"), lsr.sort_values("t"), on="t", direction="backward")
    # якщо немає — заповнимо нейтральним
    for col in ["buyVol","sellVol","buySellRatio","taker_imb"]:
        if col not in df:
            df[col] = np.nan
    return df

def join_premium(kl: pd.DataFrame, prem: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge_asof(kl.sort_values("t"), prem.sort_values("t"), on="t", direction="backward")
    return df
