# core/signals.py
import numpy as np
import pandas as pd
from .features import realized_vol, atr, vwap, pct_rank

# --- Параметри за замовчуванням (можеш міняти у Streamlit UI) ---
DEFAULTS = dict(
    rv_win=60,                 # для 1m ~60 хв
    rv_rank_win=1440,         # ~1 день 1m або гнучко
    hh_lookback=20,           # пробій локального high/low
    atr_win=14,
    atr_k=0.8,
    d_oi_min=0.005,           # +0.5%
    taker_ratio_min=1.10,     # >1.10 для long
)

def signal_impulse_participation(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    Сетап A: Пробій + ΔOI% + агресія taker → LONG/SHORT з R:R 1:2
    Очікує колонки: t,o,h,l,c,v,q, oi, d_oi_pct, buySellRatio, taker_imb
    """
    P = {**DEFAULTS, **(params or {})}
    out = df.copy()

    out["rv"] = realized_vol(out, "c", P["rv_win"])
    out["rv_rank"] = pct_rank(out["rv"], P["rv_rank_win"])
    out["atr"] = atr = (out["h"] - out["l"]).rolling(P["atr_win"]).mean()

    out["hh"] = out["h"].rolling(P["hh_lookback"]).max()
    out["ll"] = out["l"].rolling(P["hh_lookback"]).min()
    out["above_hh"] = out["c"] > out["hh"].shift(1)
    out["below_ll"] = out["c"] < out["ll"].shift(1)

    # фільтри
    vol_ok = out["rv_rank"] >= 0.6
    oi_ok  = out["d_oi_pct"] >= P["d_oi_min"]
    taker_ok_long  = out["buySellRatio"] >= P["taker_ratio_min"]
    taker_ok_short = out["buySellRatio"] <= (2 - P["taker_ratio_min"])  # симетрично ~0.90

    # сигнали
    out["long_sig"]  = vol_ok & oi_ok & taker_ok_long  & out["above_hh"]
    out["short_sig"] = vol_ok & (out["d_oi_pct"] <= -P["d_oi_min"]) & taker_ok_short & out["below_ll"]

    # цілі SL/TP
    sl_long  = (out["c"] - P["atr_k"]*atr).clip(lower=0)
    tp_long  = out["c"] + 2*(out["c"] - sl_long)
    sl_short = out["c"] + P["atr_k"]*atr
    tp_short = out["c"] - 2*(sl_short - out["c"])

    out["sl_long"], out["tp_long"]   = sl_long, tp_long
    out["sl_short"], out["tp_short"] = sl_short, tp_short
    return out

def signal_funding_meanrev(df: pd.DataFrame, premium_col: str = "premium", win_z: int = 60, z_thr: float = 2.0) -> pd.DataFrame:
    """
    Сетап B: Перегрів фандінгу/премії → mean reversion до VWAP
    Очікує: t,o,h,l,c,v,premium
    """
    out = df.copy()
    out["vwap"] = vwap(out)
    prem = out[premium_col].astype(float)
    z = (prem - prem.rolling(win_z).mean())/prem.rolling(win_z).std()
    out["prem_z"] = z

    # умови (контртрендова логіка в діапазоні)
    rng = (out["h"] - out["l"]).rolling(30).mean()
    inside = (out["c"] <= out["vwap"] + rng*0.8) & (out["c"] >= out["vwap"] - rng*0.8)

    out["short_sig"] = (z >= z_thr) & inside & (out["c"] > out["vwap"])
    out["long_sig"]  = (z <= -z_thr) & inside & (out["c"] < out["vwap"])

    # SL/TP до VWAP з R:R~1:2
    out["sl_long"]  = (out["l"] - rng*0.5)
    out["tp_long"]  = out["vwap"]
    out["sl_short"] = (out["h"] + rng*0.5)
    out["tp_short"] = out["vwap"]
    return out
