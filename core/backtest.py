# core/backtest.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

@dataclass
class BTConfig:
    fee_rate: float = 0.0004   # 0.04% тейкер умовно; підлаштуй
    slip: float = 0.5          # ковзання у тиках/USDT (підлаштуй)
    rr: float = 2.0            # TP = rr * risk
    risk_usdt: float = 40.0    # умовний ризик для sizing (для звіту)
    side_preference: Optional[str] = None  # "long"/"short"/None

def _entry_price(row, side):
    px = row["c"]
    return px + row.get("slip", 0) if side == "long" else px - row.get("slip", 0)

def _apply_fees(notional, cfg: BTConfig):
    return notional * cfg.fee_rate

def backtest_signals(df: pd.DataFrame, setup: str, cfg: BTConfig) -> Dict[str, float]:
    """
    Простий «bar-close» бек-тест: вхід на закритті тригер-бару, перевірка TP/SL на наступних барах.
    Колонки: long_sig/short_sig, sl_long/tp_long/sl_short/tp_short, c (ціна)
    """
    trades = []
    for i in range(1, len(df)-1):
        row = df.iloc[i]
        nxt  = df.iloc[i+1]

        if setup == "A":
            long_sig  = bool(row.get("long_sig", False))
            short_sig = bool(row.get("short_sig", False))
        else:
            long_sig  = bool(row.get("long_sig", False))
            short_sig = bool(row.get("short_sig", False))

        # опц. преференція сторони
        if cfg.side_preference == "long":
            short_sig = False
        elif cfg.side_preference == "short":
            long_sig = False

        for side in ["long","short"]:
            sig = long_sig if side=="long" else short_sig
            if not sig:
                continue
            entry = float(row["c"])
            sl = float(row[f"sl_{side}"])
            tp = float(row[f"tp_{side}"])

            # наступний бар — груба симуляція: чи торкнулись TP/SL
            hi, lo = float(nxt["h"]), float(nxt["l"])
            hit_tp = hi >= tp if side=="long" else lo <= tp
            hit_sl = lo <= sl if side=="long" else hi >= sl

            # порядок перевірки: перетини в середині бару невідомі, приймемо консервативно — спершу SL
            outcome = None
            exit_px = None
            if hit_sl and hit_tp:
                # невизначеність: приймаємо гірший (SL)
                outcome = "SL"
                exit_px = sl
            elif hit_sl:
                outcome = "SL"
                exit_px = sl
            elif hit_tp:
                outcome = "TP"
                exit_px = tp
            else:
                # ні SL, ні TP — закриття бару (incomplete trade)
                outcome = "NONE"
                exit_px = float(nxt["c"])

            # PnL у «R»
            risk = (entry - sl) if side=="long" else (sl - entry)
            pnl  = (exit_px - entry) if side=="long" else (entry - exit_px)
            R    = pnl / risk if risk != 0 else 0.0

            # комісії (2 сторони)
            notional = abs(entry)  # спрощено
            fees = 2 * _apply_fees(notional, cfg)

            trades.append({
                "i": i,
                "side": side,
                "entry": entry,
                "exit": exit_px,
                "outcome": outcome,
                "R": R,
                "pnl_usdt": (R * cfg.risk_usdt) - fees,
            })

    if not trades:
        return {"trades": 0, "winrate": 0.0, "avg_R": 0.0, "sum_usdt": 0.0}

    tdf = pd.DataFrame(trades)
    wins = (tdf["R"] > 0).sum()
    return {
        "trades": len(tdf),
        "winrate": wins / len(tdf),
        "avg_R": tdf["R"].mean(),
        "sum_usdt": tdf["pnl_usdt"].sum(),
    }
