# pages/10_Signals_Live.py — v1.1
import time, collections
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st

from core.data_sources import (
    fetch_klines,
    fetch_open_interest,
    fetch_taker_longshort_ratio,
    fetch_premium_index_klines,
)

st.set_page_config(page_title="⚡ LIVE Signals", layout="wide")

# ---------------- Sidebar parameters ----------------
st.sidebar.header("Parameters (quick)")
colA, colB = st.sidebar.columns(2)
oi_min = colA.number_input("ΔOI% min (A)", min_value=0.05, max_value=5.0, value=0.10, step=0.05)
taker_min = colB.number_input("Taker ratio min (A)", min_value=1.00, max_value=3.00, value=1.10, step=0.05)
colC, colD = st.sidebar.columns(2)
prem_z = colC.number_input("Premium z-threshold (B)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
z_win = colD.number_input("Z lookback (bars)", min_value=10, max_value=200, value=60, step=5)

# ---------------- Legend / guide ----------------
st.title("⚡ LIVE Signals (1–5–15m)")
st.markdown("""
### 🧭 Legend
**Setup A (Impulse + Participation)** — імпульс свічі + перевага агресорів (taker).  
**Setup B (Premium Mean Reversion)** — велика премія → очікуваний відкат.

- ΔOI% — best-effort зміна OI між опитуваннями (snapshot).  
- Taker ratio — `buySellRatio` або fallback, інакше використовуємо імпульс ціни як проксі.  
- Premium z-score — z-оцінка індексу премії в deque.
""")
st.info("💡 Узгоджуй із головним дашбордом: *Trend ↑* → LONG-пріоритет; *Trend ↓* → SHORT-пріоритет.")

# ---------------- Controls ----------------
symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)
interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=0)
setup = st.radio("Setup", ["A: Impulse + Participation", "B: Premium Mean Reversion"], horizontal=True)

# ---------------- Utils ----------------
def _ratio_period(tf: str) -> str:
    return {"1m":"5m","5m":"5m","15m":"15m"}.get(tf, "5m")

def _render_card(side: str, m1: float, m2: float, note: str = ""):
    color = "#c8ffd0" if side == "LONG" else "#ffd6d6"
    emoji = "🚀" if side == "LONG" else "🔻"
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.markdown(
        f"""
<div style='background:{color};padding:1em;border-radius:14px;margin:6px 0;border:1px solid #eee'>
  <div style='font-size:22px'><b>{emoji} {side} signal</b></div>
  <div style='font-size:15px;opacity:0.9'>
    <div>Metric #1: <b>{m1:.4f}</b></div>
    <div>Metric #2: <b>{m2:.4f}</b></div>
    {"<div style='margin-top:6px;opacity:0.8'>" + note + "</div>" if note else ""}
    <div style='margin-top:6px'><i>{now}</i></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

@st.cache_data(ttl=45, show_spinner=False)
def _load_klines_df(symbol: str, interval: str) -> pd.DataFrame:
    rows = fetch_klines(symbol, interval, limit=240)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ret"] = df["close"].pct_change()
    return df

# session state for ΔOI and premium z
if "last_oi" not in st.session_state: st.session_state.last_oi = None
if "prem_hist" not in st.session_state: st.session_state.prem_hist = collections.deque(maxlen=int(z_win))

# ---------------- Signal engines ----------------
def _try_signal_A() -> None:
    # best-effort ΔOI%
    d_oi = None
    oi_now = fetch_open_interest(symbol)
    if oi_now is not None:
        last = st.session_state.last_oi
        if last: d_oi = (oi_now - last) / max(last, 1e-9) * 100.0
        st.session_state.last_oi = oi_now

    # taker ratio (with internal fallback in data_sources; may still return None)
    ratio, imb, _ = fetch_taker_longshort_ratio(symbol, _ratio_period(interval), limit=20)

    # price impulse proxy
    df = _load_klines_df(symbol, interval)
    if df.empty:
        st.warning("Price feed degraded (will retry).")
        return
    r1 = float(df["ret"].iloc[-1])
    r3 = float(df["ret"].tail(3).sum())
    # if taker unavailable, synthesize a proxy from price impulse
    ratio_eff = ratio if ratio is not None else (1.0 + max(0.0, r3*100))  # mild >1 on up-impulse

    note = " · ".join([
        (f"ΔOI {d_oi:+.2f}%" if d_oi is not None else "ΔOI n/a"),
        f"taker ratio {ratio_eff:.2f}" + (" (proxy)" if ratio is None else ""),
        f"1-bar ret {r1:+.4f}; 3-bar {r3:+.4f}",
    ])

    if (d_oi is None or d_oi >= oi_min) and ratio_eff >= taker_min and r1 > 0:
        _render_card("LONG", (0.0 if d_oi is None else d_oi), ratio_eff, note)
    elif (d_oi is None or d_oi >= oi_min) and ratio_eff <= (1.0 / taker_min) and r1 < 0:
        _render_card("SHORT", (0.0 if d_oi is None else d_oi), ratio_eff, note)
    else:
        st.write("…no signal yet — waiting for conditions to trigger ⚙️")

def _try_signal_B() -> None:
    prem, _ = fetch_premium_index_klines(symbol, interval)
    if prem is not None:
        st.session_state.prem_hist.append(float(prem))

    if len(st.session_state.prem_hist) < max(10, int(z_win * 0.5)):
        st.info("Collecting premium samples…")
        return

    s = np.array(st.session_state.prem_hist, dtype=float)
    z = (s[-1] - s.mean()) / (s.std() + 1e-12)

    df = _load_klines_df(symbol, interval)
    ma = df["close"].rolling(10).mean().iloc[-1] if not df.empty else None
    px = df["close"].iloc[-1] if not df.empty else None

    note = f"premium z={z:.2f} over {len(s)} samples"
    if z >= prem_z and (ma is None or (px and px > ma)):
        _render_card("SHORT", z, (px or 0.0), note)
    elif z <= -prem_z and (ma is None or (px and px < ma)):
        _render_card("LONG", -z, (px or 0.0), note)
    else:
        st.write("…no signal yet — premium within normal band ⚙️")

# ---------------- Header chart + live poll loop ----------------
hdr = st.empty()
st.markdown("### ⏱ Realtime feed (auto refresh ~10s)")

for _ in range(24):  # ~4 minutes
    with hdr.container():
        dfv = _load_klines_df(symbol, interval)
        if not dfv.empty:
            st.line_chart(dfv.set_index("ts")[["close"]].tail(200), height=180, use_container_width=True)
        else:
            st.warning("Price feed degraded (will retry).")

    (_try_signal_A if setup.startswith("A") else _try_signal_B)()
    time.sleep(10)
