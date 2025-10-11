# pages/10_Signals_Live.py — v0.6
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

# --------------- Sidebar params ---------------
st.sidebar.header("Parameters (quick)")
colA, colB = st.sidebar.columns(2)
oi_min = colA.number_input("ΔOI% min (A)", 0.05, 5.0, 0.10, 0.05)
taker_min = colB.number_input("Taker ratio min (A)", 1.00, 3.00, 1.10, 0.05)
colC, colD = st.sidebar.columns(2)
prem_z = colC.number_input("Premium z-threshold (B)", 0.5, 6.0, 2.0, 0.1)
z_win = colD.number_input("Z lookback (bars)", 10, 200, 60, 5)

# --------------- Title + legend ---------------
st.title("⚡ LIVE Signals (1–5–15m)")
st.markdown("""
### 🧭 Legend & Quick Guide
**Setup A (Impulse + Participation)** — імпульс + перевага агресорів (taker).  
**Setup B (Premium Mean Reversion)** — велика премія → очікуваний відкат.
""")
st.info("💡 Зіставляй із головним дашбордом: *Trend ↑* → пріоритет LONG; *Trend ↓* → SHORT.")

# --------------- Controls ---------------
symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)
interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=0)
setup = st.radio("Setup", ["A: Impulse + Participation", "B: Premium Mean Reversion"], horizontal=True)

# period for taker endpoint (1m → 5m тощо)
def _ratio_period(tf: str) -> str:
    return {"1m":"5m","5m":"5m","15m":"15m"}.get(tf, "5m")

# --------------- Load klines ---------------
@st.cache_data(ttl=45, show_spinner=False)
def load_klines_df(symbol: str, interval: str) -> pd.DataFrame:
    rows = fetch_klines(symbol, interval, limit=240)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Empty klines")
    df["ret"] = df["close"].pct_change()
    return df

# --------------- Helpers ---------------
def render_card(kind: str, v1: float, v2: float, note: str = ""):
    color = "#c8ffd0" if kind == "LONG" else "#ffd6d6"
    emoji = "🚀" if kind == "LONG" else "🔻"
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.markdown(
        f"""
<div style='background:{color};padding:1em;border-radius:14px;margin:6px 0;border:1px solid #eee'>
  <div style='font-size:22px'><b>{emoji} {kind} signal</b></div>
  <div style='font-size:15px;opacity:0.9'>
    <div>Metric #1: <b>{v1:.4f}</b></div>
    <div>Metric #2: <b>{v2:.4f}</b></div>
    {"<div style='margin-top:6px;opacity:0.8'>" + note + "</div>" if note else ""}
    <div style='margin-top:6px'><i>{now}</i></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# keep session state for ΔOI and premium z
if "last_oi" not in st.session_state:
    st.session_state.last_oi = None
if "prem_hist" not in st.session_state:
    st.session_state.prem_hist = collections.deque(maxlen=int(z_win))

def try_signal_A() -> None:
    # ΔOI% (best-effort snapshot) + taker ratio + 1-bar impulse
    d_oi = None
    try:
        oi_now = fetch_open_interest(symbol)
        if st.session_state.last_oi:
            d_oi = (oi_now - st.session_state.last_oi) / max(st.session_state.last_oi, 1e-9) * 100.0
        st.session_state.last_oi = oi_now
    except Exception:
        pass

    ratio, imb, _ = fetch_taker_longshort_ratio(symbol, _ratio_period(interval), limit=20)
    if ratio is None:
        st.info("Waiting for taker ratio…")
        return

    try:
        df = load_klines_df(symbol, interval)
        r = float(df["ret"].iloc[-1])
    except Exception as e:
        st.warning(f"Price feed degraded (retrying): {e}")
        r = 0.0

    note_bits = []
    if d_oi is not None:
        note_bits.append(f"ΔOI {d_oi:+.2f}%")
    note_bits.append(f"taker ratio {ratio:.2f}")
    note_bits.append(f"1-bar ret {r:+.4f}")
    note = " · ".join(note_bits)

    if (d_oi is None or d_oi >= oi_min) and ratio >= taker_min and r > 0:
        render_card("LONG", (0.0 if d_oi is None else d_oi), ratio, note)
    elif (d_oi is None or d_oi >= oi_min) and ratio <= (1.0 / taker_min) and r < 0:
        render_card("SHORT", (0.0 if d_oi is None else d_oi), ratio, note)
    else:
        st.write("…no signal yet — waiting for conditions to trigger ⚙️")

def try_signal_B() -> None:
    # накопичуємо premium у deque → стабільний z-score
    val, _ts = fetch_premium_index_klines(symbol, interval)
    if val is not None:
        st.session_state.prem_hist.append(float(val))

    if len(st.session_state.prem_hist) < max(10, int(z_win*0.5)):
        st.info("Collecting premium samples…")
        return

    s = np.array(st.session_state.prem_hist, dtype=float)
    z = (s[-1] - s.mean()) / (s.std() + 1e-12)

    # простий фільтр напрямку через ковзну середню ціни
    try:
        df = load_klines_df(symbol, interval)
        ma = df["close"].rolling(10).mean().iloc[-1]
        px = df["close"].iloc[-1]
    except Exception:
        ma, px = None, None

    note = f"premium z={z:.2f} over {len(s)} samples"
    if z >= prem_z and (ma is None or px > ma):
        render_card("SHORT", z, (px or 0.0), note)
    elif z <= -prem_z and (ma is None or px < ma):
        render_card("LONG", -z, (px or 0.0), note)
    else:
        st.write("…no signal yet — premium within normal band ⚙️")

# --------------- Header chart + live loop ---------------
hdr = st.empty()
st.markdown("### ⏱ Realtime feed (auto refresh ~10s)")

for _ in range(30):  # ~5 хвилин
    with hdr.container():
        try:
            dfv = load_klines_df(symbol, interval)
            st.line_chart(dfv.set_index("ts")[["close"]].tail(200), height=180, use_container_width=True)
        except Exception as e:
            st.warning(f"Price feed degraded (will retry): {e}")

    if setup.startswith("A"):
        try_signal_A()
    else:
        try_signal_B()
    time.sleep(10)
