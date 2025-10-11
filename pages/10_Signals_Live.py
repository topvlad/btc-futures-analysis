# pages/10_Signals_Live.py — v0.5 (legend + signal cards + robust fetch)
import time
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

# ---------------- UI: Sidebar parameters ----------------
st.sidebar.header("Parameters (quick)")
colA, colB = st.sidebar.columns(2)
oi_min = colA.number_input("ΔOI% min (A)", min_value=0.05, max_value=5.0, value=0.10, step=0.05)
taker_min = colB.number_input("Taker ratio min (A)", min_value=1.00, max_value=3.00, value=1.10, step=0.05)
colC, colD = st.sidebar.columns(2)
prem_z = colC.number_input("Premium z-threshold (B)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
lookback_z = colD.number_input("Z lookback (bars)", min_value=10, max_value=200, value=60, step=5)

# ---------------- Page title + legend ----------------
st.title("⚡ LIVE Signals (1–5–15m)")
st.markdown("""
### 🧭 Legend & Quick Guide
**Setup A (Impulse + Participation)** — шукає імпульси з підтвердженням участі (taker-перевага).  
**Setup B (Premium Mean Reversion)** — коли премія фʼючерса надто відхилилась → очікуємо відкат до середньої.

- **ΔOI %** — зміна open interest (лише як hint у LIVE, може бути нестабільним на публічному API).  
- **Taker ratio** — `buySellRatio` (агресія ринку).  
- **Premium z-score** — Z-оцінка premium/basis (вище → сильніший потенціал mean-reversion).
""")
st.info("💡 Tip: комбінуй сигнали з головним дашбордом: якщо режим *Trend ↑* — пріоритезуй лонги; якщо *Trend ↓* — шорти.")

# ---------------- Controls ----------------
symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)
interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=0)
setup = st.radio("Setup", ["A: Impulse + Participation", "B: Premium Mean Reversion"], index=0, horizontal=True)

# ---------------- Data loader ----------------
@st.cache_data(ttl=45, show_spinner=False)
def load_klines_df(symbol: str, interval: str) -> pd.DataFrame:
    rows = fetch_klines(symbol, interval, limit=240)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Empty klines")
    df["ret"] = df["close"].pct_change()
    return df

# Helper: z-score over rolling window
def rolling_z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - m) / (sd.replace(0, np.nan))

# Card renderer
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

# ---------------- Live loop ----------------
ph_hdr = st.empty()
ph_tbl = st.empty()
st.markdown("### ⏱ Realtime feed (auto refresh ~10s)")

# keep small in-memory state for ΔOI
if "last_oi" not in st.session_state:
    st.session_state["last_oi"] = None

def try_signal_A() -> None:
    # ΔOI% (best-effort snapshot) + taker ratio filter + recent breakout confirmation via returns
    oi_now = None
    try:
        oi_now = fetch_open_interest(symbol)
    except Exception:
        pass

    d_oi_pct = None
    if oi_now is not None:
        last = st.session_state.get("last_oi")
        if last:
            d_oi_pct = (oi_now - last) / max(last, 1e-9) * 100.0
        st.session_state["last_oi"] = oi_now

    ratio, imb, _ = fetch_taker_longshort_ratio(symbol, interval, limit=20)
    # Basic direction: ratio>taker_min -> LONG bias; ratio<1/taker_min -> SHORT bias
    if ratio is None:
        st.info("Waiting for taker ratio…")
        return

    # Fetch recent 1 bar return for tiny impulse confirmation
    try:
        df = load_klines_df(symbol, interval)
        r = float(df["ret"].iloc[-1])
    except Exception:
        r = 0.0

    note_bits = []
    if d_oi_pct is not None:
        note_bits.append(f"ΔOI {d_oi_pct:+.2f}%")
    note_bits.append(f"taker ratio {ratio:.2f}")
    note_bits.append(f"1-bar ret {r:+.4f}")
    note = " · ".join(note_bits)

    # Generate signals
    if (d_oi_pct is None or d_oi_pct >= oi_min) and ratio >= taker_min and r > 0:
        render_card("LONG", d_oi_pct if d_oi_pct is not None else 0.0, ratio, note)
    elif (d_oi_pct is None or d_oi_pct >= oi_min) and ratio <= (1.0 / taker_min) and r < 0:
        render_card("SHORT", d_oi_pct if d_oi_pct is not None else 0.0, ratio, note)
    else:
        st.write("…no signal yet — waiting for conditions to trigger ⚙️")

def try_signal_B() -> None:
    # Pull last N premiums → z-score → LONG if deeply negative, SHORT if positive
    df = load_klines_df(symbol, interval)
    premiums = []
    # try to collect N values by calling endpoint each loop; acceptable for LIVE view
    # (if endpoint fails, we still show no-signal)
    for _ in range(min(lookback_z, 40)):  # limited calls this render
        val, _ts = fetch_premium_index_klines(symbol, interval)
        if val is not None:
            premiums.append(val)
        time.sleep(0.02)
    if len(premiums) < max(10, int(lookback_z * 0.5)):
        st.info("Not enough premium samples yet…")
        return
    s = pd.Series(premiums)
    z = (s.iloc[-1] - s.mean()) / (s.std() + 1e-12)

    note = f"premium z={z:.2f} over {len(s)} samples"
    if z >= prem_z and df["close"].iloc[-1] > df["close"].rolling(10).mean().iloc[-1]:
        render_card("SHORT", z, df["close"].iloc[-1], note)
    elif z <= -prem_z and df["close"].iloc[-1] < df["close"].rolling(10).mean().iloc[-1]:
        render_card("LONG", -z, df["close"].iloc[-1], note)
    else:
        st.write("…no signal yet — premium within normal band ⚙️")

# Render loop (5 minutes ~30 iterations)
for _ in range(30):
    with ph_hdr.container():
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
