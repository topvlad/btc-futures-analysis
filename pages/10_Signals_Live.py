# pages/10_Signals_Live.py — v0.3
import streamlit as st, pandas as pd, numpy as np, time
from datetime import datetime, timezone
from core.data_sources import (
    fetch_klines, fetch_open_interest,
    fetch_taker_longshort_ratio, fetch_premium_index_klines
)

st.set_page_config(page_title="⚡ LIVE Signals", layout="wide")

# ============ Sidebar ============ #
st.sidebar.header("Parameters (quick)")
colA, colB = st.sidebar.columns(2)
oi_min = colA.number_input("ΔOI% min (A)", 0.05, 5.0, 0.10, 0.05)
oi_max = colB.number_input("", 0.5, 5.0, 2.00, 0.1)
colC, colD = st.sidebar.columns(2)
taker_min = colC.number_input("Taker ratio min (A)", 1.0, 3.0, 1.00, 0.05)
taker_max = colD.number_input("", 1.0, 3.0, 1.50, 0.05)
colE, colF = st.sidebar.columns(2)
prem_z = colE.number_input("Premium z-threshold (B)", 0.5, 6.0, 1.00, 0.1)
prem_z_hi = colF.number_input("", 0.5, 6.0, 4.00, 0.1)

# ============ Legend ============ #
st.title("⚡ LIVE Signals (1–5–15m)")
st.markdown("""
### 🧭 Legend & Quick Guide
**Setup A (Impulse + Participation)** — імпульси, коли Open Interest різко росте, а переважають агресивні лонги/шорти.  
**Setup B (Premium Mean Reversion)** — контр-сигнал: коли премія фʼючерсу надто велика — очікується відкат.

- **ΔOI %** — зміна open interest між свічками (імпульс).  
- **Taker ratio** — перевага агресорів (лонги/шорти).  
- **Premium z-score** — наскільки премія вийшла за норму.  
- **Signal strength** — зелений/червоний колір = сила напрямку.
""")
st.info("💡 Tip: комбінуй ці сигнали з головним BTC-дашбордом — якщо режим ринку *Trend ↑*, пріоритезуй лонги; якщо *Trend ↓* — шорти.")

# ============ Core logic ============ #
symbol = st.selectbox("Symbol", ["BTCUSDT", "ETHUSDT"], index=0)
interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=0)
setup = st.radio("Setup", ["A: Impulse + Participation", "B: Premium Mean Reversion"])

@st.cache_data(ttl=120, show_spinner=False)
def _load_baseline(symbol, interval):
    kl = fetch_klines(symbol, interval, limit=200)
    df = pd.DataFrame(kl)
    df["return"] = df["close"].pct_change()
    return df

try:
    df0 = _load_baseline(symbol, interval)
except Exception as e:
    st.error(f"Cannot load klines: {e}")
    st.stop()

def signal_a(df):
    try:
        oi_now = fetch_open_interest(symbol)
        oi_prev = getattr(signal_a, "oi_prev", oi_now)
        signal_a.oi_prev = oi_now
        delta_oi = (oi_now - oi_prev) / max(oi_prev, 1e-9) * 100
        ratio, acc_ratio, _ = fetch_taker_longshort_ratio(symbol, interval)
        cond_up = delta_oi >= oi_min and ratio >= taker_min
        cond_down = delta_oi >= oi_min and ratio <= (1 / taker_min)
        if cond_up:
            return "LONG", delta_oi, ratio
        elif cond_down:
            return "SHORT", delta_oi, ratio
    except Exception:
        return None, None, None
    return None, None, None

def signal_b(df):
    try:
        prem, _ = fetch_premium_index_klines(symbol, interval)
        prem_series = getattr(signal_b, "prem_hist", [])
        prem_series.append(prem)
        if len(prem_series) > 20: prem_series.pop(0)
        signal_b.prem_hist = prem_series
        z = (prem - np.mean(prem_series)) / (np.std(prem_series) + 1e-9)
        if abs(z) >= prem_z:
            return ("SHORT" if z > 0 else "LONG"), prem, z
    except Exception:
        return None, None, None
    return None, None, None

def card(signal, val1, val2):
    color = "#c8ffd0" if signal == "LONG" else "#ffd6d6"
    emoji = "🚀" if signal == "LONG" else "🔻"
    st.markdown(f"""
    <div style='background:{color};padding:1em;border-radius:12px;margin:5px 0'>
      <b style='font-size:22px'>{emoji} {signal} signal</b><br>
      <span style='font-size:15px'>
        ΔOI / Premium: <b>{val1:.3f}</b><br>
        Ratio / z-score: <b>{val2:.3f}</b><br>
        <i>{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}</i>
      </span>
    </div>
    """, unsafe_allow_html=True)

# ============ Live loop ============ #
ph = st.empty()
st.markdown("### ⏱ Realtime feed (auto-refresh ≈ 10 s)")

for _ in range(30):  # 5 min live
    df = _load_baseline(symbol, interval)
    if setup.startswith("A"):
        sig, v1, v2 = signal_a(df)
    else:
        sig, v1, v2 = signal_b(df)
    if sig:
        card(sig, v1, v2)
    else:
        st.write("…no signal yet — waiting for conditions to trigger ⚙️")
    time.sleep(10)
