# streamlit_app.py
import json
import requests
import pandas as pd
import streamlit as st

URL_DEFAULT = "https://topvlad.github.io/btc-futures-analysis/report.json"

st.set_page_config(page_title="BTC Futures — EMA/RSI/MACD", layout="wide")
st.title("BTC Futures — EMA/RSI/MACD (15m, 1h, 4h, 1d)")

pages_url = st.text_input("GitHub Pages JSON URL:", value=URL_DEFAULT)

@st.cache_data(ttl=300)
def fetch_report(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    payload = json.loads(r.content.decode("utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Некоректна структура JSON (очікував {generated_at, stale, data}).")
    return payload

def badge(text: str):
    # Простий бейдж через markdown
    return f"<span style='padding:2px 8px;border-radius:12px;background:#eee;border:1px solid #ccc;font-size:12px;'>{text}</span>"

if pages_url:
    try:
        payload = fetch_report(pages_url)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.success("JSON завантажено ✅")
            st.write(f"**generated_at (UTC):** `{payload.get('generated_at','?')}`")
        with col_b:
            if payload.get("stale"):
                st.warning("Дані позначені як **STALE**. Показуємо, що є.")

        rows = payload.get("data", [])
        df = pd.json_normalize(rows)

        if df.empty:
            st.warning("Порожні дані в report.json.")
            st.stop()

        # Показати сирі дані
        with st.expander("Показати raw-дані (таблиця)", expanded=False):
            st.dataframe(df, use_container_width=True)

        # Відмітити джерела
        if "source" in df.columns:
            sources = sorted([str(s) for s in df["source"].dropna().unique()])
            src_line = " ".join(badge(s) for s in sources)
            st.markdown(f"**Джерела в цьому звіті:** {src_line}", unsafe_allow_html=True)

            # Попередження, якщо не основне джерело
            non_primary = df[df["source"].astype(str).str.contains("markPrice|vision", case=False, na=False)]
            if not non_primary.empty:
                st.warning(
                    "Частина даних отримана з **неосновних джерел**: "
                    "`markPriceKlines` (індикативна ціна) та/або `binance_vision` (архів). "
                    "Ці ряди придатні для огляду, але можуть відрізнятися від last-trade цін."
                )

        # Відібрати коректні рядки
        ok = df[df["error"].isna()] if "error" in df.columns else df
        if ok.empty:
            st.info("Усі рядки з помилками. Перевір логи GitHub Actions.")
            st.stop()

        # Зручний огляд метрик
        metric_cols = ["last_close", "ema20", "ema50", "ema200", "rsi14", "macd", "macd_signal", "macd_hist"]
        present_metrics = [c for c in metric_cols if c in ok.columns]

        st.subheader("Зведена таблиця (pivot)")
        pivot = ok.pivot_table(
            index=["symbol", "tf"],
            values=present_metrics,
            aggfunc="first"
        ).sort_index()
        st.dataframe(pivot, use_container_width=True)

        # Селектори та короткий річап
        st.subheader("Деталі по вибраному інструменту")
        sym = st.selectbox("Symbol", sorted(ok["symbol"].dropna().unique()))
        tf_options = sorted(ok[ok["symbol"] == sym]["tf"].dropna().unique(), key=lambda x: ["15m","1h","4h","1d"].index(x) if x in ["15m","1h","4h","1d"] else x)
        tf = st.selectbox("TF", tf_options)

        row = ok[(ok.symbol == sym) & (ok.tf == tf)].iloc[0].to_dict()

        src = str(row.get("source") or row.get("route") or "?")
        st.markdown(f"**Джерело:** {badge(src)}", unsafe_allow_html=True)

        if "last_close" in row:
            st.markdown(
                f"**{sym} {tf}** — "
                f"close: `{row['last_close']:.2f}`; "
                f"EMA20/50/200: "
                f"`{row.get('ema20', float('nan')):.2f}` / "
                f"`{row.get('ema50', float('nan')):.2f}` / "
                f"`{row.get('ema200', float('nan')):.2f}`; "
                f"RSI14: `{row.get('rsi14', float('nan')):.2f}`; "
                f"MACD: `{row.get('macd', float('nan')):.4f}` / "
                f"`{row.get('macd_signal', float('nan')):.4f}` / "
                f"`{row.get('macd_hist', float('nan')):.4f}`"
            )

        # Текстові індикатори-кроси (якщо присутні)
        crosses = []
        if "price_vs_ema200" in row:
            crosses.append(f"Price vs EMA200: **{row['price_vs_ema200']}**")
        if "ema20_cross_50" in row:
            crosses.append(f"EMA20 vs EMA50: **{row['ema20_cross_50']}**")
        if "macd_cross" in row:
            crosses.append(f"MACD cross: **{row['macd_cross']}**")

        if crosses:
            st.markdown(" — ".join(crosses))

        # Показати можливі помилки рядків (щоб розуміти, що саме відвалилось)
        err = df[~df["error"].isna()] if "error" in df.columns else pd.DataFrame()
        if not err.empty:
            with st.expander("Є рядки з помилками — переглянути", expanded=False):
                st.dataframe(err, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
