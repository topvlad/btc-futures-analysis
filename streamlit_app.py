import pandas as pd
import json, requests, streamlit as st

URL_DEFAULT = "https://topvlad.github.io/btc-futures-analysis/report.json"

st.title("BTC Futures — EMA/RSI/MACD (15m, 1h, 4h, 1d)")
pages_url = st.text_input("GitHub Pages JSON URL:", value=URL_DEFAULT)

@st.cache_data(ttl=300)
def fetch_report(url: str):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    payload = json.loads(r.content.decode("utf-8"))
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError("Некоректна структура JSON (очікував {generated_at, stale, data}).")
    return payload

if pages_url:
    try:
        payload = fetch_report(pages_url)
        if payload.get("stale"):
            st.warning("Дані позначені як STALE. Показуємо, що є.")

        rows = payload.get("data", [])
        df = pd.json_normalize(rows)  # зручно, навіть якщо є помилки в окремих рядках

        if df.empty:
            st.warning("Порожні дані в report.json.")
        else:
            st.success("JSON завантажено ✅")
            st.dataframe(df, use_container_width=True)

            ok = df[df["error"].isna()] if "error" in df.columns else df
            if not ok.empty:
                pivot = ok.pivot_table(
                    index=["symbol","tf"],
                    values=["last_close","ema20","ema50","ema200","rsi14","macd","macd_signal","macd_hist"],
                    aggfunc="first"
                )
                st.subheader("Pivot view")
                st.dataframe(pivot, use_container_width=True)

                sym = st.selectbox("Symbol", sorted(ok["symbol"].dropna().unique()))
                tf  = st.selectbox("TF", sorted(ok[ok["symbol"]==sym]["tf"].dropna().unique()))
                row = ok[(ok.symbol==sym) & (ok.tf==tf)].iloc[0]
                st.markdown(
                    f"**{sym} {tf}** — close: {row['last_close']:.2f}, "
                    f"EMA20/50/200: {row['ema20']:.2f}/{row['ema50']:.2f}/{row['ema200']:.2f}, "
                    f"RSI14: {row['rsi14']:.2f}, "
                    f"MACD: {row['macd']:.4f}/{row['macd_signal']:.4f}/{row['macd_hist']:.4f}"
                )
            else:
                st.info("Усі рядки з помилками. Перевір логи GitHub Actions.")
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
