import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="BTC Futures MTF Dashboard", layout="wide")
st.title("BTC Futures — EMA/RSI/MACD (15m, 1h, 4h, 1d)")

pages_url = st.text_input(
    "GitHub Pages JSON URL:",
    value="https://topvlad.github.io/btc-futures-analysis/report.json"  # <-- ПІДСТАВ СВІЙ
)

@st.cache_data(ttl=300)
def load_data(url: str) -> pd.DataFrame:
    # Базова валідація URL (не допускаємо кутові дужки та пробіли)
    if "<" in url or ">" in url or " " in url:
        raise ValueError("URL містить недопустимі символи. Підстав реальний username і прибери <>.")

    r = requests.get(url, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: не вдалося отримати JSON (можливо, Pages ще не опубліковані).")

    ctype = r.headers.get("Content-Type", "")
    text = r.text.strip()

    # Якщо це HTML (404 або index.html), підкажемо коректно
    if "html" in ctype.lower() or text.startswith("<!DOCTYPE") or text.startswith("<html"):
        raise RuntimeError("Схоже, за URL повертається HTML (ймовірно 404). Перевір, що report.json доступний за прямим лінком.")

    # Пробуємо спочатку як JSON через requests
    try:
        data = r.json()
        return pd.DataFrame(data)
    except Exception:
        # Фолбек на pandas (раптом там JSON Lines чи інший формат)
        try:
            return pd.read_json(text)
        except Exception as e:
            raise RuntimeError(f"Не вдалося розпарсити JSON: {e}")

if pages_url:
    try:
        df = load_data(pages_url)
        if df.empty:
            st.warning("JSON порожній. Перевір логи GitHub Actions: чи згенерувався report.json?")
        else:
            st.success("JSON завантажено ✅")
            st.dataframe(df.sort_values(["symbol","tf"]), use_container_width=True)

            pivot = df.pivot_table(index=["symbol","tf"],
                                   values=["close","ema20","ema50","ema200","rsi14","macd","signal","hist"],
                                   aggfunc="first")
            st.subheader("Pivot view")
            st.dataframe(pivot, use_container_width=True)

            sym = st.selectbox("Symbol", sorted(df["symbol"].unique()))
            tf  = st.selectbox("TF", sorted(df["tf"].unique()))
            row = df[(df.symbol==sym)&(df.tf==tf)].iloc[0]
            st.markdown(
                f"**{sym} {tf}** — close: {row['close']}, "
                f"EMA20/50/200: {row['ema20']}/{row['ema50']}/{row['ema200']}, "
                f"RSI14: {row['rsi14']}, "
                f"MACD: {row['macd']}/{row['signal']}/{row['hist']}"
            )
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
        st.info("Перевір: 1) правильність URL (без <>), 2) чи пройшов деплой Pages, 3) чи публічний репозиторій, 4) логи Actions.")
