import streamlit as st
import pandas as pd

st.set_page_config(page_title="BTC Futures MTF Dashboard", layout="wide")

st.title("BTC Futures — EMA/RSI/MACD (15m, 1h, 4h, 1d)")

pages_url = st.text_input(
    "GitHub Pages JSON URL:",
    value="https://<логін>.github.io/btc-futures-analysis/report.json"
)

@st.cache_data(ttl=300)
def load_data(url):
    return pd.read_json(url)

if pages_url:
    try:
        df = load_data(pages_url)
        st.dataframe(df.sort_values(["symbol","tf"]), use_container_width=True)

        # Приклад простого зведення по символу/TF:
        pivot = df.pivot_table(index=["symbol","tf"], values=["close","ema20","ema50","ema200","rsi14","macd","signal","hist"], aggfunc="first")
        st.subheader("Pivot view")
        st.dataframe(pivot, use_container_width=True)

        # Фільтри
        sym = st.selectbox("Symbol", sorted(df["symbol"].unique()))
        tf  = st.selectbox("TF", sorted(df["tf"].unique()))
        row = df[(df.symbol==sym)&(df.tf==tf)].iloc[0]
        st.markdown(f"**{sym} {tf}** — close: {row['close']}, EMA20/50/200: {row['ema20']}/{row['ema50']}/{row['ema200']}, RSI14: {row['rsi14']}, MACD: {row['macd']}/{row['signal']}/{row['hist']}")
    except Exception as e:
        st.error(f"Failed to load JSON: {e}")
