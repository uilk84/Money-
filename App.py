import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
import time

# --- Page Config ---
st.set_page_config(page_title="Penny Stock Live Scanner", layout="wide")
st.title("🚀 Penny Stock Live Scanner with Real Trading (Yahoo Finance + Alpaca)")
st.caption("Live AI Buy/Sell Signals & Order Execution (Refactored)")

# --- User Inputs ---
tickers = st.text_input("🔎 Enter up to 10 Tickers (comma separated):", value="RELI, MSTY, CGTX").upper().replace(" ", "").split(",")[:10]
buying_power = st.number_input("💵 Buying Power ($)", min_value=0.0, value=20.0)
refresh_rate = st.slider("⏱️ Auto Refresh (seconds)", min_value=10, max_value=300, value=60, step=10)

# --- Alpaca Credentials ---
st.sidebar.header("Alpaca API Settings")
alpaca_key = st.sidebar.text_input("Alpaca API Key", type="password")
alpaca_secret = st.sidebar.text_input("Alpaca Secret Key", type="password")
paper = st.sidebar.checkbox("Use Paper Trading?", value=True)
base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

def alpaca_headers():
    return {
        "APCA-API-KEY-ID": alpaca_key,
        "APCA-API-SECRET-KEY": alpaca_secret,
        "Content-Type": "application/json"
    }

# --- Technical Indicators ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    return df

def smart_trade_signal(df):
    row = df.iloc[-1]
    reasons = []
    if row['Close'] > row['EMA9']:
        reasons.append("Price above EMA9")
    if row['RSI'] < 30:
        reasons.append("RSI < 30 (Oversold)")
    if row['RSI'] > 70:
        reasons.append("RSI > 70 (Overbought)")
    if row['Close'] > row['EMA9'] and 35 < row['RSI'] < 65:
        return "BUY", "#10c469", "; ".join(reasons)
    elif row['RSI'] > 70:
        return "SELL", "#ff5b5b", "; ".join(reasons)
    else:
        return "WATCH", "#ffaa00", "; ".join(reasons or ["No strong signal"])

# --- Fetch Data ---
@st.cache_data(ttl=60)
def fetch_data(tickers):
    raw_data = yf.download(tickers, period="2d", interval="1m", group_by='ticker', threads=True, progress=False)
    results = []
    for t in tickers:
        try:
            df = raw_data[t] if len(tickers) > 1 else raw_data
            df = df.dropna(subset=['Close'])
            df = compute_indicators(df)
            current = df['Close'].iloc[-1]
            openp = df['Open'].iloc[0]
            volume = df['Volume'].iloc[-1]
            change = ((current - openp) / openp) * 100
            shares = int(buying_power // current)
            signal, color, reason = smart_trade_signal(df)
            results.append({
                "Ticker": t,
                "Price": round(current, 4),
                "% Change": round(change, 2),
                "Volume": int(volume),
                "Shares": shares,
                "Signal": signal,
                "Color": color,
                "Reason": reason,
                "Chart": df['Close'][-50:]
            })
        except Exception as e:
            results.append({
                "Ticker": t,
                "Price": "N/A",
                "% Change": "N/A",
                "Volume": "N/A",
                "Shares": "N/A",
                "Signal": "ERROR",
                "Color": "#666",
                "Reason": str(e),
                "Chart": pd.Series()
            })
    return pd.DataFrame(results)

# --- Order Function ---
def place_order(symbol, qty, side, type="market", time_in_force="gtc"):
    url = f"{base_url}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": type,
        "time_in_force": time_in_force
    }
    r = requests.post(url, json=payload, headers=alpaca_headers())
    return r.json()

# --- Chart Helper ---
def plot_sparkline(prices, ticker):
    fig, ax = plt.subplots(figsize=(4, 1))
    ax.plot(prices, color='dodgerblue')
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{ticker} Trend", fontsize=8)
    st.pyplot(fig)

# --- Autorefresh Trigger ---
st_autorefresh(interval=refresh_rate * 1000, key="scanner_refresh")

# --- Data Display ---
df = fetch_data(tickers)
st.markdown(f"🕒 Last updated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")

for i, row in df.iterrows():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            f"""
            <div style="padding: 10px; border: 1px solid #ccc; border-radius: 8px;">
            <strong style='font-size: 22px'>{row['Ticker']}</strong><br>
            💵 ${row['Price']} | 
            📈 <span style='color: {"#10c469" if isinstance(row["% Change"], float) and row["% Change"] > 0 else "#ff5b5b"}'>{row['% Change']}%</span> | 
            📊 Vol: {row['Volume']}<br>
            🧮 Shares: {row['Shares']}<br>
            <strong style='color: {row["Color"]}'>{row["Signal"]}</strong><br>
            <span style='font-size: 12px; color: #888'>{row["Reason"]}</span>
            </div>
            """, unsafe_allow_html=True)
        if not row['Chart'].empty:
            plot_sparkline(row['Chart'], row['Ticker'])

    with col2:
        if row['Signal'] in ["BUY", "SELL"] and alpaca_key and alpaca_secret:
            trade_type = "buy" if row['Signal'] == "BUY" else "sell"
            if st.button(f"{trade_type.capitalize()} {row['Ticker']}", key=f"{trade_type}_{i}"):
                result = place_order(row['Ticker'], row['Shares'], trade_type)
                st.success(f"{trade_type.capitalize()} order sent: {result.get('status', 'error')}")
