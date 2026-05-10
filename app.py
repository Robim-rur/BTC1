# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="BTC Quant Breakout Lab",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚀 BTC Quant Breakout Lab")
st.markdown("""
Laboratório quantitativo para detectar setups estatisticamente robustos no gráfico diário do Bitcoin.

### Estrutura do setup:
- Tendência forte
- Compressão de volatilidade
- Breakout
- Gain fixo de +3%
- Sem stop loss
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Configurações")

ticker = st.sidebar.selectbox(
    "Ativo",
    ["BTC-USD"]
)

period = st.sidebar.selectbox(
    "Período Histórico",
    ["2y", "5y", "10y"],
    index=2
)

target_gain = st.sidebar.slider(
    "Gain (%)",
    1.0,
    10.0,
    3.0,
    0.5
)

max_hold = st.sidebar.slider(
    "Máximo de candles para atingir alvo",
    1,
    30,
    10
)

adx_min = st.sidebar.slider(
    "ADX mínimo",
    10,
    50,
    22
)

compression_window = st.sidebar.slider(
    "Janela compressão ATR",
    3,
    20,
    5
)

breakout_lookback = st.sidebar.slider(
    "Rompimento de máxima",
    2,
    20,
    3
)

# =========================================================
# DOWNLOAD DATA
# =========================================================

@st.cache_data
def load_data(ticker, period):
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True
    )

    df.dropna(inplace=True)

    return df

df = load_data(ticker, period)

# =========================================================
# INDICATORS
# =========================================================

df["EMA50"] = ta.trend.ema_indicator(df["Close"], window=50)
df["EMA200"] = ta.trend.ema_indicator(df["Close"], window=200)

adx = ta.trend.ADXIndicator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

df["ADX"] = adx.adx()
df["DI_POS"] = adx.adx_pos()
df["DI_NEG"] = adx.adx_neg()

atr = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

df["ATR"] = atr.average_true_range()

# Bollinger Width
bb = ta.volatility.BollingerBands(
    close=df["Close"],
    window=20,
    window_dev=2
)

df["BB_HIGH"] = bb.bollinger_hband()
df["BB_LOW"] = bb.bollinger_lband()
df["BB_WIDTH"] = (
    (df["BB_HIGH"] - df["BB_LOW"]) / df["Close"]
)

# =========================================================
# SETUP CONDITIONS
# =========================================================

# Tendência
trend_filter = (
    (df["Close"] > df["EMA50"]) &
    (df["EMA50"] > df["EMA200"])
)

# Força
strength_filter = (
    (df["ADX"] > adx_min) &
    (df["DI_POS"] > df["DI_NEG"])
)

# Compressão ATR
df["ATR_MEAN"] = df["ATR"].rolling(50).mean()

compression_filter = (
    df["ATR"] <
    df["ATR_MEAN"] * 0.8
)

# Compressão Bollinger
df["BB_WIDTH_MEAN"] = df["BB_WIDTH"].rolling(50).mean()

bb_compression = (
    df["BB_WIDTH"] <
    df["BB_WIDTH_MEAN"] * 0.8
)

# Breakout
rolling_high = df["High"].rolling(breakout_lookback).max().shift(1)

breakout_filter = (
    df["Close"] > rolling_high
)

# Volume
df["VOL_MA20"] = df["Volume"].rolling(20).mean()

volume_filter = (
    df["Volume"] > df["VOL_MA20"]
)

# Setup Final
df["SIGNAL"] = (
    trend_filter &
    strength_filter &
    (compression_filter | bb_compression) &
    breakout_filter &
    volume_filter
)

# =========================================================
# BACKTEST
# =========================================================

results = []

signals = df[df["SIGNAL"]].copy()

for idx in signals.index:

    entry_price = df.loc[idx, "Close"]

    future = df.loc[idx:].iloc[1:max_hold + 1]

    if len(future) == 0:
        continue

    target_price = entry_price * (1 + target_gain / 100)

    hit_target = False

    mae = 0
    mfe = 0

    days_to_target = None

    for i, row in enumerate(future.itertuples()):

        low_return = (row.Low / entry_price - 1) * 100
        high_return = (row.High / entry_price - 1) * 100

        mae = min(mae, low_return)
        mfe = max(mfe, high_return)

        if row.High >= target_price:
            hit_target = True
            days_to_target = i + 1
            break

    results.append({
        "Date": idx,
        "Entry": entry_price,
        "Hit_Target": hit_target,
        "MAE_%": mae,
        "MFE_%": mfe,
        "Days_To_Target": days_to_target
    })

results_df = pd.DataFrame(results)

# =========================================================
# METRICS
# =========================================================

if len(results_df) > 0:

    total_trades = len(results_df)

    wins = results_df["Hit_Target"].sum()

    win_rate = wins / total_trades * 100

    avg_mae = results_df["MAE_%"].mean()

    worst_mae = results_df["MAE_%"].min()

    avg_mfe = results_df["MFE_%"].mean()

    avg_days = results_df["Days_To_Target"].dropna().mean()

else:

    total_trades = 0
    wins = 0
    win_rate = 0
    avg_mae = 0
    worst_mae = 0
    avg_mfe = 0
    avg_days = 0

# =========================================================
# METRICS DISPLAY
# =========================================================

st.header("📊 Estatísticas")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Trades", total_trades)
c2.metric("Win Rate", f"{win_rate:.2f}%")
c3.metric("MAE Médio", f"{avg_mae:.2f}%")
c4.metric("Pior MAE", f"{worst_mae:.2f}%")

c5, c6 = st.columns(2)

c5.metric("MFE Médio", f"{avg_mfe:.2f}%")

if not np.isnan(avg_days):
    c6.metric("Dias até alvo", f"{avg_days:.2f}")
else:
    c6.metric("Dias até alvo", "-")

# =========================================================
# CURRENT SIGNAL
# =========================================================

st.header("🟢 Situação Atual")

latest_signal = df["SIGNAL"].iloc[-1]

if latest_signal:
    st.success("SETUP ATIVO NO CANDLE MAIS RECENTE")
else:
    st.warning("Nenhum setup ativo atualmente")

# =========================================================
# CHART
# =========================================================

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.7, 0.3]
)

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="BTC"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA50"],
        name="EMA50"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA200"],
        name="EMA200"
    ),
    row=1,
    col=1
)

signal_points = df[df["SIGNAL"]]

fig.add_trace(
    go.Scatter(
        x=signal_points.index,
        y=signal_points["Close"],
        mode="markers",
        marker=dict(size=10),
        name="Sinal"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["ADX"],
        name="ADX"
    ),
    row=2,
    col=1
)

fig.update_layout(
    height=900,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# RESULTS TABLE
# =========================================================

st.header("📋 Trades")

if len(results_df) > 0:

    display_df = results_df.copy()

    display_df["Date"] = display_df["Date"].astype(str)

    display_df["Hit_Target"] = display_df["Hit_Target"].map({
        True: "✅",
        False: "❌"
    })

    st.dataframe(
        display_df.sort_values("Date", ascending=False),
        use_container_width=True
    )

else:
    st.warning("Nenhum trade encontrado.")

# =========================================================
# ANALYSIS
# =========================================================

st.header("🧠 Interpretação")

if total_trades > 0:

    if win_rate >= 70:
        st.success("""
        Setup estatisticamente robusto para o alvo configurado.
        """)

    elif win_rate >= 60:
        st.warning("""
        Setup promissor, mas ainda não suficientemente robusto.
        """)

    else:
        st.error("""
        Setup fraco para operação sem stop.
        """)

    st.markdown(f"""
    ### Resumo Quantitativo

    - Taxa de acerto: **{win_rate:.2f}%**
    - Drawdown médio antes do gain: **{avg_mae:.2f}%**
    - Pior drawdown histórico: **{worst_mae:.2f}%**
    - Expansão média máxima: **{avg_mfe:.2f}%**

    ### O que observar

    Sem stop loss, o dado MAIS IMPORTANTE é o MAE.

    Quanto menor o MAE:
    - mais confortável a operação
    - menor sofrimento psicológico
    - menor risco de colapso da estratégia

    """)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("BTC Quant Breakout Lab - Streamlit Edition")
