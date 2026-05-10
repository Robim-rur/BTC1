# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="BTC Probability Engine",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🧠 BTC Probability Engine")

st.markdown("""
Este app NÃO procura sinais antigos no gráfico.

Ele analisa a ESTRUTURA ATUAL do Bitcoin no gráfico diário e responde:

# 👉 Qual a probabilidade matemática do BTC subir +3%?

Baseado em:
- tendência
- força
- volatilidade
- compressão
- breakout
- momentum
- comportamento histórico semelhante
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Configurações")

period = st.sidebar.selectbox(
    "Histórico",
    ["2y", "5y", "10y"],
    index=2
)

target_gain = st.sidebar.slider(
    "Gain alvo (%)",
    1.0,
    10.0,
    3.0,
    0.5
)

future_bars = st.sidebar.slider(
    "Máximo candles futuros",
    1,
    30,
    10
)

# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():

    df = yf.download(
        "BTC-USD",
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    # Corrigir MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close", "Volume"]

    df = df[cols].copy()

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(inplace=True)

    return df

df = load_data()

# =========================================================
# INDICATORS
# =========================================================

close = df["Close"]
high = df["High"]
low = df["Low"]
volume = df["Volume"]

# EMA
df["EMA21"] = ta.trend.ema_indicator(close, window=21)
df["EMA50"] = ta.trend.ema_indicator(close, window=50)
df["EMA200"] = ta.trend.ema_indicator(close, window=200)

# ADX
adx = ta.trend.ADXIndicator(
    high=high,
    low=low,
    close=close,
    window=14
)

df["ADX"] = adx.adx()
df["DI_POS"] = adx.adx_pos()
df["DI_NEG"] = adx.adx_neg()

# ATR
atr = ta.volatility.AverageTrueRange(
    high=high,
    low=low,
    close=close,
    window=14
)

df["ATR"] = atr.average_true_range()

# RSI
df["RSI"] = ta.momentum.rsi(close, window=14)

# Bollinger
bb = ta.volatility.BollingerBands(
    close=close,
    window=20,
    window_dev=2
)

df["BB_WIDTH"] = (
    (bb.bollinger_hband() - bb.bollinger_lband())
    / close
)

# Volume média
df["VOL_MA20"] = volume.rolling(20).mean()

# =========================================================
# CURRENT STRUCTURE
# =========================================================

latest = df.iloc[-1]

# =========================================================
# SCORING ENGINE
# =========================================================

score = 0
max_score = 100

details = []

# ---------------------------------------------------------
# TENDÊNCIA
# ---------------------------------------------------------

if latest["Close"] > latest["EMA21"]:
    score += 10
    details.append(("Preço acima EMA21", "✅ +10"))
else:
    details.append(("Preço acima EMA21", "❌"))

if latest["EMA21"] > latest["EMA50"]:
    score += 15
    details.append(("EMA21 acima EMA50", "✅ +15"))
else:
    details.append(("EMA21 acima EMA50", "❌"))

if latest["EMA50"] > latest["EMA200"]:
    score += 20
    details.append(("EMA50 acima EMA200", "✅ +20"))
else:
    details.append(("EMA50 acima EMA200", "❌"))

# ---------------------------------------------------------
# FORÇA
# ---------------------------------------------------------

if latest["ADX"] > 22:
    score += 15
    details.append(("ADX forte", "✅ +15"))
else:
    details.append(("ADX forte", "❌"))

if latest["DI_POS"] > latest["DI_NEG"]:
    score += 10
    details.append(("DI+ acima DI−", "✅ +10"))
else:
    details.append(("DI+ acima DI−", "❌"))

# ---------------------------------------------------------
# MOMENTUM
# ---------------------------------------------------------

if latest["RSI"] > 55:
    score += 10
    details.append(("RSI momentum", "✅ +10"))
else:
    details.append(("RSI momentum", "❌"))

# ---------------------------------------------------------
# VOLUME
# ---------------------------------------------------------

if latest["Volume"] > latest["VOL_MA20"]:
    score += 10
    details.append(("Volume acima média", "✅ +10"))
else:
    details.append(("Volume acima média", "❌"))

# ---------------------------------------------------------
# COMPRESSÃO
# ---------------------------------------------------------

bb_mean = df["BB_WIDTH"].rolling(50).mean().iloc[-1]

if latest["BB_WIDTH"] < bb_mean:
    score += 10
    details.append(("Compressão volatilidade", "✅ +10"))
else:
    details.append(("Compressão volatilidade", "❌"))

# =========================================================
# HISTORICAL MATCH ENGINE
# =========================================================

historical = []

for i in range(250, len(df) - future_bars):

    row = df.iloc[i]

    local_score = 0

    if row["Close"] > row["EMA21"]:
        local_score += 10

    if row["EMA21"] > row["EMA50"]:
        local_score += 15

    if row["EMA50"] > row["EMA200"]:
        local_score += 20

    if row["ADX"] > 22:
        local_score += 15

    if row["DI_POS"] > row["DI_NEG"]:
        local_score += 10

    if row["RSI"] > 55:
        local_score += 10

    if row["Volume"] > row["VOL_MA20"]:
        local_score += 10

    local_bb_mean = (
        df["BB_WIDTH"]
        .rolling(50)
        .mean()
        .iloc[i]
    )

    if row["BB_WIDTH"] < local_bb_mean:
        local_score += 10

    similarity = abs(local_score - score)

    future = df.iloc[i + 1:i + 1 + future_bars]

    target = row["Close"] * (1 + target_gain / 100)

    hit = (future["High"] >= target).any()

    historical.append({
        "score": local_score,
        "similarity": similarity,
        "hit": hit
    })

hist_df = pd.DataFrame(historical)

# =========================================================
# MOST SIMILAR STRUCTURES
# =========================================================

similar_df = hist_df[
    hist_df["similarity"] <= 5
].copy()

total_cases = len(similar_df)

wins = similar_df["hit"].sum()

if total_cases > 0:
    probability = (wins / total_cases) * 100
else:
    probability = 0

# =========================================================
# FINAL RESULT
# =========================================================

st.header("🎯 Resultado Probabilístico Atual")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Score Estrutural",
    f"{score}/100"
)

col2.metric(
    "Casos Históricos Similares",
    total_cases
)

col3.metric(
    f"Chance de +{target_gain:.1f}%",
    f"{probability:.2f}%"
)

# =========================================================
# INTERPRETAÇÃO
# =========================================================

st.header("🧠 Interpretação")

if probability >= 70:

    st.success(f"""
ESTRUTURA MUITO FORTE.

Historicamente, estruturas semelhantes atingiram
+{target_gain:.1f}% em aproximadamente {probability:.1f}% das vezes.
""")

elif probability >= 60:

    st.warning(f"""
ESTRUTURA MODERADAMENTE POSITIVA.

Existe vantagem estatística, mas não extrema.
""")

else:

    st.error(f"""
ESTRUTURA FRACA.

Historicamente o BTC não apresentou consistência suficiente
para atingir +{target_gain:.1f}% rapidamente.
""")

# =========================================================
# DETALHES DA ESTRUTURA
# =========================================================

st.header("📋 Componentes da Estrutura Atual")

details_df = pd.DataFrame(
    details,
    columns=["Fator", "Status"]
)

st.dataframe(
    details_df,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# EXTREME WARNING
# =========================================================

st.markdown("---")

st.warning("""
IMPORTANTE:

Este modelo NÃO prevê o futuro.

Ele mede:

- quantas vezes estruturas parecidas ocorreram
- e qual foi o resultado histórico depois disso

Ou seja:

é um motor de PROBABILIDADE ESTATÍSTICA,
não uma previsão absoluta.
""")
