"""
ShiftWN Markt-Wächter – Neuaufbau
Schritt 2: Echte Marktdaten laden und als Chart anzeigen.
Noch keine Analyse-Logik – nur Datenabruf + Darstellung.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# --- Grundkonfiguration ---
st.set_page_config(page_title="ShiftWN Wächter", page_icon="⚡", layout="wide")

st.title("⚡ ShiftWN Markt-Wächter")
st.caption("Neuaufbau · Schritt 2: echte Marktdaten")

# --- Auswahlmöglichkeiten ---
MAERKTE = {
    "DAX": "^GDAXI",
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC",
    "Bitcoin": "BTC-USD",
    "Gold": "GC=F",
}

GRANULARITAET = {
    "Täglich (1 Jahr)": ("1y", "1d"),
    "15-Minuten (1 Monat)": ("1mo", "15m"),
    "5-Minuten (5 Tage)": ("5d", "5m"),
}

with st.sidebar:
    st.header("Einstellungen")
    markt_name = st.selectbox("Markt", list(MAERKTE.keys()))
    gran_name = st.selectbox("Granularität", list(GRANULARITAET.keys()))

ticker = MAERKTE[markt_name]
period, interval = GRANULARITAET[gran_name]

# --- Daten laden (mit Cache, damit nicht bei jedem Klick neu geladen wird) ---
@st.cache_data(ttl=120)
def lade_daten(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    kurse = df["Close"].values.flatten()
    kurse = kurse[~np.isnan(kurse)]
    return kurse

with st.spinner(f"Lade {markt_name}-Daten ..."):
    kurse = lade_daten(ticker, period, interval)

# --- Anzeige ---
if kurse is None or len(kurse) < 10:
    st.error("❌ Konnte keine Marktdaten laden. Bitte anderen Markt/Granularität versuchen.")
    st.stop()

aktueller_preis = float(kurse[-1])

spalte1, spalte2 = st.columns(2)
spalte1.metric("Markt", markt_name)
spalte2.metric("Aktueller Preis", f"{aktueller_preis:,.2f}")

st.markdown(f"#### Preisverlauf {markt_name}")
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=kurse,
    mode="lines",
    line=dict(color="#00ff88", width=2),
    name=markt_name,
))
fig.update_layout(
    height=450,
    template="plotly_dark",
    margin=dict(l=0, r=0, t=10, b=0),
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(f"{len(kurse)} Datenpunkte geladen · Quelle: Yahoo Finance")
st.info("Nächster Schritt: die Random-Walk-Analyse (Varianz-Ratio, Runs, ACF) einbauen.")
