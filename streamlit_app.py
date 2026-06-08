"""
ShiftWN Markt-Wächter – Neuaufbau
Schritt 1: Skelett. Startet die App, prüft das Setup.
Noch keine Daten, noch keine Logik.
"""

import streamlit as st

# --- Grundkonfiguration der Seite ---
st.set_page_config(
    page_title="ShiftWN Wächter",
    page_icon="⚡",
    layout="wide",
)

# --- Kopf ---
st.title("⚡ ShiftWN Markt-Wächter")
st.caption("Neuaufbau · Schritt 1: Skelett")

# --- Sidebar (vorerst leer, nur Platzhalter) ---
with st.sidebar:
    st.header("Einstellungen")
    st.write("Hier kommen später Markt- und Analyse-Optionen hin.")

# --- Hauptbereich ---
st.write("Wenn du das hier siehst, läuft die App. ✅")
st.info("Nächster Schritt: echte Marktdaten laden und als Chart anzeigen.")
