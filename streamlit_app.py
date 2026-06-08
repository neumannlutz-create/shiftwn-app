"""
ShiftWN Markt-Wächter – Neuaufbau
Schritt 3: Random-Walk-Analyse (Varianz-Ratio, Runs, ACF) + Permutationstest.
Zeigt erstmals einen Befund (random-walk-konsistent / Struktur erkannt) + p-Wert.
Noch ohne Verteilungs-Grafik und ohne Wächter (kommen in Schritt 4 und 5).
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# --- Grundkonfiguration ---
st.set_page_config(page_title="ShiftWN Wächter", page_icon="⚡", layout="wide")

st.title("⚡ ShiftWN Markt-Wächter")
st.caption("Neuaufbau · Schritt 3: Random-Walk-Analyse")

# ============================================================
#  ANALYSE-ENGINE
#  Misst, ob eine Kursreihe vom Random Walk abweicht.
#  Prognosefrei. Statistische Signifikanz ist NICHT gleich
#  Handelbarkeit. Kein Kauf-/Verkaufssignal.
# ============================================================

def normiere(kurse):
    c = np.asarray(kurse, float)
    ref = np.median(c) if np.median(c) > 0 else 1.0
    skala = np.median(np.abs(np.diff(c))) or 0.01 * ref
    return (c - ref) / skala

def varianz_ratio(kurse, q=5):
    """Lo-MacKinlay VR(q). =1 unter Random Walk. <1 Mean-Reversion, >1 Trending."""
    p = normiere(kurse); r = np.diff(p); n = len(r)
    if n < q + 2:
        return 1.0
    var1 = np.sum((r - r.mean())**2) / n
    rq = p[q:] - p[:-q]
    varq = np.sum((rq - rq.mean())**2) / len(rq)
    return float(varq / (q * var1)) if var1 else 1.0

def runs_z(kurse):
    """Standardisierte Vorzeichen-Run-Anzahl. |z| groß => Reihenfolge nicht zufällig."""
    r = np.diff(normiere(kurse)); s = np.sign(r); s = s[s != 0]; n = len(s)
    if n < 10:
        return 0.0
    runs = 1 + np.sum(s[1:] != s[:-1])
    npos = np.sum(s > 0); nneg = n - npos
    if npos == 0 or nneg == 0:
        return 0.0
    erwartung = 1 + 2 * npos * nneg / n
    varianz = (2 * npos * nneg * (2 * npos * nneg - n)) / (n * n * (n - 1))
    return float((runs - erwartung) / np.sqrt(varianz)) if varianz > 0 else 0.0

def acf_lag1(kurse):
    """Lag-1-Autokorrelation der Returns."""
    r = np.diff(normiere(kurse)); r = r - r.mean()
    nenner = np.sum(r * r) or 1e-12
    return float(np.sum(r[:-1] * r[1:]) / nenner)

def permutationstest(kurse, metrik, basiswert, n_perm=400, seed=0):
    """Shuffelt die Returns vielfach. p-Wert = Anteil der Surrogate, die
    mindestens so weit vom Basiswert abweichen wie die echte Reihe.
    Kleiner p-Wert => echte (reihenfolge-abhängige) Struktur."""
    rng = np.random.default_rng(seed)
    c = normiere(kurse); steps = np.diff(c)
    beobachtet = metrik(kurse)
    abweichung = abs(beobachtet - basiswert)
    zaehler = 0
    for _ in range(n_perm):
        perm = rng.permutation(steps)
        surrogat = np.concatenate([[c[0]], c[0] + np.cumsum(perm)])
        if abs(metrik(surrogat) - basiswert) >= abweichung:
            zaehler += 1
    p = (zaehler + 1) / (n_perm + 1)
    return beobachtet, p

# ============================================================
#  AUSWAHL + DATEN
# ============================================================

MAERKTE = {
    "DAX": "^GDAXI", "S&P 500": "^GSPC", "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC", "Bitcoin": "BTC-USD", "Gold": "GC=F",
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
    st.divider()
    alpha = st.slider("Signifikanzniveau α", 0.01, 0.10, 0.05, 0.01)
    n_perm = st.slider("Permutationen", 100, 800, 400, 50)

ticker = MAERKTE[markt_name]
period, interval = GRANULARITAET[gran_name]

@st.cache_data(ttl=120)
def lade_daten(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    kurse = df["Close"].values.flatten()
    return kurse[~np.isnan(kurse)]

with st.spinner(f"Lade {markt_name}-Daten ..."):
    kurse = lade_daten(ticker, period, interval)

if kurse is None or len(kurse) < 30:
    st.error("❌ Konnte keine ausreichenden Marktdaten laden. Anderen Markt/Granularität versuchen.")
    st.stop()

# auf max. 500 Punkte begrenzen (Geschwindigkeit des Permutationstests)
segment = kurse[-min(len(kurse), 500):]
aktueller_preis = float(kurse[-1])

# ============================================================
#  ANALYSE AUSFÜHREN
# ============================================================

with st.spinner("Analysiere Marktstruktur ..."):
    vr, p_vr = permutationstest(segment, varianz_ratio, 1.0, n_perm)
    rz, p_rz = permutationstest(segment, runs_z, 0.0, n_perm)
    ac, p_ac = permutationstest(segment, acf_lag1, 0.0, n_perm)

min_p = min(p_vr, p_rz, p_ac)
struktur_erkannt = min_p < alpha

if struktur_erkannt:
    if vr > 1.0 and p_vr < alpha:
        tendenz = "Trending / Momentum"
    elif (vr < 1.0 and p_vr < alpha) or (ac < 0 and p_ac < alpha):
        tendenz = "Mean-Reversion"
    elif ac > 0 and p_ac < alpha:
        tendenz = "Momentum"
    else:
        tendenz = "Struktur (Art unklar)"
    befund = "🟢 STRUKTUR ERKANNT"
else:
    tendenz = "—"
    befund = "🟠 RANDOM-WALK-KONSISTENT"

# ============================================================
#  ANZEIGE
# ============================================================

st.caption(f"Analyse {markt_name} · {gran_name}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Befund", befund.split()[0])
k1.caption(befund.split(" ", 1)[1])
k2.metric("Aktueller Preis", f"{aktueller_preis:,.2f}")
k3.metric("kleinster p-Wert", f"{min_p:.3f}")
k3.caption(f"α = {alpha}")
k4.metric("Tendenz", tendenz)

# Teststatistiken
st.markdown("#### Teststatistiken")
t1, t2, t3 = st.columns(3)
for spalte, name, wert, p in [
    (t1, "Varianz-Ratio VR(5)", vr, p_vr),
    (t2, "Runs-Test (z)", rz, p_rz),
    (t3, "ACF Lag-1", ac, p_ac),
]:
    sig = "✓ signifikant" if p < alpha else "nicht signifikant"
    spalte.metric(name, f"{wert:.3f}", f"p={p:.3f} · {sig}")

# Chart
st.markdown(f"#### Preisverlauf {markt_name}")
fig = go.Figure()
fig.add_trace(go.Scatter(y=segment, mode="lines",
              line=dict(color="#00ff88", width=2), name=markt_name))
fig.update_layout(height=420, template="plotly_dark",
                  margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.caption(f"{len(segment)} Datenpunkte analysiert · "
           "Misst Abweichung vom Random Walk. Statistische Signifikanz ≠ Handelbarkeit. "
           "Keine Anlageberatung.")
st.info("Nächster Schritt: die Permutationsverteilung als Grafik anzeigen (zweispaltiges Layout).")
