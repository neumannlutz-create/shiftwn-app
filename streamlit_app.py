"""
ShiftWN Markt-Wächter
Patentierter Kern: Triangle · Vortex · Impulse FFT · Photonics (Fusion).
ShiftWN arbeitet als geometrisches Korrektiv über bestehender Trading-
Intelligenz: es prüft die Kohärenz einer Empfehlung gegen die Marktgeometrie.
Patent EPA EP25221251.9 / SPECEPO-1/2.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import re
from datetime import datetime

st.set_page_config(page_title="ShiftWN AI", page_icon="⚡", layout="wide")
st.markdown("""
<style>
    .stApp {background-color: #0e1117; color: #e6e6e6;}
    h1, h2, h3 {color: #00ff88;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ ShiftWN AI – Geometrischer Markt-Wächter")
st.caption("Patent EPA EP25221251.9 · Triangle · Vortex · Impulse FFT · Photonics")

# ============================================================
#  PATENTIERTER SHIFTWN-KERN
#  Vier Operatoren + Photonics-Fusion. Geometrisches Korrektiv,
#  das die Marktstruktur misst und die Kohärenz einer externen
#  Empfehlung dagegen prüft. Keine Prognose, keine Anlageberatung.
# ============================================================

def _normalize(window):
    """Normalisiert ein OHLCV-Fenster auf robuste Skala (Median/MAD)."""
    c = window[:, 3]
    ref = np.median(c) if np.median(c) > 0 else 1.0
    scale = np.median(np.abs(np.diff(c))) or 0.01 * ref
    norm = np.zeros_like(window, dtype=float)
    norm[:, 0] = (window[:, 0] - ref) / scale
    norm[:, 1] = (window[:, 1] - ref) / scale
    norm[:, 2] = (window[:, 2] - ref) / scale
    norm[:, 3] = (c - ref) / scale
    norm[:, 4] = window[:, 4] / (np.median(window[:, 4]) or 1.0)
    return norm

def triangle(window):
    """Triangle Analysis: Trend-Geometrie / strukturelle Asymmetrie.
    Misst die Konvergenz der lokalen Dreiecksflächen (Steigungs-/Spannweiten-
    Struktur). Stark in Trending-Phasen."""
    c = _normalize(window)[:, 3]
    if len(c) < 3:
        return {"convergence": 0.0}
    areas = []
    for i in range(1, len(c) - 1):
        p0 = np.array([i - 1, c[i - 1]])
        p1 = np.array([i, c[i]])
        p2 = np.array([i + 1, c[i + 1]])
        area = 0.5 * ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        areas.append(area)
    x = np.arange(len(areas))
    slope = np.polyfit(x, np.abs(areas), 1)[0] if len(areas) >= 2 else 0.0
    return {"convergence": float(slope)}

def vortex(window):
    """Vortex Dynamics: rotierende Marktbewegung / zyklische Inkohärenz.
    Phasenraum (Position, Geschwindigkeit) -> Kohärenz der Drehung + Drift.
    Stark in Seitwärtsphasen / Mean-Reversion."""
    c = _normalize(window)[:, 3]
    if len(c) < 4:
        return {"coherence": 0.0, "drift_direction": 0.0}
    pos = c - np.mean(c)
    vel = np.gradient(pos)
    P = np.column_stack([pos, vel])
    ang = np.arctan2(P[:, 1], P[:, 0])
    dphi = np.diff(ang)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    coherence = float(np.clip(np.mean(np.abs(dphi)) / (np.std(dphi) + 1e-8), 0, 1))
    slope = float(np.polyfit(np.arange(len(c)), c, 1)[0])
    drift = float(np.tanh(slope * 3))
    return {"coherence": coherence, "drift_direction": drift}

def impulse(window):
    """Impulse FFT: Frequenzbrüche / Volatilitäts-Cluster.
    Rechteckwelle (Returns) -> Fourier-Spektrum -> dominante Leistung +
    spektraler Schwerpunkt. Stark in volatilen Phasen."""
    c = _normalize(window)[:, 3]
    r = np.diff(c)
    if len(r) < 4:
        return {"centroid": 0.0, "dominant_power_ratio": 0.0}
    r = r - np.mean(r)
    w = np.hanning(len(r))
    spec = np.abs(np.fft.rfft(r * w)) ** 2
    spec_sum = np.sum(spec) or 1.0
    dom = float(np.max(spec) / spec_sum)
    centroid = float(np.sum(np.linspace(0, 1, len(spec)) * spec) / spec_sum)
    return {"centroid": centroid, "dominant_power_ratio": dom}

def photonics(tri, vor, imp):
    """Photonics-Fusion: signal-adaptive Gewichtung der drei Operatoren
    über komplexe Phasen-Modulation. Bestimmt den Marktmodus (Trending /
    Ranging / Volatile), gewichtet adaptiv und fusioniert per Interferenz
    zu einem kohärenten Gesamtbild.

    S = w1*T + w2*V + w3*F  mit signal-adaptiven w_i (Summe = 1)."""
    # Operator-Scores in [0,1] bringen
    T = float(np.tanh(abs(tri["convergence"]) * 5))          # Trend-/Konvergenzstärke
    V = float(vor["coherence"])                               # Rotations-Kohärenz
    F = float(imp["dominant_power_ratio"])                    # spektrale Dominanz

    # Signal-adaptive Gewichte: der jeweils stärkste Operator bestimmt den Modus
    raw = np.array([T, V, F]) + 1e-6
    w = raw / raw.sum()

    # Phasen-Modulation: jedem Operator eine Phase zuordnen, dann
    # kohärent (komplex) aufsummieren -> konstruktive/destruktive Interferenz
    phases = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])    # 3 Operatoren, 120° versetzt
    amps = np.array([T, V, F])
    field = np.sum(w * amps * np.exp(1j * phases))
    coherence_fused = float(np.clip(np.abs(field), 0, 1))     # |resultierendes Feld|

    modus = ["TRENDING", "RANGING", "VOLATILE"][int(np.argmax([T, V, F]))]
    dominant = ["Triangle", "Vortex", "Impulse FFT"][int(np.argmax([T, V, F]))]

    return {"S": coherence_fused, "weights": {"Triangle": float(w[0]),
            "Vortex": float(w[1]), "ImpulseFFT": float(w[2])},
            "modus": modus, "dominant": dominant, "T": T, "V": V, "F": F,
            "drift": vor["drift_direction"]}

def measure_shiftwn(window):
    tri = triangle(window)
    vor = vortex(window)
    imp = impulse(window)
    fus = photonics(tri, vor, imp)
    return {"triangle": tri, "vortex": vor, "impulse": imp, "photonics": fus}

def build_window(closes):
    """Baut ein OHLCV-Fenster aus Schlusskursen (HLC approximiert um Close)."""
    n = min(60, len(closes))
    seg = closes[-n:]
    window = np.zeros((n, 5))
    window[:, 3] = seg
    window[:, 0] = seg * 0.997   # Open
    window[:, 1] = seg * 1.005   # High
    window[:, 2] = seg * 0.995   # Low
    window[:, 4] = 1.0           # Volumen (neutral, falls nicht vorhanden)
    return window

def fibonacci_levels(closes):
    hi, lo = np.max(closes), np.min(closes)
    d = hi - lo
    return {"0.0%": hi, "23.6%": hi-0.236*d, "38.2%": hi-0.382*d, "50.0%": hi-0.5*d,
            "61.8%": hi-0.618*d, "78.6%": hi-0.786*d, "100.0%": lo}

# ============================================================
#  KI-WÄCHTER: Empfehlung lesen + Kohärenz gegen ShiftWN prüfen
# ============================================================

def extrahiere_empfehlung(text):
    out = {"richtung": "HOLD", "konfidenz": None, "ziel": None, "stop": None}
    if not text:
        return out
    tl = text.lower()
    m = re.search(r"(?:konfidenz|confidence)\s*[:=]?\s*(\d{1,3})\s*%?", tl)
    out["konfidenz"] = int(m.group(1)) if m else None
    tlp = re.sub(r"(?:konfidenz|confidence)\s*[:=]?\s*\d{1,3}\s*%?", " ", tl)
    num = r"(\d{1,3}(?:\.\d{3})+|\d{4,6})"
    f = lambda s: float(s.replace(".", ""))
    z = re.search(rf"(?:ziel|target|kursziel)\s*[:=]?\s*{num}", tlp); out["ziel"] = f(z.group(1)) if z else None
    s = re.search(rf"(?:stop[- ]?loss|stop)\s*[:=]?\s*{num}", tlp); out["stop"] = f(s.group(1)) if s else None
    sell = ["sell", "short", "verkaufen", "verkauf", "bearish"]
    buy = ["buy", "long", "kaufen", "kauf", "bullish"]
    hs = any(re.search(rf"\b{re.escape(w)}\b", tl) for w in sell)
    hb = any(re.search(rf"\b{re.escape(w)}\b", tl) for w in buy)
    out["richtung"] = "SELL" if hs and not hb else "BUY" if hb and not hs else "CONFLICT" if hb and hs else "HOLD"
    return out

# ============================================================
#  DATEN
# ============================================================

MAERKTE = {
    "DAX": ["^GDAXI", "EXS1.DE"], "S&P 500": ["^GSPC"], "Dow Jones": ["^DJI"],
    "Nasdaq": ["^IXIC"], "TecDAX": ["^TECDAX"], "Bitcoin": ["BTC-USD"], "Gold": ["GC=F"],
}
GRAN = {"Täglich (1 Jahr)": ("1y", "1d"), "15-Minuten (1 Monat)": ("1mo", "15m"),
        "5-Minuten (5 Tage)": ("5d", "5m")}

with st.sidebar:
    st.header("Einstellungen")
    markt_name = st.selectbox("Markt", list(MAERKTE.keys()))
    gran_name = st.selectbox("Granularität", list(GRAN.keys()))
    st.divider()
    coherence_min = st.slider("Vortex-Kohärenz (Minimum für Signal)", 0.50, 0.95, 0.70, 0.01)
    drift_min = st.slider("Drift (Minimum für Richtung)", 0.04, 0.30, 0.06, 0.01)
    st.divider()
    st.subheader("🛡️ KI-Wächter")
    externe_empfehlung = st.text_area("KI-Empfehlung einfügen", height=150,
        placeholder="Empfehlung von ChatGPT, Grok etc. ...")
    waechter_an = st.checkbox("Wächter aktivieren", value=True)

ticker_liste = MAERKTE[markt_name]
period, interval = GRAN[gran_name]

@st.cache_data(ttl=120)
def lade_daten(ticker_liste, period, interval, versuche=3):
    for ticker in ticker_liste:
        for _ in range(versuche):
            try:
                df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            except Exception:
                df = None
            if df is not None and not df.empty:
                k = df["Close"].values.flatten(); k = k[~np.isnan(k)]
                if len(k) >= 30:
                    return k, ticker
    return None, None

with st.spinner(f"Lade {markt_name}-Daten ..."):
    kurse, used = lade_daten(tuple(ticker_liste), period, interval)

if kurse is None or len(kurse) < 30:
    st.error("❌ Konnte keine Marktdaten laden. Anderen Markt/Granularität versuchen (oder erneut).")
    st.stop()

closes = kurse[-min(len(kurse), 200):]
aktueller_preis = float(kurse[-1])

# ============================================================
#  SHIFTWN-MESSUNG
# ============================================================
window = build_window(closes)
r = measure_shiftwn(window)
ph = r["photonics"]
coherence = ph["S"]
drift = ph["drift"]

# Signal aus ShiftWN-Geometrie (Korrektiv-Logik)
if coherence > coherence_min and drift > drift_min:
    signal, sig_icon = "STRUKTUR ↑ (BUY-kohärent)", "🟢"; struktur_dir = "BUY"
elif coherence > coherence_min and drift < -drift_min:
    signal, sig_icon = "STRUKTUR ↓ (SELL-kohärent)", "🔴"; struktur_dir = "SELL"
else:
    signal, sig_icon = "KEINE KOHÄRENTE STRUKTUR", "🟠"; struktur_dir = "NEUTRAL"

st.caption(f"ShiftWN-Analyse {markt_name} · {gran_name} · {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
if used and used not in ("^GDAXI","^GSPC","^DJI","^IXIC","^TECDAX","BTC-USD","GC=F"):
    st.caption(f"Hinweis: Daten über Ersatz-Ticker `{used}` geladen.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Marktmodus", ph["modus"]); k1.caption(f"dominant: {ph['dominant']}")
k2.metric("Aktueller Preis", f"{aktueller_preis:,.2f}")
k3.metric("Photonics-Kohärenz S", f"{coherence:.3f}")
k4.metric("Drift", f"{drift:+.3f}")

# Operator-Scores + adaptive Gewichte
st.markdown("#### Operatoren & Photonics-Gewichtung")
o1, o2, o3 = st.columns(3)
o1.metric("Triangle (T)", f"{ph['T']:.3f}", f"Gewicht {ph['weights']['Triangle']:.2f}")
o2.metric("Vortex (V)", f"{ph['V']:.3f}", f"Gewicht {ph['weights']['Vortex']:.2f}")
o3.metric("Impulse FFT (F)", f"{ph['F']:.3f}", f"Gewicht {ph['weights']['ImpulseFFT']:.2f}")

links, rechts = st.columns(2)
with links:
    st.markdown(f"#### Preisverlauf {markt_name} mit Fibonacci")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=closes, mode="lines", line=dict(color="#00ff88", width=2), name=markt_name))
    for nm, pr in fibonacci_levels(closes).items():
        fig.add_hline(y=pr, line_dash="dash", line_color="rgba(255,255,0,0.35)", annotation_text=nm)
    fig.update_layout(height=420, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with rechts:
    st.markdown("#### Photonics-Fusion (adaptive Gewichte)")
    fig2 = go.Figure(go.Bar(
        x=[ph["weights"]["Triangle"], ph["weights"]["Vortex"], ph["weights"]["ImpulseFFT"]],
        y=["Triangle", "Vortex", "Impulse FFT"], orientation="h",
        marker_color=["#00ff88", "#00b3ff", "#ff9500"]))
    fig2.update_layout(height=420, template="plotly_dark", margin=dict(l=0,r=0,t=10,b=0),
                       xaxis_title="adaptives Gewicht w_i", xaxis_range=[0,1])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown(f"### {sig_icon} ShiftWN-Befund: {signal}")
st.caption("ShiftWN misst die Marktgeometrie über vier patentierte Operatoren (Triangle, Vortex, "
           "Impulse FFT) und fusioniert sie adaptiv über Photonics. Geometrisches Korrektiv, keine "
           "Prognose und keine Anlageberatung.")

# ============================================================
#  WÄCHTER-URTEIL
# ============================================================
if externe_empfehlung and waechter_an:
    emp = extrahiere_empfehlung(externe_empfehlung)
    richtung = emp["richtung"]
    st.markdown("---")
    st.subheader("🛡️ Wächter-Urteil")
    wa, wb = st.columns(2)
    with wa:
        st.write("**Externe KI sagt (extrahiert):**")
        st.info(externe_empfehlung)
        konf = f" · Konfidenz {emp['konfidenz']} %" if emp["konfidenz"] is not None else ""
        st.caption(f"Richtung: **{richtung}**{konf} · Ziel {emp['ziel']} · Stop {emp['stop']}")
    with wb:
        st.write("**ShiftWN misst (Geometrie):**")
        st.success(f"{sig_icon} {ph['modus']} · Kohärenz {coherence:.3f} · dominant {ph['dominant']}")

    if richtung == "CONFLICT":
        st.warning("⚠️ Empfehlung mehrdeutig (Kauf- und Verkaufsbegriffe). Bitte manuell prüfen.")
    elif struktur_dir == "NEUTRAL":
        if richtung in ("BUY", "SELL"):
            st.error(f"❌ ShiftWN widerspricht: Empfehlung sagt **{richtung}**, die Marktgeometrie zeigt "
                     f"jedoch keine kohärente Struktur (Kohärenz {coherence:.3f} < {coherence_min}). "
                     f"Die behauptete Richtung ist geometrisch nicht gedeckt.")
        else:
            st.info("Beide Seiten sehen keine kohärente Struktur. Übereinstimmung.")
    elif richtung == struktur_dir:
        st.success(f"✅ ShiftWN bestätigt: Richtung **{richtung}** ist mit der Marktgeometrie kohärent "
                   f"(Modus {ph['modus']}, Kohärenz {coherence:.3f}). Hinweis: geometrische Kohärenz, "
                   f"kein Kaufsignal und keine Handelbarkeitsgarantie.")
    elif richtung == "HOLD":
        st.warning(f"⚠️ Empfehlung sagt HOLD – ShiftWN sieht jedoch kohärente Struktur Richtung {struktur_dir}.")
    else:
        st.error(f"❌ ShiftWN widerspricht: Empfehlung sagt **{richtung}**, die Marktgeometrie tendiert "
                 f"jedoch **{struktur_dir}** (Modus {ph['modus']}).")

st.caption("ShiftWN AI · Patentierter geometrischer Kern (Triangle · Vortex · Impulse FFT · Photonics). "
           "Geometrisches Korrektiv über bestehender Trading-Intelligenz. Keine Anlageberatung.")
