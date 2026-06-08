"""
ShiftWN AI – Geometrische Marktanalyse
Patentierter Kern (v4.3-Basis, unveraendert): Triangle, Vortex, Impulse, Photonics-Fusion.
Erweitert: besserer Look, Live-Daten, robuster Datenabruf, KI-Waechter mit korrektem Parser.
Patent EPA EP25221251.9 / SPECEPO-1/2.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import re
from datetime import datetime

st.set_page_config(page_title="ShiftWN AI", layout="wide", page_icon="⚡")

# ---------- Look ----------
st.markdown("""
<style>
    .stApp {background-color: #0a0e16; color: #e8edf4;}
    section[data-testid="stSidebar"] {background-color: #0f1521;}
    h1 {color: #00ff9d; font-weight: 800; letter-spacing: -0.5px;}
    h2, h3 {color: #d6f5e6;}
    .stMetric {background: #141b29; border: 1px solid #1e2838; border-radius: 12px;
               padding: 14px 16px;}
    [data-testid="stMetricValue"] {color: #00ff9d;}
    [data-testid="stMetricLabel"] {color: #9aa6b8;}
    .stButton>button {background:#00ff9d;color:#06210f;font-weight:700;border:none;border-radius:8px;}
    div[data-testid="stMarkdownContainer"] code {color:#00ff9d;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ ShiftWN AI")
st.caption("Geometrische Marktanalyse · Patent EPA EP25221251.9 · Triangle · Vortex · Impulse FFT · Photonics")

# ============================================================
#  PATENTIERTER SHIFTWN-KERN  (unveraendert aus v4.3)
# ============================================================
def _normalize(window):
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
    c = _normalize(window)[:, 3]
    if len(c) < 3: return {"convergence": 0.0}
    areas = []
    for i in range(1, len(c) - 1):
        p0 = np.array([i-1, c[i-1]]); p1 = np.array([i, c[i]]); p2 = np.array([i+1, c[i+1]])
        areas.append(0.5*((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])))
    x = np.arange(len(areas))
    slope = np.polyfit(x, np.abs(areas), 1)[0] if len(areas) >= 2 else 0.0
    return {"convergence": float(slope)}

def vortex(window):
    c = _normalize(window)[:, 3]
    if len(c) < 4: return {"coherence": 0.0, "drift_direction": 0.0}
    pos = c - np.mean(c); vel = np.gradient(pos)
    P = np.column_stack([pos, vel])
    ang = np.arctan2(P[:, 1], P[:, 0])
    dphi = np.diff(ang); dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    coherence = float(np.clip(np.mean(np.abs(dphi)) / (np.std(dphi) + 1e-8), 0, 1))
    slope = float(np.polyfit(np.arange(len(c)), c, 1)[0])
    drift = float(np.tanh(slope * 3))
    return {"coherence": coherence, "drift_direction": drift}

def impulse(window):
    c = _normalize(window)[:, 3]; r = np.diff(c)
    if len(r) < 4: return {"centroid": 0.0, "dominant_power_ratio": 0.0}
    r = r - np.mean(r); w = np.hanning(len(r))
    spec = np.abs(np.fft.rfft(r * w)) ** 2
    ss = np.sum(spec) or 1.0
    return {"centroid": float(np.sum(np.linspace(0,1,len(spec))*spec)/ss),
            "dominant_power_ratio": float(np.max(spec)/ss)}

def photonics_weights(tri, vor, imp):
    """Photonics: signal-adaptive Gewichtung der drei Operatoren + Marktmodus.
    Bestimmt, welcher Operator im aktuellen Marktzustand fuehrt."""
    T = float(np.tanh(abs(tri["convergence"]) * 8))
    V = float(vor["coherence"])
    F = float(imp["dominant_power_ratio"])
    raw = np.array([T, V, F]) + 1e-6
    w = raw / raw.sum()
    modus = ["TRENDING", "RANGING", "VOLATILE"][int(np.argmax([T, V, F]))]
    dominant = ["Triangle", "Vortex", "Impulse FFT"][int(np.argmax([T, V, F]))]
    return {"w": w, "T": T, "V": V, "F": F, "modus": modus, "dominant": dominant}

def measure_shiftwn(window):
    tri, vor, imp = triangle(window), vortex(window), impulse(window)
    return {"triangle": tri, "vortex": vor, "impulse": imp, "photonics": photonics_weights(tri, vor, imp)}

def fibonacci_levels(closes):
    hi, lo = np.max(closes), np.min(closes); d = hi - lo
    return {"0.0%": hi, "23.6%": hi-0.236*d, "38.2%": hi-0.382*d, "50.0%": hi-0.5*d,
            "61.8%": hi-0.618*d, "78.6%": hi-0.786*d, "100.0%": lo}

# ============================================================
#  KI-WÄCHTER PARSER  (Bug gefixt: Wortgrenzen)
# ============================================================
def parse_ki(text):
    out = {"direction": "HOLD", "confidence": None, "target": None, "stop": None}
    if not text: return out
    tl = text.lower()
    m = re.search(r"(?:konfidenz|confidence)\s*[:=]?\s*(\d{1,3})\s*%?", tl)
    out["confidence"] = int(m.group(1)) if m else None
    tlp = re.sub(r"(?:konfidenz|confidence)\s*[:=]?\s*\d{1,3}\s*%?", " ", tl)
    num = r"(\d{1,3}(?:\.\d{3})+|\d{4,6})"
    f = lambda s: float(s.replace(".", ""))
    z = re.search(rf"(?:ziel|target|kursziel)\s*[:=]?\s*{num}", tlp); out["target"] = f(z.group(1)) if z else None
    s = re.search(rf"(?:stop[- ]?loss|stop)\s*[:=]?\s*{num}", tlp); out["stop"] = f(s.group(1)) if s else None
    sell = ["sell","short","verkaufen","verkauf","bearish"]; buy = ["buy","long","kaufen","kauf","bullish"]
    hs = any(re.search(rf"\b{re.escape(w)}\b", tl) for w in sell)
    hb = any(re.search(rf"\b{re.escape(w)}\b", tl) for w in buy)
    out["direction"] = "SELL" if hs and not hb else "BUY" if hb and not hs else "CONFLICT" if hb and hs else "HOLD"
    return out

# ============================================================
#  DATEN  (Live, robust)
# ============================================================
MARKETS = {
    "DAX": ["^GDAXI", "EXS1.DE"], "S&P 500": ["^GSPC"], "Dow Jones": ["^DJI"],
    "Nasdaq": ["^IXIC"], "TecDAX": ["^TECDAX"], "Bitcoin (BTC-USD)": ["BTC-USD"], "Gold": ["GC=F"],
}
GRAN = {"Täglich (1 Jahr)": ("1y","1d"), "15-Minuten (1 Monat)": ("1mo","15m"), "5-Minuten (5 Tage)": ("5d","5m")}

with st.sidebar:
    st.header("Einstellungen")
    market_name = st.selectbox("Markt", list(MARKETS.keys()))
    gran_name = st.selectbox("Granularität", list(GRAN.keys()))
    st.divider()
    st.subheader("Signal-Schwellen")
    vortex_th = st.slider("Vortex Coherence (Minimum)", 0.50, 0.95, 0.70, 0.01)
    drift_th = st.slider("Drift (Minimum für Signal)", 0.04, 0.30, 0.06, 0.01)
    st.divider()
    st.subheader("🛡️ KI-Wächter")
    external = st.text_area("KI-Empfehlung einfügen", height=140,
        placeholder="Empfehlung von ChatGPT, Grok etc. einfügen...")
    ki_on = st.checkbox("ShiftWN als KI-Wächter aktivieren", value=True)

ticker_list = MARKETS[market_name]
period, interval = GRAN[gran_name]

@st.cache_data(ttl=120)
def get_data(ticker_list, period, interval, tries=3):
    for tk in ticker_list:
        for _ in range(tries):
            try:
                df = yf.download(tk, period=period, interval=interval, progress=False, auto_adjust=True)
            except Exception:
                df = None
            if df is not None and not df.empty:
                k = df["Close"].values.flatten(); k = k[~np.isnan(k)]
                if len(k) >= 50: return k, tk
    return None, None

with st.spinner(f"Lade {market_name}-Daten ..."):
    closes_full, used = get_data(tuple(ticker_list), period, interval)

if closes_full is None or len(closes_full) < 50:
    st.error("❌ Konnte keine Marktdaten laden. Anderen Markt/Granularität versuchen (oder erneut).")
    st.stop()

closes = closes_full[-min(len(closes_full), 200):]
current_price = float(closes_full[-1])

# ============================================================
#  ANALYSE  (v4.3-Signallogik, unveraendert)
# ============================================================
window_size = min(50, len(closes))
window = np.zeros((window_size, 5))
seg = closes[-window_size:]
window[:, 3] = seg; window[:, 0] = seg*0.97; window[:, 1] = seg*1.08
window[:, 2] = seg*0.92; window[:, 4] = 15000

r = measure_shiftwn(window)
vs = r["vortex"]["coherence"]; drift = r["vortex"]["drift_direction"]; ph = r["photonics"]

if vs > vortex_th and drift > drift_th:
    signal, sig_icon, hold, sig_dir = "HEDGE_BUY", "🟢", "3–8 Tage halten, dann verkaufen", "BUY"
elif vs > vortex_th and drift < -drift_th:
    signal, sig_icon, hold, sig_dir = "HEDGE_SELL", "🔴", "2–6 Tage halten, dann eindecken", "SELL"
else:
    signal, sig_icon, hold, sig_dir = "HOLD", "🟠", "Abwarten – kein klares Signal", "NEUTRAL"

# ---------- Kopfzeile ----------
st.markdown(f"#### Analyse {market_name} · {gran_name} · {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
if used and used not in ("^GDAXI","^GSPC","^DJI","^IXIC","^TECDAX","BTC-USD","GC=F"):
    st.caption(f"Hinweis: Daten über Ersatz-Ticker `{used}` geladen.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Signal", f"{sig_icon} {signal}")
c2.metric("Aktueller Preis", f"{current_price:,.2f}")
c3.metric("Vortex Coherence", f"{vs:.3f}")
c4.metric("Drift", f"{drift:+.3f}")
st.info(f"📅 **Empfohlene Haltedauer:** {hold}  ·  **Marktmodus (Photonics):** {ph['modus']} (dominant: {ph['dominant']})")

# ---------- Charts ----------
left, right = st.columns([3, 2])
with left:
    st.markdown(f"##### Preisverlauf {market_name} mit Fibonacci")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=closes, mode="lines", line=dict(color="#00ff9d", width=2.5), name=market_name))
    for nm, pr in fibonacci_levels(closes).items():
        fig.add_hline(y=pr, line_dash="dash", line_color="rgba(255,210,0,0.30)", annotation_text=nm,
                      annotation_font_color="rgba(255,210,0,0.7)")
    fig.update_layout(height=440, template="plotly_dark", paper_bgcolor="#0a0e16", plot_bgcolor="#0a0e16",
                      margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("##### Photonics-Gewichtung der Operatoren")
    fig2 = go.Figure(go.Bar(
        x=[ph["w"][0], ph["w"][1], ph["w"][2]], y=["Triangle", "Vortex", "Impulse FFT"],
        orientation="h", marker_color=["#00ff9d", "#00b3ff", "#ff9d3c"],
        text=[f"{ph['w'][0]:.0%}", f"{ph['w'][1]:.0%}", f"{ph['w'][2]:.0%}"], textposition="auto"))
    fig2.update_layout(height=200, template="plotly_dark", paper_bgcolor="#0a0e16", plot_bgcolor="#0a0e16",
                       margin=dict(l=0,r=10,t=10,b=20), xaxis_range=[0,1], xaxis_title="adaptives Gewicht")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"T={ph['T']:.2f} · V={ph['V']:.2f} · F={ph['F']:.2f}  — Photonics gewichtet signal-adaptiv "
               f"und bestimmt den führenden Operator.")

# ============================================================
#  KI-WÄCHTER
# ============================================================
if external and ki_on:
    ki = parse_ki(external)
    st.markdown("---")
    st.subheader("🛡️ KI-Wächter Auswertung")
    a, b = st.columns(2)
    with a:
        st.markdown("**Externe KI sagt:**")
        st.info(external)
        konf = f" · Konfidenz {ki['confidence']} %" if ki["confidence"] is not None else ""
        st.caption(f"Erkannte Richtung: **{ki['direction']}**{konf} · Ziel {ki['target']} · Stop {ki['stop']}")
    with b:
        st.markdown("**ShiftWN sagt:**")
        st.success(f"{sig_icon} {signal} · {hold}")
        st.caption(f"Vortex {vs:.3f} · Drift {drift:+.3f} · Modus {ph['modus']}")

    d = ki["direction"]
    if d == "CONFLICT":
        st.warning("⚠️ Externe Empfehlung mehrdeutig (Kauf- und Verkaufsbegriffe). Bitte manuell prüfen.")
    elif sig_dir == "NEUTRAL":
        if d in ("BUY","SELL"):
            st.error(f"❌ ShiftWN widerspricht: Externe KI sagt **{d}**, ShiftWN sieht kein kohärentes Signal "
                     f"(Vortex {vs:.3f}, Drift {drift:+.3f}).")
        else:
            st.info("Beide Seiten sehen kein klares Signal (HOLD).")
    elif d == sig_dir:
        st.success(f"✅ ShiftWN bestätigt: Externe Richtung **{d}** ist mit der Marktgeometrie kohärent "
                   f"(Signal {signal}, Modus {ph['modus']}).")
    elif d == "HOLD":
        st.warning(f"⚠️ Externe KI sagt HOLD – ShiftWN sieht jedoch ein klares Signal ({signal}).")
    else:
        st.error(f"❌ ShiftWN widerspricht: Externe KI sagt **{d}**, ShiftWN sagt **{signal}**.")

st.markdown("---")
st.caption("ShiftWN AI · Patentierter geometrischer Kern (Triangle · Vortex · Impulse FFT · Photonics-Fusion). "
           "Geometrisches Korrektiv über bestehender Trading-Intelligenz. Keine Anlageberatung.")
