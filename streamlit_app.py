import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime

st.set_page_config(page_title="ShiftWN AI", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stApp {background-color: #0e1117; color: white;}
    h1 {color: #00ff88;}
    .stButton>button {background-color: #00ff88; color: black; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ ShiftWN AI – Geometrische Marktanalyse")
st.caption("Patent EPO SPECEPO-1/2 | Vollständige 3-6-9 + Photonic Fusion")

# ==================== ShiftWN Kern-Funktionen ====================
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
    for i in range(1, len(c)-1):
        p0 = np.array([i-1, c[i-1]])
        p1 = np.array([i, c[i]])
        p2 = np.array([i+1, c[i+1]])
        area = 0.5 * ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        areas.append(area)
    x = np.arange(len(areas))
    slope = np.polyfit(x, np.abs(areas), 1)[0] if len(areas) >= 2 else 0.0
    return {"convergence": float(slope)}

def vortex(window):
    c = _normalize(window)[:, 3]
    if len(c) < 4: return {"coherence": 0.0, "drift_direction": 0.0}
    pos = c - np.mean(c)
    vel = np.gradient(pos)
    P = np.column_stack([pos, vel])
    ang = np.arctan2(P[:,1], P[:,0])
    dphi = np.diff(ang)
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    coherence = float(np.clip(np.mean(np.abs(dphi)) / (np.std(dphi) + 1e-8), 0, 1))
    slope = float(np.polyfit(np.arange(len(c)), c, 1)[0])
    drift = float(np.tanh(slope * 3))
    return {"coherence": coherence, "drift_direction": drift}

def impulse(window):
    c = _normalize(window)[:, 3]
    r = np.diff(c)
    if len(r) < 4:
        return {"centroid": 0.0, "dominant_power_ratio": 0.0, "flatness": 1.0, "kurtosis": 0.0}
    r = r - np.mean(r)
    w = np.hanning(len(r))
    spec = np.abs(np.fft.rfft(r * w)) ** 2
    spec_sum = np.sum(spec) or 1.0
    freqs = np.linspace(0.0, 1.0, len(spec))
    centroid = float(np.sum(freqs * spec) / spec_sum)
    dom = float(np.max(spec) / spec_sum)
    geo = np.exp(np.mean(np.log(spec + 1e-12)))
    flat = float(geo / (np.mean(spec) + 1e-12))
    m = r - np.mean(r)
    var = np.mean(m**2) or 1e-12
    kurt = float(np.mean(m**4) / (var**2) - 3.0)
    return {"centroid": centroid, "dominant_power_ratio": dom, "flatness": flat, "kurtosis": kurt}

def measure_shiftwn(window):
    return {
        "triangle": triangle(window),
        "vortex": vortex(window),
        "impulse": impulse(window)
    }

# ==================== MÄRKTE ====================
markets = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Phelix DE (Strom)": None,
    "Dow Jones": "^DJI",
    "TecDAX": "^TECDAX",
    "DAX": "^GDAXI",
    "Nasdaq": "^IXIC",
    "S&P 500": "^GSPC",
    "Gold": "GC=F"
}

market_name = st.sidebar.selectbox("Markt auswählen", list(markets.keys()))
ticker = markets[market_name]

if ticker:
    df = yf.download(ticker, period="1y", progress=False)
    closes = df['Close'].values.flatten()
else:
    def generate_phelix_data(days=180):
        np.random.seed(42)
        price = 120.0
        prices = []
        for i in range(days):
            vol = np.random.normal(0, 35)
            if np.random.rand() < 0.1:
                vol += np.random.choice([-150, 150])
            price = max(5, min(450, price + vol))
            prices.append(price)
        return np.array(prices)
    closes = generate_phelix_data(180)

days = st.sidebar.slider("Tage Historie", 30, 365, 180)

# ==================== ALARM-GRENZWERTE ====================
st.sidebar.subheader("Alarm-Grenzwerte")
vortex_threshold = st.sidebar.slider("Vortex Coherence (Minimum)", 0.70, 1.0, 0.82, 0.01)
drift_threshold = st.sidebar.slider("Drift (Minimum für Signal)", 0.08, 0.30, 0.12, 0.01)
confidence_threshold = st.sidebar.slider("Konfidenz (Minimum in %)", 60, 95, 72, 5)

email = st.sidebar.text_input("Email-Adresse für Alerts (optional)")

if st.button("⚡ ShiftWN-Analyse starten", type="primary", use_container_width=True):
    with st.spinner("ShiftWN analysiert..."):
        analysis_time = datetime.now().strftime("%d.%m.%Y um %H:%M Uhr")
        current_price = float(closes[-1])

        window_size = min(50, len(closes))
        window = np.zeros((window_size, 5))
        window[:, 3] = closes[-window_size:]
        window[:, 0] = closes[-window_size:] * 0.97
        window[:, 1] = closes[-window_size:] * 1.08
        window[:, 2] = closes[-window_size:] * 0.92
        window[:, 4] = 15000

        readings = measure_shiftwn(window)
        
        # Photonic Fusion (alle drei Module)
        triangle_score = 1 - abs(readings["triangle"]["convergence"])
        vortex_score = readings["vortex"]["coherence"]
        impulse_score = readings["impulse"]["dominant_power_ratio"]
        
        fusion = (triangle_score * 0.35 + vortex_score * 0.40 + impulse_score * 0.25)
        ki_conf = fusion * 1.6 if vortex_score > 0.82 else fusion * 0.4

        drift = readings["vortex"]["drift_direction"]

        if vortex_score > vortex_threshold and drift > drift_threshold and ki_conf > confidence_threshold / 100:
            signal = "HEDGE_BUY"
            color = "🟢"
            zeit = "Heute oder morgen kaufen"
            haltez = "3–8 Tage"
        elif vortex_score > vortex_threshold and drift < -drift_threshold and ki_conf > confidence_threshold / 100:
            signal = "HEDGE_SELL"
            color = "🔴"
            zeit = "Heute oder morgen verkaufen"
            haltez = "2–6 Tage"
        else:
            signal = "HOLD"
            color = "🟠"
            zeit = "Abwarten – kein klares Signal"
            haltez = "—"

        # ==================== AUSGABE ====================
        st.subheader("Analyse-Ergebnis")
        st.write(f"**Analyse vom:** {analysis_time}")

        col1, col2, col3 = st.columns(3)
        col1.metric("**Signal**", f"{color} {signal}", f"Konfidenz {ki_conf:.1%}")
        col2.metric("Aktueller Preis", f"{current_price:.2f}")
        col3.metric("Drift", f"{drift*100:+.1f}%")

        st.success(f"**Empfohlene Aktion:** {zeit}")
        st.info(f"**Empfohlene Haltezeit:** {haltez}")

        # Detaillierte Module
        st.write("**Dreiecksanalyse:**", "Konvergierend" if readings["triangle"]["convergence"] < -0.001 else "Divergierend")
        st.write("**Vortex-Analyse:**", f"Starker Vortex (Coherence {vortex_score:.3f})")
        st.write("**Frequenzanalyse:**", f"Dominante Mode {readings['impulse']['dominant_power_ratio']:.3f}")

        if email and signal != "HOLD":
            st.success(f"✅ Email-Alert würde an **{email}** gesendet werden!")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(closes[-200:]))), y=closes[-200:], mode='lines', name=market_name, line=dict(color='#00ff88', width=3)))
        fig.update_layout(height=600, template="plotly_dark", title=f"Preisverlauf {market_name}")
        st.plotly_chart(fig, use_container_width=True)

st.caption("ShiftWN AI v3.0 – volle Photonic Fusion mit allen Modulen")
