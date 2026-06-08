import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import time

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
st.caption("Patent EPO SPECEPO-1/2 | v4.0 – Haltedauer klar sichtbar")

# ==================== Kern-Funktionen ====================
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
        return {"centroid": 0.0, "dominant_power_ratio": 0.0}
    r = r - np.mean(r)
    w = np.hanning(len(r))
    spec = np.abs(np.fft.rfft(r * w)) ** 2
    spec_sum = np.sum(spec) or 1.0
    dom = float(np.max(spec) / spec_sum)
    return {"centroid": float(np.sum(np.linspace(0,1,len(spec)) * spec) / spec_sum), "dominant_power_ratio": dom}

def measure_shiftwn(window):
    return {"triangle": triangle(window), "vortex": vortex(window), "impulse": impulse(window)}

def fibonacci_levels(closes):
    high = np.max(closes)
    low = np.min(closes)
    diff = high - low
    levels = {"0.0%": high, "23.6%": high - 0.236*diff, "38.2%": high - 0.382*diff,
              "50.0%": high - 0.5*diff, "61.8%": high - 0.618*diff,
              "78.6%": high - 0.786*diff, "100.0%": low}
    return levels

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

@st.cache_data(ttl=60)
def get_data(ticker):
    if ticker:
        try:
            df = yf.download(ticker, period="1y", progress=False)
            return df['Close'].values.flatten()
        except:
            return np.array([])
    else:
        np.random.seed(42)
        price = 120.0
        prices = []
        for _ in range(180):
            vol = np.random.normal(0, 35)
            if np.random.rand() < 0.1: vol += np.random.choice([-150, 150])
            price = max(5, min(450, price + vol))
            prices.append(price)
        return np.array(prices)

closes_full = get_data(ticker)
days = st.sidebar.slider("Tage Historie", 30, 365, 180)
closes = closes_full[-days:] if len(closes_full) > days else closes_full

st.sidebar.subheader("🔄 Echtzeit-Update")
dauer_refresh = st.sidebar.checkbox("Dauer-Auto-Refresh aktivieren (alle 60 Sekunden)", value=True)

st.sidebar.subheader("KI-Wächter Modus")
external_message = st.sidebar.text_area("Hier KI-Empfehlung einfügen", height=120, placeholder="Kopiere hier die Empfehlung von ChatGPT, Grok etc. hinein...")
ki_control = st.sidebar.checkbox("ShiftWN als KI-Wächter aktivieren", value=True)

st.sidebar.subheader("Alarm-Grenzwerte")
vortex_threshold = st.sidebar.slider("Vortex Coherence (Minimum)", 0.60, 1.0, 0.70, 0.01)
drift_threshold = st.sidebar.slider("Drift (Minimum für Signal)", 0.04, 0.30, 0.06, 0.01)
confidence_threshold = st.sidebar.slider("Konfidenz (Minimum in %)", 55, 95, 62, 1)

# ==================== ANALYSE ====================
if len(closes) == 0:
    st.error("❌ Keine Marktdaten verfügbar.")
    st.stop()

analysis_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
current_price = float(closes[-1])

window_size = min(50, len(closes))
window = np.zeros((window_size, 5))
window[:, 3] = closes[-window_size:]
window[:, 0] = closes[-window_size:] * 0.97
window[:, 1] = closes[-window_size:] * 1.08
window[:, 2] = closes[-window_size:] * 0.92
window[:, 4] = 15000

readings = measure_shiftwn(window)
vortex_score = readings["vortex"]["coherence"]
drift = readings["vortex"]["drift_direction"]

if vortex_score > vortex_threshold and drift > drift_threshold:
    signal = "HEDGE_BUY"
    color = "🟢"
    haltez = "3–8 Tage halten, dann verkaufen"
elif vortex_score > vortex_threshold and drift < -drift_threshold:
    signal = "HEDGE_SELL"
    color = "🔴"
    haltez = "2–6 Tage halten, dann verkaufen"
else:
    signal = "HOLD"
    color = "🟠"
    haltez = "Abwarten – kein klares Signal"

st.subheader(f"Analyse um {analysis_time}")

col1, col2, col3 = st.columns(3)
col1.metric("**Signal**", f"{color} {signal}")
col2.metric("Aktueller Preis", f"{current_price:.2f}")
col3.metric("Vortex Coherence", f"{vortex_score:.3f}")

# === GROßE HALTE-DAUER ANZEIGE ===
st.info(f"**📅 Empfohlene Haltedauer:** {haltez}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(closes[-200:]))), y=closes[-200:], mode='lines', name=market_name, line=dict(color='#00ff88', width=3)))
fib_levels = fibonacci_levels(closes[-200:])
for name, price in fib_levels.items():
    fig.add_hline(y=price, line_dash="dash", line_color="yellow", annotation_text=name)
fig.update_layout(height=550, template="plotly_dark", title=f"Preisverlauf {market_name} mit Fibonacci")
st.plotly_chart(fig, use_container_width=True)

if external_message and ki_control:
    st.subheader("🛡️ KI-Wächter Auswertung")
    st.info(external_message)

st.success(f"Automatisch aktualisiert um {datetime.now().strftime('%H:%M:%S')}")

if dauer_refresh:
    st.info("🔄 Dauer-Auto-Refresh aktiv – nächste Aktualisierung in 60 Sekunden")
    time.sleep(1)
    st.rerun()

st.caption("ShiftWN AI v4.0 – Haltedauer klar sichtbar")
