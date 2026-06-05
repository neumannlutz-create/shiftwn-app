import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

st.set_page_config(page_title="ShiftWN AI", layout="wide", page_icon="⚡")

st.title("⚡ ShiftWN AI – Geometrische Marktanalyse")
st.markdown("**Patent EPO SPECEPO-1/2** | 3-6-9 + Photonic Fusion")

# ==================== CORE (gleich geblieben) ====================
def _normalize(window):
    o, h, l, c, v = window[:,0], window[:,1], window[:,2], window[:,3], window[:,4]
    ref = np.median(c) if np.median(c) > 0 else 1.0
    scale = np.median(np.abs(np.diff(c))) or 0.01
    norm = np.empty_like(window, dtype=np.float64)
    norm[:,0] = (o - ref) / scale
    norm[:,1] = (h - ref) / scale
    norm[:,2] = (l - ref) / scale
    norm[:,3] = (c - ref) / scale
    norm[:,4] = v / (np.median(v) or 1.0)
    return norm

def triangle_analysis(window):
    c = _normalize(window)[:,3]
    if len(c) < 3: return {"detail": "Zu wenig Daten"}
    areas = [0.5 * ((i-1 - i)*(c[i+1]-c[i-1]) - (i+1 - i)*(c[i]-c[i-1])) for i in range(1, len(c)-1)]
    x = np.arange(len(areas))
    slope = np.polyfit(x, np.abs(areas), 1)[0] if len(areas) >= 2 else 0.0
    return {"convergence": float(slope), "detail": "Starkes Dreieck" if abs(slope) < 0.01 else "Divergierend"}

def vortex_analysis(window):
    c = _normalize(window)[:,3]
    if len(c) < 4: return {"coherence": 0.0, "drift_direction": 0.0, "detail": "Zu wenig Daten"}
    pos = c - np.mean(c)
    vel = np.gradient(pos)
    P = np.column_stack([pos, vel])
    ang = np.arctan2(P[:,1], P[:,0])
    dphi = np.diff(ang)
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    step = np.abs(dphi)
    coherence = float(np.clip(np.mean(step) / (np.std(step) + np.mean(step) + 1e-12) * 2.0, 0, 1))
    slope = float(np.polyfit(np.arange(len(c)), c, 1)[0] if len(c) >= 2 else 0.0)
    drift = float(np.tanh(slope * 3.0))
    return {"coherence": coherence, "drift_direction": drift, "detail": "Starker Vortex" if coherence > 0.85 else "Schwacher Vortex"}

def impulse_analysis(window):
    c = _normalize(window)[:,3]
    r = np.diff(c)
    if len(r) < 4: return {"detail": "Zu wenig Daten"}
    r = r - np.mean(r)
    w = np.hanning(len(r))
    spec = np.abs(np.fft.rfft(r * w)) ** 2
    dominant = float(np.max(spec) / (np.sum(spec) or 1.0))
    return {"detail": "Starke Impulsbewegung" if dominant > 0.15 else "Schwache Frequenz"}

def measure_shiftwn(window):
    return {
        "triangle": triangle_analysis(window),
        "vortex": vortex_analysis(window),
        "impulse": impulse_analysis(window)
    }

# ==================== UI ====================
st.sidebar.header("Markt")
market = st.sidebar.radio("", ["Bitcoin (BTC-USD)", "Phelix DE (Strom)"])

if market == "Bitcoin (BTC-USD)":
    df = yf.download("BTC-USD", period="1y", progress=False)
    closes = df['Close'].values.flatten()
else:
    uploaded = st.sidebar.file_uploader("CSV hochladen", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        closes = df['Close'].values.flatten()
    else:
        closes = np.random.normal(120, 35, 180).cumsum() + 100

days = st.sidebar.slider("Tage Historie", 30, 365, 180)

email = st.sidebar.text_input("Email für Alerts (optional)")
email_password = st.sidebar.text_input("Email-Passwort (optional)", type="password")

if st.button("⚡ ShiftWN-Analyse starten", type="primary"):
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
    fusion = vortex_score * 0.65 + (1 - abs(readings["triangle"]["convergence"])) * 0.35
    ki_conf = fusion * 1.5 if vortex_score > 0.82 else fusion * 0.25

    current_price = float(closes[-1])

    if ki_conf > 0.75 and drift > 0.18:
        signal = "HEDGE_BUY"
        color = "🟢"
        zeit = "Heute oder morgen kaufen"
        haltez = "3–8 Tage"
    elif ki_conf > 0.75 and drift < -0.18:
        signal = "HEDGE_SELL"
        color = "🔴"
        zeit = "Heute oder morgen verkaufen"
        haltez = "2–6 Tage"
    else:
        signal = "HOLD"
        color = "🟠"
        zeit = "Abwarten"
        haltez = "Keine Position"

    # Schöne Ausgabe
    st.subheader("Analyse-Ergebnis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Signal", f"{color} {signal}", f"{ki_conf:.1%}")
    col2.metric("Vortex Coherence", f"{vortex_score:.3f}")
    col3.metric("Drift", f"{drift:.3f}")

    st.write("**Dreiecksanalyse**:", readings["triangle"]["detail"])
    st.write("**Vortex-Analyse**:", readings["vortex"]["detail"])
    st.write("**Frequenzanalyse**:", readings["impulse"]["detail"])

    st.success(f"**Empfohlene Aktion:** {zeit}")
    st.info(f"**Haltezeit:** {haltez}")

    if email and email_password and signal != "HOLD":
        try:
            msg = MIMEText(f"ShiftWN Signal: {signal}\nMarkt: {market}\nKonfidenz: {ki_conf:.1%}\nZeit: {datetime.now()}")
            msg['Subject'] = f"ShiftWN Alert: {signal}"
            msg['From'] = email
            msg['To'] = email
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(email, email_password)
            server.sendmail(email, email, msg.as_string())
            server.quit()
            st.success("Email-Alert gesendet!")
        except:
            st.warning("Email konnte nicht gesendet werden.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(closes[-100:]))), y=closes[-100:], mode='lines', name=market))
    st.plotly_chart(fig, width='stretch')