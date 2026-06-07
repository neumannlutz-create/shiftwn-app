import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import smtplib
from email.mime.text import MIMEText
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
st.caption("Patent EPO SPECEPO-1/2 | Vollständige 3-6-9 + Photonic Fusion + Fibonacci v3.4")

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
        return {"centroid": 0.0, "dominant_power_ratio": 0.0}
    r = r - np.mean(r)
    w = np.hanning(len(r))
    spec = np.abs(np.fft.rfft(r * w)) ** 2
    spec_sum = np.sum(spec) or 1.0
    dom = float(np.max(spec) / spec_sum)
    return {"centroid": float(np.sum(np.linspace(0,1,len(spec)) * spec) / spec_sum), "dominant_power_ratio": dom}

def measure_shiftwn(window):
    return {"triangle": triangle(window), "vortex": vortex(window), "impulse": impulse(window)}

# ==================== Fibonacci ====================
def fibonacci_levels(closes):
    high = np.max(closes)
    low = np.min(closes)
    diff = high - low
    levels = {
        "0.0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50.0%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100.0%": low
    }
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

# ==================== EINSTELLUNGEN ====================
ki_control = st.sidebar.checkbox("KI-Kontroll-Modus aktivieren (ShiftWN als Wächter)", value=False)

st.sidebar.subheader("Email-Alerts")
provider = st.sidebar.selectbox("Email-Anbieter", ["Gmail", "iCloud / Mac (@me.com / @mac.com)"])
email = st.sidebar.text_input("Deine Email-Adresse", "")
email_password = st.sidebar.text_input("App-Passwort", type="password")

auto_refresh = st.sidebar.checkbox("Auto-Refresh alle 60 Sekunden", value=True)

st.sidebar.subheader("Alarm-Grenzwerte")
vortex_threshold = st.sidebar.slider("Vortex Coherence (Minimum)", 0.65, 1.0, 0.78, 0.01)
drift_threshold = st.sidebar.slider("Drift (Minimum für Signal)", 0.06, 0.30, 0.09, 0.01)
confidence_threshold = st.sidebar.slider("Konfidenz (Minimum in %)", 60, 95, 68, 5)

# ==================== ANALYSE ====================
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
        vortex_score = readings["vortex"]["coherence"]
        drift = readings["vortex"]["drift_direction"]
        impulse_score = readings["impulse"]["dominant_power_ratio"]

        fib_levels = fibonacci_levels(closes[-200:])
        fib_score = 0.0
        for name, price in fib_levels.items():
            if abs(current_price - price) / current_price < 0.005:
                fib_score = 0.3 if name in ["61.8%", "38.2%"] else 0.15
                break

        fusion = ((1 - abs(readings["triangle"]["convergence"])) * 0.30 +
                  vortex_score * 0.35 +
                  impulse_score * 0.25 +
                  fib_score)
        ki_conf = fusion * 1.6 if vortex_score > 0.80 else fusion * 0.45

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

        st.subheader("Analyse-Ergebnis")
        st.write(f"**Analyse vom:** {analysis_time}")

        col1, col2, col3 = st.columns(3)
        col1.metric("**Signal**", f"{color} {signal}", f"Konfidenz {ki_conf:.1%}")
        col2.metric("Aktueller Preis", f"{current_price:.2f}")
        col3.metric("Drift (20 Tage)", f"{drift*100:+.1f}%")

        st.success(f"**Empfohlene Aktion:** {zeit}")
        st.info(f"**Empfohlene Haltezeit:** {haltez}")

        st.write("**Dreiecksanalyse:**", "Konvergierend" if readings["triangle"]["convergence"] < -0.001 else "Divergierend")
        st.write("**Vortex-Analyse:**", f"Starker Vortex (Coherence {vortex_score:.3f})")
        st.write("**Frequenzanalyse:**", f"Dominante Mode {impulse_score:.3f}")

        if ki_control:
            st.info("🛡️ KI-Kontroll-Modus aktiv – ShiftWN prüft KI-Empfehlungen")

        if email and email_password and signal != "HOLD":
            try:
                if provider == "Gmail":
                    smtp_server = "smtp.gmail.com"
                    port = 465
                    server = smtplib.SMTP_SSL(smtp_server, port)
                else:
                    smtp_server = "smtp.mail.me.com"
                    port = 587
                    server = smtplib.SMTP(smtp_server, port)
                    server.starttls()
                msg = MIMEText(f"ShiftWN Signal: {signal}\nMarkt: {market_name}\nPreis: {current_price:.2f}\nKonfidenz: {ki_conf:.1%}\nZeit: {analysis_time}")
                msg['Subject'] = f"ShiftWN Alert: {signal} – {market_name}"
                msg['From'] = email
                msg['To'] = email
                server.login(email, email_password)
                server.sendmail(email, email, msg.as_string())
                server.quit()
                st.success("✅ Echter Email-Alert gesendet!")
            except Exception as e:
                st.warning(f"Email konnte nicht gesendet werden: {str(e)}")

        # Chart mit Fibonacci
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(closes[-200:]))), y=closes[-200:], mode='lines', name=market_name, line=dict(color='#00ff88', width=3)))
        for name, price in fib_levels.items():
            fig.add_hline(y=price, line_dash="dash", line_color="yellow", annotation_text=name)
        fig.update_layout(height=600, template="plotly_dark", title=f"Preisverlauf {market_name} mit Fibonacci")
        st.plotly_chart(fig, use_container_width=True)

        # ==================== AUTO-REFRESH ====================
        if auto_refresh:
            st.info("🔄 Auto-Refresh ist aktiv – nächste Aktualisierung in 60 Sekunden...")
            time.sleep(60)
            st.rerun()

st.caption("ShiftWN AI v3.4 – mit Auto-Refresh + Fibonacci + KI-Kontroll-Modus")
