"""
ShiftWN AI – Geometrische Marktanalyse mit adaptivem Photonics-Kern
Patentierter Kern: Triangle · Vortex · Impulse FFT · Photonics.
Photonics gewichtet signal-adaptiv nach AUSSAGEKRAFT (Blindheits-Erkennung)
und detektiert Regime-Brüche/Schocks (Vol-Sprung UND Kohärenz-Einbruch).
Universelles Prinzip: erkennt, wann ein Modul nichts mehr analysiert, und
verschiebt autonom auf die tragenden Module.
Patent EPA EP25221251.9 / SPECEPO-1/2.  Keine Anlageberatung.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import re
from datetime import datetime

st.set_page_config(page_title="ShiftWN AI", layout="wide", page_icon="⚡")
st.markdown("""
<style>
    .stApp {background-color:#0a0e16;color:#e8edf4;}
    section[data-testid="stSidebar"]{background-color:#0f1521;}
    h1{color:#00ff9d;font-weight:800;letter-spacing:-.5px;}
    h2,h3{color:#d6f5e6;}
    .stMetric{background:#141b29;border:1px solid #1e2838;border-radius:12px;padding:14px 16px;}
    [data-testid="stMetricValue"]{color:#00ff9d;}
    [data-testid="stMetricLabel"]{color:#9aa6b8;}
    .stButton>button{background:#00ff9d;color:#06210f;font-weight:700;border:none;border-radius:8px;}
    div[data-testid="stMarkdownContainer"] code{color:#00ff9d;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ ShiftWN AI")
st.caption("Adaptive geometrische Marktanalyse · Patent EPA EP25221251.9 · Triangle · Vortex · Impulse FFT · Photonics")

# ============================================================
#  PATENTIERTER SHIFTWN-KERN
#  Jeder Operator liefert score + kappa (Aussagekraft, 0..1).
#  Photonics gewichtet nach kappa: blinde Module verlieren Gewicht.
# ============================================================
def _normalize(window):
    c = window[:, 3]
    ref = np.median(c) if np.median(c) > 0 else 1.0
    scale = np.median(np.abs(np.diff(c))) or 0.01 * ref
    n = np.zeros_like(window, dtype=float)
    n[:,0]=(window[:,0]-ref)/scale; n[:,1]=(window[:,1]-ref)/scale
    n[:,2]=(window[:,2]-ref)/scale; n[:,3]=(c-ref)/scale
    n[:,4]=window[:,4]/(np.median(window[:,4]) or 1.0); return n

def triangle(window):
    """Trend-/Form-Geometrie. kappa = Güte der geometrischen Form
    (klarer Trend ODER zulaufender Keil = hoch; Rauschen = niedrig)."""
    c=_normalize(window)[:,3]; x=np.arange(len(c))
    if len(c)<5: return {"score":0.0,"kappa":0.0,"dir":0.0}
    lin=np.polyfit(x,c,1); fit=np.polyval(lin,x)
    ss_tot=np.sum((c-np.mean(c))**2)+1e-9
    r2=1-np.sum((c-fit)**2)/ss_tot
    h=len(c)//2; re=np.ptp(c[:h]); rl=np.ptp(c[h:])
    wedge=np.clip((re-rl)/(re+1e-9),0,1)
    kappa=float(np.clip(max(r2, wedge*0.9),0,1))
    return {"score":float(np.clip(abs(lin[0])*6,0,1)),"kappa":kappa,"dir":float(np.tanh(lin[0]*3))}

def vortex(window):
    """Rotation/Kohärenz im Phasenraum. kappa = stabile Rotation, fällt bei
    Volatilitäts-Bruch (Regime-Wechsel = Vortex wird blind)."""
    c=_normalize(window)[:,3]
    if len(c)<6: return {"score":0.0,"kappa":0.0,"dir":0.0,"vol_break":0.0}
    pos=c-np.mean(c); vel=np.gradient(pos); ang=np.arctan2(vel,pos)
    dphi=np.diff(ang); dphi=(dphi+np.pi)%(2*np.pi)-np.pi
    score=float(np.clip(np.mean(np.abs(dphi))/(np.std(dphi)+1e-8),0,1))
    rot_stab=np.clip(1.0-np.std(dphi)/(np.pi/2),0,1)
    h=len(c)//2; v1=np.std(np.diff(c[:h]))+1e-9; v2=np.std(np.diff(c[h:]))+1e-9
    vol_break=float(np.clip(abs(np.log(v2/v1))/1.5,0,1))
    kappa=float(np.clip(rot_stab*(1-vol_break),0,1))
    return {"score":score,"kappa":kappa,"dir":float(np.tanh(np.polyfit(np.arange(len(c)),c,1)[0]*3)),"vol_break":vol_break}

def impulse(window):
    """Frequenz/Eruption. kappa = spektrale Konzentration (scharfer Peak vs
    flaches weißes Rauschen). Wird bei Eruptionen laut."""
    c=_normalize(window)[:,3]; r=np.diff(c)
    if len(r)<5: return {"score":0.0,"kappa":0.0,"dir":0.0}
    r=r-np.mean(r); w=np.hanning(len(r)); spec=np.abs(np.fft.rfft(r*w))**2
    ss=np.sum(spec) or 1.0; p=spec/ss
    ent=-np.sum(p*np.log(p+1e-12))/np.log(len(p))
    return {"score":float(np.max(p)),"kappa":float(np.clip((1-ent)*1.8,0,1)),"dir":0.0}

def photonics(tri, vor, imp, prev_mean_kappa=None):
    """Photonics-Fusion: Gewicht folgt der Aussagekraft kappa (nicht der
    Score-Höhe). Erkennt Regime-Bruch/Schock, wenn Volatilität springt UND
    die mittlere Aussagekraft gegenüber dem Vorfenster einbricht."""
    ks=np.array([tri["kappa"], vor["kappa"], imp["kappa"]])
    ss=np.array([tri["score"], vor["score"], imp["score"]])
    w = np.ones(3)/3 if ks.sum()<1e-6 else ks/ks.sum()
    phases=np.array([0, 2*np.pi/3, 4*np.pi/3])
    den=np.sum(w*ks*ss)+1e-9
    field_coh=float(np.clip(np.abs(np.sum(w*ks*ss*np.exp(1j*phases)))/den,0,1))
    mean_kappa=float(ks.mean())
    kappa_drop = 0.0 if prev_mean_kappa is None else max(0.0, prev_mean_kappa-mean_kappa)
    vol_break = vor.get("vol_break",0.0)
    regime_break = (vol_break>0.45 and kappa_drop>0.12)   # (C): beides zusammen
    if regime_break:
        modus,dominant,lead = "REGIME-BRUCH / SCHOCK","Impulse FFT",2; drift=vor["dir"]
    else:
        lead=int(np.argmax(ks))
        modus=["TRENDING","RANGING","VOLATILE"][lead]
        dominant=["Triangle","Vortex","Impulse FFT"][lead]
        drift=[tri["dir"],vor["dir"],vor["dir"]][lead]
    return {"w":w,"kappa":ks,"modus":modus,"dominant":dominant,"field_coh":field_coh,
            "mean_kappa":mean_kappa,"kappa_drop":float(kappa_drop),"vol_break":vol_break,
            "regime_break":regime_break,"drift":drift,
            "T":tri["kappa"],"V":vor["kappa"],"F":imp["kappa"]}

def _build(seg):
    n=len(seg); w=np.zeros((n,5))
    w[:,3]=seg; w[:,0]=seg*0.97; w[:,1]=seg*1.08; w[:,2]=seg*0.92; w[:,4]=15000; return w

def analyse(closes, win=50):
    cur=closes[-win:]
    prev=closes[-2*win:-win] if len(closes)>=2*win else None
    pmk=None
    if prev is not None and len(prev)>=20:
        pmk=np.mean([triangle(_build(prev))["kappa"], vortex(_build(prev))["kappa"], impulse(_build(prev))["kappa"]])
    return photonics(triangle(_build(cur)), vortex(_build(cur)), impulse(_build(cur)), pmk)

def fibonacci_levels(closes):
    hi,lo=np.max(closes),np.min(closes); d=hi-lo
    return {"0.0%":hi,"23.6%":hi-0.236*d,"38.2%":hi-0.382*d,"50.0%":hi-0.5*d,"61.8%":hi-0.618*d,"78.6%":hi-0.786*d,"100.0%":lo}

# ============================================================
#  KI-WÄCHTER PARSER  (Wortgrenzen-Fix)
# ============================================================
def parse_ki(text):
    out={"direction":"HOLD","confidence":None,"target":None,"stop":None}
    if not text: return out
    tl=text.lower()
    m=re.search(r"(?:konfidenz|confidence)\s*[:=]?\s*(\d{1,3})\s*%?", tl)
    out["confidence"]=int(m.group(1)) if m else None
    tlp=re.sub(r"(?:konfidenz|confidence)\s*[:=]?\s*\d{1,3}\s*%?"," ",tl)
    num=r"(\d{1,3}(?:\.\d{3})+|\d{4,6})"; f=lambda s: float(s.replace(".",""))
    z=re.search(rf"(?:ziel|target|kursziel)\s*[:=]?\s*{num}",tlp); out["target"]=f(z.group(1)) if z else None
    s=re.search(rf"(?:stop[- ]?loss|stop)\s*[:=]?\s*{num}",tlp); out["stop"]=f(s.group(1)) if s else None
    sell=["sell","short","verkaufen","verkauf","bearish"]; buy=["buy","long","kaufen","kauf","bullish"]
    hs=any(re.search(rf"\b{re.escape(x)}\b",tl) for x in sell); hb=any(re.search(rf"\b{re.escape(x)}\b",tl) for x in buy)
    out["direction"]="SELL" if hs and not hb else "BUY" if hb and not hs else "CONFLICT" if hb and hs else "HOLD"
    return out

# ============================================================
#  DATEN
# ============================================================
MARKETS={"DAX":["^GDAXI","EXS1.DE"],"S&P 500":["^GSPC"],"Dow Jones":["^DJI"],"Nasdaq":["^IXIC"],
         "TecDAX":["^TECDAX"],"Bitcoin (BTC-USD)":["BTC-USD"],"Gold":["GC=F"]}
GRAN={"Täglich (2 Jahre)":("2y","1d"),"15-Minuten (1 Monat)":("1mo","15m"),"5-Minuten (5 Tage)":("5d","5m")}

with st.sidebar:
    st.header("Einstellungen")
    market_name=st.selectbox("Markt",list(MARKETS.keys()))
    gran_name=st.selectbox("Granularität",list(GRAN.keys()))
    st.divider()
    st.subheader("Signal-Schwellen")
    drift_th=st.slider("Drift (Minimum für Richtung)",0.04,0.30,0.06,0.01)
    kappa_min=st.slider("Mindest-Aussagekraft für Signal",0.20,0.70,0.35,0.01)
    st.divider()
    st.subheader("🛡️ KI-Wächter")
    external=st.text_area("KI-Empfehlung einfügen",height=140,placeholder="Empfehlung von ChatGPT, Grok etc. ...")
    ki_on=st.checkbox("ShiftWN als KI-Wächter aktivieren",value=True)

ticker_list=MARKETS[market_name]; period,interval=GRAN[gran_name]

@st.cache_data(ttl=120)
def get_data(ticker_list,period,interval,tries=3):
    for tk in ticker_list:
        for _ in range(tries):
            try: df=yf.download(tk,period=period,interval=interval,progress=False,auto_adjust=True)
            except Exception: df=None
            if df is not None and not df.empty:
                k=df["Close"].values.flatten(); k=k[~np.isnan(k)]
                if len(k)>=100: return k,tk
    return None,None

with st.spinner(f"Lade {market_name}-Daten ..."):
    closes_full,used=get_data(tuple(ticker_list),period,interval)

if closes_full is None or len(closes_full)<100:
    st.error("❌ Konnte keine Marktdaten laden. Anderen Markt/Granularität versuchen (oder erneut)."); st.stop()

closes=closes_full[-min(len(closes_full),300):]
current_price=float(closes_full[-1])

# ============================================================
#  ANALYSE
# ============================================================
ph=analyse(closes, win=50)
drift=ph["drift"]; mk=ph["mean_kappa"]

if ph["regime_break"]:
    signal,sig_icon,hold,sig_dir = "REGIME-BRUCH / SCHOCK","⚠️","Vorsicht – gewohnte Struktur gilt nicht. Risiko reduzieren.","SHOCK"
elif mk>kappa_min and drift>drift_th:
    signal,sig_icon,hold,sig_dir = "HEDGE_BUY","🟢","3–8 Perioden halten, dann schließen","BUY"
elif mk>kappa_min and drift<-drift_th:
    signal,sig_icon,hold,sig_dir = "HEDGE_SELL","🔴","2–6 Perioden halten, dann eindecken","SELL"
else:
    signal,sig_icon,hold,sig_dir = "HOLD","🟠","Abwarten – keine ausreichend klare Struktur","NEUTRAL"

st.markdown(f"#### Analyse {market_name} · {gran_name} · {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
if used and used not in ("^GDAXI","^GSPC","^DJI","^IXIC","^TECDAX","BTC-USD","GC=F"):
    st.caption(f"Hinweis: Daten über Ersatz-Ticker `{used}` geladen.")

c1,c2,c3,c4=st.columns(4)
c1.metric("Signal",f"{sig_icon} {signal}")
c2.metric("Aktueller Preis",f"{current_price:,.2f}")
c3.metric("Drift",f"{drift:+.3f}")
c4.metric("Aussagekraft Ø",f"{mk:.2f}")

if ph["regime_break"]:
    st.error(f"⚠️ **REGIME-BRUCH erkannt** – Volatilitätssprung ({ph['vol_break']:.2f}) und Einbruch der "
             f"Aussagekraft ({ph['kappa_drop']:.2f}). Die gewohnte Marktstruktur gilt nicht mehr. "
             f"Photonics schaltet auf Impulse FFT (Eruptions-Erkennung).")
else:
    st.info(f"📅 **Haltedauer:** {hold}  ·  **Marktmodus (Photonics):** {ph['modus']} (führend: {ph['dominant']})")

left,right=st.columns([3,2])
with left:
    st.markdown(f"##### Preisverlauf {market_name} mit Fibonacci")
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=closes,mode="lines",line=dict(color="#00ff9d",width=2.5),name=market_name))
    for nm,pr in fibonacci_levels(closes).items():
        fig.add_hline(y=pr,line_dash="dash",line_color="rgba(255,210,0,.30)",annotation_text=nm,
                      annotation_font_color="rgba(255,210,0,.7)")
    fig.update_layout(height=440,template="plotly_dark",paper_bgcolor="#0a0e16",plot_bgcolor="#0a0e16",
                      margin=dict(l=0,r=0,t=10,b=0),showlegend=False)
    st.plotly_chart(fig,use_container_width=True)
with right:
    st.markdown("##### Photonics: Aussagekraft & Gewichtung")
    fig2=go.Figure(go.Bar(x=[ph["w"][0],ph["w"][1],ph["w"][2]],y=["Triangle","Vortex","Impulse FFT"],
        orientation="h",marker_color=["#00ff9d","#00b3ff","#ff9d3c"],
        text=[f"{ph['w'][0]:.0%}",f"{ph['w'][1]:.0%}",f"{ph['w'][2]:.0%}"],textposition="auto"))
    fig2.update_layout(height=200,template="plotly_dark",paper_bgcolor="#0a0e16",plot_bgcolor="#0a0e16",
                       margin=dict(l=0,r=10,t=10,b=20),xaxis_range=[0,1],xaxis_title="Gewicht (folgt Aussagekraft κ)")
    st.plotly_chart(fig2,use_container_width=True)
    st.caption(f"κ: Triangle {ph['T']:.2f} · Vortex {ph['V']:.2f} · Impulse {ph['F']:.2f}. "
               f"Photonics verschiebt Gewicht auf die Module, die etwas sehen – blinde Module verlieren Einfluss.")

st.markdown(f"### {sig_icon} ShiftWN-Befund: {signal}")

# ============================================================
#  KI-WÄCHTER
# ============================================================
if external and ki_on:
    ki=parse_ki(external)
    st.markdown("---"); st.subheader("🛡️ KI-Wächter Auswertung")
    a,b=st.columns(2)
    with a:
        st.markdown("**Externe KI sagt:**"); st.info(external)
        konf=f" · Konfidenz {ki['confidence']} %" if ki["confidence"] is not None else ""
        st.caption(f"Richtung: **{ki['direction']}**{konf} · Ziel {ki['target']} · Stop {ki['stop']}")
    with b:
        st.markdown("**ShiftWN sagt:**"); st.success(f"{sig_icon} {signal}")
        st.caption(f"Modus {ph['modus']} · Drift {drift:+.3f} · Aussagekraft {mk:.2f}")
    d=ki["direction"]
    if d=="CONFLICT":
        st.warning("⚠️ Externe Empfehlung mehrdeutig (Kauf- und Verkaufsbegriffe). Manuell prüfen.")
    elif sig_dir=="SHOCK":
        st.error(f"⚠️ ShiftWN warnt: REGIME-BRUCH. Die externe Empfehlung (**{d}**) beruht auf der gewohnten "
                 f"Marktstruktur, die gerade nicht mehr gilt. Genau hier laufen normale Modelle blind hinein.")
    elif sig_dir=="NEUTRAL":
        st.error(f"❌ ShiftWN widerspricht: Externe KI sagt **{d}**, ShiftWN sieht keine ausreichend klare Struktur "
                 f"(Aussagekraft {mk:.2f}).") if d in ("BUY","SELL") else st.info("Beide Seiten: kein klares Signal.")
    elif d==sig_dir:
        st.success(f"✅ ShiftWN bestätigt: Richtung **{d}** ist mit der Marktgeometrie kohärent (Modus {ph['modus']}).")
    elif d=="HOLD":
        st.warning(f"⚠️ Externe KI sagt HOLD – ShiftWN sieht ein klares Signal ({signal}).")
    else:
        st.error(f"❌ ShiftWN widerspricht: Externe KI sagt **{d}**, ShiftWN sagt **{signal}**.")

st.markdown("---")
st.caption("ShiftWN AI · Patentierter geometrischer Kern (Triangle · Vortex · Impulse FFT · Photonics). "
           "Photonics gewichtet nach Aussagekraft und erkennt Regime-Brüche. Geometrisches Korrektiv über "
           "bestehender Trading-Intelligenz. Keine Anlageberatung.")
