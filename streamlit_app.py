"""
ShiftWN AI – Geometrische Marktanalyse mit adaptivem Photonics-Kern
Struktur:  1) Multi-Markt-Radar  2) Detailanalyse  3) KI-Wächter
Patentierter Kern: Triangle · Vortex · Impulse FFT · Photonics (Aussagekraft-
Gewichtung + Regime-Bruch-Erkennung).  Patent EPA EP25221251.9 / SPECEPO-1/2.
Keine Anlageberatung.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import re
import time
from datetime import datetime

st.set_page_config(page_title="ShiftWN AI", layout="wide", page_icon="⚡")

# ---------------- Look: ruhig, dunkel, klare Karten ----------------
st.markdown("""
<style>
    .stApp {background:#0a0e16; color:#e8edf4;}
    section[data-testid="stSidebar"]{background:#0f1521;}
    h1{color:#00ff9d;font-weight:800;letter-spacing:-.5px;margin-bottom:0;}
    h2{color:#d6f5e6;font-weight:700;border-bottom:1px solid #1e2838;padding-bottom:6px;margin-top:8px;}
    h3{color:#d6f5e6;font-weight:600;}
    .stMetric{background:#141b29;border:1px solid #1e2838;border-radius:12px;padding:14px 16px;}
    [data-testid="stMetricValue"]{color:#00ff9d;font-size:1.4rem;}
    [data-testid="stMetricLabel"]{color:#9aa6b8;}
    .stButton>button{background:#00ff9d;color:#06210f;font-weight:700;border:none;border-radius:8px;}
    div[data-testid="stMarkdownContainer"] code{color:#00ff9d;}
    .block-container{padding-top:2.2rem;}
    /* Radar-Karten */
    .radar{background:#121a28;border:1px solid #1e2838;border-radius:12px;padding:12px 14px;margin-bottom:8px;}
    .radar b{color:#e8edf4;font-size:1.02rem;}
    .pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.78rem;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ============================================================
#  PATENTIERTER SHIFTWN-KERN
# ============================================================
def _normalize(window):
    c=window[:,3]; ref=np.median(c) if np.median(c)>0 else 1.0
    scale=np.median(np.abs(np.diff(c))) or 0.01*ref
    n=np.zeros_like(window,dtype=float)
    n[:,0]=(window[:,0]-ref)/scale; n[:,1]=(window[:,1]-ref)/scale
    n[:,2]=(window[:,2]-ref)/scale; n[:,3]=(c-ref)/scale
    n[:,4]=window[:,4]/(np.median(window[:,4]) or 1.0); return n

def triangle(window):
    c=_normalize(window)[:,3]; x=np.arange(len(c))
    if len(c)<5: return {"score":0.0,"kappa":0.0,"dir":0.0}
    lin=np.polyfit(x,c,1); fit=np.polyval(lin,x); sst=np.sum((c-np.mean(c))**2)+1e-9
    r2=1-np.sum((c-fit)**2)/sst; h=len(c)//2; re_=np.ptp(c[:h]); rl=np.ptp(c[h:])
    wedge=np.clip((re_-rl)/(re_+1e-9),0,1)
    return {"score":float(np.clip(abs(lin[0])*6,0,1)),"kappa":float(np.clip(max(r2,wedge*0.9),0,1)),"dir":float(np.tanh(lin[0]*3))}

def vortex(window):
    c=_normalize(window)[:,3]
    if len(c)<6: return {"score":0.0,"kappa":0.0,"dir":0.0,"vol_break":0.0}
    pos=c-np.mean(c); vel=np.gradient(pos); ang=np.arctan2(vel,pos)
    dphi=np.diff(ang); dphi=(dphi+np.pi)%(2*np.pi)-np.pi
    score=float(np.clip(np.mean(np.abs(dphi))/(np.std(dphi)+1e-8),0,1))
    rot=np.clip(1.0-np.std(dphi)/(np.pi/2),0,1); h=len(c)//2
    v1=np.std(np.diff(c[:h]))+1e-9; v2=np.std(np.diff(c[h:]))+1e-9
    vb=float(np.clip(abs(np.log(v2/v1))/1.5,0,1))
    return {"score":score,"kappa":float(np.clip(rot*(1-vb),0,1)),"dir":float(np.tanh(np.polyfit(np.arange(len(c)),c,1)[0]*3)),"vol_break":vb}

def impulse(window):
    c=_normalize(window)[:,3]; r=np.diff(c)
    if len(r)<5: return {"score":0.0,"kappa":0.0,"dir":0.0}
    r=r-np.mean(r); w=np.hanning(len(r)); spec=np.abs(np.fft.rfft(r*w))**2
    ss=np.sum(spec) or 1.0; p=spec/ss; ent=-np.sum(p*np.log(p+1e-12))/np.log(len(p))
    return {"score":float(np.max(p)),"kappa":float(np.clip((1-ent)*1.8,0,1)),"dir":0.0}

def photonics(tri,vor,imp,prev_mk=None):
    ks=np.array([tri["kappa"],vor["kappa"],imp["kappa"]]); ss=np.array([tri["score"],vor["score"],imp["score"]])
    w=np.ones(3)/3 if ks.sum()<1e-6 else ks/ks.sum()
    phs=np.array([0,2*np.pi/3,4*np.pi/3]); den=np.sum(w*ks*ss)+1e-9
    field=float(np.clip(np.abs(np.sum(w*ks*ss*np.exp(1j*phs)))/den,0,1))
    mk=float(ks.mean()); drop=0.0 if prev_mk is None else max(0.0,prev_mk-mk); vb=vor.get("vol_break",0.0)
    rb=(vb>0.45 and drop>0.12)
    if rb: modus,dom,lead="REGIME-BRUCH / SCHOCK","Impulse FFT",2; drift=vor["dir"]
    else:
        lead=int(np.argmax(ks)); modus=["TRENDING","RANGING","VOLATILE"][lead]
        dom=["Triangle","Vortex","Impulse FFT"][lead]; drift=[tri["dir"],vor["dir"],vor["dir"]][lead]
    return {"w":w,"kappa":ks,"modus":modus,"dominant":dom,"field_coh":field,"mean_kappa":mk,
            "kappa_drop":float(drop),"vol_break":vb,"regime_break":rb,"drift":drift,
            "T":tri["kappa"],"V":vor["kappa"],"F":imp["kappa"]}

def _build(seg):
    n=len(seg); w=np.zeros((n,5)); w[:,3]=seg; w[:,0]=seg*0.97; w[:,1]=seg*1.08; w[:,2]=seg*0.92; w[:,4]=15000; return w

def analyse(closes,win=50):
    cur=closes[-win:]; prev=closes[-2*win:-win] if len(closes)>=2*win else None; pmk=None
    if prev is not None and len(prev)>=20:
        pmk=np.mean([triangle(_build(prev))["kappa"],vortex(_build(prev))["kappa"],impulse(_build(prev))["kappa"]])
    return photonics(triangle(_build(cur)),vortex(_build(cur)),impulse(_build(cur)),pmk)

def signal_from(ph,drift_th=0.06,kappa_min=0.35):
    if ph["regime_break"]: return "SCHOCK","⚠️","#e0883a"
    if ph["mean_kappa"]>kappa_min and ph["drift"]>drift_th: return "HEDGE_BUY","🟢","#00ff9d"
    if ph["mean_kappa"]>kappa_min and ph["drift"]<-drift_th: return "HEDGE_SELL","🔴","#e25c5c"
    return "HOLD","🟠","#9aa6b8"

def fibonacci_levels(c):
    hi,lo=np.max(c),np.min(c); d=hi-lo
    return {"0.0%":hi,"23.6%":hi-.236*d,"38.2%":hi-.382*d,"50.0%":hi-.5*d,"61.8%":hi-.618*d,"78.6%":hi-.786*d,"100.0%":lo}

# ============================================================
#  KI-WÄCHTER PARSER
# ============================================================
def parse_ki(text):
    out={"direction":"HOLD","confidence":None,"target":None,"stop":None}
    if not text: return out
    tl=text.lower()
    m=re.search(r"(?:konfidenz|confidence)\s*[:=]?\s*(\d{1,3})\s*%?",tl); out["confidence"]=int(m.group(1)) if m else None
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
         "TecDAX":["^TECDAX"],"Bitcoin":["BTC-USD"],"Gold":["GC=F"]}
GRAN={"Täglich (2 Jahre)":("2y","1d"),"15-Minuten (1 Monat)":("1mo","15m"),"5-Minuten (5 Tage)":("5d","5m")}
DIREKT=("^GDAXI","^GSPC","^DJI","^IXIC","^TECDAX","BTC-USD","GC=F")

@st.cache_data(ttl=120)
def get_data(ticker_list,period,interval,tries=2):
    for tk in ticker_list:
        for _ in range(tries):
            try: df=yf.download(tk,period=period,interval=interval,progress=False,auto_adjust=True)
            except Exception: df=None
            if df is not None and not df.empty:
                k=df["Close"].values.flatten(); k=k[~np.isnan(k)]
                if len(k)>=100: return k,tk
    return None,None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Einstellungen")
    market_name=st.selectbox("Detailmarkt",list(MARKETS.keys()))
    gran_name=st.selectbox("Granularität",list(GRAN.keys()))
    st.divider()
    st.subheader("Signal-Schwellen")
    drift_th=st.slider("Drift (Minimum)",0.04,0.30,0.06,0.01)
    kappa_min=st.slider("Mindest-Aussagekraft",0.20,0.70,0.35,0.01)
    st.divider()
    st.subheader("🔄 Auto-Refresh")
    auto=st.checkbox("Automatisch aktualisieren",value=False)
    takt=st.select_slider("Takt",options=[15,30,60,120,300],value=60,format_func=lambda s:f"{s}s")
    st.divider()
    st.subheader("🛡️ KI-Wächter")
    external=st.text_area("KI-Empfehlung",height=130,placeholder="Empfehlung von ChatGPT, Grok ...")
    ki_on=st.checkbox("Wächter aktivieren",value=True)

period,interval=GRAN[gran_name]

# ---------------- Kopf ----------------
st.title("⚡ ShiftWN AI")
st.caption(f"Adaptive geometrische Marktanalyse · Patent EPA EP25221251.9 · "
           f"Stand {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")

# ============================================================
#  1) MULTI-MARKT-RADAR
# ============================================================
st.markdown("## 1 · Markt-Radar")
st.caption("Alle Märkte gleichzeitig – wo ist ein Signal, wo ein Schock?")

radar=[]
for nm,tl in MARKETS.items():
    cf,used=get_data(tuple(tl),period,interval)
    if cf is None: radar.append((nm,None,None,None,None)); continue
    cc=cf[-min(len(cf),300):]; ph=analyse(cc,50); sig,icon,col=signal_from(ph,drift_th,kappa_min)
    radar.append((nm,float(cf[-1]),sig,icon,col,ph))

cols=st.columns(len(radar))
for i,row in enumerate(radar):
    with cols[i]:
        if row[1] is None:
            st.markdown(f"<div class='radar'><b>{row[0]}</b><br><span style='color:#9aa6b8'>keine Daten</span></div>",unsafe_allow_html=True)
        else:
            nm,price,sig,icon,col,ph=row
            st.markdown(f"<div class='radar'><b>{nm}</b><br>"
                        f"<span style='color:#9aa6b8;font-size:.85rem'>{price:,.0f}</span><br>"
                        f"<span class='pill' style='background:{col}22;color:{col};border:1px solid {col}55'>{icon} {sig}</span><br>"
                        f"<span style='color:#6b7688;font-size:.72rem'>{ph['modus']} · κØ {ph['mean_kappa']:.2f}</span></div>",
                        unsafe_allow_html=True)

# Schock-Warnung global
schocks=[r[0] for r in radar if r[1] is not None and r[2]=="SCHOCK"]
if schocks:
    st.error(f"⚠️ Regime-Bruch erkannt in: {', '.join(schocks)} – gewohnte Struktur gilt dort nicht mehr.")

# ============================================================
#  2) DETAILANALYSE
# ============================================================
st.markdown(f"## 2 · Detailanalyse — {market_name}")
cf,used=get_data(tuple(MARKETS[market_name]),period,interval)
if cf is None:
    st.error("Keine Daten für den gewählten Detailmarkt.")
else:
    closes=cf[-min(len(cf),300):]; price=float(cf[-1])
    ph=analyse(closes,50); sig,icon,col=signal_from(ph,drift_th,kappa_min)
    if used and used not in DIREKT: st.caption(f"Daten über Ersatz-Ticker `{used}`.")

    m1,m2,m3,m4=st.columns(4)
    m1.metric("Signal",f"{icon} {sig}")
    m2.metric("Preis",f"{price:,.2f}")
    m3.metric("Drift",f"{ph['drift']:+.3f}")
    m4.metric("Aussagekraft Ø",f"{ph['mean_kappa']:.2f}")

    if ph["regime_break"]:
        st.error(f"⚠️ **Regime-Bruch** – Volatilitätssprung ({ph['vol_break']:.2f}) und Einbruch der "
                 f"Aussagekraft ({ph['kappa_drop']:.2f}). Photonics schaltet auf Impulse FFT.")
    else:
        st.info(f"**Marktmodus (Photonics):** {ph['modus']} · führend: {ph['dominant']}")

    left,right=st.columns([3,2])
    with left:
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=closes,mode="lines",line=dict(color="#00ff9d",width=2.4)))
        for nmf,pr in fibonacci_levels(closes).items():
            fig.add_hline(y=pr,line_dash="dash",line_color="rgba(255,210,0,.28)",
                          annotation_text=nmf,annotation_font_color="rgba(255,210,0,.65)")
        fig.update_layout(height=420,template="plotly_dark",paper_bgcolor="#0a0e16",plot_bgcolor="#0a0e16",
                          margin=dict(l=0,r=0,t=8,b=0),showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("**Photonics — Gewichtung nach Aussagekraft**")
        fig2=go.Figure(go.Bar(x=[ph["w"][0],ph["w"][1],ph["w"][2]],y=["Triangle","Vortex","Impulse FFT"],
            orientation="h",marker_color=["#00ff9d","#00b3ff","#ff9d3c"],
            text=[f"{ph['w'][0]:.0%}",f"{ph['w'][1]:.0%}",f"{ph['w'][2]:.0%}"],textposition="auto"))
        fig2.update_layout(height=180,template="plotly_dark",paper_bgcolor="#0a0e16",plot_bgcolor="#0a0e16",
                           margin=dict(l=0,r=10,t=4,b=20),xaxis_range=[0,1])
        st.plotly_chart(fig2,use_container_width=True)
        st.caption(f"κ: T {ph['T']:.2f} · V {ph['V']:.2f} · F {ph['F']:.2f}. Blinde Module verlieren Gewicht.")

    # ============================================================
    #  3) KI-WÄCHTER
    # ============================================================
    if external and ki_on:
        st.markdown("## 3 · KI-Wächter")
        ki=parse_ki(external); d=ki["direction"]
        a,b=st.columns(2)
        with a:
            st.markdown("**Externe KI sagt:**"); st.info(external)
            konf=f" · Konfidenz {ki['confidence']} %" if ki["confidence"] is not None else ""
            st.caption(f"Richtung **{d}**{konf} · Ziel {ki['target']} · Stop {ki['stop']}")
        with b:
            st.markdown("**ShiftWN sagt:**"); st.success(f"{icon} {sig}")
            st.caption(f"Modus {ph['modus']} · Drift {ph['drift']:+.3f} · κØ {ph['mean_kappa']:.2f}")
        sdir={"HEDGE_BUY":"BUY","HEDGE_SELL":"SELL","HOLD":"NEUTRAL","SCHOCK":"SHOCK"}[sig]
        if d=="CONFLICT": st.warning("⚠️ Empfehlung mehrdeutig (Kauf und Verkauf). Manuell prüfen.")
        elif sdir=="SHOCK": st.error(f"⚠️ Regime-Bruch: Die Empfehlung (**{d}**) beruht auf der gewohnten Struktur, die gerade nicht gilt. Hier laufen normale Modelle blind hinein.")
        elif sdir=="NEUTRAL": (st.error(f"❌ ShiftWN widerspricht: KI sagt **{d}**, ShiftWN sieht keine klare Struktur (κØ {ph['mean_kappa']:.2f}).") if d in ("BUY","SELL") else st.info("Beide: kein klares Signal."))
        elif d==sdir: st.success(f"✅ ShiftWN bestätigt: **{d}** ist mit der Marktgeometrie kohärent (Modus {ph['modus']}).")
        elif d=="HOLD": st.warning(f"⚠️ KI sagt HOLD – ShiftWN sieht ein klares Signal ({sig}).")
        else: st.error(f"❌ ShiftWN widerspricht: KI sagt **{d}**, ShiftWN sagt **{sig}**.")

st.markdown("---")
st.caption("ShiftWN AI · Patentierter geometrischer Kern (Triangle · Vortex · Impulse FFT · Photonics). "
           "Photonics gewichtet nach Aussagekraft und erkennt Regime-Brüche. Keine Anlageberatung.")

# ---------------- Auto-Refresh (am Ende, ohne Zusatzpaket) ----------------
if auto:
    time.sleep(takt)
    st.cache_data.clear()
    st.rerun()
