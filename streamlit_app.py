"""
ShiftWN AI – Geometrische Marktanalyse mit adaptivem Photonics-Kern
Struktur: 1) Markt-Radar  2) Detailanalyse  3) KI-Wächter
Look: gedämpft warmes Hellgrau + Petrol (hell, edel, ruhig).
Alerts: visuell in der App (neues Signal / Schock werden hervorgehoben).
        E-Mail-Zustellung = dokumentierter nächster Schritt (separater Dienst,
        erst nach rechtlicher Klärung – siehe Hinweis unten im Code).
Patentierter Kern: Triangle · Vortex · Impulse FFT · Photonics.
Patent EPA EP25221251.9 / SPECEPO-1/2.  Keine Anlageberatung.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import re
import time
from datetime import datetime

st.set_page_config(page_title="ShiftWN AI", layout="wide", page_icon="⚡")

# ---------------- Farbpalette: gedämpft warmes Hellgrau + Petrol ----------------
BG="#e9e7e2"; BG_ALT="#e3e1db"; CARD="#f1efea"; BORDER="#cfccc4"
TEAL="#1f6f6b"; TEXT="#2b2f36"; MUTED="#6b6f78"; FAINT="#969aa2"
BUY="#1f6f6b"; SELL="#b5524a"; HOLD="#8a8077"; SHOCK="#b07a2e"

st.markdown(f"""
<style>
    .stApp {{background:{BG};color:{TEXT};}}
    section[data-testid="stSidebar"]{{background:{BG_ALT};border-right:1px solid {BORDER};}}
    h1{{color:{TEXT};font-weight:700;letter-spacing:-.3px;margin-bottom:0;}}
    .app-sub {{color:{MUTED};}}
    h2{{color:{MUTED};font-weight:600;font-size:.95rem;text-transform:uppercase;
        letter-spacing:2px;border-bottom:1px solid {BORDER};padding-bottom:8px;margin-top:18px;}}
    h2 .num{{color:{TEAL};font-weight:700;margin-right:8px;}}
    h3{{color:{TEXT};font-weight:600;}}
    .stMetric{{background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:16px 18px;
               box-shadow:0 1px 3px rgba(43,47,54,.06);}}
    [data-testid="stMetricValue"]{{color:{TEAL};font-size:1.35rem;font-weight:700;}}
    [data-testid="stMetricLabel"]{{color:{MUTED};font-size:.78rem;text-transform:uppercase;letter-spacing:.6px;}}
    .stButton>button{{background:{TEAL};color:#f4f3ef;font-weight:600;border:none;border-radius:8px;}}
    div[data-testid="stMarkdownContainer"] code{{color:{TEAL};}}
    .block-container{{padding-top:2.4rem;max-width:1400px;}}
    .radar{{background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:14px 16px;
            margin-bottom:8px;box-shadow:0 1px 3px rgba(43,47,54,.05);transition:box-shadow .2s,transform .2s;}}
    .radar:hover{{box-shadow:0 4px 14px rgba(43,47,54,.12);transform:translateY(-1px);}}
    .radar .nm{{color:{TEXT};font-size:1.04rem;font-weight:600;}}
    .radar .px{{color:{MUTED};font-size:.82rem;}}
    .radar .md{{color:{FAINT};font-size:.72rem;letter-spacing:.3px;}}
    .pill{{display:inline-block;padding:3px 12px;border-radius:999px;font-size:.76rem;font-weight:700;margin:6px 0;}}
    .alertbar{{border-radius:12px;padding:14px 18px;margin:6px 0 14px 0;font-weight:600;}}
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
    if ph["regime_break"]: return "SCHOCK","⚠",SHOCK
    if ph["mean_kappa"]>kappa_min and ph["drift"]>drift_th: return "HEDGE_BUY","▲",BUY
    if ph["mean_kappa"]>kappa_min and ph["drift"]<-drift_th: return "HEDGE_SELL","▼",SELL
    return "HOLD","■",HOLD

def fibonacci_levels(c):
    hi,lo=np.max(c),np.min(c); d=hi-lo
    return {"0.0%":hi,"23.6%":hi-.236*d,"38.2%":hi-.382*d,"50.0%":hi-.5*d,"61.8%":hi-.618*d,"78.6%":hi-.786*d,"100.0%":lo}

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

MARKETS={"DAX":["^GDAXI","EXS1.DE"],"S&P 500":["^GSPC"],"Dow Jones":["^DJI"],"Nasdaq":["^IXIC"],
         "TecDAX":["^TECDAX"],"Bitcoin":["BTC-USD"],"Gold":["GC=F"]}
GRAN={"Täglich (2 Jahre)":("2y","1d"),"15-Minuten (1 Monat)":("1mo","15m"),"5-Minuten (5 Tage)":("5d","5m")}
DIREKT=("^GDAXI","^GSPC","^DJI","^IXIC","^TECDAX","BTC-USD","GC=F")

@st.cache_data(ttl=120, show_spinner=False)
def get_data(ticker_list, period, interval, tries=1):
    for tk in ticker_list:
        for _ in range(tries):
            try:
                df = yf.download(tk, period=period, interval=interval,
                                 progress=False, auto_adjust=True, timeout=8)
            except Exception:
                df = None
            if df is not None and not df.empty:
                k = df["Close"].values.flatten(); k = k[~np.isnan(k)]
                if len(k) >= 100:
                    return k, tk
    return None, None

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
    st.subheader("🔔 Alerts")
    alert_on=st.checkbox("Signal-Wechsel hervorheben",value=True)
    st.caption("Visueller Alert in der App. E-Mail-Zustellung folgt als separater Dienst nach rechtlicher Klärung.")
    st.divider()
    st.subheader("🛡️ KI-Wächter")
    external=st.text_area("KI-Empfehlung",height=120,placeholder="Empfehlung von ChatGPT, Grok ...")
    ki_on=st.checkbox("Wächter aktivieren",value=True)

period,interval=GRAN[gran_name]

# ---------------- Kopf ----------------
st.title("⚡ ShiftWN AI")
st.markdown(f"<div class='app-sub'>Adaptive geometrische Marktanalyse · Patent EPA EP25221251.9 · "
            f"Stand {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</div>", unsafe_allow_html=True)

# ============================================================
#  0) LIVE-BITCOIN  (Live-Analyse, 5-Minuten, mit Signal-Verlauf)
#     Reine Analyse – kein Geld, keine Order-Ausführung.
# ============================================================
st.markdown(f"## <span class='num'>0</span> Live-Bitcoin", unsafe_allow_html=True)

@st.cache_data(ttl=60, show_spinner=False)
def get_btc_live():
    """Bitcoin auf 5-Minuten-Basis, kurzer Zeitraum, mit Timeout."""
    try:
        df = yf.download("BTC-USD", period="5d", interval="5m",
                         progress=False, auto_adjust=True, timeout=8)
    except Exception:
        df = None
    if df is not None and not df.empty:
        k = df["Close"].values.flatten(); k = k[~np.isnan(k)]
        if len(k) >= 100:
            return k
    return None

with st.spinner("Lade Live-Bitcoin (5-Min) ..."):
    btc = get_btc_live()

if btc is None:
    st.warning("Live-Bitcoin-Daten gerade nicht verfügbar. Bitte kurz erneut versuchen.")
else:
    btc_closes = btc[-min(len(btc),300):]
    btc_price = float(btc[-1])
    btc_ph = analyse(btc_closes, 50)
    btc_sig, btc_icon, btc_col = signal_from(btc_ph, drift_th, kappa_min)

    # Signal-Verlauf (innerhalb der Sitzung)
    if "btc_history" not in st.session_state:
        st.session_state.btc_history = []
    hist = st.session_state.btc_history
    now_str = datetime.now().strftime("%H:%M:%S")
    if not hist or hist[-1]["signal"] != btc_sig:
        hist.append({"zeit": now_str, "signal": btc_sig, "preis": btc_price,
                     "modus": btc_ph["modus"], "drift": btc_ph["drift"]})
        st.session_state.btc_history = hist[-30:]  # letzte 30 Wechsel behalten

    bL, bR = st.columns([3,2])
    with bL:
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Bitcoin (5-Min)", f"{btc_price:,.0f}")
        b2.metric("ShiftWN-Signal", f"{btc_icon} {btc_sig}")
        b3.metric("Drift", f"{btc_ph['drift']:+.3f}")
        b4.metric("Aussagekraft Ø", f"{btc_ph['mean_kappa']:.2f}")
        if btc_ph["regime_break"]:
            st.markdown(f"<div class='alertbar' style='background:{SHOCK}1a;border:1px solid {SHOCK}66;color:{SHOCK}'>"
                        f"Regime-Bruch bei Bitcoin – die gewohnte Struktur gilt gerade nicht.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alertbar' style='background:{CARD};border:1px solid {BORDER};color:{MUTED}'>"
                        f"Marktmodus (Photonics): <b style='color:{TEXT}'>{btc_ph['modus']}</b> · führend: {btc_ph['dominant']}</div>",
                        unsafe_allow_html=True)
        fig_btc = go.Figure()
        fig_btc.add_trace(go.Scatter(y=btc_closes, mode="lines", line=dict(color=TEAL, width=2)))
        fig_btc.update_layout(height=240, template="plotly_white", paper_bgcolor=BG, plot_bgcolor=CARD,
                              margin=dict(l=0,r=0,t=6,b=0), showlegend=False,
                              xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
        st.plotly_chart(fig_btc, use_container_width=True)
    with bR:
        st.markdown("**Signal-Verlauf (diese Sitzung)**")
        if len(st.session_state.btc_history) <= 1:
            st.caption("Noch keine Signalwechsel erfasst. Der Verlauf füllt sich, sobald sich das Signal ändert "
                       "(bei aktiviertem Auto-Refresh automatisch).")
        for h in reversed(st.session_state.btc_history):
            sig_col = {"HEDGE_BUY":BUY,"HEDGE_SELL":SELL,"HOLD":HOLD,"SCHOCK":SHOCK}.get(h["signal"],MUTED)
            st.markdown(f"<div class='radar' style='padding:8px 12px;margin-bottom:6px'>"
                        f"<span style='color:{MUTED};font-size:.78rem'>{h['zeit']}</span> · "
                        f"<span style='color:{sig_col};font-weight:700'>{h['signal']}</span><br>"
                        f"<span style='color:{FAINT};font-size:.74rem'>{h['preis']:,.0f} · {h['modus']} · Drift {h['drift']:+.2f}</span></div>",
                        unsafe_allow_html=True)
    st.caption("Live-Analyse zur Beobachtung – ShiftWN ist Signalgeber, keine Order-Ausführung und keine Anlageberatung.")

# ============================================================
#  1) MARKT-RADAR  (+ Alert-Erkennung via session_state)
# ============================================================
st.markdown(f"## <span class='num'>1</span> Markt-Radar", unsafe_allow_html=True)

if "last_signals" not in st.session_state: st.session_state.last_signals={}

@st.cache_data(ttl=120, show_spinner=False)
def load_all_markets(market_items, period, interval):
    """Lädt alle Märkte parallel – sieben Abrufe dauern zusammen so lang wie einer."""
    from concurrent.futures import ThreadPoolExecutor
    def one(item):
        nm, tl = item
        cf, used = get_data(tuple(tl), period, interval)
        return nm, cf
    out = {}
    with ThreadPoolExecutor(max_workers=len(market_items)) as ex:
        for nm, cf in ex.map(one, market_items):
            out[nm] = cf
    return out

with st.spinner("Lade Markt-Radar ..."):
    market_data = load_all_markets(tuple(MARKETS.items()), period, interval)

radar=[]; current_signals={}
for nm in MARKETS:
    cf = market_data.get(nm)
    if cf is None:
        radar.append((nm,None)); continue
    cc=cf[-min(len(cf),300):]; ph=analyse(cc,50); sig,icon,col=signal_from(ph,drift_th,kappa_min)
    radar.append((nm,float(cf[-1]),sig,icon,col,ph)); current_signals[nm]=sig

# Alert: welche Signale haben sich seit dem letzten Lauf geändert?
changes=[]
if alert_on and st.session_state.last_signals:
    for nm,sig in current_signals.items():
        prev=st.session_state.last_signals.get(nm)
        if prev is not None and prev!=sig:
            changes.append((nm,prev,sig))
st.session_state.last_signals=current_signals

# Alert-Banner
shocks=[r[0] for r in radar if len(r)>2 and r[2]=="SCHOCK"]
if shocks:
    st.markdown(f"<div class='alertbar' style='background:{SHOCK}1a;border:1px solid {SHOCK}66;color:{SHOCK}'>"
                f"Regime-Bruch erkannt: {', '.join(shocks)} – gewohnte Struktur gilt dort nicht mehr.</div>",
                unsafe_allow_html=True)
if changes:
    txt=" · ".join([f"{nm}: {p}→{s}" for nm,p,s in changes])
    st.markdown(f"<div class='alertbar' style='background:{TEAL}1a;border:1px solid {TEAL}66;color:{TEAL}'>"
                f"Signal-Wechsel: {txt}</div>", unsafe_allow_html=True)

cols=st.columns(len(radar))
for i,row in enumerate(radar):
    with cols[i]:
        if row[1] is None:
            st.markdown(f"<div class='radar'><div class='nm'>{row[0]}</div>"
                        f"<div class='px' style='color:{FAINT}'>lädt …</div></div>",unsafe_allow_html=True)
        else:
            nm,price,sig,icon,col,ph=row
            st.markdown(f"<div class='radar'><div class='nm'>{nm}</div>"
                        f"<div class='px'>{price:,.0f}</div>"
                        f"<span class='pill' style='background:{col}1f;color:{col};border:1px solid {col}55'>{icon} {sig}</span>"
                        f"<div class='md'>{ph['modus']} · κØ {ph['mean_kappa']:.2f}</div></div>",
                        unsafe_allow_html=True)

# ============================================================
#  2) DETAILANALYSE
# ============================================================
st.markdown(f"## <span class='num'>2</span> Detailanalyse — {market_name}", unsafe_allow_html=True)
with st.spinner(f"Lade {market_name} ..."):
    cf,used=get_data(tuple(MARKETS[market_name]),period,interval)
if cf is None:
    st.warning(f"Für {market_name} konnten gerade keine Daten geladen werden. "
               f"Bitte einen anderen Markt oder eine andere Granularität wählen – oder kurz erneut versuchen.")
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
        st.markdown(f"<div class='alertbar' style='background:{SHOCK}1a;border:1px solid {SHOCK}66;color:{SHOCK}'>"
                    f"⚠ Regime-Bruch – Volatilitätssprung ({ph['vol_break']:.2f}) und Einbruch der Aussagekraft "
                    f"({ph['kappa_drop']:.2f}). Photonics schaltet auf Impulse FFT.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alertbar' style='background:{CARD};border:1px solid {BORDER};color:{MUTED}'>"
                    f"Marktmodus (Photonics): <b style='color:{TEXT}'>{ph['modus']}</b> · führend: {ph['dominant']}</div>",
                    unsafe_allow_html=True)

    left,right=st.columns([3,2])
    with left:
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=closes,mode="lines",line=dict(color=TEAL,width=2.2)))
        for nmf,pr in fibonacci_levels(closes).items():
            fig.add_hline(y=pr,line_dash="dot",line_color="rgba(107,111,120,.30)",
                          annotation_text=nmf,annotation_font_color="rgba(107,111,120,.75)",annotation_font_size=10)
        fig.update_layout(height=420,template="plotly_white",paper_bgcolor=BG,plot_bgcolor=CARD,
                          margin=dict(l=0,r=0,t=8,b=0),showlegend=False,
                          xaxis=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("**Photonics — Gewichtung nach Aussagekraft**")
        fig2=go.Figure(go.Bar(x=[ph["w"][0],ph["w"][1],ph["w"][2]],y=["Triangle","Vortex","Impulse FFT"],
            orientation="h",marker_color=[TEAL,"#3f7d9c","#b07a2e"],
            text=[f"{ph['w'][0]:.0%}",f"{ph['w'][1]:.0%}",f"{ph['w'][2]:.0%}"],textposition="auto"))
        fig2.update_layout(height=180,template="plotly_white",paper_bgcolor=BG,plot_bgcolor=CARD,
                           margin=dict(l=0,r=10,t=4,b=20),xaxis_range=[0,1],
                           xaxis=dict(gridcolor=BORDER),yaxis=dict(gridcolor=BORDER))
        st.plotly_chart(fig2,use_container_width=True)
        st.caption(f"κ: T {ph['T']:.2f} · V {ph['V']:.2f} · F {ph['F']:.2f}. Blinde Module verlieren Gewicht.")

    # ============================================================
    #  3) KI-WÄCHTER
    # ============================================================
    if external and ki_on:
        st.markdown(f"## <span class='num'>3</span> KI-Wächter", unsafe_allow_html=True)
        ki=parse_ki(external); d=ki["direction"]
        a,b=st.columns(2)
        with a:
            st.markdown("**Externe KI sagt:**"); st.info(external)
            konf=f" · Konfidenz {ki['confidence']} %" if ki["confidence"] is not None else ""
            st.caption(f"Richtung {d}{konf} · Ziel {ki['target']} · Stop {ki['stop']}")
        with b:
            st.markdown("**ShiftWN sagt:**"); st.success(f"{icon} {sig}")
            st.caption(f"Modus {ph['modus']} · Drift {ph['drift']:+.3f} · κØ {ph['mean_kappa']:.2f}")
        sdir = {"HEDGE_BUY":"BUY", "HEDGE_SELL":"SELL", "HOLD":"NEUTRAL", "SCHOCK":"SHOCK"}[sig]
        kappa_txt = f"{ph['mean_kappa']:.2f}"
        if d == "CONFLICT":
            st.warning("Empfehlung mehrdeutig (Kauf und Verkauf zugleich). Bitte manuell prüfen.")
        elif sdir == "SHOCK":
            st.error(f"Regime-Bruch: Die Empfehlung ({d}) beruht auf der gewohnten Struktur, "
                     f"die gerade nicht gilt. Hier laufen normale Modelle blind hinein.")
        elif sdir == "NEUTRAL":
            if d in ("BUY", "SELL"):
                st.error(f"ShiftWN widerspricht: KI sagt {d}, ShiftWN sieht keine klare Struktur "
                         f"(Aussagekraft {kappa_txt}).")
            else:
                st.info("Beide Seiten sehen kein klares Signal.")
        elif d == sdir:
            st.success(f"ShiftWN bestätigt: {d} ist mit der Marktgeometrie kohärent (Modus {ph['modus']}).")
        elif d == "HOLD":
            st.warning(f"KI sagt HOLD – ShiftWN sieht jedoch ein klares Signal ({sig}).")
        else:
            st.error(f"ShiftWN widerspricht: KI sagt {d}, ShiftWN sagt {sig}.")

st.markdown("---")
st.caption("ShiftWN AI · Patentierter geometrischer Kern (Triangle · Vortex · Impulse FFT · Photonics). "
           "Photonics gewichtet nach Aussagekraft und erkennt Regime-Brüche. Keine Anlageberatung.")

# ============================================================
#  E-MAIL-ALERT — KONZEPT / NÄCHSTER SCHRITT (bewusst nicht aktiv)
#  Streamlit Cloud läuft nur bei offener Seite; echte Push-/E-Mail-Alerts
#  brauchen einen separaten Dauerdienst (z.B. ein kleiner Scheduler, der
#  analyse() periodisch ausführt und bei Signalwechsel/Schock mailt).
#  Vor Aktivierung rechtliche Einordnung klären (Signaldienst/Beratung).
# ============================================================

# ---------------- Auto-Refresh ----------------
if auto:
    time.sleep(takt); st.cache_data.clear(); st.rerun()
