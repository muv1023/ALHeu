import io
import os
import re
import hmac
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
from PIL import Image
import math
import json
from xml.sax.saxutils import escape
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

st.set_page_config(page_title="BAN 517 Algorithms & Heuristics Lab", page_icon="🧭", layout="wide")

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.lab-hero {background: linear-gradient(135deg,#003b5c 0%,#006c67 55%,#00a6a6 100%);padding:28px;border-radius:18px;color:white;margin-bottom:20px;}
.lab-hero h1 {color:white;margin:0 0 6px 0;}
.lab-card {border:1px solid #dfe5ec;border-radius:14px;padding:18px 20px;margin:12px 0 18px 0;background:white;}
.small-note {color:#52606d;font-size:0.92rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="lab-hero">
<div style="font-size:14px;color:#d7f4f2;">BAN 517 · Module 5</div>
<h1>Algorithms & Heuristics Decision Lab</h1>
<div style="font-size:18px;">Solve outside the app. Enter your results here to verify your work.</div>
</div>
""", unsafe_allow_html=True)

# ---------- configuration ----------
def get_cfg():
    try:
        lab = st.secrets["lab"]
        ans = st.secrets["answers"]
        return lab, ans
    except Exception:
        st.error("Instructor setup is incomplete. The private Streamlit Secrets have not been configured.")
        st.stop()

lab_cfg, answers = get_cfg()
DIST_TOL = float(lab_cfg.get("distance_tolerance", 0.05))
PCT_TOL = float(lab_cfg.get("percent_tolerance", 0.10))
SALT = str(lab_cfg.get("verification_salt", ""))


def norm_route(s: str) -> str:
    parts = [p.strip().upper() for p in re.split(r"(?:->|→|,|;|-)+", s or "") if p.strip()]
    return "-".join(parts)


def near(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def status(ok):
    return "Correct" if ok else "Incorrect"


def make_verification(name, score, timestamp, payload):
    msg = f"{name}|{score:.2f}|{timestamp}|{payload}".encode()
    return hmac.new(SALT.encode(), msg, hashlib.sha256).hexdigest()[:16].upper()


def make_pdf(record):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Centered", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#003b5c")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#003b5c"), spaceBefore=10, spaceAfter=6))
    story = [Paragraph("BAN 517 Algorithms & Heuristics Decision Lab", styles["Centered"]), Spacer(1, 10)]
    meta = [
        ["Student", record["name"]],
        ["Timestamp (UTC)", record["timestamp"]], ["Verification Code", record["verification"]],
        ["Automated Score", f'{record["score"]:.2f} / 10.00']
    ]
    t = Table(meta, colWidths=[130, 340])
    t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#dfe5ec')),('BACKGROUND',(0,0),(0,-1),colors.HexColor('#eef7ff')),('VALIGN',(0,0),(-1,-1),'TOP'),('PADDING',(0,0),(-1,-1),6)]))
    story += [t, Spacer(1,12)]

    for sec in record["sections"]:
        story.append(Paragraph(sec["title"], styles["Section"]))
        data = [["Entry", "Submitted", "Result", "Points"]] + sec["rows"]
        data = [[Paragraph(str(v), styles["BodyText"]) for v in row] for row in data]
        tab = Table(data, colWidths=[145,175,85,65], repeatRows=1)
        tab.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#003b5c')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#dfe5ec')),('VALIGN',(0,0),(-1,-1),'TOP'),('PADDING',(0,0),(-1,-1),5)
        ]))
        story += [tab, Spacer(1,8)]

    story.append(Paragraph("Submission Reminder", styles["Section"]))
    story.append(Paragraph("Submit this verified PDF together with the Excel workbook required in Canvas.", styles["BodyText"]))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ---------- identity ----------
st.subheader("1. Student Information")
name = st.text_input("Student name")

if not name.strip():
    st.info("Enter your name to begin.")
    st.stop()

st.subheader("2. Download Assignment Data")
cols = st.columns(3)
for col, fn, label in [
    (cols[0], "routing_data.csv", "Routing / VRP data"),
    (cols[1], "bin_packing_items.csv", "Bin packing data"),
    (cols[2], "dijkstra_edges.csv", "Dijkstra network data"),
]:
    p = DATA / fn
    col.download_button(label, p.read_bytes(), file_name=fn, mime="text/csv", use_container_width=True)

st.caption("Complete calculations outside the app. Use the interactive sweep interface to record your vehicle clusters. The app will not reveal the private answer key.")

# ---------- part 1 ----------
st.subheader("3. Nearest Neighbor + 2-Opt")
nn_route = st.text_input("Nearest Neighbor route", placeholder="Depot-F-...-Depot")
nn_dist = st.number_input("Nearest Neighbor total distance", min_value=0.0, step=0.01, format="%.2f")
opt_route = st.text_input("2-Opt improved route", placeholder="Depot-F-...-Depot")
opt_dist = st.number_input("2-Opt improved total distance", min_value=0.0, step=0.01, format="%.2f")
pct_imp = st.number_input("Percentage improvement from NN to 2-Opt (%)", min_value=0.0, step=0.01, format="%.2f")

# ---------- Sweep heuristic directly on the assignment image ----------
# Positions are pixel locations on assets/sweep_pickup_diagram.png (1728 x 910).
# IDs are stable labels for recording the clusters; quantities are printed in the image.
# These coordinates are traced from the supplied illustration, NOT routing_data.csv.
SWEEP_IMAGE = BASE / "assets" / "sweep_pickup_diagram.png"
SWEEP_WIDTH, SWEEP_HEIGHT = 1728, 910
SWEEP_DEPOT = (886, 529)
SWEEP_POINTS = [
    {"id":"1", "x":799,"y":132,"demand":4000},
    {"id":"2", "x":487,"y":295,"demand":3500},
    {"id":"3", "x":682,"y":257,"demand":3000},
    {"id":"4", "x":950,"y":253,"demand":2500},
    {"id":"5", "x":1169,"y":314,"demand":5000},
    {"id":"6", "x":804,"y":387,"demand":1000},
    {"id":"7", "x":435,"y":422,"demand":5000},
    {"id":"8", "x":568,"y":543,"demand":2500},
    {"id":"9", "x":604,"y":655,"demand":5000},
    {"id":"10","x":825,"y":678,"demand":2500},
    {"id":"11","x":1120,"y":482,"demand":2000},
    {"id":"12","x":1056,"y":598,"demand":4000},
    {"id":"13","x":1045,"y":745,"demand":2000},
]

def sweep_angle(p):
    # Image Y grows downward; negate Y to use conventional mathematical angles.
    return math.degrees(math.atan2(SWEEP_DEPOT[1]-p["y"], p["x"]-SWEEP_DEPOT[0])) % 360

def sweep_order(start, clockwise):
    return sorted(SWEEP_POINTS, key=lambda p: (((start-sweep_angle(p)) if clockwise else (sweep_angle(p)-start)) % 360, int(p["id"])))

def sweep_figure(angle, groups, active):
    fig=go.Figure()
    # Plotly image occupies the exact original image extent; no second map.
    fig.add_layout_image(dict(source=Image.open(SWEEP_IMAGE),xref="x",yref="y",x=0,y=0,
                              sizex=SWEEP_WIDTH,sizey=SWEEP_HEIGHT,sizing="stretch",layer="below"))
    rad=math.radians(angle)
    radius=620
    xend=SWEEP_DEPOT[0]+radius*math.cos(rad)
    yend=SWEEP_DEPOT[1]-radius*math.sin(rad)
    fig.add_trace(go.Scatter(x=[SWEEP_DEPOT[0],xend],y=[SWEEP_DEPOT[1],yend],mode="lines",
                             line=dict(color="#e11d48",width=4),name="Movable sweep ray",hoverinfo="skip"))
    assigned={v for group in groups for v in group}|set(active)
    palette=["#2563eb","#16a34a","#d97706","#9333ea","#0891b2","#db2777"]
    for i,group in enumerate(groups+([active] if active else [])):
        if not group: continue
        pts=[p for p in SWEEP_POINTS if p["id"] in group]
        fig.add_trace(go.Scatter(x=[p["x"] for p in pts],y=[p["y"] for p in pts],mode="markers",
             marker=dict(size=24,color=palette[i%len(palette)],opacity=.30,line=dict(width=2,color=palette[i%len(palette)])),
             text=[f'Pickup {p["id"]}: {p["demand"]:,} units' for p in pts],hovertemplate="%{text}<extra></extra>",name=f"Cluster {i+1}"))
    # Tiny numbered tags next to the original black pickup dots for unambiguous answers.
    fig.add_trace(go.Scatter(x=[p["x"]+22 for p in SWEEP_POINTS],y=[p["y"]-16 for p in SWEEP_POINTS],
        mode="text",text=[p["id"] for p in SWEEP_POINTS],textfont=dict(color="#075985",size=15,family="Arial Black"),
        hoverinfo="skip",showlegend=False))
    fig.update_xaxes(range=[0,SWEEP_WIDTH],visible=False,fixedrange=True)
    fig.update_yaxes(range=[SWEEP_HEIGHT,0],visible=False,fixedrange=True,scaleanchor="x",scaleratio=1)
    fig.update_layout(height=570,margin=dict(l=0,r=0,t=5,b=5),paper_bgcolor="white",
        plot_bgcolor="white",showlegend=True,legend=dict(orientation="h",y=-.04),dragmode=False)
    return fig

st.subheader("4. Sweep Algorithm / VRP")
st.caption("Rotate the red ray directly over the assignment figure. Pickup points are numbered 1-13 in blue; their pickup quantities remain printed in the diagram. Each vehicle can carry at most 15,000 units.")
if not SWEEP_IMAGE.is_file():
    st.error("Missing assets/sweep_pickup_diagram.png")
    st.stop()

for k,default in {"sweep_clusters":[],"sweep_current":[],"sweep_submitted":False,"sweep_angle":0,"sweep_direction":"Counterclockwise"}.items():
    if k not in st.session_state: st.session_state[k]=default

def rotate_ray(amount):
    if not st.session_state.sweep_submitted:
        st.session_state.sweep_angle=(st.session_state.sweep_angle+amount)%360

c1,c2=st.columns([2,1])
with c1:
    angle=st.slider("Ray angle (0 degrees points right)",0,359,key="sweep_angle",disabled=st.session_state.sweep_submitted)
with c2:
    direction=st.radio("Sweep direction",["Counterclockwise","Clockwise"],horizontal=True,key="sweep_direction",disabled=st.session_state.sweep_submitted)

r1,r2,r3,r4=st.columns(4)
with r1: st.button("Rotate 5 degrees clockwise",on_click=rotate_ray,args=(-5,),disabled=st.session_state.sweep_submitted,use_container_width=True)
with r2: st.button("Rotate 5 degrees anticlockwise",on_click=rotate_ray,args=(5,),disabled=st.session_state.sweep_submitted,use_container_width=True)
with r3: st.button("Rotate 15 degrees clockwise",on_click=rotate_ray,args=(-15,),disabled=st.session_state.sweep_submitted,use_container_width=True)
with r4: st.button("Rotate 15 degrees anticlockwise",on_click=rotate_ray,args=(15,),disabled=st.session_state.sweep_submitted,use_container_width=True)

# Changing the start ray while building a cluster would silently change the
# sequence. Instead, the ray remains freely movable for exploration; only
# the start angle used for assignments is frozen at first assignment.
if "sweep_locked_start" not in st.session_state: st.session_state.sweep_locked_start=None
if "sweep_locked_direction" not in st.session_state: st.session_state.sweep_locked_direction=None
if st.session_state.sweep_locked_start is None:
    start_angle=st.session_state.sweep_angle
    clockwise=direction=="Clockwise"
else:
    start_angle=st.session_state.sweep_locked_start
    clockwise=st.session_state.sweep_locked_direction=="Clockwise"

ordered=sweep_order(start_angle,clockwise)
assigned={v for group in st.session_state.sweep_clusters for v in group}|set(st.session_state.sweep_current)
remaining=[p for p in ordered if p["id"] not in assigned]
st.plotly_chart(sweep_figure(st.session_state.sweep_angle,st.session_state.sweep_clusters,st.session_state.sweep_current),use_container_width=True,config={"displayModeBar":False})
st.caption("The red ray rotates over the original image. Colored halos indicate assigned pickup points. The blue numbers identify pickup points for the recorded answer.")
if st.session_state.sweep_locked_start is None:
    st.write("**Proposed sweep order:** "+" -> ".join(p["id"] for p in ordered))
else:
    st.write("**Locked sweep order:** "+" -> ".join(p["id"] for p in ordered))

lookup={p["id"]:p for p in SWEEP_POINTS}
def load_of(ids): return sum(lookup[v]["demand"] for v in ids)
for idx,group in enumerate(st.session_state.sweep_clusters,1):
    st.write(f"**Cluster {idx}:** "+", ".join(group)+f" | Load {load_of(group):,} / 15,000")
cur=st.session_state.sweep_current
st.write("**Current cluster:** "+(", ".join(cur) if cur else "(empty)")+f" | Load {load_of(cur):,} / 15,000")
if not st.session_state.sweep_submitted:
    a,b,c=st.columns(3)
    with a:
        if st.button("Add next pickup",disabled=not remaining,use_container_width=True):
            candidate=remaining[0]
            if load_of(cur)+candidate["demand"]>15000:
                st.warning("Vehicle capacity exceeded. Close this cluster before adding the next pickup.")
            else:
                if st.session_state.sweep_locked_start is None:
                    st.session_state.sweep_locked_start=st.session_state.sweep_angle
                    st.session_state.sweep_locked_direction=direction
                st.session_state.sweep_current=cur+[candidate["id"]]
                # Do not mutate the slider widget key after it has been instantiated.
                st.rerun()
    with b:
        if st.button("Close cluster / next vehicle",disabled=not cur,use_container_width=True):
            st.session_state.sweep_clusters.append(list(cur))
            st.session_state.sweep_current=[]
            st.rerun()
    with c:
        if st.button("Undo last pickup",disabled=not cur,use_container_width=True):
            st.session_state.sweep_current=cur[:-1]
            st.rerun()
    if st.button("Reset sweep",use_container_width=False):
        st.session_state.sweep_clusters=[]
        st.session_state.sweep_current=[]
        st.session_state.sweep_locked_start=None
        st.session_state.sweep_locked_direction=None
        st.rerun()
    if st.button("Submit Sweep Answer",type="primary",disabled=bool(remaining)):
        if cur:
            st.session_state.sweep_clusters.append(list(cur))
            st.session_state.sweep_current=[]
        st.session_state.sweep_submitted=True
        st.rerun()
else:
    st.success("Sweep clusters submitted and locked for the final PDF.")
sweep_cluster_text="; ".join(f"Cluster {i}: {', '.join(group)}" for i,group in enumerate(st.session_state.sweep_clusters,1))
if st.session_state.sweep_submitted: st.code(sweep_cluster_text,language=None)
vrp_vehicles=st.number_input("Number of vehicles required",min_value=0,step=1,
    value=len(st.session_state.sweep_clusters) if st.session_state.sweep_submitted else 0,key="vrp_vehicle_count")
vrp_distance=st.number_input("Total distance traveled by all vehicles",min_value=0.0,step=0.01,format="%.2f")

# ---------- part 3 ----------
st.subheader("5. Bin Packing")
st.caption("Enter the number of trucks/bins produced by each heuristic. Your Excel workbook should show the complete allocation steps.")
b1,b2,b3,b4 = st.columns(4)
with b1: nf = st.number_input("Next Fit", min_value=0, step=1)
with b2: bf = st.number_input("Best Fit", min_value=0, step=1)
with b3: wf = st.number_input("Worst Fit", min_value=0, step=1)
with b4: ffd = st.number_input("First Fit Decreasing", min_value=0, step=1)

# ---------- part 4 ----------
st.subheader("6. Dijkstra's Algorithm")
dij_path = st.text_input("Shortest path from CS to WL", placeholder="CS-...-WL")
dij_dist = st.number_input("Shortest-path distance", min_value=0.0, step=1.0, format="%.0f")

# ---------- part 5 ----------
st.subheader("7. Decision Comparison")
st.caption("Use your own results from Parts 1–4.")
better_distance = st.number_input("Distance saved by 2-Opt compared with Nearest Neighbor", min_value=0.0, step=0.01, format="%.2f")

confirmed = st.checkbox("I have checked these entries against my Excel calculations and am ready to grade them.")

if st.button("Grade Results and Generate Report", type="primary", disabled=not confirmed or not st.session_state.get("sweep_submitted", False)):
    rows1=[]; score=0
    checks=[]
    def add(rows,label,submitted,ok,pts):
        nonlocal_dummy = None
        rows.append([label, str(submitted), status(ok), f"{pts if ok else 0:.2f}"])
        return pts if ok else 0

    ok_nn_route = norm_route(nn_route) == norm_route(str(answers["nn_route"]))
    ok_nn_dist = near(nn_dist, answers["nn_distance"], DIST_TOL)
    ok_opt_route = norm_route(opt_route) == norm_route(str(answers["opt_route"]))
    ok_opt_dist = near(opt_dist, answers["opt_distance"], DIST_TOL)
    ok_pct = near(pct_imp, answers["percent_improvement"], PCT_TOL)
    score += add(rows1,"NN route",nn_route,ok_nn_route,0.50)
    score += add(rows1,"NN distance",f"{nn_dist:.2f}",ok_nn_dist,0.50)
    score += add(rows1,"2-Opt route",opt_route,ok_opt_route,0.50)
    score += add(rows1,"2-Opt distance",f"{opt_dist:.2f}",ok_opt_dist,0.50)
    score += add(rows1,"% improvement",f"{pct_imp:.2f}%",ok_pct,0.50)

    rows2=[]
    rows2.append(["Sweep clusters", sweep_cluster_text or "Not submitted", "Recorded", "Manual review"])
    ok_v = int(vrp_vehicles) == int(answers["vrp_vehicles"])
    ok_vd = near(vrp_distance, answers["vrp_total_distance"], DIST_TOL)
    score += add(rows2,"Vehicles",vrp_vehicles,ok_v,0.75)
    score += add(rows2,"Total VRP distance",f"{vrp_distance:.2f}",ok_vd,1.25)

    rows3=[]
    for label,val,key in [("Next Fit",nf,"next_fit_bins"),("Best Fit",bf,"best_fit_bins"),("Worst Fit",wf,"worst_fit_bins"),("First Fit Decreasing",ffd,"ffd_bins")]:
        ok=int(val)==int(answers[key]); score += add(rows3,label,val,ok,0.625)

    rows4=[]
    ok_dp = norm_route(dij_path) == norm_route(str(answers["dijkstra_path"]))
    ok_dd = near(dij_dist, answers["dijkstra_distance"], DIST_TOL)
    score += add(rows4,"Shortest path",dij_path,ok_dp,1.0)
    score += add(rows4,"Shortest distance",f"{dij_dist:.0f}",ok_dd,1.0)

    rows5=[]
    expected_saved=float(answers["nn_distance"])-float(answers["opt_distance"])
    ok_saved=near(better_distance,expected_saved,DIST_TOL)
    score += add(rows5,"2-Opt distance saved",f"{better_distance:.2f}",ok_saved,1.0)

    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload=f"{sweep_cluster_text}|{nn_route}|{nn_dist}|{opt_route}|{opt_dist}|{pct_imp}|{vrp_vehicles}|{vrp_distance}|{nf}|{bf}|{wf}|{ffd}|{dij_path}|{dij_dist}|{better_distance}"
    verification=make_verification(name.strip(),score,timestamp,payload)

    sections=[
        {"title":"Nearest Neighbor + 2-Opt","rows":rows1},
        {"title":"Sweep Algorithm / VRP","rows":rows2},
        {"title":"Bin Packing","rows":rows3},
        {"title":"Dijkstra's Algorithm","rows":rows4},
        {"title":"Decision Comparison","rows":rows5},
    ]
    rec={"name":name.strip(),"timestamp":timestamp,"verification":verification,"score":score,"sections":sections}
    st.session_state["graded_record"]=rec
    st.session_state["graded_pdf"]=make_pdf(rec)

if "graded_record" in st.session_state:
    rec=st.session_state["graded_record"]
    st.success(f'Automated score: {rec["score"]:.2f} / 10.00')
    st.caption("Incorrect entries are identified, but the private correct answers are not displayed.")
    st.download_button("Download Verified PDF", st.session_state["graded_pdf"], file_name=f"BAN517_Algorithms_Heuristics_{re.sub(r'[^A-Za-z0-9_-]+', '_', name.strip())}.pdf", mime="application/pdf", use_container_width=True)
