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
import math
import json
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


def make_verification(name, student_id, score, timestamp, payload):
    msg = f"{name}|{student_id}|{score:.2f}|{timestamp}|{payload}".encode()
    return hmac.new(SALT.encode(), msg, hashlib.sha256).hexdigest()[:16].upper()


def make_pdf(record):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Centered", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#003b5c")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], textColor=colors.HexColor("#003b5c"), spaceBefore=10, spaceAfter=6))
    story = [Paragraph("BAN 517 Algorithms & Heuristics Decision Lab", styles["Centered"]), Spacer(1, 10)]
    meta = [
        ["Student", record["name"]], ["Student ID", record["student_id"]],
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
c1, c2 = st.columns(2)
with c1:
    name = st.text_input("Student name")
with c2:
    student_id = st.text_input("Student ID")

if not name.strip() or not student_id.strip():
    st.info("Enter your name and student ID to begin.")
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

# ---------- interactive sweep heuristic ----------
def read_sweep_points(csv_path):
    """Read routing_data.csv without assuming a single header convention.

    Expected: a customer/node identifier and numeric x,y coordinates;
    optionally a demand/pickup quantity. Depot may be marked by a depot
    flag or identifier such as Depot / 0. If absent, use (0,0) and say so.
    """
    frame = pd.read_csv(csv_path)
    frame.columns = [str(c).strip() for c in frame.columns]
    canon = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in frame.columns}
    def choose(names):
        for n in names:
            if n in canon: return canon[n]
        return None
    id_col = choose(["customer", "customerid", "node", "nodeid", "location", "locationid", "stop", "stopid", "id", "point"])
    x_col = choose(["x", "xcoordinate", "coordx", "longitude", "lon"])
    y_col = choose(["y", "ycoordinate", "coordy", "latitude", "lat"])
    q_col = choose(["demand", "quantity", "pickup", "pickupquantity", "volume", "load", "units"])
    depot_col = choose(["isdepot", "depot", "type", "nodetype"])
    if not (id_col and x_col and y_col):
        raise ValueError("routing_data.csv needs customer/node ID, X, and Y columns. Found: " + ", ".join(frame.columns))
    frame = frame.copy()
    frame[x_col] = pd.to_numeric(frame[x_col], errors="coerce")
    frame[y_col] = pd.to_numeric(frame[y_col], errors="coerce")
    frame = frame.dropna(subset=[x_col,y_col])
    if frame.empty: raise ValueError("No valid coordinates were found in routing_data.csv")
    def is_depot(r):
        label = str(r[id_col]).strip().lower()
        if label in ("depot", "warehouse", "origin", "dc", "0"): return True
        if depot_col:
            v = str(r[depot_col]).strip().lower()
            return v in ("true", "yes", "1", "depot", "warehouse", "dc")
        return False
    depots = frame[frame.apply(is_depot, axis=1)]
    if len(depots):
        d = depots.iloc[0]
        origin = (float(d[x_col]), float(d[y_col]))
        frame = frame[~frame.apply(is_depot, axis=1)]
        depot_note = "Depot identified in routing data."
    else:
        origin = (0.0, 0.0)
        depot_note = "No depot row identified; using (0, 0) as the sweep origin. Check the assignment figure."
    points = []
    for _,r in frame.iterrows():
        q = None if not q_col or pd.isna(r[q_col]) else float(r[q_col])
        points.append({"id":str(r[id_col]).strip(), "x":float(r[x_col]), "y":float(r[y_col]), "demand":q})
    if len({v["id"] for v in points}) != len(points): raise ValueError("Customer IDs must be unique.")
    return points, origin, depot_note


def sweep_order(points, origin, start_deg, clockwise):
    # Angle zero points right (+x); positive angles are counterclockwise.
    def angle(p):
        return math.degrees(math.atan2(p["y"]-origin[1],p["x"]-origin[0])) % 360
    return sorted(points, key=lambda p: (((start_deg-angle(p)) if clockwise else (angle(p)-start_deg)) % 360, p["id"]))


def draw_sweep(points, origin, angle_deg, clusters, current):
    fig=go.Figure()
    all_xy=[origin]+[(p["x"],p["y"]) for p in points]
    radius=max([math.hypot(x-origin[0],y-origin[1]) for x,y in all_xy] + [1.0])*1.13
    colorset=["#2563eb", "#059669", "#d97706", "#9333ea", "#dc2626", "#0891b2", "#4f46e5"]
    lookup={p["id"]:p for p in points}
    for j, ids in enumerate(clusters + ([current] if current else [])):
        if not ids: continue
        xy=[lookup[k] for k in ids if k in lookup]
        fig.add_trace(go.Scatter(x=[p["x"] for p in xy],y=[p["y"] for p in xy],mode="markers+text",
            text=[p["id"] for p in xy],textposition="top center",marker=dict(size=15,color=colorset[j%len(colorset)]),
            name=f"Cluster {j+1}"+(" (building)" if j==len(clusters) else ""),hovertemplate="Customer %{text}<extra></extra>"))
    assigned={k for c in clusters for k in c}|set(current)
    remaining=[p for p in points if p["id"] not in assigned]
    if remaining:
        fig.add_trace(go.Scatter(x=[p["x"] for p in remaining],y=[p["y"] for p in remaining],mode="markers+text",
            text=[p["id"] for p in remaining],textposition="top center",marker=dict(size=12,color="#94a3b8"),
            name="Unassigned",hovertemplate="Customer %{text}<extra></extra>"))
    rad=math.radians(angle_deg)
    fig.add_trace(go.Scatter(x=[origin[0],origin[0]+radius*math.cos(rad)],y=[origin[1],origin[1]+radius*math.sin(rad)],
        mode="lines",line=dict(color="#e11d48",width=3),name="Sweep ray"))
    fig.add_trace(go.Scatter(x=[origin[0]],y=[origin[1]],mode="markers+text",text=["Depot"],textposition="bottom center",
        marker=dict(symbol="star",size=17,color="#0f172a"),name="Depot"))
    fig.update_layout(height=490,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="X coordinate",yaxis_title="Y coordinate",
        legend=dict(orientation="h",y=-0.19),plot_bgcolor="white")
    fig.update_yaxes(scaleanchor="x",scaleratio=1)
    return fig


# ---------- part 2 ----------
st.subheader("4. Sweep Algorithm / VRP")
st.caption("Use the assignment pickup diagram as a visual reference, then set the sweep ray and assign customers in the interactive map. Cluster memberships are saved in your PDF report.")

# Static assignment diagram. The interactive map below uses the CSV coordinates
# so that sweep ordering and vehicle capacity checks remain data-driven.
SWEEP_IMAGE = BASE / "assets" / "sweep_pickup_diagram.png"
if SWEEP_IMAGE.is_file():
    st.image(str(SWEEP_IMAGE), caption="Assignment pickup diagram (visual reference; demands shown beside each pickup point)", use_container_width=True)
else:
    st.warning("Reference diagram missing. Add assets/sweep_pickup_diagram.png to your GitHub repository.")

try:
    sweep_points, sweep_origin, depot_note = read_sweep_points(DATA / "routing_data.csv")
    st.caption(depot_note)
    if "sweep_clusters" not in st.session_state: st.session_state.sweep_clusters=[]
    if "sweep_current" not in st.session_state: st.session_state.sweep_current=[]
    if "sweep_submitted" not in st.session_state: st.session_state.sweep_submitted=False
    c1,c2=st.columns([2,1])
    with c1:
        sweep_start=st.slider("Starting ray (degrees from positive X-axis)",0,359,0,1,key="sweep_start",disabled=st.session_state.sweep_submitted)
    with c2:
        sweep_direction=st.radio("Sweep direction",["Counterclockwise","Clockwise"],horizontal=True,key="sweep_direction",disabled=st.session_state.sweep_submitted)
    sweep_settings=(sweep_start,sweep_direction)
    if st.session_state.get("sweep_previous_settings") not in (None,sweep_settings) and not st.session_state.sweep_submitted:
        st.session_state.sweep_clusters=[]
        st.session_state.sweep_current=[]
    st.session_state.sweep_previous_settings=sweep_settings
    ordered=sweep_order(sweep_points,sweep_origin,sweep_start,sweep_direction=="Clockwise")
    assigned_ids={k for group in st.session_state.sweep_clusters for k in group}|set(st.session_state.sweep_current)
    remaining=[p for p in ordered if p["id"] not in assigned_ids]
    # While building a cluster, show the ray at the next point; otherwise show the selected starting ray.
    shown_angle=sweep_start
    if st.session_state.sweep_current and remaining:
        nxt=remaining[0]
        shown_angle=math.degrees(math.atan2(nxt["y"]-sweep_origin[1],nxt["x"]-sweep_origin[0]))%360
    st.plotly_chart(draw_sweep(sweep_points,sweep_origin,shown_angle,st.session_state.sweep_clusters,st.session_state.sweep_current),use_container_width=True)
    st.caption("Gray = unassigned; colored = assigned clusters; red = sweep ray. Adjust the starting ray and direction before assigning customers.")
    st.write("**Sweep order:** " + " → ".join(p["id"] for p in ordered))
    capacity=15000.0
    lookup={p["id"]:p for p in sweep_points}
    def load_of(ids):
        vals=[lookup[k]["demand"] for k in ids]
        return sum(vals) if all(v is not None for v in vals) else None
    for idx,grp in enumerate(st.session_state.sweep_clusters,1):
        ld=load_of(grp)
        st.write(f"Cluster {idx}: " + ", ".join(grp) + (f" | Load: {ld:,.0f} / {capacity:,.0f}" if ld is not None else ""))
    cur=st.session_state.sweep_current
    st.write("**Current cluster:** " + (", ".join(cur) if cur else "(empty)"))
    if cur and load_of(cur) is not None: st.caption(f"Current load: {load_of(cur):,.0f} / {capacity:,.0f}")
    if not st.session_state.sweep_submitted:
        a,b,c=st.columns(3)
        with a:
            if st.button("Add next customer",disabled=not remaining,use_container_width=True):
                candidate=remaining[0]
                newload=load_of(cur+[candidate["id"]])
                if newload is not None and newload>capacity:
                    st.warning("Adding this customer would exceed the 15,000-unit vehicle capacity. Close the current cluster first.")
                else:
                    st.session_state.sweep_current=cur+[candidate["id"]]
                    st.rerun()
        with b:
            if st.button("Close cluster / next vehicle",disabled=not cur,use_container_width=True):
                st.session_state.sweep_clusters.append(list(cur))
                st.session_state.sweep_current=[]
                st.rerun()
        with c:
            if st.button("Undo last assignment",disabled=not cur,use_container_width=True):
                st.session_state.sweep_current=cur[:-1]
                st.rerun()
        if st.button("Reset sweep clusters",use_container_width=False):
            st.session_state.sweep_clusters=[]
            st.session_state.sweep_current=[]
            st.rerun()
        all_assigned=(len(remaining)==0)
        if st.button("Submit Sweep Answer",type="primary",disabled=not all_assigned):
            if cur:
                st.session_state.sweep_clusters.append(list(cur))
                st.session_state.sweep_current=[]
            st.session_state.sweep_submitted=True
            st.rerun()
    else:
        st.success("Sweep answer submitted. Cluster memberships are locked for the final report.")
    sweep_cluster_text="; ".join(f"Cluster {i}: {', '.join(group)}" for i,group in enumerate(st.session_state.sweep_clusters,1))
    if st.session_state.sweep_submitted:
        st.code(sweep_cluster_text,language=None)
    vrp_vehicles=st.number_input("Number of vehicles required",min_value=0,step=1,
        value=len(st.session_state.sweep_clusters) if st.session_state.sweep_submitted else 0,
        key="vrp_vehicle_count")
    vrp_distance=st.number_input("Total distance traveled by all vehicles",min_value=0.0,step=0.01,format="%.2f")
except (FileNotFoundError, ValueError, KeyError) as e:
    st.error(f"Sweep map unavailable: {e}")
    st.info("Upload a routing_data.csv with customer ID, X, Y, and optional Demand columns to the data folder.")
    sweep_cluster_text=""
    vrp_vehicles=st.number_input("Number of vehicles required",min_value=0,step=1)
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
    verification=make_verification(name.strip(),student_id.strip(),score,timestamp,payload)

    sections=[
        {"title":"Nearest Neighbor + 2-Opt","rows":rows1},
        {"title":"Sweep Algorithm / VRP","rows":rows2},
        {"title":"Bin Packing","rows":rows3},
        {"title":"Dijkstra's Algorithm","rows":rows4},
        {"title":"Decision Comparison","rows":rows5},
    ]
    rec={"name":name.strip(),"student_id":student_id.strip(),"timestamp":timestamp,"verification":verification,"score":score,"sections":sections}
    st.session_state["graded_record"]=rec
    st.session_state["graded_pdf"]=make_pdf(rec)

if "graded_record" in st.session_state:
    rec=st.session_state["graded_record"]
    st.success(f'Automated score: {rec["score"]:.2f} / 10.00')
    st.caption("Incorrect entries are identified, but the private correct answers are not displayed.")
    st.download_button("Download Verified PDF", st.session_state["graded_pdf"], file_name=f"BAN517_Algorithms_Heuristics_{student_id.strip()}.pdf", mime="application/pdf", use_container_width=True)
