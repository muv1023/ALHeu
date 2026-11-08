import io
import os
import re
import hmac
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import base64
import math
import json
from xml.sax.saxutils import escape
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
import streamlit.components.v1 as components
import html

st.set_page_config(page_title="BAN 517 Algorithms & Heuristics Lab", page_icon="🧭", layout="wide")

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

st.markdown("""
<style>
.block-container {max-width: 1350px; padding-top: 2rem;}
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
st.subheader("Student Information")
name = st.text_input("Student name")

if not name.strip():
    st.info("Enter your name to begin.")
    st.stop()

st.subheader("Download Assignment Data")
cols = st.columns(3)
for col, fn, label in [
    (cols[0], "routing_distance_matrix.csv", "NN / 2-Opt distance matrix"),
    (cols[1], "bin_packing_items.csv", "Bin packing data"),
    (cols[2], "dijkstra_edges.csv", "Dijkstra network data"),
]:
    p = DATA / fn
    col.download_button(label, p.read_bytes(), file_name=fn, mime="text/csv", use_container_width=True)

st.caption("Complete calculations outside the app. Use the interactive sweep interface to record your vehicle clusters. The app will not reveal the private answer key.")

# ---------- Route diagram and distance-matrix helpers ----------
NETWORK_NODES = {
    "D": (610, 470), "1": (856, 369), "2": (955, 320),
    "3": (510, 271), "4": (462, 174), "5": (116, 470),
    "6": (363, 567), "7": (363, 666), "8": (560, 764),
    "9": (708, 814), "10": (758, 714), "11": (856, 714),
    "12": (954, 566), "13": (1004, 470),
}
CUSTOMERS = {str(i) for i in range(1, 14)}
MATRIX = pd.read_csv(DATA / "routing_distance_matrix.csv").set_index("From/To")
MATRIX.index = MATRIX.index.map(str)
MATRIX.columns = MATRIX.columns.map(str)


def parse_tour(value):
    tokens = [x.strip().upper() for x in re.split(r"\s*(?:->|→|,|;|[-–])\s*", value.strip()) if x.strip()]
    if len(tokens) != 15 or tokens[0] != "D" or tokens[-1] != "D":
        return None, "Use D-...-D with all 13 customers exactly once."
    if set(tokens[1:-1]) != CUSTOMERS or len(set(tokens[1:-1])) != 13:
        return None, "Each customer from 1 through 13 must appear exactly once."
    return tokens, None


def tour_distance(tour):
    return sum(float(MATRIX.loc[a,b]) for a,b in zip(tour[:-1],tour[1:]))


def route_network_svg(tour=None, emphasize=None):
    # Original network image is the backdrop; route segments are transparent SVG overlays.
    image_uri = "data:image/png;base64," + base64.b64encode(
        (BASE / "assets" / "nearest_neighbor_network.png").read_bytes()).decode("ascii")
    lines = []
    if tour:
        for idx, (a,b) in enumerate(zip(tour[:-1],tour[1:])):
            x1,y1=NETWORK_NODES[a]; x2,y2=NETWORK_NODES[b]
            is_selected = emphasize is not None and idx in emphasize
            color = "#e11d48" if is_selected else "#0d9488"
            width = 7 if is_selected else 4
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="white" stroke-width="{width+3}" stroke-opacity=".85"/>')
            lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" stroke-opacity=".9" stroke-linecap="round"/>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1080 1068" style="width:100%;height:auto;display:block">'
            f'<image href="{image_uri}" x="0" y="0" width="1080" height="1068" preserveAspectRatio="none"/>'
            + ''.join(lines) + '</svg>')


# ---------- Question 4: NN / 2-Opt is rendered after Questions 1-3 ----------
# ---------- Sweep heuristic: draggable ray on assignment image ----------
SWEEP_IMAGE = BASE / "assets" / "sweep_pickup_diagram.png"
SWEEP_DEPOT = (886, 529)
SWEEP_POINTS = [
    {"id":"1", "x":799,"y":132}, {"id":"2", "x":487,"y":295},
    {"id":"3", "x":682,"y":257}, {"id":"4", "x":950,"y":253},
    {"id":"5", "x":1169,"y":314}, {"id":"6", "x":804,"y":387},
    {"id":"7", "x":435,"y":422}, {"id":"8", "x":568,"y":543},
    {"id":"9", "x":604,"y":655}, {"id":"10","x":825,"y":678},
    {"id":"11","x":1120,"y":482}, {"id":"12","x":1056,"y":598},
    {"id":"13","x":1045,"y":745},
]


def point_angle(p):
    return math.degrees(math.atan2(SWEEP_DEPOT[1]-p["y"], p["x"]-SWEEP_DEPOT[0])) % 360

def angle_distance(a, b):
    return abs((a-b+180) % 360-180)

@st.cache_data
def sweep_image_uri():
    return "data:image/png;base64," + base64.b64encode(SWEEP_IMAGE.read_bytes()).decode("ascii")

sweep_component = components.declare_component(
    "ban517_sweep_ray", path=str(BASE / "sweep_component")
)

st.subheader("Question 1. Sweep Heuristic / Vehicle Routing")
st.caption("Drag the ray on the pickup-point diagram. When it reaches a customer, that node highlights. Decide which truck receives each node. Calculate truck loads yourself in Excel; this interface does not calculate or display capacity.")
if not SWEEP_IMAGE.is_file():
    st.error("Missing assets/sweep_pickup_diagram.png")
    st.stop()

for key, default in {
    "sweep_clusters": [], "sweep_current": [], "sweep_submitted": False,
    "sweep_angle_v6": 0, "sweep_event_v6": -1,
    "sweep_start_v6": None, "sweep_direction_v6": "Counterclockwise",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

assigned = {item for cluster in st.session_state.sweep_clusters for item in cluster}
assigned.update(st.session_state.sweep_current)

left, right = st.columns([1.75, 1], gap="large")
with left:
    st.markdown("**Sweep map — drag the red ray**")
    # The HTML component communicates its final angle to Streamlit on pointer release.
    sweep_result = sweep_component(
        image=sweep_image_uri(), nodes=SWEEP_POINTS,
        angle=st.session_state.sweep_angle_v6,
        assigned=sorted(assigned), clusters=st.session_state.sweep_clusters,
        current=st.session_state.sweep_current,
        disabled=st.session_state.sweep_submitted,
        key="sweep_ray_component_v6", default=None,
    )
    if isinstance(sweep_result, dict):
        event = int(sweep_result.get("event", -1))
        if event != st.session_state.sweep_event_v6:
            st.session_state.sweep_event_v6 = event
            st.session_state.sweep_angle_v6 = int(round(float(sweep_result.get("angle", 0)))) % 360
    st.caption("The ray starts at the depot. Drag it to rotate, or use the 5° and 15° controls. No load totals or capacity warnings are shown.")

angle = st.session_state.sweep_angle_v6
available = [p for p in SWEEP_POINTS if p["id"] not in assigned]
closest = min(available, key=lambda p: angle_distance(angle, point_angle(p)), default=None)
hit = closest if closest and angle_distance(angle, point_angle(closest)) <= 7 else None

with right:
    st.markdown("**Truck assignments**")
    if not st.session_state.sweep_submitted:
        if st.session_state.sweep_start_v6 is None:
            st.radio("Sweep direction", ["Counterclockwise", "Clockwise"], key="sweep_direction_v6", horizontal=True)
        else:
            st.caption(f"Sweep direction: {st.session_state.sweep_direction_v6}")
        st.write(f"**Ray angle:** {angle}°")
        st.write(f"**Highlighted customer:** {hit['id'] if hit else 'None'}")
        if st.button("Add highlighted node to current truck", disabled=hit is None, use_container_width=True):
            if st.session_state.sweep_start_v6 is None:
                st.session_state.sweep_start_v6 = angle
            st.session_state.sweep_current.append(hit["id"])
            st.rerun()
    st.markdown(f"**Truck {len(st.session_state.sweep_clusters)+1} — Current**")
    st.write(" → ".join(st.session_state.sweep_current) if st.session_state.sweep_current else "No nodes assigned")
    if not st.session_state.sweep_submitted:
        if st.button("Undo last node", disabled=not st.session_state.sweep_current, use_container_width=True):
            st.session_state.sweep_current.pop()
            st.rerun()
        if st.button("Close current truck", disabled=not st.session_state.sweep_current, use_container_width=True):
            st.session_state.sweep_clusters.append(list(st.session_state.sweep_current))
            st.session_state.sweep_current = []
            st.rerun()
    st.markdown("**Closed trucks**")
    if not st.session_state.sweep_clusters:
        st.caption("None yet")
    for i, cluster in enumerate(st.session_state.sweep_clusters, 1):
        st.write(f"Truck {i}: Depot → {' → '.join(cluster)} → Depot")
    st.caption(f"Customers assigned: {len(assigned)} of {len(SWEEP_POINTS)}")
    if not st.session_state.sweep_submitted:
        if st.button("Reset all trucks", use_container_width=True):
            st.session_state.sweep_clusters = []
            st.session_state.sweep_current = []
            st.session_state.sweep_start_v6 = None
            st.rerun()
        if st.button("Submit sweep answer", type="primary", use_container_width=True,
                     disabled=(len(assigned) != len(SWEEP_POINTS) or bool(st.session_state.sweep_current))):
            st.session_state.sweep_submitted = True
            st.rerun()
        if len(assigned) == len(SWEEP_POINTS) and st.session_state.sweep_current:
            st.info("Close the current truck before submitting.")
    else:
        st.success("Sweep answer recorded and locked.")

sweep_cluster_text = "; ".join(
    f"Truck {i}: Depot → {', '.join(group)} → Depot"
    for i, group in enumerate(st.session_state.sweep_clusters, 1)
)
if st.session_state.sweep_submitted:
    st.code(sweep_cluster_text, language=None)

vrp_vehicles = st.number_input("Number of vehicles required", min_value=0, step=1, key="vrp_vehicle_count")


# ---------- Question 2: Shortest path ----------
st.subheader("Question 2. Shortest Path — Dijkstra's Algorithm")
if "dijkstra_locked" not in st.session_state:
    st.session_state.dijkstra_locked = None
if st.session_state.dijkstra_locked is None:
    dij_path = st.text_input("Shortest path from CS to WL", placeholder="CS-...-WL", key="dij_path_input")
    dij_dist = st.number_input("Shortest-path distance", min_value=0.0, step=1.0, format="%.0f", key="dij_distance_input")
    if st.button("Submit Dijkstra Answer", type="primary", disabled=not dij_path.strip()):
        st.session_state.dijkstra_locked = {"path": dij_path.strip(), "distance": float(dij_dist)}
        st.rerun()
else:
    dij_path = st.session_state.dijkstra_locked["path"]
    dij_dist = st.session_state.dijkstra_locked["distance"]
    st.success("Dijkstra answer submitted and recorded.")
    st.write(f"Path: {dij_path} | Distance: {dij_dist:.0f}")
    if st.button("Edit Dijkstra Answer"):
        st.session_state.dijkstra_locked = None
        st.rerun()

# ---------- part 3 ----------
st.subheader("Question 3. Bin Packing")
st.caption("Enter the number of trucks/bins produced by each heuristic. Your Excel workbook should show the complete allocation steps.")
b1,b2,b3,b4 = st.columns(4)
with b1: nf = st.number_input("Next Fit", min_value=0, step=1)
with b2: bf = st.number_input("Best Fit", min_value=0, step=1)
with b3: wf = st.number_input("Worst Fit", min_value=0, step=1)
with b4: ffd = st.number_input("First Fit Decreasing", min_value=0, step=1)

# ---------- Question 4: Nearest Neighbor and 2-Opt ----------
st.subheader("Question 4. Nearest Neighbor and 2-Opt")
st.caption("Enter your nearest-neighbor tour and submit it to display its arcs on the spatial network. Use the distance matrix for all mileage calculations; the diagram is not drawn to distance scale.")
with st.expander("View the 14 x 14 distance matrix (miles)"):
    st.dataframe(MATRIX, use_container_width=True)

if "nn_locked_v8" not in st.session_state:
    st.session_state.nn_locked_v8 = None

nn_left, nn_right = st.columns([1.0, 1.18], gap="large")
with nn_left:
    st.markdown("**Nearest Neighbor solution**")
    nn_route = st.text_input("Nearest Neighbor route", placeholder="D-3-4-...-D", key="nn_route_v8")
    nn_dist = st.number_input("Nearest Neighbor total distance (miles)", min_value=0.0, step=0.01, format="%.2f", key="nn_distance_v8")
    if st.button("Submit NN Solution", type="primary", use_container_width=True):
        parsed, err = parse_tour(nn_route)
        if err:
            st.error(err)
        else:
            st.session_state.nn_locked_v8 = parsed
            st.rerun()
    if st.session_state.nn_locked_v8:
        st.success("NN route recorded. Its arcs are displayed on the network.")
        if st.button("Clear displayed NN route", use_container_width=True):
            st.session_state.nn_locked_v8 = None
            st.rerun()
    st.divider()
    st.markdown("**2-Opt improvement**")
    st.caption("Inspect the displayed NN route for crossing arcs, then enter your improved 2-Opt route and results below.")
    opt_route = st.text_input("2-Opt improved route", placeholder="D-...-D", key="opt_route_v8")
    opt_dist = st.number_input("2-Opt improved distance (miles)", min_value=0.0, step=0.01, format="%.2f", key="opt_distance_v8")
    pct_imp = st.number_input("Percentage savings from NN to 2-Opt (%)", min_value=0.0, step=0.01, format="%.2f", key="pct_improvement_v8")
    if st.button("Submit 2-Opt Solution", use_container_width=True):
        parsed, err = parse_tour(opt_route)
        if err:
            st.error(err)
        else:
            st.session_state.opt_submitted_v8 = True
            st.success("2-Opt route recorded for the final report.")

with nn_right:
    st.markdown("**Spatial network and submitted NN route**")
    tour = st.session_state.nn_locked_v8
    svg = route_network_svg(tour)
    components.html(f'<html><body style="margin:0;background:#f5f3eb">{svg}</body></html>', height=610, scrolling=False)
    if tour:
        st.caption("Teal: submitted NN tour arcs. Inspect the diagram for crossings; use the distance matrix to calculate route distance.")
    else:
        st.caption("Submit the NN route in the left panel to overlay its arcs on this diagram.")

st.subheader("Final comparison")
better_distance = st.number_input("Distance saved by 2-Opt compared with Nearest Neighbor", min_value=0.0, step=0.01, format="%.2f")

if st.button("Grade Results and Generate Report", type="primary", disabled=not st.session_state.get("sweep_submitted", False) or st.session_state.get("dijkstra_locked") is None or st.session_state.get("nn_locked_v8") is None):
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
    score += add(rows2,"Vehicles",vrp_vehicles,ok_v,2.0)

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
    payload=f"{sweep_cluster_text}|{nn_route}|{nn_dist}|{opt_route}|{opt_dist}|{pct_imp}|{vrp_vehicles}|{nf}|{bf}|{wf}|{ffd}|{dij_path}|{dij_dist}|{better_distance}"
    verification=make_verification(name.strip(),score,timestamp,payload)

    sections=[
        {"title":"Question 1 - Sweep Heuristic / VRP","rows":rows2},
        {"title":"Question 2 - Shortest Path / Dijkstra","rows":rows4},
        {"title":"Question 3 - Bin Packing","rows":rows3},
        {"title":"Question 4 - Nearest Neighbor and 2-Opt","rows":rows1},
        {"title":"Final Comparison","rows":rows5},
    ]
    rec={"name":name.strip(),"timestamp":timestamp,"verification":verification,"score":score,"sections":sections}
    st.session_state["graded_record"]=rec
    st.session_state["graded_pdf"]=make_pdf(rec)

if "graded_record" in st.session_state:
    rec=st.session_state["graded_record"]
    st.success(f'Automated score: {rec["score"]:.2f} / 10.00')
    st.caption("Incorrect entries are identified, but the private correct answers are not displayed.")
    st.download_button("Download Verified PDF", st.session_state["graded_pdf"], file_name=f"BAN517_Algorithms_Heuristics_{re.sub(r'[^A-Za-z0-9_-]+', '_', name.strip())}.pdf", mime="application/pdf", use_container_width=True)
