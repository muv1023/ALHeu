import io
import os
import re
import hmac
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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
        tab = Table(data, colWidths=[175,150,90,55], repeatRows=1)
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

st.caption("Complete your calculations outside the app. Enter only your final results below. The app will not reveal the private answer key.")

# ---------- part 1 ----------
st.subheader("3. Nearest Neighbor + 2-Opt")
nn_route = st.text_input("Nearest Neighbor route", placeholder="Depot-F-...-Depot")
nn_dist = st.number_input("Nearest Neighbor total distance", min_value=0.0, step=0.01, format="%.2f")
opt_route = st.text_input("2-Opt improved route", placeholder="Depot-F-...-Depot")
opt_dist = st.number_input("2-Opt improved total distance", min_value=0.0, step=0.01, format="%.2f")
pct_imp = st.number_input("Percentage improvement from NN to 2-Opt (%)", min_value=0.0, step=0.01, format="%.2f")

# ---------- part 2 ----------
st.subheader("4. Sweep Algorithm / VRP")
vrp_vehicles = st.number_input("Number of vehicles required", min_value=0, step=1)
vrp_distance = st.number_input("Total distance traveled by all vehicles", min_value=0.0, step=0.01, format="%.2f")

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

if st.button("Grade Results and Generate Report", type="primary", disabled=not confirmed):
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
    payload=f"{nn_route}|{nn_dist}|{opt_route}|{opt_dist}|{pct_imp}|{vrp_vehicles}|{vrp_distance}|{nf}|{bf}|{wf}|{ffd}|{dij_path}|{dij_dist}|{better_distance}"
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
