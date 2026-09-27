import streamlit as st
import cv2
from PIL import Image
import numpy as np
import hashlib
import random
from datetime import datetime
from ultralytics import YOLO
import time
import os
import urllib.request

st.set_page_config(page_title="IBVAP | Video Analytics Platform", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

yolo_model = load_model()
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# --- GREEN / BLACK VMS-STYLE UI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    :root {
        --bg: #060A08;
        --panel: #0B120E;
        --panel-2: #0F1912;
        --line: #16241C;
        --green: #22C55E;
        --green-bright: #4ADE80;
        --green-dim: #86EFAC;
        --text: #E7F5EC;
        --muted: #6B8577;
    }

    .stApp { background-color: var(--bg); color: var(--text); }
    .block-container { padding-top: 1.2rem; max-width: 100%; }

    /* ===== Sidebar as camera/category list ===== */
    div[data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--line);
    }
    div[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
        color: var(--text) !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        border-bottom: none;
        padding-bottom: 6px;
    }
    div[data-testid="stSidebar"] label { color: #CFE8DA !important; font-size: 13.5px; }
    div[data-testid="stSidebar"] .stCheckbox { padding: 2px 0; }
    div[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p { margin-bottom: 0; }

    div[data-testid="stTextInput"] input {
        background-color: #060A08;
        border: 1px solid var(--line);
        border-radius: 8px;
        color: var(--text);
    }

    div[data-testid="stAlert"] {
        background-color: var(--panel-2);
        border: 1px solid var(--line);
        border-radius: 10px;
        color: var(--green-dim);
    }

    /* Checkbox accent color */
    div[data-testid="stCheckbox"] label span[role="checkbox"] {
        border-color: var(--line) !important;
    }
    div[data-testid="stCheckbox"] input:checked + div span[role="checkbox"] {
        background-color: var(--green) !important;
        border-color: var(--green) !important;
    }

    /* Headings */
    h1 { color: var(--text) !important; font-weight: 800; letter-spacing: -0.01em; border-bottom: none; }
    h2, h3 {
        color: var(--text) !important;
        font-weight: 700;
        letter-spacing: -0.01em;
        border-bottom: 1px solid var(--line);
        padding-bottom: 8px;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* ===== Top nav bar ===== */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 22px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        margin-bottom: 18px;
    }
    .nav-left { display: flex; align-items: center; gap: 34px; }
    .brand { display: flex; align-items: center; gap: 10px; }
    .brand-mark {
        width: 26px; height: 26px;
        background: linear-gradient(135deg, var(--green-bright), var(--green));
        clip-path: polygon(100% 0, 100% 100%, 30% 50%);
    }
    .brand-name { font-weight: 800; font-size: 17px; letter-spacing: -0.02em; color: var(--text); }
    .nav-tabs { display: flex; gap: 4px; }
    .nav-tab {
        padding: 7px 14px;
        border-radius: 8px;
        font-size: 13.5px;
        font-weight: 600;
        color: var(--muted);
    }
    .nav-tab.active { background: var(--green); color: #04140A; }
    .nav-right { display: flex; align-items: center; gap: 20px; }
    .meter-group { display: flex; flex-direction: column; gap: 4px; }
    .meter-row { display: flex; align-items: center; gap: 8px; font-size: 10.5px; color: var(--muted); }
    .meter-bar { width: 70px; height: 5px; border-radius: 3px; background: #16241C; overflow: hidden; }
    .meter-fill { height: 100%; border-radius: 3px; }
    .icon-btn {
        width: 30px; height: 30px;
        display: flex; align-items: center; justify-content: center;
        border: 1px solid var(--line);
        border-radius: 50%;
        color: var(--muted);
        font-size: 13px;
    }

    /* Filter chip row */
    .chip-row { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
    .chip {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 7px 14px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 9px;
        font-size: 12.5px;
        color: var(--muted);
    }
    .chip.active { border-color: var(--green); color: var(--green-dim); }
    .chip b { color: var(--text); background: rgba(34,197,94,0.15); padding: 1px 7px; border-radius: 5px; }

    /* Status pill */
    .status-pill {
        display: inline-flex; align-items: center; gap: 7px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: var(--green-bright);
        font-size: 12.5px; font-weight: 700;
        padding: 5px 12px; border-radius: 999px;
    }
    .live-dot {
        height: 8px; width: 8px;
        background-color: var(--green-bright);
        border-radius: 50%;
        display: inline-block;
        animation: pulse 1.6s ease-out infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.55); }
        70%  { box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }
        100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
    }

    /* ===== Panels / cards ===== */
    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .status-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 3px solid var(--green);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .status-card-critical {
        background: rgba(239, 68, 68, 0.06);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-left: 3px solid #EF4444;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .status-title { font-weight: 700; font-size: 14.5px; color: var(--text); margin-bottom: 4px; }
    .status-title-critical { font-weight: 700; font-size: 14.5px; color: #FCA5A5; margin-bottom: 4px; }
    .status-body { color: var(--muted); font-size: 13.5px; line-height: 1.5; }

    .log-entry {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: #9FC6AC;
        padding: 8px 0;
        border-bottom: 1px solid var(--line);
    }
    .log-entry:last-child { border-bottom: none; }
    .log-hash {
        color: var(--green-bright);
        background: rgba(74, 222, 128, 0.08);
        padding: 2px 6px;
        border-radius: 4px;
    }

    .vector-readout {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: var(--green-dim);
        background: rgba(34, 197, 94, 0.06);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 8px;
    }

    /* Video tile — framed like a grid camera panel */
    .video-frame {
        background: #000;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 10px;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- TOP NAVBAR ---
st.markdown("""
    <div class="navbar">
        <div class="nav-left">
            <div class="brand">
                <div class="brand-mark"></div>
                <div class="brand-name">ibvap</div>
            </div>
            <div class="nav-tabs">
                <div class="nav-tab active">Dashboard</div>
                <div class="nav-tab">Events</div>
                <div class="nav-tab">Archive</div>
                <div class="nav-tab">Configuration</div>
                <div class="nav-tab">Settings</div>
            </div>
        </div>
        <div class="nav-right">
            <div class="meter-group">
                <div class="meter-row">CPU
                    <div class="meter-bar"><div class="meter-fill" style="width:55%; background:#4ADE80;"></div></div>
                </div>
                <div class="meter-row">MEM
                    <div class="meter-bar"><div class="meter-fill" style="width:38%; background:#4ADE80;"></div></div>
                </div>
            </div>
            <div class="icon-btn">?</div>
            <div class="icon-btn">&#9993;</div>
            <div class="icon-btn">&#9679;</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- FILTER / STATUS CHIP ROW ---
st.markdown("""
    <div class="chip-row">
        <div class="chip active">&#9660; All Cameras <b>1</b></div>
        <div class="chip">&#9660; Rule Wise View</div>
        <div class="chip">&#9660; Today</div>
        <div style="flex:1;"></div>
        <div class="status-pill"><span class="live-dot"></span> LIVE — NODE ALPHA</div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR: styled as a camera/category list ---
st.sidebar.markdown('<div class="brand" style="margin-bottom:14px;"><div class="brand-mark"></div><div class="brand-name">ibvap</div></div>', unsafe_allow_html=True)
st.sidebar.text_input("Search", placeholder="Search cameras / rules", label_visibility="collapsed")
st.sidebar.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

st.sidebar.markdown("**NODE ALPHA — DETECTION RULES**")
enable_tripwire = st.sidebar.checkbox("Motion Tripwire", value=True)
enable_nightvision = st.sidebar.checkbox("Low-Light Enhancement", value=False)
enable_frs = st.sidebar.checkbox("Face Recognition (FRS)", value=True)
enable_blockchain = st.sidebar.checkbox("Evidence Hashing", value=True)

st.sidebar.markdown("---")
st.sidebar.info("**Status:** Online\n\n**Engine:** PyTorch / YOLOv8\n\n**Network:** Optimized")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Feed")
    video_placeholder = st.empty()
    fps_text = st.empty()
    vector_display = st.empty()

with col2:
    st.subheader("Threat Status")
    threat_status = st.empty()
    st.subheader("Evidence Log")
    evidence_log = st.empty()

# Automatically download a sample test video if it doesn't exist yet
video_path = "border_test.mp4"
if not os.path.exists(video_path):
    st.info("Downloading sample video for the first run — this takes a few seconds.")
    video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
    urllib.request.urlretrieve(video_url, video_path)

cap = cv2.VideoCapture(video_path)
logs = []

# --- MAIN VIDEO LOOP ---
while cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame is None:
        st.warning("Video feed ended or connection lost. Refresh the page to restart.")
        break

    # 1. Low-light enhancement
    if enable_nightvision:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.putText(frame, "LOW-LIGHT ENHANCED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    motion_detected = True
    current_fps = "30 FPS — full compute"

    # 2. Tripwire
    if enable_tripwire:
        fgmask = fgbg.apply(frame)
        if cv2.countNonZero(fgmask) < 1000:
            motion_detected = False
            current_fps = "5 FPS — idle mode (saving ~85% CPU)"
            cv2.putText(frame, "IDLE — NO MOTION", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

    threat_score = 0
    threat_level = "Nominal"
    simulated_vector = ""

    # 3. YOLOv8 inference
    if motion_detected:
        results = yolo_model(frame, verbose=False)
        humans = 0
        vehicles = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    humans += 1
                elif cls in [2, 3, 5, 7]:
                    vehicles += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)

                if cls == 0 and enable_frs:
                    cv2.putText(frame, "Person — extracting vector", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74, 222, 128), 2)
                    simulated_vector = f"**Vector generated (2 KB):** `[0.{random.randint(100,999)}, -0.{random.randint(100,999)}, 0.{random.randint(100,999)} ... 512d]`"
                elif cls in [2, 3, 5, 7]:
                    cv2.putText(frame, "Vehicle — ANPR active", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (134, 239, 172), 2)

        threat_score = (humans * 50) + (vehicles * 30)

        if threat_score > 40:
            threat_level = f"Elevated (score {threat_score})"

            if enable_blockchain and len(logs) < 4:
                frame_bytes = frame.tobytes()
                crypto_hash = hashlib.sha256(frame_bytes).hexdigest()
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = (
                    f"<div class='log-entry'><b>{timestamp}</b> &nbsp;·&nbsp; "
                    f"<span class='log-hash'>{crypto_hash[:24]}…</span></div>"
                )
                if not any(crypto_hash[:24] in log for log in logs):
                    logs.insert(0, log_entry)

    # --- UI RENDERING ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    fps_text.markdown(f"<span style='color:#6B8577; font-size:13px;'>{current_fps}</span>", unsafe_allow_html=True)

    if simulated_vector:
        vector_display.markdown(f"<div class='vector-readout'>{simulated_vector}</div>", unsafe_allow_html=True)
    else:
        vector_display.empty()

    if "Elevated" in threat_level:
        threat_status.markdown(
            f"<div class='status-card-critical'>"
            f"<div class='status-title-critical'>⚠ {threat_level}</div>"
            f"<div class='status-body'>Person detected inside the monitored zone. Trajectory tracking engaged.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        threat_status.markdown(
            f"<div class='status-card'>"
            f"<div class='status-title'>● {threat_level}</div>"
            f"<div class='status-body'>Area clear. Awaiting motion trigger.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if enable_blockchain and logs:
        log_text = "".join(logs)
        evidence_log.markdown(f"<div class='panel'>{log_text}</div>", unsafe_allow_html=True)
    elif not enable_blockchain:
        evidence_log.markdown(
            "<div class='panel' style='border-left: 3px solid #6B8577;'>"
            "<span style='color:#6B8577;'>Evidence hashing is turned off.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        evidence_log.markdown(
            "<div class='panel'><span style='color:#6B8577;'>No events logged yet.</span></div>",
            unsafe_allow_html=True,
        )

    time.sleep(0.05)

cap.release()
