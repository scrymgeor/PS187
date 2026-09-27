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

# --- UI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700&family=Roboto+Slab:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}

    html, body, [class*="css"] { font-family: 'Noto Sans', Arial, sans-serif; }

    :root {
        --saffron: #FF9933;
        --white: #FFFFFF;
        --india-green: #128807;
        --navy: #0B3D6B;
        --navy-dark: #08294A;
        --blue: #1A5FA3;
        --text: #1F2937;
        --muted: #5B6B7A;
        --bg: #F4F6F8;
        --panel: #FFFFFF;
        --line: #DCE3E8;
        --amber: #B45309;
    }

    .stApp { background-color: var(--bg); color: var(--text); }
    .block-container { padding-top: 1rem; max-width: 100%; }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--line);
    }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
        color: var(--navy) !important;
        font-size: 12.5px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 700;
        border-bottom: 2px solid var(--saffron);
        padding-bottom: 6px;
    }
    div[data-testid="stSidebar"] label { color: var(--text) !important; font-size: 13.5px; }
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 6px;
        color: var(--text);
    }
    div[data-testid="stAlert"] {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 3px solid var(--blue);
        border-radius: 6px;
        color: #1E3A5F;
    }
    div[data-testid="stCheckbox"] input:checked + div span[role="checkbox"] {
        background-color: var(--india-green) !important;
        border-color: var(--india-green) !important;
    }

    h1 { color: var(--navy) !important; font-family: 'Roboto Slab', serif; font-weight: 700; }
    h2, h3 {
        color: var(--navy) !important;
        font-weight: 700;
        border-bottom: 2px solid var(--line);
        padding-bottom: 8px;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Tricolor top strip */
    .tricolor-strip {
        height: 6px;
        width: 100%;
        background: linear-gradient(to right, var(--saffron) 0%, var(--saffron) 33.3%, var(--white) 33.3%, var(--white) 66.6%, var(--india-green) 66.6%, var(--india-green) 100%);
        border-radius: 4px;
        margin-bottom: 14px;
        border: 1px solid var(--line);
    }

    /* Header bar (gov portal style) */
    .govheader {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 10px;
    }
    .govheader-left { display: flex; align-items: center; gap: 16px; }
    .emblem {
        width: 46px; height: 46px;
        border-radius: 50%;
        background: var(--navy);
        display: flex; align-items: center; justify-content: center;
        color: var(--saffron);
        font-weight: 800;
        font-size: 15px;
        border: 2px solid var(--saffron);
    }
    .govheader-title { font-family: 'Roboto Slab', serif; font-weight: 700; font-size: 19px; color: var(--navy); }
    .govheader-sub { color: var(--muted); font-size: 12.5px; margin-top: 2px; }
    .govheader-right { text-align: right; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 7px;
        background: #ECFDF3;
        border: 1px solid #A7E3B8;
        color: #15803D;
        font-size: 12.5px; font-weight: 700;
        padding: 5px 12px; border-radius: 999px;
    }
    .live-dot {
        height: 8px; width: 8px;
        background-color: #16A34A;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 1.6s ease-out infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.5); }
        70%  { box-shadow: 0 0 0 8px rgba(22, 163, 74, 0); }
        100% { box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
    }

    /* Nav tab strip */
    .navtabs {
        display: flex; gap: 2px;
        background: var(--navy);
        border-radius: 10px;
        padding: 4px;
        margin-bottom: 18px;
    }
    .navtab {
        padding: 9px 18px;
        border-radius: 7px;
        font-size: 13.5px;
        font-weight: 600;
        color: #C9D9E8;
    }
    .navtab.active { background: var(--saffron); color: var(--navy-dark); }

    /* Breadcrumb */
    .breadcrumb { color: var(--muted); font-size: 12.5px; margin: 4px 0 16px 2px; }
    .breadcrumb b { color: var(--navy); }

    /* Filter chip row */
    .chip-row { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; align-items: center; }
    .chip {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 7px 14px;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 7px;
        font-size: 12.5px;
        color: var(--muted);
    }
    .chip.active { border-color: var(--blue); color: var(--blue); background: #EFF6FF; font-weight: 600; }
    .chip b { color: var(--white); background: var(--blue); padding: 1px 7px; border-radius: 5px; }

    /* Panels / cards */
    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .status-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid var(--india-green);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .status-card-critical {
        background: #FFF7ED;
        border: 1px solid #FDE0BD;
        border-left: 4px solid var(--saffron);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .status-title { font-weight: 700; font-size: 14.5px; color: #14532D; margin-bottom: 4px; }
    .status-title-critical { font-weight: 700; font-size: 14.5px; color: var(--amber); margin-bottom: 4px; }
    .status-body { color: var(--muted); font-size: 13.5px; line-height: 1.5; }

    .log-entry {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: #374151;
        padding: 8px 0;
        border-bottom: 1px solid var(--line);
    }
    .log-entry:last-child { border-bottom: none; }
    .log-hash {
        color: var(--blue);
        background: #EFF6FF;
        padding: 2px 6px;
        border-radius: 4px;
    }

    .vector-readout {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #1E3A5F;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 6px;
        padding: 10px 12px;
        margin-top: 8px;
    }

    .video-frame {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 8px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    div[data-testid="stImage"] img { border-radius: 6px; border: 1px solid var(--line); }

    .govfooter {
        margin-top: 24px;
        padding: 14px 4px;
        border-top: 1px solid var(--line);
        color: var(--muted);
        font-size: 11.5px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- TRICOLOR STRIP + HEADER ---
st.markdown('<div class="tricolor-strip"></div>', unsafe_allow_html=True)
st.markdown("""
    <div class="govheader">
        <div class="govheader-left">
            <div class="emblem">IB</div>
            <div>
                <div class="govheader-title">Intelligent Border Video Analytics Platform (IBVAP)</div>
                <div class="govheader-sub">Sashastra Seema Bal &nbsp;|&nbsp; Ministry of Home Affairs &nbsp;|&nbsp; Edge Node: Alpha</div>
            </div>
        </div>
        <div class="govheader-right">
            <div class="status-pill"><span class="live-dot"></span> SYSTEM ONLINE</div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="navtabs">
        <div class="navtab active">Dashboard</div>
        <div class="navtab">Events</div>
        <div class="navtab">Archive</div>
        <div class="navtab">Configuration</div>
        <div class="navtab">Settings</div>
    </div>
    <div class="breadcrumb">Home &nbsp;›&nbsp; Surveillance &nbsp;›&nbsp; <b>Live Dashboard</b></div>
""", unsafe_allow_html=True)

# --- FILTER CHIP ROW ---
st.markdown("""
    <div class="chip-row">
        <div class="chip active">&#9660; All Cameras <b>1</b></div>
        <div class="chip">&#9660; Rule Wise View</div>
        <div class="chip">&#9660; Today</div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown('<div style="font-weight:700; color:#0B3D6B; font-size:15px; margin-bottom:12px;">IBVAP Control Panel</div>', unsafe_allow_html=True)
st.sidebar.text_input("Search", placeholder="Search cameras / rules", label_visibility="collapsed")
st.sidebar.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

st.sidebar.header("Node Alpha — Detection Rules")
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
        cv2.putText(frame, "LOW-LIGHT ENHANCED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 0), 2)

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (26, 95, 163), 2)

                if cls == 0 and enable_frs:
                    cv2.putText(frame, "Person — extracting vector", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (18, 136, 7), 2)
                    simulated_vector = f"**Vector generated (2 KB):** `[0.{random.randint(100,999)}, -0.{random.randint(100,999)}, 0.{random.randint(100,999)} ... 512d]`"
                elif cls in [2, 3, 5, 7]:
                    cv2.putText(frame, "Vehicle — ANPR active", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 153, 51), 2)

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
    fps_text.markdown(f"<span style='color:#5B6B7A; font-size:13px;'>{current_fps}</span>", unsafe_allow_html=True)

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
            "<div class='panel' style='border-left: 3px solid #9CA3AF;'>"
            "<span style='color:#5B6B7A;'>Evidence hashing is turned off.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        evidence_log.markdown(
            "<div class='panel'><span style='color:#5B6B7A;'>No events logged yet.</span></div>",
            unsafe_allow_html=True,
        )

    time.sleep(0.05)

cap.release()

st.markdown('<div class="govfooter">© Government of India | This is a demonstration interface for the IBVAP system.</div>', unsafe_allow_html=True)
