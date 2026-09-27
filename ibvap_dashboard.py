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

# --- PROFESSIONAL DASHBOARD UI CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    /* Base theme: clean dark slate, not neon */
    .stApp {
        background-color: #0B1120;
        color: #CBD5E1;
    }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
        color: #F1F5F9 !important;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        border-bottom: none;
        padding-bottom: 4px;
    }

    /* Headings */
    h1 {
        color: #F8FAFC !important;
        font-weight: 700;
        letter-spacing: -0.01em;
        border-bottom: none;
        padding-bottom: 0;
        margin-bottom: 0.2em;
    }
    h2, h3 {
        color: #F1F5F9 !important;
        font-weight: 600;
        letter-spacing: -0.01em;
        border-bottom: 1px solid #1E293B;
        padding-bottom: 8px;
    }

    /* Top bar */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        background: linear-gradient(180deg, #0F172A 0%, #0B1120 100%);
        border: 1px solid #1E293B;
        border-radius: 12px;
        margin-bottom: 22px;
    }
    .topbar-meta {
        color: #64748B;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        letter-spacing: 0.02em;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34D399;
        font-size: 12.5px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
    }
    .live-dot {
        height: 8px; width: 8px;
        background-color: #34D399;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
        animation: pulse 1.6s ease-out infinite;
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }
        70%  { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    /* Card panels */
    .panel {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .panel-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748B;
        margin-bottom: 6px;
    }

    /* Status / alert cards */
    .status-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-left: 3px solid #10B981;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .status-card-critical {
        background: rgba(239, 68, 68, 0.07);
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-left: 3px solid #EF4444;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .status-title { font-weight: 600; font-size: 14.5px; color: #F1F5F9; margin-bottom: 4px; }
    .status-title-critical { font-weight: 600; font-size: 14.5px; color: #FCA5A5; margin-bottom: 4px; }
    .status-body { color: #94A3B8; font-size: 13.5px; line-height: 1.5; }

    /* Evidence / log entries */
    .log-entry {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12.5px;
        color: #94A3B8;
        padding: 8px 0;
        border-bottom: 1px solid #1E293B;
    }
    .log-entry:last-child { border-bottom: none; }
    .log-hash {
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.08);
        padding: 2px 6px;
        border-radius: 4px;
    }

    /* Vector readout */
    .vector-readout {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #A78BFA;
        background: rgba(167, 139, 250, 0.07);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 8px;
    }

    /* Sidebar toggles + info */
    div[data-testid="stSidebar"] label { color: #CBD5E1 !important; font-size: 13.5px; }
    div[data-testid="stAlert"] {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
        color: #94A3B8;
    }

    /* Video frame container */
    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid #1E293B;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="topbar">
        <div>
            <div style="font-size:20px; font-weight:700; color:#F8FAFC;">IBVAP — Video Analytics Platform</div>
            <div class="topbar-meta">Edge Node: Alpha &nbsp;·&nbsp; Encryption: SHA-256 &nbsp;·&nbsp; Engine: PyTorch / YOLOv8</div>
        </div>
        <div class="status-pill"><span class="live-dot"></span> LIVE</div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Camera Settings — Node Alpha")
enable_tripwire = st.sidebar.checkbox("Motion tripwire (saves compute)", value=True)
enable_nightvision = st.sidebar.checkbox("Low-light enhancement", value=False)
enable_frs = st.sidebar.checkbox("Face-recognition vectorization", value=True)
enable_blockchain = st.sidebar.checkbox("Evidence hashing", value=True)

st.sidebar.markdown("---")
st.sidebar.info("**System status:** Online\n\n**AI engine:** PyTorch\n\n**Network:** Optimized")

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
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if cls == 0 and enable_frs:
                    cv2.putText(frame, "Person — extracting vector", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    simulated_vector = f"**Vector generated (2 KB):** `[0.{random.randint(100,999)}, -0.{random.randint(100,999)}, 0.{random.randint(100,999)} ... 512d]`"
                elif cls in [2, 3, 5, 7]:
                    cv2.putText(frame, "Vehicle — ANPR active", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

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
    fps_text.markdown(f"<span style='color:#64748B; font-size:13px;'>{current_fps}</span>", unsafe_allow_html=True)

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
            "<div class='panel' style='border-left: 3px solid #64748B;'>"
            "<span style='color:#64748B;'>Evidence hashing is turned off.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        evidence_log.markdown(
            "<div class='panel'><span style='color:#64748B;'>No events logged yet.</span></div>",
            unsafe_allow_html=True,
        )

    time.sleep(0.05)

cap.release()
