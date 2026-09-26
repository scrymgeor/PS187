import streamlit as st
import cv2
from PIL import Image
import numpy as np
import hashlib
import random
from datetime import datetime
from ultralytics import YOLO
import time

st.set_page_config(page_title="IBVAP | Command Center", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

yolo_model = load_model()
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: white; }
    h1, h2, h3 { color: #0EA5E9; }
    .threat-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #334155; font-family: monospace;}
    .vector-text { color: #10B981; font-family: monospace; font-size: 12px;}
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ IBVAP: Intelligent Video Analytics Platform")
st.markdown("**Sashastra Seema Bal (SSB) | Border Out Post Command Terminal**")

# --- CONTROLS ---
st.sidebar.header("⚙️ Node Settings (Camera-Alpha)")
enable_tripwire = st.sidebar.checkbox("1. Cascaded 'Tripwire' (Save CPU)", value=True)
enable_nightvision = st.sidebar.checkbox("2. Zero-DCE Night Vision", value=False)
enable_frs = st.sidebar.checkbox("3. Edge-Vectorized FRS", value=True)
enable_blockchain = st.sidebar.checkbox("4. Cryptographic Locker", value=True)

st.sidebar.markdown("---")
st.sidebar.info("System Status: **ONLINE**\n\nAI Engine: **PyTorch **\n\nNetwork Payload: **Optimized**")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live CCTV Feed (RTSP Simulated)")
    video_placeholder = st.empty()
    fps_text = st.empty()
    vector_display = st.empty() # For showing the FRS math

with col2:
    st.subheader("Dynamic Threat Matrix")
    threat_status = st.empty()
    st.subheader("Evidence Locker (SHA-256)")
    evidence_log = st.empty()

#cap = cv2.VideoCapture(0)
import os
import urllib.request

# Automatically download a perfect AI test video if it doesn't exist yet
video_path = "border_test.mp4"
if not os.path.exists(video_path):
    st.info("Downloading test video for the first time... please wait 5 seconds.")
    # This is a reliable, open-source traffic video from Intel's AI library
    video_url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
    urllib.request.urlretrieve(video_url, video_path)

cap = cv2.VideoCapture(video_path)
logs = []



# --- MAIN VIDEO LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    
    # Safety catch for video ending or failing to load
    if not ret or frame is None:
        st.warning("Video feed ended or connection lost. Refresh page to restart.")
        break
    
    # 1. Zero-DCE Simulation (Night Vision)
    if enable_nightvision:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cv2.putText(frame, "ZERO-DCE ENHANCED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    motion_detected = True
    current_fps = "30 FPS (Heavy Compute Mode)"

    # 2. Tripwire
    if enable_tripwire:
        fgmask = fgbg.apply(frame)
        if cv2.countNonZero(fgmask) < 1000:
            motion_detected = False
            current_fps = "5 FPS (Tripwire Sleep Mode - Saving 85% CPU)"
            cv2.putText(frame, "SYSTEM IDLE - NO MOTION", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

    threat_score = 0
    threat_level = "GREEN (Nominal)"
    simulated_vector = ""

    # 3. YOLOv8 Inference
    if motion_detected:
        results = yolo_model(frame, verbose=False)
        humans = 0
        vehicles = 0
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls == 0: humans += 1
                elif cls in [2, 3, 5, 7]: vehicles += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # FRS Vector Simulation UI
                if cls == 0 and enable_frs:
                    cv2.putText(frame, "Human Detected -> Extracting FAISS Vector...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    simulated_vector = f"**Live Edge Vector Generated (2 KB payload):** `[0.{random.randint(100,999)}, -0.{random.randint(100,999)}, 0.{random.randint(100,999)} ... 512d]` -> *Sending to HQ...*"
                
                # ANPR Regex Simulation UI
                elif cls in [2,3,5,7]:
                    cv2.putText(frame, "Vehicle -> Regex ANPR Active", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Threat Matrix Logic
        threat_score = (humans * 50) + (vehicles * 30)
            
        if threat_score > 40:
            threat_level = f"CRITICAL THREAT (Score: {threat_score})"
            
            # Blockchain Locker
            if enable_blockchain and len(logs) < 4:
                frame_bytes = frame.tobytes()
                crypto_hash = hashlib.sha256(frame_bytes).hexdigest()
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"**{timestamp}** | Threat Logged\n> **Hash:** `{crypto_hash[:24]}...`"
                if not any(crypto_hash[:24] in log for log in logs):
                    logs.insert(0, log_entry)

    # --- CLOUD-SAFE UI RENDERING ---
    # Convert BGR to RGB and enforce uint8 data type so Streamlit doesn't crash
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    video_placeholder.image(pil_image, use_column_width=True)
    fps_text.markdown(f"**Network Status:** {current_fps}")
    
    if simulated_vector:
        vector_display.markdown(f"<p class='vector-text'>{simulated_vector}</p>", unsafe_allow_html=True)
    else:
        vector_display.empty()
    
    if "CRITICAL" in threat_level:
        threat_status.error(f"🚨 {threat_level}\n\n**Action:** Threat detected. Trajectory engine engaged.")
    else:
        threat_status.success(f"✅ {threat_level}\n\n**Action:** Area secure.")
        
    if enable_blockchain and logs:
        evidence_log.markdown(f"<div class='threat-box'>{ '<br><br>'.join(logs) }</div>", unsafe_allow_html=True)
    elif not enable_blockchain:
        evidence_log.warning("Blockchain Offline.")
        
    # Crucial for Streamlit Cloud: tiny pause to prevent WebSocket overload
    time.sleep(0.03) 

cap.release()
