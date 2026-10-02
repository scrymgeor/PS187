# 🛡️ IBVAP: Intelligent Border Video Analytics Platform
**Team: The W.A.L.L.** | **Smart India Hackathon 2026**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR-STREAMLIT-APP-URL-HERE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Architecture](https://img.shields.io/badge/Architecture-Edge--Fog-orange)]()

### *A Software-Defined, Hardware-Agnostic Edge-Fog Architecture for Legacy CCTV.*

IBVAP is a lightweight, edge-optimized video analytics platform designed for border security perimeters. It retrofits existing "dumb" legacy CCTV cameras with advanced AI, providing real-time threat detection, spatial tracking, and immutable chain-of-custody for digital evidence, all while strictly minimizing network bandwidth and compute overhead.

---

## 🌟 Core Features & USPs

*   📉 **Cascaded Tripwire Engine:** Saves massive compute power by idling standard feeds at ~5 FPS. Instantly upscales to 30 FPS inference only upon motion detection.
*   🧠 **Contextual AI Filtering:** Eliminates false positives by filtering out wildlife, shadows, and debris, triggering alerts exclusively for armed humans, vehicles, and restricted watchlist matches.
*   🎯 **Dynamic Threat Matrix:** Calculates a live 1-100 danger score mathematically derived from object type, loitering duration, weapon presence, and geo-fence zone intrusion.
*   ⛓️ **Blockchain Evidence Locker:** Cryptographically hashes (SHA-256) critical threat footage into an immutable local ledger, guaranteeing 100% tamper-proof, court-admissible evidence.
*   🗺️ **Live 2D Spatial Reconstruction:** Transforms raw bounding-box coordinate data into actionable visual intelligence via a real-time top-down 2D mapping engine.
*   🌙 **Algorithmic Night Vision:** Enhances low-light RGB feeds to achieve thermal-level human detection in near-total darkness without expensive hardware upgrades.
*   👤 **Edge-Vectorized FRS:** Converts facial data into lightweight numerical vectors for instant matching, preserving massive network bandwidth by eliminating raw video transmission to the cloud.

---

## 🏗️ The 4-Layer Pipeline Architecture

IBVAP operates on a strictly defined, highly optimized processing pipeline designed for resource-constrained edge nodes:

1.  **Layer 1: Ingestion & Profiling:** Connects to RTSP/Video streams and dynamically reads the edge node's CPU/RAM specs to cap AI processing layers automatically (e.g., scaling YOLO from 640px to 320px on low-end hardware).
2.  **Layer 2: The Tripwire:** Utilizes OpenCV Background Subtraction to look for movement. *(No motion = 5fps sleep mode | Motion Detected = Pass to Phase 3).*
3.  **Layer 3: Unified AI Classification:** YOLOv8 extracts bounding boxes and classifies objects. Passes human coordinates to the Threat Vector Tracker and applies Facial Recognition against the edge-synced watchlist.
4.  **Layer 4: Threat Assessment & Security:** Calculates the final threat score. If Critical, it triggers the Blockchain Locker (saving the clip + hash) and calculates vector heading to simulate waking up adjacent cascaded cameras.

---

## 💻 Technical Stack
*   **Core Engine & UI:** Python, Streamlit, OpenCV
*   **Object & Weapon Tracking:** YOLOv8 (Ultralytics) + ByteTrack
*   **Facial Recognition:** YuNet (Detection) + SFace (Vectorization)
*   **Hardware Profiling:** `psutil`
*   **Security:** `hashlib` (SHA-256 Ledger)

---

## 🚀 Quick Start (Run the Prototype)

**1. Clone & Setup Environment**
```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
