"""IBVAP v2 - Intelligent Border Video Analytics Platform (demo)
Run:  pip install streamlit ultralytics opencv-python psutil
      streamlit run ibvap_v2.py
Optional: drop face photos into ./watchlist/ (file name = person name, e.g. rahul.jpg)
"""
import streamlit as st, cv2, numpy as np, hashlib, json, os, time, math, urllib.request, psutil
from collections import deque, defaultdict
from datetime import datetime
from ultralytics import YOLO

st.set_page_config(page_title="IBVAP v2", layout="wide")
os.makedirs("watchlist", exist_ok=True); os.makedirs("evidence", exist_ok=True)
LEDGER = "evidence/ledger.json"
ZONES = {"Lower half": [(0, .5), (1, .5), (1, 1), (0, 1)],
         "Centre corridor": [(.3, 0), (.7, 0), (.7, 1), (.3, 1)],
         "Right half": [(.5, 0), (1, 0), (1, 1), (.5, 1)]}
YU = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SF = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# ---------------- UI ----------------
st.markdown("""<style>
#MainMenu,footer,header{visibility:hidden}
.stApp{background:#F4F6F8}.block-container{padding-top:1rem;max-width:100%}
.tri{height:6px;border-radius:4px;background:linear-gradient(to right,#FF9933 33%,#fff 33% 66%,#128807 66%);border:1px solid #DCE3E8;margin-bottom:12px}
.hdr{display:flex;justify-content:space-between;align-items:center;background:#fff;border:1px solid #DCE3E8;border-radius:10px;padding:14px 20px;margin-bottom:12px}
.hdr b{font-size:19px;color:#0B3D6B}.hdr span{color:#5B6B7A;font-size:12.5px}
.pill{background:#ECFDF3;border:1px solid #A7E3B8;color:#15803D;font-size:12px;font-weight:700;padding:5px 12px;border-radius:99px}
.card{background:#fff;border:1px solid #DCE3E8;border-radius:8px;padding:12px 14px;margin-bottom:10px;font-size:13px;color:#374151}
.NOMINAL{border-left:4px solid #128807}.ELEVATED{border-left:4px solid #FF9933}
.HIGH{border-left:4px solid #EA580C;background:#FFF7ED}.CRITICAL{border-left:4px solid #DC2626;background:#FEF2F2}
.mono{font-family:monospace;font-size:12px}.ok{color:#15803D;font-weight:700}.bad{color:#DC2626;font-weight:700}
h3{color:#0B3D6B!important;font-size:14px!important;text-transform:uppercase;letter-spacing:.05em}
</style>""", unsafe_allow_html=True)

# ---------------- hardware profile ----------------
cores = psutil.cpu_count(logical=False) or 2
ram = psutil.virtual_memory().total / 1e9
IMGSZ = 640 if cores >= 8 and ram >= 16 else 480 if cores >= 4 else 320
cv2.setNumThreads(cores)

st.markdown('<div class="tri"></div>', unsafe_allow_html=True)
st.markdown(f'<div class="hdr"><div><b>Intelligent Border Video Analytics Platform v2</b><br>'
            f'<span>SSB | MHA | Edge Node Alpha | HW profile: {cores} cores, {ram:.0f} GB RAM → inference {IMGSZ}px</span></div>'
            f'<div class="pill">● SYSTEM ONLINE</div></div>', unsafe_allow_html=True)

sb = st.sidebar
sb.header("Source")
src = sb.text_input("Video file / RTSP URL", "border_test.mp4")
gps = sb.text_input("Node GPS (lat, lon)", "31.6340, 74.8723")
zone_name = sb.selectbox("Restricted zone (geo-fence)", list(ZONES))
sb.header("Pipeline")
auto_dn = sb.checkbox("Auto day/night switch", True)
use_trip = sb.checkbox("Motion tripwire (idle saver)", True)
use_frs = sb.checkbox("Face recognition (watchlist)", True)
use_ev = sb.checkbox("Evidence locker (hash-chain)", True)
up = sb.file_uploader("Add watchlist face", type=["jpg", "jpeg", "png"])
if up:
    open(f"watchlist/{os.path.splitext(up.name)[0]}.jpg", "wb").write(up.getbuffer())
    st.cache_resource.clear()
reverify = sb.button("Re-verify ledger")

# ---------------- models ----------------
def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

@st.cache_resource
def load_models():
    yolo = YOLO("yolov8n.pt")
    det = rec = None; wl = {}
    try:
        fetch(YU, "yunet.onnx"); fetch(SF, "sface.onnx")
        det = cv2.FaceDetectorYN.create("yunet.onnx", "", (320, 320), 0.8)
        rec = cv2.FaceRecognizerSF.create("sface.onnx", "")
        for f in os.listdir("watchlist"):
            img = cv2.imread(f"watchlist/{f}")
            e = embed(det, rec, img) if img is not None else None
            if e is not None: wl[os.path.splitext(f)[0]] = e
    except Exception as ex:
        print("Face models unavailable:", ex)
    return yolo, det, rec, wl

def embed(det, rec, img):
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None: return None
    return rec.feature(rec.alignCrop(img, faces[0]))

yolo, fdet, frec, watch = load_models()
if use_frs and fdet is None:
    sb.warning("Face models failed to download (needs internet). Face recognition disabled.")
    use_frs = False
sb.caption(f"Watchlist: {len(watch)} face(s) enrolled")

# ---------------- evidence ledger ----------------
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def block_hash(b):
    return hashlib.sha256(f"{b['idx']}|{b['ts']}|{b['gps']}|{b['clip_sha']}|{b['prev']}".encode()).hexdigest()

def load_chain():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else []

def verify(chain):
    out, prev = [], "GENESIS"
    for b in chain:
        ok = (b["prev"] == prev and block_hash(b) == b["hash"]
              and os.path.exists(b["clip"]) and sha_file(b["clip"]) == b["clip_sha"])
        out.append(ok); prev = b["hash"]
    return out

def commit_clip(ring, reason):
    chain = load_chain(); idx = len(chain)
    path = f"evidence/clip_{idx:04d}.mp4"
    hh, ww = ring[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (ww, hh))
    for fr in ring: vw.write(fr)
    vw.release()
    b = {"idx": idx, "ts": datetime.now().isoformat(timespec="seconds"), "gps": gps, "reason": reason,
         "clip": path, "clip_sha": sha_file(path), "prev": chain[-1]["hash"] if chain else "GENESIS"}
    b["hash"] = block_hash(b)
    chain.append(b); json.dump(chain, open(LEDGER, "w"), indent=1)
    return chain

# ---------------- helpers ----------------
def is_night(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return g.mean() < 60

def enhance(f):
    l, a, b = cv2.split(cv2.cvtColor(f, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(3.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def identify(frame, box):
    x1, y1, x2, y2 = box
    crop = frame[max(0, y1):y1 + int((y2 - y1) * .45), max(0, x1):x2]
    if crop.size == 0 or crop.shape[0] < 30: return None
    if crop.shape[0] < 160: crop = cv2.resize(crop, None, fx=2, fy=2)
    e = embed(fdet, frec, crop)
    if e is None: return None
    best, bs = "UNKNOWN", 0.0
    for name, w in watch.items():
        s = frec.match(e, w, cv2.FaceRecognizer_FR_COSINE) if hasattr(cv2, "FaceRecognizer_FR_COSINE") \
            else frec.match(e, w, 0)
        if s > .363 and s > bs: best, bs = name, s
    return best, bs

LEVELS = [(100, "CRITICAL"), (60, "HIGH"), (30, "ELEVATED"), (0, "NOMINAL")]
def level(s): return next(n for t, n in LEVELS if s >= t)

# ---------------- layout ----------------
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Live Threat Feed"); vid = st.empty(); modeline = st.empty()
with c2:
    st.subheader("Threat Status"); status_ph = st.empty()
    st.subheader("Predictive Cascade"); casc_ph = st.empty()
    st.subheader("2D Map"); map_ph = st.empty()
    st.subheader("Immutable Evidence Log"); led_ph = st.empty()

if src == "border_test.mp4" and not os.path.exists(src):
    with st.spinner("Downloading sample video..."):
        fetch("https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4", src)

chain = load_chain(); vres = verify(chain)

def draw_ledger(chain, vres):
    if not use_ev: return "<div class='card'>Evidence locker off.</div>"
    if not chain: return "<div class='card'>No critical events committed yet.</div>"
    rows = ""
    for b, ok in list(zip(chain, vres))[-4:][::-1]:
        badge = "<span class='ok'>✔ VERIFIED</span>" if ok else "<span class='bad'>✖ TAMPERED</span>"
        rows += (f"<div class='mono' style='padding:6px 0;border-bottom:1px solid #eee'>#{b['idx']} {b['ts'][11:]} {badge}<br>"
                 f"sha256 {b['clip_sha'][:20]}…<br>prev {b['prev'][:12]}… {b['reason']}</div>")
    return f"<div class='card'>{rows}</div>"

# ---------------- main loop ----------------
is_stream = src.startswith("rtsp")
cap = cv2.VideoCapture(src)
fgbg = cv2.createBackgroundSubtractorMOG2(500, 50, False)
ring = deque(maxlen=150); tracks = defaultdict(lambda: deque(maxlen=30)); faces = {}
hold = fn = 0; last_commit = 0; map_trail = deque(maxlen=60)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        if is_stream: break
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
    fn += 1
    if frame.shape[1] > 960: frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
    h, w = frame.shape[:2]
    night = auto_dn and is_night(frame)
    if night: frame = enhance(frame)

    motion = True
    if use_trip:
        mask = fgbg.apply(cv2.resize(frame, (640, 360)))
        hold = 30 if cv2.countNonZero(mask) > 600 else max(0, hold - 1)
        motion = hold > 0
        if not motion:
            for _ in range(5): cap.grab()  # ~5 FPS idle

    poly = (np.array(ZONES[zone_name]) * [w, h]).astype(np.int32)
    ov = frame.copy(); cv2.fillPoly(ov, [poly], (0, 140, 255))
    frame = cv2.addWeighted(ov, .12, frame, .88, 0)
    cv2.polylines(frame, [poly], True, (0, 140, 255), 2)

    score, top, people, vehicles, heading = 0, None, [], [], 0
    if motion:
        r = yolo.track(frame, persist=True, imgsz=IMGSZ, tracker="bytetrack.yaml",
                       classes=[0, 2, 3, 5, 7, 34, 43], conf=.35, verbose=False)[0]
        weapons, B = [], r.boxes
        ids = B.id.int().tolist() if B is not None and B.id is not None else [-1] * (len(B) if B is not None else 0)
        for box, tid in zip(B if B is not None else [], ids):
            cls = int(box.cls[0]); x1, y1, x2, y2 = map(int, box.xyxy[0]); cx, foot = (x1 + x2) // 2, y2
            if cls in (34, 43): weapons.append((cx, (y1 + y2) // 2)); continue
            inz = cv2.pointPolygonTest(poly, (float(cx), float(foot)), False) >= 0
            tr = tracks[tid]; tr.append((cx, foot))
            spd = math.hypot(tr[-1][0] - tr[0][0], tr[-1][1] - tr[0][1]) / max(1, len(tr) - 1) / w
            d = dict(id=tid, box=(x1, y1, x2, y2), zone=inz, fast=spd > .006, dx=tr[-1][0] - tr[0][0], name=None, armed=False)
            (people if cls == 0 else vehicles).append(d)
        for p in people:
            x1, y1, x2, y2 = p["box"]
            p["armed"] = any(x1 <= wx <= x2 and y1 <= wy <= y2 for wx, wy in weapons)
            if use_frs and p["id"] != -1:
                c = faces.get(p["id"])
                if c is None or (c[0] == "UNKNOWN" and fn - c[2] > 20):
                    res = identify(frame, p["box"])
                    if res: faces[p["id"]] = (res[0], res[1], fn)
                    elif c is None: faces[p["id"]] = ("UNKNOWN", 0, fn)
                p["name"] = faces.get(p["id"], (None,))[0]
        for p in people:  # Threat Scoring Matrix
            s = 20 + 40 * p["zone"] + 15 * p["fast"]
            if p["armed"] or (p["name"] and p["name"] != "UNKNOWN"): s = 100
            p["score"] = min(100, s)
        for v in vehicles:
            v["score"] = 25 + 30 * v["zone"]
        allt = people + vehicles
        if allt:
            top = max(allt, key=lambda t: t["score"]); score = top["score"]; heading = top["dx"]
            map_trail.append((((top["box"][0] + top["box"][2]) / 2) / w, top["box"][3] / h))
        for t in allt:
            col = {"CRITICAL": (0, 0, 220), "HIGH": (30, 90, 234), "ELEVATED": (51, 153, 255)}.get(level(t["score"]), (163, 95, 26))
            x1, y1, x2, y2 = t["box"]; cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            tag = ("PERSON" if t in people else "VEHICLE") + f" #{t['id']} {t['score']}"
            if t.get("armed"): tag += " ARMED"
            if t.get("name"): tag += f" [{t['name']}]"
            cv2.putText(frame, tag, (x1, max(15, y1 - 6)), 0, .5, col, 2)
    else:
        cv2.putText(frame, "IDLE - NO MOTION", (10, 28), 0, .7, (150, 150, 150), 2)
    if night: cv2.putText(frame, "NIGHT MODE (IR/CLAHE)", (10, h - 12), 0, .6, (0, 180, 0), 2)

    ring.append(cv2.resize(frame, (480, int(480 * h / w))))
    lvl = level(score)
    if use_ev and lvl == "CRITICAL" and time.time() - last_commit > 15 and len(ring) > 30:
        why = ("watchlist match " + top["name"]) if top.get("name") not in (None, "UNKNOWN") else "armed person"
        chain = commit_clip(list(ring), why); vres = verify(chain); last_commit = time.time()
    elif reverify:
        vres = verify(chain)

    vid.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    if fn % 3 == 0:
        modeline.caption(("ACTIVE: full-rate inference" if motion else "IDLE: ~5 FPS sampling") +
                         f" | {'NIGHT' if night else 'DAY'} model | people {len(people)} vehicles {len(vehicles)}")
        status_ph.markdown(f"<div class='card {lvl}'><b>{lvl}</b> (score {score})<br>"
                           f"{'Person in restricted zone.' if top and top['zone'] else 'No intrusion.'}</div>", unsafe_allow_html=True)
        if top and abs(heading) > 8 and lvl != "NOMINAL":
            east = heading > 0
            casc_ph.markdown(f"<div class='card'><b class='ok'>Cam B ({'East' if east else 'West'})</b>: PRE-WAKE, max res<br>"
                             f"Cam C ({'West' if east else 'East'}): asleep (bandwidth saved)</div>", unsafe_allow_html=True)
        else:
            casc_ph.markdown("<div class='card'>All neighbours in sleep mode.</div>", unsafe_allow_html=True)
        m = np.full((260, 260, 3), 250, np.uint8)
        cv2.fillPoly(m, [(np.array(ZONES[zone_name]) * 260).astype(np.int32)], (225, 235, 255))
        pts = [(int(x * 260), int(y * 260)) for x, y in map_trail]
        for a, b in zip(pts, pts[1:]): cv2.line(m, a, b, (120, 120, 120), 1)
        if pts: cv2.circle(m, pts[-1], 7, (0, 0, 220), -1)
        map_ph.image(m[:, :, ::-1], width=260)
        led_ph.markdown(draw_ledger(chain, vres), unsafe_allow_html=True)
    time.sleep(.01)

cap.release()
st.warning("Feed ended.")
