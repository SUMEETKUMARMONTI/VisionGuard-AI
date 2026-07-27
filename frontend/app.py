# =============================================================================
# VisionGuard AI — Flask Backend (Multithreaded: capture / inference / render)
# Author  : Sumeet Kumar
# =============================================================================

import os, cv2, time, csv, threading, numpy as np
from flask import Flask, Response, render_template, jsonify, request
from datetime import datetime
from ultralytics import YOLO

try:
    import winsound
    AUDIO = True
except ImportError:
    AUDIO = False

app = Flask(__name__)

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE       = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODEL_PATH = os.path.join(BASE, "models", "yolov8n.pt")
OUT_DIR    = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── GLOBALS ───────────────────────────────────────────────────────────────────
model        = None
COLORS       = None
latest_frame = None          # latest JPEG bytes for stream
frame_lock   = threading.Lock()
ALERT_THRESHOLD = 3

# All shared app state in one dict — protected by `lock`
state = {
    "running"      : False,
    "phase"        : 1,
    "cam"          : 0,
    "fps"          : 0.0,
    "persons"      : 0,
    "objects"      : 0,
    "alert"        : False,
    "total_alerts" : 0,
    "last_alert"   : "",
}
lock = threading.Lock()

# ── CAPTURE BUFFER ────────────────────────────────────────────────────────────
# Raw camera frames live here. Only capture_worker() writes to this.
# "id" increments on every new frame so inference_worker can tell whether
# it has already seen the current frame without sharing an Event with
# render_worker (which needs its own blocking wait at full camera speed).
cap_buf = {
    "frame": None,
    "ret"  : False,
    "id"   : 0,
}
cap_lock    = threading.Lock()
new_capture = threading.Event()   # signaled by capture_worker, consumed by render_worker

# ── DETECTION CACHE ───────────────────────────────────────────────────────────
# Latest finished YOLO result. Only inference_worker() writes to this.
# render_worker() reads it on every frame it draws — which may be a frame
# or two newer than the image the detections were actually computed from.
# That lag is what lets rendering run at full camera speed instead of
# being capped by however long YOLO takes per frame.
det_cache = {
    "boxes"   : [],     # list of (cls_id, conf, x1, y1, x2, y2)
    "persons" : 0,
    "objects" : 0,
    "alert"   : False,
    "phase"   : 1,
}
det_lock = threading.Lock()

last_beep = 0.0

# ── COLOR HELPER ──────────────────────────────────────────────────────────────
def make_colors(n):
    cols = []
    for i in range(n):
        h   = int(179 * i / n)
        bgr = cv2.cvtColor(np.array([[[h,200,200]]], dtype=np.uint8),
                           cv2.COLOR_HSV2BGR)[0][0]
        cols.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return cols

# ── DRAW HELPERS ──────────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ly = max(y1, th+8)
    cv2.rectangle(frame, (x1,ly-th-6), (x1+tw+6,ly+2), color, cv2.FILLED)
    b  = 0.3*color[2]+0.6*color[1]+0.1*color[0]
    tc = (0,0,0) if b>140 else (255,255,255)
    cv2.putText(frame, label, (x1+3,ly-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tc, 1, cv2.LINE_AA)

def draw_hud(frame, phase, fps, persons, objects, alert, total_alerts):
    fh, fw = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (fw,48), (15,15,15), cv2.FILLED)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, f"FPS:{fps:.1f}", (10,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0,220,100), 2, cv2.LINE_AA)
    labels = {1:"Phase 1: Object Detection",
              2:"Phase 2: Person Detection",
              3:"Phase 3: Crowd Alert"}
    pt = labels[phase]
    (pw,_),_ = cv2.getTextSize(pt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(frame, pt, (fw-pw-10,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,200,255), 1, cv2.LINE_AA)
    if phase == 1:
        ct, cc = f"Objects:{objects}", (0,220,255)
    elif phase == 2:
        ct, cc = f"Persons:{persons}", (0,230,0)
    else:
        ct, cc = f"Alerts:{total_alerts}", ((0,60,255) if alert else (160,160,160))
    (cw,_),_ = cv2.getTextSize(ct, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    cv2.putText(frame, ct, (fw//2-cw//2,32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, cc, 2, cv2.LINE_AA)
    if phase in (2,3):
        px,py  = 10, fh-96
        cc2    = (0,60,255) if (alert and phase==3) else (0,220,80)
        ov2    = frame.copy()
        cv2.rectangle(ov2,(px,py),(px+205,py+84),(15,15,15),cv2.FILLED)
        cv2.addWeighted(ov2,0.82,frame,0.18,0,frame)
        cv2.rectangle(frame,(px,py),(px+205,py+84),cc2,2)
        cv2.putText(frame,"PERSONS",(px+7,py+19),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,150,150),1,cv2.LINE_AA)
        cs = str(persons)
        (csw,_),_ = cv2.getTextSize(cs,cv2.FONT_HERSHEY_SIMPLEX,2.1,4)
        cv2.putText(frame,cs,(px+(205-csw)//2,py+70),
                    cv2.FONT_HERSHEY_SIMPLEX,2.1,cc2,4,cv2.LINE_AA)
        st = "ALERT ACTIVE" if (alert and phase == 3) else f"Threshold: >{ALERT_THRESHOLD}"     
        cv2.putText(frame,st,(px+7,py+83),
                    cv2.FONT_HERSHEY_SIMPLEX,0.33,cc2,1,cv2.LINE_AA)

def draw_banner(frame, persons, blink):
    fh,fw = frame.shape[:2]
    y1,y2 = fh//2-50, fh//2+50
    ov    = frame.copy()
    cv2.rectangle(ov,(0,y1),(fw,y2),(0,0,160),cv2.FILLED)
    alpha = 0.88 if blink else 0.35
    cv2.addWeighted(ov,alpha,frame,1-alpha,0,frame)
    cv2.line(frame,(0,y1),(fw,y1),(255,255,255),2)
    cv2.line(frame,(0,y2),(fw,y2),(255,255,255),2)
    t1="!! CROWD ALERT !!"
    (w1,_),_=cv2.getTextSize(t1,cv2.FONT_HERSHEY_SIMPLEX,0.95,3)
    cv2.putText(frame,t1,(fw//2-w1//2,fh//2-4),
                cv2.FONT_HERSHEY_SIMPLEX,0.95,(255,255,255),3,cv2.LINE_AA)
    t2=f"{persons} PERSONS — THRESHOLD EXCEEDED"
    (w2,_),_=cv2.getTextSize(t2,cv2.FONT_HERSHEY_SIMPLEX,0.50,2)
    cv2.putText(frame,t2,(fw//2-w2//2,fh//2+34),
                cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),2,cv2.LINE_AA)

# ── CAPTURE WORKER ────────────────────────────────────────────────────────────
def capture_worker():
    """
    Dedicated thread that ONLY talks to the camera.
    Opens/closes the camera based on state["running"]/state["cam"], and
    reads frames in a tight loop as fast as the camera allows, always
    publishing the newest one to cap_buf.
    """
    cap     = None
    cur_cam = -1

    print("[THREAD] Capture worker started")

    while True:
        with lock:
            running = state["running"]
            cam_idx = state["cam"]

        if not running:
            if cap is not None:
                cap.release()
                cap     = None
                cur_cam = -1
                with cap_lock:
                    cap_buf["frame"] = None
                    cap_buf["ret"]   = False
                print("[THREAD] Camera released — idle")
            time.sleep(0.1)
            continue

        if cap is None or not cap.isOpened() or cur_cam != cam_idx:
            if cap is not None:
                cap.release()
            print(f"[THREAD] Opening camera index {cam_idx} ...")
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if cap.isOpened():
                cur_cam = cam_idx
                print(f"[THREAD] Camera {cam_idx} opened OK")
            else:
                print(f"[THREAD] ERROR: Cannot open camera {cam_idx}")
                cap = None
                time.sleep(1.0)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[THREAD] Frame read failed — retrying")
            time.sleep(0.01)
            continue

        with cap_lock:
            cap_buf["frame"] = frame
            cap_buf["ret"]   = True
            cap_buf["id"]   += 1
        new_capture.set()

# ── INFERENCE WORKER ──────────────────────────────────────────────────────────
def inference_worker():
    """
    Runs YOLO on the newest captured frame, as fast as the model allows,
    and caches the result in det_cache. Polls cap_buf's frame id instead
    of sharing new_capture with render_worker, so this slow stage never
    holds up the fast one.
    """
    global last_beep

    last_seen_id = -1
    print("[THREAD] Inference worker started")

    while True:
        with lock:
            running = state["running"]
            phase   = state["phase"]

        if not running:
            with det_lock:
                det_cache["boxes"]   = []
                det_cache["persons"] = 0
                det_cache["objects"] = 0
                det_cache["alert"]   = False
            last_seen_id = -1
            time.sleep(0.1)
            continue

        with cap_lock:
            cur_id = cap_buf["id"]
            ret    = cap_buf["ret"]
            frame  = cap_buf["frame"]

        if not ret or frame is None or cur_id == last_seen_id:
            time.sleep(0.005)
            continue
        last_seen_id = cur_id
        frame = frame.copy()

        # ── YOLO detection ──────────────────────────────────────────────────
        results    = model(frame, conf=0.40, verbose=False)
        detections = results[0].boxes
        obj_cnt = pers_cnt = 0
        alert   = False
        boxes_out = []

        if detections is not None:
            for box in detections:
                cid  = int(box.cls[0])
                conf = float(box.conf[0])
                x1   = int(box.xyxy[0][0]); y1 = int(box.xyxy[0][1])
                x2   = int(box.xyxy[0][2]); y2 = int(box.xyxy[0][3])

                if phase == 1:
                    obj_cnt += 1
                    boxes_out.append((cid, conf, x1, y1, x2, y2))
                elif phase in (2,3):
                    if cid != 0: continue
                    pers_cnt += 1
                    boxes_out.append((cid, conf, x1, y1, x2, y2))

        # ── Phase 3 alert ───────────────────────────────────────────────────
        if phase == 3 and pers_cnt > ALERT_THRESHOLD:
            alert = True
            now   = time.time()
            if now - last_beep >= 3.0:
                last_beep = now
                if AUDIO:
                    threading.Thread(
                        target=lambda: winsound.Beep(1000,500), daemon=True
                    ).start()
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn   = f"ALERT_{pers_cnt}p_{ts}.jpg"
                cv2.imwrite(os.path.join(OUT_DIR, fn), frame)
                try:
                    lf = os.path.join(OUT_DIR,"crowd_log.csv")
                    ex = os.path.isfile(lf)
                    with open(lf,'a',newline='',encoding='utf-8') as f_:
                        w = csv.writer(f_)
                        if not ex:
                            w.writerow(["timestamp","persons","status","file"])
                        w.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            pers_cnt, "ALERT", fn
                        ])
                except Exception as e:
                    print(f"[THREAD] CSV error: {e}")
                with lock:
                    state["total_alerts"] += 1
                    state["last_alert"]    = datetime.now().strftime("%H:%M:%S")
                print(f"[THREAD] ALERT fired — {pers_cnt} persons")

        with det_lock:
            det_cache["boxes"]   = boxes_out
            det_cache["persons"] = pers_cnt
            det_cache["objects"] = obj_cnt
            det_cache["alert"]   = alert
            det_cache["phase"]   = phase

# ── RENDER WORKER (drawing + encoding) ────────────────────────────────────────
def render_worker():
    """
    Runs on every new captured frame and draws whatever the most recent
    cached detections are (from inference_worker). This is what the
    browser's FPS number reflects — capture/render speed, decoupled from
    how long YOLO takes. Boxes may trail live motion by a frame or two;
    that lag is the trade-off for keeping this loop fast.
    """
    global latest_frame

    fps_t    = time.time()
    fps_n    = 0
    blink_n  = 0
    blink_on = True
    cur_fps  = 0.0

    print("[THREAD] Render worker started")

    while True:
        with lock:
            running = state["running"]

        if not running:
            with frame_lock:
                latest_frame = None
            new_capture.clear()
            fps_n = 0
            fps_t = time.time()
            time.sleep(0.1)
            continue

        if not new_capture.wait(timeout=0.5):
            continue
        new_capture.clear()

        with cap_lock:
            ret   = cap_buf["ret"]
            frame = cap_buf["frame"]

        if not ret or frame is None:
            continue
        frame = frame.copy()

        with det_lock:
            boxes_out = det_cache["boxes"]
            pers_cnt  = det_cache["persons"]
            obj_cnt   = det_cache["objects"]
            alert     = det_cache["alert"]
            det_phase = det_cache["phase"]

        # ── Draw cached boxes ────────────────────────────────────────────────
        pers_n = 0
        for cid, conf, x1, y1, x2, y2 in boxes_out:
            if det_phase == 1:
                name = model.names[cid]
                draw_box(frame, x1, y1, x2, y2, f"{name} {conf:.0%}", COLORS[cid])
            elif det_phase in (2, 3):
                pers_n += 1
                col = (0,60,255) if (det_phase == 3 and pers_cnt > ALERT_THRESHOLD) else (0,230,0)
                draw_box(frame, x1, y1, x2, y2, f"#{pers_n} Person {conf:.0%}", col)

        # ── Blink effect ────────────────────────────────────────────────────
        blink_n += 1
        if blink_n >= 15: blink_n=0; blink_on=not blink_on
        if alert: draw_banner(frame, pers_cnt, blink_on)

        # ── HUD overlay ─────────────────────────────────────────────────────
        with lock:
            ta = state["total_alerts"]
        draw_hud(frame, det_phase, cur_fps, pers_cnt, obj_cnt, alert, ta)

        # ── Update state ────────────────────────────────────────────────────
        with lock:
            state["persons"] = pers_cnt
            state["objects"] = obj_cnt
            state["alert"]   = alert

        # ── FPS (render/stream throughput) ───────────────────────────────────
        fps_n += 1
        elapsed = time.time() - fps_t
        if elapsed >= 1.0:
            cur_fps = round(fps_n / elapsed, 1)
            with lock: state["fps"] = cur_fps
            fps_n = 0
            fps_t = time.time()

        # ── Encode JPEG and store ────────────────────────────────────────────
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with frame_lock:
                latest_frame = buf.tobytes()

# ── MJPEG STREAM GENERATOR ────────────────────────────────────────────────────
def gen_stream():
    last_sent = None
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.04)
            continue
        if frame is not last_sent:
            last_sent = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.005)

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    d = request.get_json() or {}
    with lock:
        state["cam"]          = int(d.get('camera', 0))
        state["phase"]        = int(d.get('phase',  1))
        state["running"]      = True
        state["total_alerts"] = 0
        state["last_alert"]   = ""
        state["persons"]      = 0
        state["objects"]      = 0
        state["alert"]        = False
        state["fps"]          = 0.0
    print(f"[SERVER] Start — phase={state['phase']} cam={state['cam']}")
    return jsonify({"ok": True})

@app.route('/stop', methods=['POST'])
def stop():
    with lock:
        state["running"] = False
    print("[SERVER] Stopped")
    return jsonify({"ok": True})

@app.route('/phase', methods=['POST'])
def set_phase():
    d = request.get_json() or {}
    with lock:
        state["phase"] = int(d.get('phase', 1))
    return jsonify({"ok": True})

@app.route('/camera', methods=['POST'])
def set_camera():
    d = request.get_json() or {}
    with lock:
        state["cam"] = int(d.get('camera', 0))
    return jsonify({"ok": True})

@app.route('/status')
def status():
    with lock:
        return jsonify(dict(state))

@app.route('/screenshot', methods=['POST'])
def screenshot():
    with frame_lock:
        f = latest_frame
    if not f:
        return jsonify({"ok": False, "msg": "No frame available"})
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"screenshot_{ts}.jpg"
    with open(os.path.join(OUT_DIR, fn), 'wb') as file:
        file.write(f)
    return jsonify({"ok": True, "file": fn})

# ── STARTUP ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[INFO] Model path : {MODEL_PATH}")
    print(f"[INFO] Output dir : {OUT_DIR}")
    print("[INFO] Loading YOLOv8n model ...")
    model  = YOLO(MODEL_PATH)
    COLORS = make_colors(len(model.names))
    print("[INFO] Model loaded successfully.")

    # Three long-running background threads:
    #   capture_worker    - only touches the camera
    #   inference_worker  - only touches YOLO, at whatever pace it can manage
    #   render_worker     - draws + encodes on every new captured frame,
    #                       using the latest cached detections, so it runs
    #                       at full camera speed instead of YOLO's speed
    threading.Thread(target=capture_worker,   daemon=True, name="cam-capture").start()
    threading.Thread(target=inference_worker, daemon=True, name="cam-inference").start()
    threading.Thread(target=render_worker,    daemon=True, name="cam-render").start()

    print("\n" + "="*52)
    print("  Browser  : http://localhost:5000")
    print("  Network  : http://YOUR_IP_ADDRESS:5000")
    print("  Stop     : Ctrl + C")
    print("="*52 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
