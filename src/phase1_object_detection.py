import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

CAMERA_INDEX   = 0
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
CONFIDENCE_MIN = 0.40
WINDOW_NAME    = "Phase 1 - Object Detection"
SCREEN_WIDTH   = 1266
SCREEN_HEIGHT  = 768

# ONNX model — faster on CPU than standard .pt format
# imgsz=320 means YOLO processes 320x320 internally — 2x faster than 640x640
MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "yolov8n.onnx")
)
OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
)


# ── FRAME BUFFER ──────────────────────────────────────────────────────────────
class FrameBuffer:
    def __init__(self):
        self.frame   = None
        self.running = True
        self.lock    = threading.Lock()

    def update(self, frame):
        with self.lock:
            self.frame = frame.copy()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


# ── CAMERA THREAD ─────────────────────────────────────────────────────────────
def capture_frames(cap, buffer: FrameBuffer):
    while buffer.running:
        ret, frame = cap.read()
        if ret:
            buffer.update(frame)
        else:
            time.sleep(0.01)


# ── COLOR GENERATION ──────────────────────────────────────────────────────────
def generate_colors(num_classes: int) -> list:
    colors = []
    for i in range(num_classes):
        hue       = int(179 * i / num_classes)
        hsv_array = np.array([[[hue, 220, 200]]], dtype=np.uint8)
        bgr_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR)
        b, g, r   = bgr_array[0][0]
        colors.append((int(b), int(g), int(r)))
    return colors


# ── RESIZE TO FILL SCREEN ─────────────────────────────────────────────────────
def resize_to_screen(frame):
    return cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_LINEAR)


# ── DRAW DETECTION BOX ────────────────────────────────────────────────────────
def draw_detection(frame, box, confidence: float, class_name: str, color: tuple) -> None:
    x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
    x2, y2 = int(box.xyxy[0][2]), int(box.xyxy[0][3])

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{class_name}  {confidence:.0%}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    label_y = max(y1, text_h + 10)

    cv2.rectangle(frame,
                  (x1, label_y - text_h - 8),
                  (x1 + text_w + 6, label_y),
                  color, cv2.FILLED)

    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)

    cv2.putText(frame, label, (x1 + 3, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)


# ── DRAW HUD ──────────────────────────────────────────────────────────────────
def draw_hud(frame, fps: float, total_detections: int) -> None:
    fw, fh = frame.shape[1], frame.shape[0]

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 55), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    baseline_y = 35
    gap        = 25  # px between elements so labels never collide

    # FPS (left-aligned)
    fps_text = f"FPS: {fps:.1f}"
    fps_scale, fps_thick, fps_color = 0.8, 2, (0, 255, 100)
    cv2.putText(frame, fps_text, (15, baseline_y), font, fps_scale, fps_color, fps_thick, cv2.LINE_AA)
    (fps_w, _), _ = cv2.getTextSize(fps_text, font, fps_scale, fps_thick)

    # Objects (placed right after FPS, width measured dynamically)
    obj_text = f"Objects: {total_detections}"
    obj_scale, obj_thick, obj_color = 0.8, 2, (0, 220, 255)
    obj_x = 15 + fps_w + gap
    (obj_w, _), _ = cv2.getTextSize(obj_text, font, obj_scale, obj_thick)
    cv2.putText(frame, obj_text, (obj_x, baseline_y), font, obj_scale, obj_color, obj_thick, cv2.LINE_AA)

    # Project name (right-aligned to frame edge, so it never overlaps the others)
    proj_text = "Phase 1"
    proj_scale, proj_thick, proj_color = 0.55, 1, (180, 180, 255)
    (proj_w, _), _ = cv2.getTextSize(proj_text, font, proj_scale, proj_thick)
    proj_x = max(obj_x + obj_w + gap, fw - proj_w - 15)
    cv2.putText(frame, proj_text, (proj_x, baseline_y), font, proj_scale, proj_color, proj_thick, cv2.LINE_AA)

    # Bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, fh - 34), (fw, fh), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        "Press 'S' to save screenshot  |  Press 'Q' to quit",
        (12, fh - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Phase 1: Real-Time Object Detection")
    print("  Model   : YOLOv8n ONNX (optimized for CPU)")
    print("  imgsz   : 320x320 (faster inference)")
    print("  Mode    : Multithreaded camera capture")
    print(f"  Display : {SCREEN_WIDTH}x{SCREEN_HEIGHT} (fill screen)")
    print(f"  Model   : {MODEL_PATH}")
    print(f"  Output  : {OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check ONNX model exists
    if not os.path.isfile(MODEL_PATH):
        print("\n[ERROR] ONNX model not found at:")
        print(f"        {MODEL_PATH}")
        print("\n[FIX] Run this command once to export it:")
        print("      python -c \"from ultralytics import YOLO; m = YOLO('models/yolov8n.pt'); m.export(format='onnx', imgsz=320, opset=12)\"")
        return

    print("\n[INFO] Loading YOLOv8n ONNX model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    class_names = model.names
    print(f"[INFO] Model can detect {len(class_names)} object types.")
    colors = generate_colors(len(class_names))

    print(f"\n[INFO] Opening camera (index {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try CAMERA_INDEX = 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from camera.")
        cap.release()
        return

    h, w = test_frame.shape[:2]
    print(f"[INFO] Camera opened. Resolution: {w}x{h}")

    # Start background camera thread
    buffer = FrameBuffer()
    cam_thread = threading.Thread(target=capture_frames, args=(cap, buffer), daemon=True)
    cam_thread.start()
    print("[INFO] Camera thread started.")

    print("[INFO] Waiting for first frame...")
    while buffer.read() is None:
        time.sleep(0.05)
    print("[INFO] First frame received. Starting detection.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"\n[INFO] Detection running. Press Q to quit, S to save.")
    print("-" * 60)

    fps_start   = time.time()
    fps_counter = 0
    fps_display = 0.0

    while True:
        frame = buffer.read()
        if frame is None:
            continue

        # imgsz=320 forces YOLO to process at 320x320 internally
        # This is 4x less pixels than 640x640 — significantly faster
        results    = model(frame, conf=CONFIDENCE_MIN, imgsz=320, verbose=False)
        detections = results[0].boxes
        total_count = 0

        if detections is not None:
            for box in detections:
                class_id   = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[class_id]
                color      = colors[class_id]
                draw_detection(frame, box, confidence, class_name, color)
                total_count += 1

        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        draw_hud(frame, fps_display, total_count)

        display_frame = resize_to_screen(frame)
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n[INFO] Stopping detection...")
            break
        elif key == ord('s') or key == ord('S'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.join(OUTPUT_DIR, f"phase1_screenshot_{ts}.jpg")
            cv2.imwrite(fn, display_frame)
            print(f"[INFO] Screenshot saved: {fn}")

    buffer.running = False
    cam_thread.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Phase 1 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()