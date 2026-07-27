import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
CONFIDENCE_MIN  = 0.40
WINDOW_NAME     = "Phase 2 - Person Detection "

MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "yolov8n.pt")
)
OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
)

PERSON_CLASS_ID = 0
PERSON_COLOR    = (0, 230, 0)
NUMBER_COLOR    = (255, 255, 255)


# ----------------------------------------------------------------------
# Threaded camera reader.
# cap.read() blocks while waiting on the next frame from the driver.
# Running it on its own thread means the main loop is never stuck
# waiting on the camera - it just grabs whatever the latest frame is.
# ----------------------------------------------------------------------
class VideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # not every backend honors this; harmless if ignored
        except Exception:
            pass

        self.lock = threading.Lock()
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# ----------------------------------------------------------------------
# Threaded YOLO inference worker.
# Runs detection on its own thread on whatever frame was last submitted.
# The display loop never waits on the model - it just reads the most
# recent finished result, so video stays smooth even if inference lags.
# ----------------------------------------------------------------------
class DetectionWorker:
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf

        self.lock = threading.Lock()
        self.input_frame = None
        self.output_boxes = None

        self._new_frame = threading.Event()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame):
        with self.lock:
            self.input_frame = frame.copy()
        self._new_frame.set()

    def _run(self):
        while not self.stopped:
            if not self._new_frame.wait(timeout=0.5):
                continue
            self._new_frame.clear()

            with self.lock:
                frame = self.input_frame

            if frame is None:
                continue

            results = self.model(frame, conf=self.conf, verbose=False)
            boxes = results[0].boxes

            with self.lock:
                self.output_boxes = boxes

    def get_latest(self):
        with self.lock:
            return self.output_boxes

    def stop(self):
        self.stopped = True
        self._new_frame.set()
        self.thread.join(timeout=1.0)


def draw_person_box(frame, box, person_number: int, confidence: float) -> None:
    x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
    x2, y2 = int(box.xyxy[0][2]), int(box.xyxy[0][3])

    cv2.rectangle(frame, (x1, y1), (x2, y2), PERSON_COLOR, 3)

    label = f"#{person_number}  Person  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    label_y = max(y1 - 2, th + 10)

    cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 2), (20, 20, 20), cv2.FILLED)
    cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 2), PERSON_COLOR, 1)
    cv2.putText(frame, label, (x1 + 4, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, NUMBER_COLOR, 2, cv2.LINE_AA)


def draw_count_panel(frame, person_count: int, display_fps: float) -> None:
    fw, fh = frame.shape[1], frame.shape[0]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 50), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, f"FPS: {display_fps:.1f}", (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 100), 2, cv2.LINE_AA)

    phase_text = "Phase 2: Person Detection"
    (pw, _), _ = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, phase_text, (fw - pw - 12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 255), 1, cv2.LINE_AA)

    if person_count == 0:   count_color = (160, 160, 160)
    elif person_count <= 2: count_color = (0, 220, 80)
    elif person_count <= 4: count_color = (0, 200, 255)
    else:                   count_color = (0, 60, 255)

    px, py = 10, fh - 100
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (px, py), (px + 210, py + 85), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (px, py), (px + 210, py + 85), count_color, 2)

    cv2.putText(frame, "PERSONS DETECTED", (px + 8, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    count_str = str(person_count)
    (cw, _), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 4)
    cv2.putText(frame, count_str, (px + (210 - cw) // 2, py + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, count_color, 4, cv2.LINE_AA)

    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (0, fh - 30), (fw, fh), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(overlay3, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Press 'S' to save screenshot  |  Press 'Q' to quit",
                (12, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    print("=" * 60)
    print("  Phase 2: Person-Only Detection (Multithreaded)")
    print(f"  Model path : {MODEL_PATH}")
    print(f"  Output dir : {OUTPUT_DIR}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[INFO] Loading YOLOv8n model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    print(f"\n[INFO] Opening camera (index {CAMERA_INDEX})...")
    stream = VideoStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

    if not stream.isOpened():
        print("[ERROR] Cannot open camera. Try CAMERA_INDEX = 1.")
        return

    ret, test_frame = stream.read()
    if not ret or test_frame is None:
        print("[ERROR] Could not read from camera.")
        stream.stop()
        return

    h, w = test_frame.shape[:2]
    print(f"[INFO] Camera ready. Resolution: {w}x{h}")

    stream.start()
    worker = DetectionWorker(model, CONFIDENCE_MIN).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n[INFO] Person detection running in fullscreen (multithreaded).")
    print("[INFO] Press 'Q' to quit, 'S' to save screenshot.")
    print("-" * 60)

    fps_start = time.time()
    fps_counter = 0
    fps_display = 0.0
    last_person_count = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("[WARNING] Lost camera connection.")
                break

            worker.submit(frame)
            boxes = worker.get_latest()

            person_count = 0
            person_number = 0

            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) != PERSON_CLASS_ID:
                        continue
                    confidence = float(box.conf[0])
                    person_count += 1
                    person_number += 1
                    draw_person_box(frame, box, person_number, confidence)

            last_person_count = person_count

            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            draw_count_panel(frame, person_count, fps_display)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print(f"\n[INFO] Quit. Last frame had {last_person_count} person(s).")
                break
            elif key == ord('s') or key == ord('S'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(OUTPUT_DIR, f"phase2_persons_{last_person_count}_{ts}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[INFO] Screenshot saved: {filename}")
    finally:
        worker.stop()
        stream.stop()
        cv2.destroyAllWindows()

    print("[INFO] Phase 2 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
