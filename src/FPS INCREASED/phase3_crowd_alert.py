import os
import cv2
import time
import csv
import threading
import numpy as np
from datetime import datetime
from ultralytics import YOLO

try:
    import winsound
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
CONFIDENCE_MIN  = 0.40
WINDOW_NAME     = "Phase 3 - Crowd Alert System "

MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "yolov8n.pt")
)
OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
)

CROWD_THRESHOLD = 2
ALERT_COOLDOWN  = 3.0
BEEP_FREQUENCY  = 1000
BEEP_DURATION   = 500

COLOR_PERSON_SAFE  = (0, 230, 0)
COLOR_PERSON_ALERT = (0, 60, 255)
COLOR_COUNT_SAFE   = (0, 220, 80)
COLOR_COUNT_DANGER = (0, 60, 255)
COLOR_ALERT_BG     = (0, 0, 180)

PERSON_CLASS_ID = 0


def setup_output_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, "_write_test.tmp")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"[INFO] Output folder ready: {path}")
        return True
    except Exception as e:
        print(f"[ERROR] Output folder problem: {e}")
        return False


def play_beep_async(frequency: int, duration: int) -> None:
    if not AUDIO_AVAILABLE:
        return
    def _beep():
        try:
            winsound.Beep(frequency, duration)
        except Exception as e:
            print(f"[WARNING] Beep error: {e}")
    threading.Thread(target=_beep, daemon=True).start()


def save_screenshot(frame, output_dir: str, person_count: int) -> str:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = f"CROWD_ALERT_{person_count}persons_{ts}.jpg"
        fp = os.path.join(output_dir, fn)
        if cv2.imwrite(fp, frame):
            print(f"[INFO] Screenshot saved: {fp}")
            return fp
        return ""
    except Exception as e:
        print(f"[WARNING] Screenshot error: {e}")
        return ""


def log_to_csv(output_dir: str, person_count: int, screenshot_path: str) -> None:
    log_file = os.path.join(output_dir, "crowd_log.csv")
    file_exists = os.path.isfile(log_file)
    try:
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "persons_detected", "alert_status", "screenshot_file"])
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                person_count, "CROWD ALERT",
                os.path.basename(screenshot_path) if screenshot_path else "none"
            ])
        print("[INFO] Event logged to crowd_log.csv")
    except Exception as e:
        print(f"[WARNING] CSV log error: {e}")


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
# The display/alert loop never waits on the model - it just reads the
# most recent finished result, so video stays smooth even if inference
# lags a little behind the live feed.
# ----------------------------------------------------------------------
class DetectionWorker:
    def __init__(self, model, conf):
        self.model = model
        self.conf = conf

        self.lock = threading.Lock()
        self.input_frame = None
        self.output_boxes = None
        self.output_fps = 0.0

        self._new_frame = threading.Event()
        self.stopped = False
        self.thread = threading.Thread(target=self._run, daemon=True)

        self._fps_counter = 0
        self._fps_start = time.time()

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

            self._fps_counter += 1
            elapsed = time.time() - self._fps_start
            if elapsed >= 1.0:
                self.output_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_start = time.time()

    def get_latest(self):
        with self.lock:
            return self.output_boxes, self.output_fps

    def stop(self):
        self.stopped = True
        self._new_frame.set()
        self.thread.join(timeout=1.0)


def draw_person_box(frame, box, person_number: int, confidence: float, alert_active: bool) -> None:
    x1, y1 = int(box.xyxy[0][0]), int(box.xyxy[0][1])
    x2, y2 = int(box.xyxy[0][2]), int(box.xyxy[0][3])
    box_color = COLOR_PERSON_ALERT if alert_active else COLOR_PERSON_SAFE

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
    label = f"#{person_number}  Person  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    label_y = max(y1 - 2, th + 10)

    cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 2), (15, 15, 15), cv2.FILLED)
    cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 2), box_color, 1)
    cv2.putText(frame, label, (x1 + 4, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)


def draw_alert_banner(frame, person_count: int, blink_on: bool) -> None:
    fh, fw = frame.shape[:2]
    y1, y2 = fh // 2 - 55, fh // 2 + 55
    opacity = 0.88 if blink_on else 0.40

    ov = frame.copy()
    cv2.rectangle(ov, (0, y1), (fw, y2), COLOR_ALERT_BG, cv2.FILLED)
    cv2.addWeighted(ov, opacity, frame, 1 - opacity, 0, frame)
    cv2.line(frame, (0, y1), (fw, y1), (255, 255, 255), 2)
    cv2.line(frame, (0, y2), (fw, y2), (255, 255, 255), 2)

    line1 = "!! CROWD ALERT !!"
    (w1, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.putText(frame, line1, (fw // 2 - w1 // 2, fh // 2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)

    line2 = f"{person_count} PERSONS DETECTED - THRESHOLD EXCEEDED"
    (w2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, line2, (fw // 2 - w2 // 2, fh // 2 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, person_count: int, fps: float, inference_fps: float, alert_active: bool, total_alerts: int) -> None:
    fh, fw = frame.shape[:2]
    count_color = COLOR_COUNT_DANGER if alert_active else COLOR_COUNT_SAFE

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (fw, 50), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}", (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 100), 2, cv2.LINE_AA)

    # infer_str = f"Inference: {inference_fps:.1f} fps"
    # (iw, _), _ = cv2.getTextSize(infer_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # cv2.putText(frame, infer_str, (fw // 2 - iw // 2 - 70, 33),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 200, 255), 1, cv2.LINE_AA)

    alert_str = f"Alerts fired: {total_alerts}"
    (aw, _), _ = cv2.getTextSize(alert_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(frame, alert_str, (fw // 2 - aw // 2 + 70, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 60, 255) if alert_active else (160, 160, 160), 1, cv2.LINE_AA)

    phase_str = "Phase 3: Crowd Alert System"
    (pw, _), _ = cv2.getTextSize(phase_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(frame, phase_str, (fw - pw - 12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 255), 1, cv2.LINE_AA)

    px, py = 10, fh - 108
    ov2 = frame.copy()
    cv2.rectangle(ov2, (px, py), (px + 215, py + 93), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ov2, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (px, py), (px + 215, py + 93), count_color, 2)

    cv2.putText(frame, "PERSONS DETECTED", (px + 8, py + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    cstr = str(person_count)
    (cw, _), _ = cv2.getTextSize(cstr, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 4)
    cv2.putText(frame, cstr, (px + (215 - cw) // 2, py + 78),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, count_color, 4, cv2.LINE_AA)

    status_str = "ALERT ACTIVE" if alert_active else f"Threshold: >{CROWD_THRESHOLD}"
    cv2.putText(frame, status_str, (px + 8, py + 91),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (0, 60, 255) if alert_active else (130, 130, 130), 1, cv2.LINE_AA)

    ov3 = frame.copy()
    cv2.rectangle(ov3, (0, fh - 30), (fw, fh), (15, 15, 15), cv2.FILLED)
    cv2.addWeighted(ov3, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "Press 'S' to save screenshot  |  Press 'Q' to quit",
                (12, fh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    print("=" * 60)
    print("  Phase 3: Crowd Alert System (Multithreaded)")
    print(f"  Alert threshold : >{CROWD_THRESHOLD} persons")
    print(f"  Alert cooldown  : {ALERT_COOLDOWN} seconds")
    print(f"  Audio available : {AUDIO_AVAILABLE}")
    print(f"  Output folder   : {OUTPUT_DIR}")
    print("=" * 60)

    if not setup_output_dir(OUTPUT_DIR):
        print("[ERROR] Cannot write to output folder. Exiting.")
        return

    print("\n[INFO] Loading YOLOv8n model...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model loaded successfully.")

    print(f"\n[INFO] Opening camera (index {CAMERA_INDEX})...")
    stream = VideoStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)

    if not stream.isOpened():
        print("[ERROR] Cannot open camera. Try CAMERA_INDEX = 1.")
        return

    ret, test = stream.read()
    if not ret or test is None:
        print("[ERROR] Camera opened but cannot read frames.")
        stream.stop()
        return

    h, w = test.shape[:2]
    print(f"[INFO] Camera ready. Resolution: {w}x{h}")

    stream.start()
    worker = DetectionWorker(model, CONFIDENCE_MIN).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"\n[INFO] Monitoring started in fullscreen (multithreaded). Alert fires when >{CROWD_THRESHOLD} persons.")
    print("[INFO] Press Q to quit | S to save screenshot manually")
    print("-" * 60)

    fps_start = time.time()
    fps_counter = 0
    fps_display = 0.0
    last_alert_time = 0.0
    total_alerts = 0
    blink_counter = 0
    blink_on = True
    alert_active = False
    last_person_count = 0

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("[WARNING] Lost camera feed.")
                break

            worker.submit(frame)
            boxes, inference_fps = worker.get_latest()

            person_count = 0
            person_number = 0

            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) != PERSON_CLASS_ID:
                        continue
                    person_count += 1
                    person_number += 1
                    draw_person_box(frame, box, person_number, float(box.conf[0]), alert_active)

            last_person_count = person_count

            now = time.time()
            if person_count > CROWD_THRESHOLD:
                alert_active = True
                if (now - last_alert_time) >= ALERT_COOLDOWN:
                    total_alerts += 1
                    last_alert_time = now
                    print(f"\n[ALERT #{total_alerts}] {datetime.now().strftime('%H:%M:%S')} "
                          f"- {person_count} persons detected!")
                    play_beep_async(BEEP_FREQUENCY, BEEP_DURATION)
                    saved_path = save_screenshot(frame, OUTPUT_DIR, person_count)
                    log_to_csv(OUTPUT_DIR, person_count, saved_path)
            else:
                alert_active = False

            blink_counter += 1
            if blink_counter >= 15:
                blink_counter = 0
                blink_on = not blink_on

            if alert_active:
                draw_alert_banner(frame, person_count, blink_on)

            draw_hud(frame, person_count, fps_display, inference_fps, alert_active, total_alerts)

            fps_counter += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print(f"\n[INFO] Session ended. Total alerts: {total_alerts}")
                print(f"[INFO] Log saved at: {OUTPUT_DIR}\\crowd_log.csv")
                break
            elif key == ord('s') or key == ord('S'):
                path = save_screenshot(frame, OUTPUT_DIR, last_person_count)
                print(f"[INFO] Manual screenshot: {path}")
    finally:
        worker.stop()
        stream.stop()
        cv2.destroyAllWindows()

    print("[INFO] Phase 3 complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()