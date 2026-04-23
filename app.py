import base64
import logging
import signal
import sys
import time

import cv2
from flask import Flask, send_file
from flask_socketio import SocketIO, emit

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── App / SocketIO ────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me-in-production"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# ── Cascade classifiers ───────────────────────────────────────────────────────
def _load_cascade(filename: str) -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + filename
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"Failed to load cascade: {path}")
    return clf

face_cascade = _load_cascade("haarcascade_frontalface_default.xml")
eye_cascade  = _load_cascade("haarcascade_eye.xml")

# ── State ─────────────────────────────────────────────────────────────────────
connected_clients: int = 0
streaming: bool = False

# ── Detection helpers ─────────────────────────────────────────────────────────
def detect_and_draw(frame: cv2.Mat) -> cv2.Mat:
    """Run face + eye detection and annotate *frame* in-place."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray  = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15)
        )
        for (ex, ey, ew, eh) in eyes:
            cx, cy = ex + ew // 2, ey + eh // 2
            cv2.circle(roi_color, (cx, cy), 20, (0, 255, 0), 2)

    return frame


def encode_frame(frame: cv2.Mat, quality: int = 75) -> str:
    """JPEG-encode *frame* and return a base-64 string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buffer = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buffer).decode("utf-8")


# ── Background streaming task ─────────────────────────────────────────────────
TARGET_FPS   = 30
FRAME_PERIOD = 1.0 / TARGET_FPS


def generate_frames() -> None:
    """Capture frames, annotate them, and broadcast via SocketIO."""
    global streaming
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        log.error("Cannot open camera — background task exiting.")
        streaming = False
        return

    log.info("Camera opened — streaming started.")

    try:
        while streaming:
            t0 = time.monotonic()

            # Skip frame if no clients are watching
            if connected_clients == 0:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to grab frame — retrying…")
                time.sleep(0.05)
                continue

            try:
                annotated  = detect_and_draw(frame)
                b64_frame  = encode_frame(annotated)
                socketio.emit("video_frame", b64_frame)
            except Exception as exc:
                log.exception("Error processing frame: %s", exc)

            # Throttle to TARGET_FPS
            elapsed = time.monotonic() - t0
            sleep_for = FRAME_PERIOD - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        cap.release()
        log.info("Camera released — streaming stopped.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_file("index.html")


# ── Socket events ─────────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    global connected_clients
    connected_clients += 1
    log.info("Client connected  (total: %d)", connected_clients)


@socketio.on("disconnect")
def on_disconnect():
    global connected_clients
    connected_clients = max(0, connected_clients - 1)
    log.info("Client disconnected (total: %d)", connected_clients)


# ── Graceful shutdown ─────────────────────────────────────────────────────────
def _shutdown(sig, frame):
    global streaming
    log.info("Shutting down…")
    streaming = False
    sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    streaming = True
    socketio.start_background_task(generate_frames)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
