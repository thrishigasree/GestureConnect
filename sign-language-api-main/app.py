"""
Sign Language API (Flask + MediaPipe + TFLite)

POST /predict_frame  -> send one webcam frame (base64) + a session_id,
                        get back buffering / prediction status.
POST /reset          -> clear a session's frame buffer.
GET  /health         -> liveness + model info.
"""
import base64
import binascii
import json
import os
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import tflite_runtime.interpreter as tflite
except ImportError:  # fallback for dev machines that have full TensorFlow
    import tensorflow.lite as tflite  # type: ignore

app = Flask(__name__)

# ================= CONFIG =================
FRAMES = int(os.environ.get("FRAMES", 30))
FEATURES = 126                      # 2 hands x 21 landmarks x 3 coords
MODEL_PATH = os.environ.get("MODEL_PATH", "sign_transformer.tflite")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.json")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.88))
MAX_HANDS = int(os.environ.get("MAX_HANDS", 1))          # set 2 if your signs use two hands
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", 60))
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 2_000_000))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")  # e.g. "https://myapp.com,http://localhost:5173"

app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024  # reject huge request bodies
CORS(app, origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")] if ALLOWED_ORIGINS != "*" else "*")

# ================= MODEL (fail fast if it can't load) =================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
NUM_CLASSES = int(output_details[0]["shape"][1])
interpreter_lock = threading.Lock()  # TFLite interpreters are not thread-safe
print("✅ TFLite model loaded successfully!")


def load_labels():
    """Label priority: labels.json -> dataset/ folder names -> Sign_<i> placeholders."""
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if len(labels) != NUM_CLASSES:
            print(f"⚠️  labels.json has {len(labels)} entries but model has {NUM_CLASSES} classes")
        return labels
    if os.path.isdir("dataset"):
        return sorted(os.listdir("dataset"))
    print("⚠️  No labels.json or dataset/ folder found - using placeholder labels (Sign_0, Sign_1, ...).")
    return [f"Sign_{i}" for i in range(NUM_CLASSES)]


SIGNS = load_labels()

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
# static_image_mode=True: each frame is detected independently. Frames from many
# different users share this one detector, so cross-frame *tracking* would mix them up.
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5,
)
hands_lock = threading.Lock()  # MediaPipe solutions are not thread-safe

# ================= PER-SESSION BUFFERS =================
# {session_id: {"frames": [np.ndarray, ...], "last_seen": float}}
sessions = {}
sessions_lock = threading.Lock()


def cleanup_sessions(now):
    """Drop sessions nobody has touched recently (prevents memory leaks)."""
    stale = [sid for sid, s in sessions.items() if now - s["last_seen"] > SESSION_TTL_SECONDS]
    for sid in stale:
        del sessions[sid]


def get_session(session_id):
    now = time.time()
    with sessions_lock:
        cleanup_sessions(now)
        session = sessions.setdefault(session_id, {"frames": [], "last_seen": now})
        session["last_seen"] = now
        return session


def extract_hand_landmarks(results):
    """Return a normalised (FEATURES,) vector, or None if no hand was found."""
    if not results.multi_hand_landmarks:
        return None
    features = np.zeros(FEATURES, dtype=np.float32)
    # Sort hands left-to-right so the feature slots are consistent between frames
    hand_list = sorted(results.multi_hand_landmarks[:2], key=lambda h: h.landmark[0].x)
    idx = 0
    for hand in hand_list:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32)
        pts -= pts[0]                          # translate: wrist becomes origin
        pts /= (np.linalg.norm(pts) + 1e-6)    # scale: size independent
        features[idx:idx + 63] = pts.flatten()
        idx += 63
    return features


def decode_image(b64_string):
    """Decode a base64 image (with or without a data-URL prefix). Raises ValueError."""
    if not isinstance(b64_string, str) or not b64_string:
        raise ValueError("'image' must be a non-empty base64 string")
    if b64_string.startswith("data:"):
        b64_string = b64_string.split(",", 1)[-1]
    try:
        raw = base64.b64decode(b64_string, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("'image' is not valid base64")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode image")
    return frame


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "classes": NUM_CLASSES, "frames": FRAMES})


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id", "default"))
    with sessions_lock:
        sessions.pop(session_id, None)
    return jsonify({"status": "reset"})


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "JSON body with an 'image' field is required"}), 400

    session_id = str(data.get("session_id", "default"))[:100]

    try:
        frame = decode_image(data["image"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with hands_lock:
            results = hands.process(rgb)
        landmarks = extract_hand_landmarks(results)

        session = get_session(session_id)

        if landmarks is None:
            # Hand lost -> sequence is broken, start over
            session["frames"].clear()
            return jsonify({"status": "buffering", "frames_collected": 0, "hand_detected": False})

        session["frames"].append(landmarks)
        session["frames"] = session["frames"][-FRAMES:]

        if len(session["frames"]) < FRAMES:
            return jsonify({
                "status": "buffering",
                "frames_collected": len(session["frames"]),
                "hand_detected": True,
            })

        input_data = np.expand_dims(np.stack(session["frames"]), axis=0).astype(np.float32)
        with interpreter_lock:
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])[0].copy()

        confidence = float(np.max(preds))
        idx = int(np.argmax(preds))
        label = SIGNS[idx] if idx < len(SIGNS) else str(idx)

        return jsonify({
            "status": "success" if confidence >= CONFIDENCE_THRESHOLD else "low_confidence",
            "prediction": label,
            "confidence": confidence,
            "hand_detected": True,
        })
    except Exception as e:  # noqa: BLE001
        app.logger.exception("predict_frame failed")
        return jsonify({"error": "internal error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
