"""
WeaponShield AI – app.py
Flask backend: handles file I/O, YOLO inference, weapon-type classification.

Weapon Type Detection Strategy
───────────────────────────────
The YOLO model was trained with classes=1, so it only outputs one class:
'weapon'.  To also predict the SPECIFIC weapon type we apply a secondary
visual classifier on the detected bounding-box crop:
  1. Aspect ratio  – long thin objects → rifle vs. compact → pistol
  2. Elongation    – very tall thin → knife/bat
  3. Dominant HSV color – dark metallic vs. brown/wood tones → sub-type hint
"""

import os
import logging
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image

# ── Load Environment Variables ────────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Configure Gemini AI ───────────────────────────────────────────────────────
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    log_gemini = True
else:
    log_gemini = False

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("WeaponShield")

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Config ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER    = os.path.join(os.path.dirname(__file__), 'uploads')
WEIGHTS_PATH     = "yolov3_training_2000.weights"
CONFIG_PATH      = "yolov3_testing.cfg"
NAMES_PATH        = "weapon.names"
CONFIDENCE_THRESH = 0.5   # Base threshold for YOLO trigger
NMS_THRESH        = 0.4
INPUT_SIZE        = (416, 416)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load YOLO Model (once at startup) ────────────────────────────────────────
log.info("Loading YOLO model…")
try:
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names   = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    log.info("YOLO model loaded successfully.")
except Exception as e:
    log.critical(f"Failed to load YOLO model: {e}")
    net = None

# ── Load class labels ─────────────────────────────────────────────────────────
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

WEAPON_LABELS = {c.lower() for c in classes}


# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI AI VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify_with_gemini(frame: np.ndarray) -> tuple:
    """
    Asks Gemini 1.5 Flash to verify if there's a weapon in the frame.
    Returns (is_weapon: bool, weapon_type: str, confidence: float)
    """
    if not log_gemini:
        return None, None, None

    try:
        # Convert OpenCV BGR to RGB PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = PIL.Image.fromarray(rgb_frame)

        prompt = (
            "Analyze this image carefully. Is there a visible weapon (gun, rifle, knife, bat, etc.)? "
            "Answer ONLY in this JSON format: "
            "{\"weapon_detected\": boolean, \"weapon_type\": \"string or null\", \"confidence\": float_between_0_1}"
        )

        response = gemini_model.generate_content([prompt, pil_img])
        text = response.text.strip()
        
        # Basic JSON extraction (Gemini might wrap in markdown ```json)
        if "{" in text and "}" in text:
            json_str = text[text.find("{"):text.rfind("}")+1]
            import json
            data = json.loads(json_str)
            return data.get('weapon_detected'), data.get('weapon_type'), data.get('confidence')
        
        return False, None, 0.0
    except Exception as e:
        log.error(f"Gemini verification failed: {e}")
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
#  SECONDARY VISUAL WEAPON-TYPE CLASSIFIER (Local Heuristics)
# ══════════════════════════════════════════════════════════════════════════════

def _dominant_hsv(crop: np.ndarray):
    if crop is None or crop.size == 0: return 0, 0, 0
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return cv2.mean(hsv)[:3]

def _elongation(crop: np.ndarray):
    if crop is None or crop.size == 0: return 1.0
    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 1.0
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5: return 1.0
    try:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        return max(MA, ma) / (min(MA, ma) + 1e-6)
    except: return 1.0

def classify_weapon_type(crop: np.ndarray, box_w: int, box_h: int) -> str:
    if crop is None or crop.size == 0: return "Firearm"
    aspect = box_w / max(box_h, 1)
    elong = _elongation(crop)

    if aspect > 2.0: return "Rifle / AK-47 Type"
    if 1.0 <= aspect <= 2.0: return "Pistol / Handgun"
    if aspect < 1.0:
        if elong > 3.0: return "Knife / Blade"
        return "Pistol / Handgun"
    return "Firearm"


# ── Core YOLO Inference ───────────────────────────────────────────────────────
def process_frame(frame: np.ndarray):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, INPUT_SIZE, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > CONFIDENCE_THRESH:
                cx, cy, bw, bh = int(det[0]*w), int(det[1]*h), int(det[2]*w), int(det[3]*h)
                boxes.append([int(cx-bw/2), int(cy-bh/2), bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESH, NMS_THRESH)
    if len(indices) > 0:
        indices = indices.flatten().tolist()
        return [boxes[i] for i in indices], [confidences[i] for i in indices], [class_ids[i] for i in indices]
    return [], [], []

def scan_for_weapons(frame: np.ndarray, boxes, confidences, class_ids):
    if not class_ids: return None, None, None
    best_conf, best_name, best_type = -1.0, None, None
    h, w = frame.shape[:2]

    for i, cid in enumerate(class_ids):
        if confidences[i] > best_conf:
            best_conf = confidences[i]
            best_name = "weapon"
            x, y, bw, bh = boxes[i]
            crop = frame[max(y,0):min(y+bh,h), max(x,0):min(x+bw,w)]
            best_type = classify_weapon_type(crop, bw, bh)
    return best_name, best_type, best_conf


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_weapon():
    if net is None: return jsonify({'error': 'Model not loaded'}), 500
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    safe_name = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(file_path)

    try:
        ext = os.path.splitext(safe_name)[1].lower()
        if ext in ('.png', '.jpg', '.jpeg'):
            image = cv2.imread(file_path)
            boxes, confs, ids = process_frame(image)
            wn, wt, c = scan_for_weapons(image, boxes, confs, ids)
            
            # ── Gemini Second Opinion ──
            if wn:
                log.info("YOLO detected something. Consulting Gemini for verification...")
                g_det, g_type, g_conf = verify_with_gemini(image)
                if g_det is False:
                    log.info("Gemini rejected YOLO detection (False Positive).")
                    wn, wt, c = None, None, None
                elif g_det is True:
                    log.info(f"Gemini confirmed: {g_type} ({g_conf})")
                    wt = g_type if g_type else wt
                    c  = g_conf if g_conf > 0 else c

            return jsonify({'result': {'weapon_detected': wn is not None, 'weapon_name': wt or 'No weapon detected', 'confidence': c}, 'frame_count': 1})

        elif ext in ('.mp4', '.avi', '.mov'):
            cap = cv2.VideoCapture(file_path)
            frame_count, detections_found = 0, 0
            weapon_name, weapon_type, confidence = None, None, None
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1
                if frame_count % 10 != 0: continue
                
                boxes, confs, ids = process_frame(frame)
                wn, wt, c = scan_for_weapons(frame, boxes, confs, ids)
                
                if wn:
                    # Verify with Gemini
                    g_det, g_type, g_conf = verify_with_gemini(frame)
                    if g_det:
                        weapon_name, weapon_type, confidence = wn, g_type or wt, g_conf or c
                        break # Confirmed weapon found
            cap.release()
            return jsonify({'result': {'weapon_detected': weapon_name is not None, 'weapon_name': weapon_type or 'No weapon detected', 'confidence': confidence}, 'frame_count': frame_count})

    finally:
        if os.path.exists(file_path): os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True, port=5004)
