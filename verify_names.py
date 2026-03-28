import cv2
import numpy as np
import os

# Config from app.py
WEIGHTS_PATH     = "yolov3_training_2000.weights"
CONFIG_PATH      = "yolov3_testing.cfg"
NAMES_PATH        = "weapon.names"
CONFIDENCE_THRESH = 0.5
INPUT_SIZE        = (416, 416)

net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
layer_names   = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def _dominant_hsv(crop: np.ndarray):
    if crop is None or crop.size == 0: return 0, 0, 0
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return cv2.mean(hsv)[:3]

def _elongation(crop: np.ndarray):
    if crop is None or crop.size == 0: return 1.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 1.0
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5: return 1.0
    try:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(cnt)
        return max(MA, ma) / (min(MA, ma) + 1e-6)
    except: return 1.0

def classify_weapon_type(crop, box_w, box_h):
    aspect = box_w / max(box_h, 1)
    hue, sat, val = _dominant_hsv(crop)
    elong = _elongation(crop)
    
    if aspect > 3.0: return "Rifle / Long Gun (AK-47 Type)"
    if 1.2 <= aspect <= 3.0: return "Pistol / Handgun"
    if aspect < 0.8: return "Knife / Blade"
    return "Firearm (Generic)"

def test_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return
    
    found = False
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, INPUT_SIZE, (0,0,0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        for output in outputs:
            for det in output:
                scores = det[5:]
                if scores[0] > CONFIDENCE_THRESH:
                    bw, bh = int(det[2]*w), int(det[3]*h)
                    x, y = int(det[0]*w - bw/2), int(det[1]*h - bh/2)
                    crop = frame[max(0,y):min(h,y+bh), max(0,x):min(w,x+bw)]
                    name = classify_weapon_type(crop, bw, bh)
                    print(f"Detected in {path}: {name} (conf: {scores[0]:.2f}, aspect: {bw/bh:.2f})")
                    found = True
                    break
            if found: break
        if found: break
    cap.release()

test_video("ak47.mp4")
test_video("pistol.mp4")
