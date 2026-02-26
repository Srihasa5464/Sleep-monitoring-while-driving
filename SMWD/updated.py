import os
import cv2
import time
import math
import csv
import threading
import numpy as np
import pyttsx3
import requests
from collections import deque
from datetime import datetime
import mediapipe as mp
import platform
import base64
import pygame   # ✅ ADDED

# ---------------------- CONFIG ----------------------
CAM_INDEX = 0

YAWN_SECONDS = 1.5
LIVENESS_WINDOW = 4.0
EAR_ABS_THRESHOLD = 0.22
MAR_THRESHOLD = 0.60
SMOOTH_WINDOW = 6
BLINK_EAR_DIFF = 0.12
BLINK_MIN_INTERVAL = 0.15
FRAME_MOTION_THRESH = 6.0
LANDMARK_MOVE_THRESH = 0.002
SNAPSHOT_DIR = "snapshots"
SNAPSHOT_MIN_INTERVAL = 1.0
TTS_RATE = 160
LOG_FILE = "drowsy_events.csv"
SHOW_FACE_MESH = True
DRAW_HUD = True
EYES_CLOSED_ALERT_INTERVAL = 2.0

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ---------------------- SOUND SETUP ----------------------
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")  # Make sure alarm.mp3 is in same folder

def beep_alert():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)  # loop continuously

def stop_alarm():
    pygame.mixer.music.stop()

# ---------------------- LOCATION ----------------------
def get_location():
    try:
        res = requests.get("https://ipinfo.io", timeout=4)
        data = res.json()
        city = data.get("city","")
        region = data.get("region","")
        country = data.get("country","")
        parts = [p for p in (city, region, country) if p]
        return ", ".join(parts) if parts else "Unknown Location"
    except:
        return "Unknown Location"

location_text = get_location()

# ---------------------- TTS ----------------------
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate',TTS_RATE)
except:
    tts_engine = None

tts_lock = threading.Lock()
tts_flag = threading.Event()

def tts_one_shot(msg):
    def _play():
        with tts_lock:
            if tts_engine:
                try:
                    tts_engine.say(msg)
                    tts_engine.runAndWait()
                except:
                    pass
    threading.Thread(target=_play,daemon=True).start()

# ---------------------- MEDIA PIPE ----------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_IDX = [33,160,158,133,153,144]
RIGHT_EYE_IDX = [362,385,387,263,373,380]
MOUTH_TOP=13; MOUTH_BOTTOM=14; MOUTH_LEFT=61; MOUTH_RIGHT=291

# ---------------------- HELPERS ----------------------
def landmarks_to_np(landmarks,w,h):
    return np.array([[lm.x*w,lm.y*h] for lm in landmarks],dtype=np.float32)

def compute_ear(landmarks, eye_idx, w, h):
    pts = landmarks_to_np(landmarks,w,h)[eye_idx]
    A = np.linalg.norm(pts[1]-pts[5])
    B = np.linalg.norm(pts[2]-pts[4])
    C = np.linalg.norm(pts[0]-pts[3])
    return (A+B)/(2*C) if C>0 else 0.0

def compute_mar(landmarks,w,h):
    pts = landmarks_to_np(landmarks,w,h)
    mouth_h = np.linalg.norm(pts[MOUTH_TOP]-pts[MOUTH_BOTTOM])
    mouth_w = np.linalg.norm(pts[MOUTH_LEFT]-pts[MOUTH_RIGHT])
    return mouth_h/mouth_w if mouth_w>0 else 0.0

# ---------------------- STATE ----------------------
ear_history = deque(maxlen=SMOOTH_WINDOW)
closed_start_time=None
alarm_active=False

# ---------------------- VIDEO ----------------------
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened(): raise SystemExit("Cannot open camera")

print("Drowsiness Monitor Started")

try:
    while True:
        ret,frame = cap.read()
        if not ret: break
        fh,fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        nowt=time.time()

        if results and results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = compute_ear(landmarks, LEFT_EYE_IDX, fw, fh)
            right_ear = compute_ear(landmarks, RIGHT_EYE_IDX, fw, fh)
            avg_ear=(left_ear+right_ear)/2.0
            ear_history.append(avg_ear)
            smoothed_ear = np.mean(ear_history)

            closed = smoothed_ear < EAR_ABS_THRESHOLD

            if closed:
                if closed_start_time is None:
                    closed_start_time=nowt

                if (nowt-closed_start_time)>=EYES_CLOSED_ALERT_INTERVAL:
                    if not alarm_active:
                        tts_one_shot("Wake up! Please open your eyes.")
                        beep_alert()
                        alarm_active=True
            else:
                closed_start_time=None
                if alarm_active:
                    stop_alarm()
                    alarm_active=False

        cv2.imshow("Drowsiness Monitor", frame)
        key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),27):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()