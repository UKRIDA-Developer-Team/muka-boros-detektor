import os
import time
from collections import deque

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase
from streamlit_webrtc.credentials import get_available_ice_servers

from model import align_and_crop, detect_face, load_face_detector, predict_age

WINDOW_SECONDS = 2.0
MIN_SAMPLES = 8
COLOR_ADULT = (40, 180, 70)
COLOR_MINOR = (220, 50, 50)
COLOR_WAITING = (240, 190, 40)
TURN_SECRET_KEYS = ("HF_TOKEN", "CLOUDFLARE_TURN_KEY_ID", "CLOUDFLARE_TURN_KEY_API_TOKEN", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN")

def get_rtc_configuration():
    for key in TURN_SECRET_KEYS:
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value and not os.getenv(key):
            os.environ[key] = str(value)
    return {"iceServers": get_available_ice_servers()}

class LiveAgeProcessor(VideoProcessorBase):
    def __init__(self, model, age_limit, tta_flip=False):
        self.model = model
        self.tta_flip = tta_flip
        self.age_limit = age_limit
        self.detector = load_face_detector()
        self.history = deque()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format='rgb24').copy()
        now = time.monotonic()
        while self.history and now - self.history[0][0] > WINDOW_SECONDS:
            self.history.popleft()

        face = detect_face(self.detector, Image.fromarray(image))
        if face is None:
            self.history.clear()
            draw_label(image, 'No face detected', (16, 36), COLOR_WAITING)
            return av.VideoFrame.from_ndarray(image, format='rgb24')

        age = predict_age(self.model, align_and_crop(Image.fromarray(image), face), tta_flip=self.tta_flip)
        self.history.append((now, age))
        median_age = round(float(np.median([a for _, a in self.history])))
        if len(self.history) < MIN_SAMPLES:
            color, label = COLOR_WAITING, f'~{median_age} y, hold still...'
        elif median_age >= self.age_limit:
            color, label = COLOR_ADULT, f'{median_age} y, allowed'
        else:
            color, label = COLOR_MINOR, f'{median_age} y, not allowed'

        x, y, w, h = (int(v) for v in face[:4])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        draw_label(image, label, (x, max(28, y - 12)), color)
        return av.VideoFrame.from_ndarray(image, format='rgb24')

def draw_label(image, text, origin, color):
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    x, y = origin
    cv2.rectangle(image, (x - 4, y - text_h - 8), (x + text_w + 4, y + baseline), color, -1)
    cv2.putText(image, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
