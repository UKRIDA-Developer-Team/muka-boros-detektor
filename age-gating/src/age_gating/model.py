import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'age_verification_model.pt' # path dari model nya
FACE_DETECTOR_PATH = Path(__file__).resolve().parents[2] / 'models' / 'face_detection_yunet_2023mar.onnx'
AGE_RANGE = (0, 116) # range usia manusia yang realistis, dari 0 sampai 116 tahun
FACE_CROP_SCALE = 1.0
FACE_CROP_SHIFT_Y = 0.05
FACE_MIN_SCORE = 0.7
DETECT_MAX_SIDE = 640

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_model(median_age, dropout=0.2):
    model = models.mobilenet_v3_small(weights=None)
    age_head = nn.Linear(model.classifier[3].in_features, 1)
    nn.init.constant_(age_head.bias, median_age)
    model.classifier[3] = age_head
    model.classifier[2].p = dropout
    return model

def load_model(path=MODEL_PATH, dropout=0.2):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['median_age'], dropout=dropout)
    model.load_state_dict(checkpoint['model'])
    return model.eval()

def load_face_detector(path=FACE_DETECTOR_PATH):
    return cv2.FaceDetectorYN.create(str(path), '', (320, 320), FACE_MIN_SCORE)

def detect_face(detector, image: Image.Image) -> np.ndarray | None:
    image = image.convert('RGB')
    resize_ratio = min(1.0, DETECT_MAX_SIDE / max(image.size))
    small = image.resize((max(1, round(image.width * resize_ratio)), max(1, round(image.height * resize_ratio))))
    detector.setInputSize(small.size)
    _, faces = detector.detect(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3]) / resize_ratio

def crop_face(detector, image: Image.Image) -> Image.Image | None:
    image = image.convert('RGB')
    face = detect_face(detector, image)
    return None if face is None else align_and_crop(image, face)

def align_and_crop(image: Image.Image, face: np.ndarray) -> Image.Image:
    x, y, w, h = face[:4]
    right_eye, left_eye = face[4:6], face[6:8]
    center_x, center_y = x + w / 2, y + h / 2
    eye_angle = math.degrees(math.atan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0]))
    side = FACE_CROP_SCALE * max(w, h)
    left, top = round(center_x - side), round(center_y - side)
    region = image.crop((left, top, left + round(2 * side), top + round(2 * side)))
    center_x, center_y = center_x - left, center_y - top
    aligned = region.rotate(eye_angle, resample=Image.BILINEAR, center=(center_x, center_y))
    center_y += FACE_CROP_SHIFT_Y * h
    return aligned.crop((round(center_x - side / 2), round(center_y - side / 2),
                         round(center_x + side / 2), round(center_y + side / 2)))

@torch.inference_mode()
def predict_age(model, image: Image.Image, tta_flip: bool = False) -> float:
    image_tensor = PREPROCESS(image.convert('RGB')).unsqueeze(0)
    raw_age = model(image_tensor).item()
    if tta_flip:
        flipped = torch.flip(image_tensor, dims=[3])
        raw_age = (raw_age + model(flipped).item()) / 2
    return min(max(raw_age, AGE_RANGE[0]), AGE_RANGE[1])
