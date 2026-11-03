from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parents[2] / 'models' / 'age_verification_model.pt' # path dari model nya
AGE_RANGE = (0, 116) # range usia manusia yang realistis, dari 0 sampai 116 tahun

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_model(median_age):
    model = models.mobilenet_v3_small(weights=None)
    age_head = nn.Linear(model.classifier[3].in_features, 1)
    nn.init.constant_(age_head.bias, median_age)
    model.classifier[3] = age_head
    return model

def load_model(path=MODEL_PATH):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = build_model(checkpoint['median_age'])
    model.load_state_dict(checkpoint['model'])
    return model.eval()

@torch.inference_mode()
def predict_age(model, image: Image.Image) -> float:
    image_tensor = PREPROCESS(image.convert('RGB')).unsqueeze(0)
    raw_age = model(image_tensor).item()
    return min(max(raw_age, AGE_RANGE[0]), AGE_RANGE[1])
