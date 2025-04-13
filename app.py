import os
os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'

import logging
import cv2
import numpy as np
import tensorflow as tf
import requests
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from tensorflow.keras import layers, models
from deepface import DeepFace
import imghdr

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(title="Image Classification API")

# Paths & Constants
MODEL_PATH = "/tmp/model/final_model_11_4_2025.keras"
MODEL_URL = "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng"
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('/tmp/.deepface', exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Response Models
class PredictionResult(BaseModel):
    image: str
    predicted_class: str
    confidence: float
    probabilities: dict

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool

# Download Model
def download_model_if_needed():
    if not Path(MODEL_PATH).exists():
        logger.info("Downloading model...")
        response = requests.get(MODEL_URL)
        if response.status_code != 200:
            raise RuntimeError("Model download failed")
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        logger.info("Model downloaded and saved")

# Attention Modules
class EfficientChannelAttention(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = models.Sequential([
            layers.Dense(channels // reduction, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])

    def call(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return tf.reshape(out, [-1, 1, 1, self.channels]) * x

class FixedSpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=3)
        return self.conv(concat) * inputs

class FixedHybridBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.eca = EfficientChannelAttention(filters)
        self.sa = FixedSpatialAttention()
        self.norm1 = layers.BatchNormalization()
        self.norm2 = layers.BatchNormalization()
        self.act = layers.Activation('swish')
        self.res_conv = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.res_conv = layers.Conv2D(self.filters, 1)

    def call(self, inputs):
        residual = self.res_conv(inputs) if self.res_conv else inputs
        x = self.act(self.norm1(self.conv1(inputs)))
        x = self.eca(x)
        x = self.sa(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)

# Load Model
def load_model():
    download_model_if_needed()
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            'EfficientChannelAttention': EfficientChannelAttention,
            'FixedSpatialAttention': FixedSpatialAttention,
            'FixedHybridBlock': FixedHybridBlock
        }
    )

model = load_model()

# Face Cropper
class FaceCropper:
    def safe_crop(self, path, target_size=(224, 224)):
        try:
            faces = DeepFace.extract_faces(path, detector_backend='opencv', enforce_detection=False)
            if not faces:
                img = cv2.imread(path)
                return cv2.resize(img, target_size)
            return cv2.resize(faces[0]['face'], target_size)
        except Exception as e:
            logger.warning(f"Cropping fallback: {e}")
            return np.zeros((*target_size, 3), dtype=np.float32)

cropper = FaceCropper()

def is_valid_image_file(path):
    ext = os.path.splitext(path)[1].lower()[1:]
    if ext not in {'jpg', 'jpeg', 'png', 'bmp'}:
        return False
    if imghdr.what(path) not in ['jpeg', 'png', 'bmp']:
        return False
    return cv2.imread(path) is not None

def preprocess_images(paths: List[str], target_size=(224, 224)):
    def load_and_resize(path):
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, target_size).numpy()
        except Exception:
            return np.zeros((*target_size, 3), dtype=np.float32)

    return (
        np.array([load_and_resize(p) for p in paths]),
        np.array([cropper.safe_crop(p, target_size) for p in paths])
    )

@app.post("/predict", response_model=List[PredictionResult])
async def predict(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        with open(path, "wb") as f:
            f.write(await file.read())

        if not is_valid_image_file(path):
            raise HTTPException(status_code=400, detail="Invalid image")

        full, face = preprocess_images([path])
        pred = model.predict([full, face])[0]
        return [{
            "image": file.filename,
            "predicted_class": ['AI', 'FAKE', 'REAL'][np.argmax(pred)],
            "confidence": float(np.max(pred)),
            "probabilities": {
                'AI': float(pred[0]),
                'FAKE': float(pred[1]),
                'REAL': float(pred[2])
            }
        }]
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.get("/", response_model=HealthCheck)
async def health():
    return {
        "status": "API is running",
        "model_loaded": Path(MODEL_PATH).exists()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
