import sys
import os
import asyncio
import gc
import tempfile
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from deepface import DeepFace
import imghdr
import requests

# Environment setup
os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tensorflow optimization
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# App initialization
app = FastAPI(
    title="Deepfake Detection API",
    description="Detects Deepfake, AI, Real images (CPU optimized)",
    version="1.0",
    contact={"name": "Support", "email": "support@example.com"},
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0.html"}
)

# Constants
MODEL_URL = "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng"
MODEL_DIR = os.path.join(os.getenv('MODEL_DIR', '/tmp/model'))
MODEL_PATH = os.path.join(MODEL_DIR, "final_model_11_4_2025.keras")
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_WORKERS = 1  # strictly 0.5 vCPU
TIMEOUT_SECONDS = 30

# Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]
    error: Optional[str] = None

    class Config:
        allow_population_by_field_name = True

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str]
    model_exists: Optional[bool]

# Custom Layers
class EfficientChannelAttention(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = models.Sequential([
            layers.Dense(channels//reduction, activation='relu'),
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
        attention = self.conv(concat)
        return attention * inputs

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
        residual = inputs if self.res_conv is None else self.res_conv(inputs)
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act(x)
        x = self.eca(x)
        x = self.sa(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(x + residual)

def download_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        temp_path = MODEL_PATH + ".tmp"
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        os.rename(temp_path, MODEL_PATH)
    else:
        logger.info("Model already downloaded.")

def load_model():
    custom_objects = {
        'EfficientChannelAttention': EfficientChannelAttention,
        'FixedSpatialAttention': FixedSpatialAttention,
        'FixedHybridBlock': FixedHybridBlock
    }
    with tf.device('/CPU:0'):
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    return model

class FaceCropper:
    def __init__(self):
        logger.info("DeepFace initialized.")

    def safe_crop(self, img_array: np.ndarray) -> np.ndarray:
        try:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(img_path=img_rgb, detector_backend='opencv', enforce_detection=False, align=False)
            if not faces:
                return cv2.resize(img_array, TARGET_SIZE)
            face_img = faces[0]['face']
            return cv2.resize(face_img, TARGET_SIZE)
        except Exception:
            return cv2.resize(img_array, TARGET_SIZE)

def is_valid_image(file_bytes: bytes) -> bool:
    if len(file_bytes) > MAX_FILE_SIZE:
        return False
    image_type = imghdr.what(None, h=file_bytes)
    return image_type in ['jpeg', 'jpg', 'png', 'bmp']

async def process_single_file(file: UploadFile) -> dict:
    result = {"filename": file.filename, "class": None, "confidence": None, "probabilities": None, "error": None}
    try:
        file_bytes = await file.read()
        await file.close()
        if not is_valid_image(file_bytes):
            raise ValueError("Invalid image file.")
        img_arr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        full_img = preprocess_image(img_arr)
        face_img = app.state.cropper.safe_crop(img_arr)
        with tf.device('/CPU:0'):
            prediction = app.state.model.predict([np.array([full_img]), np.array([face_img])], verbose=0)[0]
        result.update({
            "class": CLASS_NAMES[np.argmax(prediction)],
            "confidence": float(np.max(prediction)),
            "probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, prediction)}
        })
    except Exception as e:
        result["error"] = str(e)
    finally:
        del file_bytes, img_arr, full_img, face_img, prediction
        gc.collect()
        tf.keras.backend.clear_session()
    return result

def preprocess_image(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, TARGET_SIZE)

@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
    if len(files) > 100:
        raise HTTPException(413, "Max 100 files allowed.")
    results = await asyncio.gather(*(process_single_file(file) for file in files))
    gc.collect()
    return results

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, "model"),
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH)
    }

@app.on_event("startup")
async def startup_event():
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    download_model()
    app.state.model = load_model()
    app.state.cropper = FaceCropper()
    app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, "executor"):
        app.state.executor.shutdown(wait=False)
    if hasattr(app.state, "model"):
        del app.state.model
        tf.keras.backend.clear_session()
    cv2.destroyAllWindows()
    gc.collect()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        contact=app.contact,
        license_info=app.license_info
    )
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development"},
        {"url": "https://your-production-url.com", "description": "Production"}
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1, limit_concurrency=4, timeout_keep_alive=TIMEOUT_SECONDS)
