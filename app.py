import sys
import os
import asyncio
os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import cv2
import imghdr
from deepface import DeepFace
from typing import List, Dict, Optional
import requests
from pydantic import BaseModel, Field
import logging
from concurrent.futures import ThreadPoolExecutor
import gc

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/model/final_model_11_4_2025.keras')
TARGET_SIZE = tuple(map(int, os.environ.get('TARGET_SIZE', '224,224').split(',')))
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 10485760))  # 10MB
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 2))
CLASS_NAMES = ['AI', 'FAKE', 'REAL']

app = FastAPI(
    title="Deepfake Detection API",
    description="Optimized API for detecting deepfake images",
    version="2.0"
)

# Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]
    error: Optional[str]

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    memory_usage: str

# Custom Layers with proper serialization
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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'reduction': self.reduction
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            channels=config.get('channels', config.get('config', {}).get('channels')),
            reduction=config.get('reduction', config.get('config', {}).get('reduction', 8))
        )

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
    
    def get_config(self):
        return super().get_config()

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
        super().build(input_shape)

    def call(self, inputs):
        residual = inputs
        
        if self.res_conv is not None:
            residual = self.res_conv(inputs)
            
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.eca(x)
        x = self.sa(x)
            
        x = self.conv2(x)
        x = self.norm2(x)
        
        return self.act(x + residual)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            filters=config.get('filters', config.get('config', {}).get('filters'))
        )

# Model Loading with proper custom objects
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    custom_objects = {
        'EfficientChannelAttention': EfficientChannelAttention,
        'FixedSpatialAttention': FixedSpatialAttention,
        'FixedHybridBlock': FixedHybridBlock,
        'mixed_float16': tf.keras.mixed_precision.Policy('mixed_float16')
    }
    
    try:
        with keras.utils.custom_object_scope(custom_objects):
            model = tf.keras.models.load_model(MODEL_PATH)
            
            # Warm up the model
            dummy_input = [
                tf.convert_to_tensor(np.zeros((1, *TARGET_SIZE, 3))),
                tf.convert_to_tensor(np.zeros((1, *TARGET_SIZE, 3)))
            ]
            model.predict(dummy_input, steps=1)
            
            return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Face Processing
class FaceCropper:
    def __init__(self):
        logger.info("Initializing face detector...")
        self.detector = DeepFace.build_model('Facenet')
        
    def safe_crop(self, img_array: np.ndarray) -> np.ndarray:
        try:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(
                img_path=img_rgb,
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )
            if faces:
                face_img = faces[0]['face']
                resized = cv2.resize(face_img, TARGET_SIZE)
                return resized
            return cv2.resize(img_array, TARGET_SIZE)
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return cv2.resize(img_array, TARGET_SIZE)

# Image Processing
def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, TARGET_SIZE)

def process_single_file(file: UploadFile):
    result = {"filename": file.filename, "error": None}
    try:
        if file.size > MAX_FILE_SIZE:
            raise ValueError("File size exceeds limit")
            
        file_bytes = file.file.read(MAX_FILE_SIZE)
        if not imghdr.what(None, file_bytes):
            raise ValueError("Invalid image format")
            
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        full_img = preprocess_image(img)
        face_img = app.state.cropper.safe_crop(img)
        
        prediction = app.state.model.predict([
            np.expand_dims(full_img, 0), 
            np.expand_dims(face_img, 0)
        ], verbose=0)[0]
        
        result.update({
            "class": CLASS_NAMES[np.argmax(prediction)],
            "confidence": float(np.max(prediction)),
            "probabilities": dict(zip(CLASS_NAMES, prediction.astype(float)))
        })
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error processing {file.filename}: {e}")
    finally:
        gc.collect()
        return result

# API Endpoints
@app.on_event("startup")
async def startup_event():
    try:
        tf.config.optimizer.set_jit(True)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        app.state.model = load_model()
        app.state.cropper = FaceCropper()
        app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logger.info("API initialized successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise RuntimeError(f"Initialization error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'model'):
        del app.state.model
        tf.keras.backend.clear_session()
    if hasattr(app.state, 'executor'):
        app.state.executor.shutdown(wait=False)
    gc.collect()

@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
    if len(files) > 5:
        raise HTTPException(413, "Maximum 5 files allowed")
    
    futures = []
    for file in files:
        futures.append(
            asyncio.get_event_loop().run_in_executor(
                app.state.executor,
                process_single_file,
                file
            )
        )
    
    results = await asyncio.gather(*futures)
    gc.collect()
    return results

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    import psutil
    return {
        "status": "healthy",
        "model_loaded": hasattr(app.state, 'model'),
        "memory_usage": f"{psutil.Process().memory_info().rss / 1024 ** 2:.1f}MB"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=4,
        log_level="warning"
    )
