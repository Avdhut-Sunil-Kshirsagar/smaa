import os
import sys
import asyncio
import tempfile
import logging
import numpy as np
import cv2
import imghdr
from concurrent.futures import ThreadPoolExecutor

# Configure environment and mixed precision FIRST
os.environ.update({
    'DEEPFACE_HOME': '/tmp/.deepface',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'TF_ENABLE_ONEDNN_OPTS': '1',
    'TF_NUM_INTEROP_THREADS': '1',
    'TF_NUM_INTRAOP_THREADS': '1',
    'OMP_NUM_THREADS': '1'
})

# Set mixed precision policy before any TF operations
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Now import other dependencies
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras import layers, models
from deepface import DeepFace
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants loaded from environment
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10 * 1024 * 1024))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 2))

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake images with mixed precision support",
    version="1.0",
    docs_url=None,
    redoc_url=None
)

# Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float]
    error: Optional[str] = None

    class Config:
        allow_population_by_field_name = True

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_size_mb: Optional[float]

# Custom layers with mixed precision support
class EfficientChannelAttention(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = models.Sequential([
            layers.Dense(channels//reduction, activation='relu', dtype=policy),
            layers.Dense(channels, activation='sigmoid', dtype=policy)
        ])
        
    def call(self, x):
        if x.dtype != policy.compute_dtype:
            x = tf.cast(x, policy.compute_dtype)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return tf.reshape(out, [-1, 1, 1, self.channels]) * x
    
    def get_config(self):
        return {'channels': self.channels, 'reduction': self.reduction}

class FixedSpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid', dtype=policy)
        
    def call(self, inputs):
        if inputs.dtype != policy.compute_dtype:
            inputs = tf.cast(inputs, policy.compute_dtype)
        avg_out = tf.reduce_mean(inputs, axis=3, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=3, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=3)
        return self.conv(concat) * inputs

class FixedHybridBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding='same', dtype=policy)
        self.conv2 = layers.Conv2D(filters, 3, padding='same', dtype=policy)
        self.eca = EfficientChannelAttention(filters)
        self.sa = FixedSpatialAttention()
        self.norm1 = layers.BatchNormalization(dtype='float32')
        self.norm2 = layers.BatchNormalization(dtype='float32')
        self.act = layers.Activation('swish', dtype=policy)
        self.res_conv = None

    def call(self, inputs):
        if inputs.dtype != policy.compute_dtype:
            inputs = tf.cast(inputs, policy.compute_dtype)
            
        residual = inputs
        if self.res_conv is not None:
            residual = self.res_conv(residual)
            
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = tf.cast(x, policy.compute_dtype)
        x = self.act(x)
        
        x = self.eca(x)
        x = self.sa(x)
            
        x = self.conv2(x)
        x = self.norm2(x)
        x = tf.cast(x, policy.compute_dtype)
        
        return self.act(x + residual)
    
    def get_config(self):
        return {'filters': self.filters}

# Resource management
def cleanup_resources():
    """Clean up all temporary resources"""
    tf.keras.backend.clear_session()
    if hasattr(app.state, 'temp_files'):
        for f in app.state.temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except Exception:
                pass
        app.state.temp_files = []

# Optimized FaceCropper
class FaceCropper:
    __instance = None
    
    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance._initialized = False
        return cls.__instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            logger.info("DeepFace detector initialized")

    def safe_crop(self, img_array: np.ndarray) -> np.ndarray:
        try:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(
                img_path=img_rgb,
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )
            result = cv2.resize(faces[0]['face'] if faces else img_array, TARGET_SIZE)
            del img_rgb, faces
            return result
        except Exception as e:
            logger.warning(f"Face processing fallback: {str(e)}")
            return cv2.resize(img_array, TARGET_SIZE)

# Image processing
def is_valid_image(file_bytes):
    return len(file_bytes) <= MAX_FILE_SIZE and imghdr.what(None, h=file_bytes) in ['jpeg', 'jpg', 'png']

def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) / 255.0)
    return cv2.resize(img, TARGET_SIZE)

# API endpoints
@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
    if len(files) > 5:
        raise HTTPException(413, "Maximum 5 files allowed")

    async def process_file(file: UploadFile):
        result = {
            "filename": file.filename,
            "class": None,
            "confidence": None,
            "probabilities": None,
            "error": None
        }
        
        try:
            file_bytes = await file.read()
            if not is_valid_image(file_bytes):
                raise ValueError("Invalid image format or size")
            
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid image data")
            
            # Process with cleanup
            full_img = preprocess_image(img)
            face_img = app.state.cropper.safe_crop(img)
            
            # Convert to mixed precision
            full_img = tf.cast(full_img, policy.compute_dtype)
            face_img = tf.cast(face_img, policy.compute_dtype)
            
            prediction = app.state.model.predict([
                np.array([full_img]), 
                np.array([face_img])
            ], verbose=0)[0]
            
            # Cleanup
            del full_img, face_img, img, nparr, file_bytes
            
            result.update({
                "class": CLASS_NAMES[np.argmax(prediction)],
                "confidence": float(np.max(prediction)),
                "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, prediction)}
            })
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {file.filename}: {str(e)}")
        
        return result
    
    try:
        results = await asyncio.gather(*(process_file(f) for f in files))
        cleanup_resources()
        return results
    except Exception as e:
        cleanup_resources()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, "Processing failed")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    model_size = None
    if hasattr(app.state, 'model'):
        model_size = os.path.getsize(os.getenv('MODEL_PATH')) / (1024 * 1024)
    
    return {
        "status": "healthy" if hasattr(app.state, 'model') else "unhealthy",
        "model_loaded": hasattr(app.state, 'model'),
        "model_size_mb": round(model_size, 2) if model_size else None
    }

@app.on_event("startup")
async def startup_event():
    try:
        # Configure TensorFlow
        tf.config.optimizer.set_experimental_options({
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': False,
            'arithmetic_optimization': True,
        })
        
        # Load model with custom objects
        model_path = os.getenv('MODEL_PATH')
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("Model file not found")
            
        custom_objects = {
            'EfficientChannelAttention': EfficientChannelAttention,
            'FixedSpatialAttention': FixedSpatialAttention,
            'FixedHybridBlock': FixedHybridBlock
        }
        
        app.state.model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects
        )
        app.state.cropper = FaceCropper()
        app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        app.state.temp_files = []
        
        # Warm up model
        dummy_input = [
            tf.cast(np.zeros((1, *TARGET_SIZE, 3)), policy.compute_dtype),
            tf.cast(np.zeros((1, *TARGET_SIZE, 3)), policy.compute_dtype)
        ]
        app.state.model.predict(dummy_input, steps=1)
        del dummy_input
        
        logger.info("Service initialized with mixed precision support")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Initialization error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    cleanup_resources()
    if hasattr(app.state, 'executor'):
        app.state.executor.shutdown(wait=False)
    logger.info("Service shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        limit_concurrency=2
    )
