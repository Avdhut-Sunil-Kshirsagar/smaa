import os
import sys
import asyncio
import tempfile
import logging
import numpy as np
import cv2
import imghdr
from concurrent.futures import ThreadPoolExecutor

# Configure environment before any imports
os.environ.update({
    'DEEPFACE_HOME': '/tmp/.deepface',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'TF_ENABLE_ONEDNN_OPTS': '1'
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from deepface import DeepFace
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

# Set mixed precision policy (must keep as-is)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake, real and AI-generated images (CPU Optimized)",
    version="1.0",
    contact={
        "name": "API Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    }
)

# Constants from environment
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10 * 1024 * 1024))
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 2))

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
    model_path: Optional[str]
    model_exists: Optional[bool]

# Custom layers (must keep original implementation)
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

# Resource management functions
def cleanup_tensors():
    """Explicitly clear TensorFlow tensors and Keras backend"""
    tf.keras.backend.clear_session()
    if hasattr(app.state, 'temp_arrays'):
        for arr in app.state.temp_arrays:
            del arr
        app.state.temp_arrays = []

def cleanup_files():
    """Clean up any temporary files"""
    if hasattr(app.state, 'temp_files'):
        for f in app.state.temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except:
                pass
        app.state.temp_files = []

def cleanup_resources():
    """Clean all temporary resources"""
    cleanup_tensors()
    cleanup_files()
    if hasattr(app.state, 'last_images'):
        del app.state.last_images
    if hasattr(app.state, 'last_predictions'):
        del app.state.last_predictions

# FaceCropper with resource management
class FaceCropper:
    def __init__(self):
        logger.info("DeepFace detector initialized successfully.")

    def safe_crop(self, img_array: np.ndarray) -> np.ndarray:
        """Process image with cleanup"""
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
            logger.error(f"Face processing error: {str(e)}")
            return cv2.resize(img_array, TARGET_SIZE)

# Image processing with cleanup
def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """Process image with cleanup"""
    try:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        result = cv2.resize(img, TARGET_SIZE)
        del img
        return result
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return np.zeros((*TARGET_SIZE, 3), dtype=np.float32)

def is_valid_image(file_bytes):
    try:
        if len(file_bytes) > MAX_FILE_SIZE:
            return False
        image_type = imghdr.what(None, h=file_bytes)
        return image_type in ['jpeg', 'jpg', 'png', 'bmp']
    except Exception:
        return False

# File processing with resource management
def process_single_file(file: UploadFile):
    """Process file with explicit cleanup"""
    result = {
        "filename": file.filename,
        "class": None,
        "confidence": None,
        "probabilities": None,
        "error": None
    }
    
    try:
        # Read and validate
        file_bytes = file.file.read(MAX_FILE_SIZE + 1)
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError("File size exceeds limit")
            
        if not is_valid_image(file_bytes):
            raise ValueError("Invalid image format")
        
        # Convert to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        
        # Process with cleanup
        full_img = preprocess_image(img)
        face_img = app.state.cropper.safe_crop(img)
        
        # Store temp arrays for later cleanup
        if not hasattr(app.state, 'temp_arrays'):
            app.state.temp_arrays = []
        app.state.temp_arrays.extend([full_img, face_img])
        
        # Predict with cleanup
        prediction = app.state.model.predict([
            np.array([full_img]), 
            np.array([face_img])
        ], verbose=0)[0]
        
        # Cleanup immediately
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

@app.on_event("startup")
async def startup_event():
    try:
        # Configure TensorFlow
        tf.config.optimizer.set_experimental_options({
            'layout_optimizer': False,
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': False,
            'arithmetic_optimization': True,
        })
        
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        # Load model with custom objects
        model_path = os.getenv('MODEL_PATH')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        custom_objects = {
            'EfficientChannelAttention': EfficientChannelAttention,
            'FixedSpatialAttention': FixedSpatialAttention,
            'FixedHybridBlock': FixedHybridBlock
        }
        
        app.state.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        app.state.cropper = FaceCropper()
        app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Initialize resource tracking
        app.state.temp_files = []
        app.state.temp_arrays = []
        
        # Warm up model
        dummy_input = [
            tf.convert_to_tensor(np.zeros((1, *TARGET_SIZE, 3))),
            tf.convert_to_tensor(np.zeros((1, *TARGET_SIZE, 3)))
        ]
        app.state.model.predict(dummy_input, steps=1)
        
        logger.info("Service initialized with resource tracking")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Initialization error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    cleanup_resources()
    if hasattr(app.state, 'executor'):
        app.state.executor.shutdown(wait=False)
    logger.info("Service shutdown complete")

@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(413, "Maximum 10 files allowed")
    
    async def process_wrapper(file: UploadFile):
        return await asyncio.get_event_loop().run_in_executor(
            app.state.executor,
            process_single_file,
            file
        )
    
    try:
        results = await asyncio.gather(*(process_wrapper(f) for f in files))
        cleanup_resources()  # Explicit cleanup after processing
        return results
    except Exception as e:
        cleanup_resources()  # Cleanup even on error
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, "Processing failed")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        model_exists = os.path.exists(os.getenv('MODEL_PATH'))
        return {
            "status": "healthy",
            "model_loaded": hasattr(app.state, "model"),
            "model_path": os.getenv('MODEL_PATH'),
            "model_exists": model_exists
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(503, detail=f"Service unavailable: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        limit_concurrency=4,
    )
