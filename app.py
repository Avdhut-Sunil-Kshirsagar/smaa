import os
import sys
import asyncio
import logging
import numpy as np
import cv2
import imghdr
from concurrent.futures import ThreadPoolExecutor
import shutil
from pathlib import Path

# Configure environment before any imports
os.environ.update({
    'DEEPFACE_HOME': '/tmp/.deepface',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'TF_ENABLE_ONEDNN_OPTS': '1',
    'DEEPFACE_CACHE': '0'  # Disable DeepFace caching
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras import layers, models
from deepface import DeepFace
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# Set mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

app = FastAPI(
    title="Deepfake Detection API",
    description="API with strict resource management for detecting deepfake images",
    version="1.0"
)

# Constants
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_WORKERS = 2

# Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]
    error: Optional[str] = None

# Custom layers (unchanged from your working version)
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

class RequestResourceManager:
    """Manages all resources for a single request"""
    def __init__(self):
        self.temp_files = []
        self.temp_arrays = []
        self.cleanup_complete = False
    
    def add_temp_file(self, path):
        self.temp_files.append(path)
    
    def add_temp_array(self, arr):
        self.temp_arrays.append(arr)
    
    def cleanup(self):
        """Ensure all resources are released"""
        if self.cleanup_complete:
            return
            
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        
        # Delete numpy arrays
        for arr in self.temp_arrays:
            del arr
        
        # Remove temporary files
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except:
                pass
                
        # Clear DeepFace cache
        cache_path = os.path.join(os.environ['DEEPFACE_HOME'], '.deepface')
        if os.path.exists(cache_path):
            try:
                shutil.rmtree(cache_path)
            except:
                pass
                
        self.cleanup_complete = True
        logger.info("Request resources cleaned up")

def is_valid_image(file_bytes):
    """Validate image without creating temporary files"""
    try:
        if len(file_bytes) > MAX_FILE_SIZE:
            return False
        return imghdr.what(None, h=file_bytes) in ['jpeg', 'jpg', 'png']
    except:
        return False

def process_image_in_memory(file_bytes, resource_mgr):
    """Process image entirely in memory"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        
        # Preprocess full image
        full_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        full_img = full_img.astype(np.float32) / 255.0
        full_img = cv2.resize(full_img, TARGET_SIZE)
        
        # Process face crop in memory
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = DeepFace.extract_faces(
            img_path=img_rgb,  # Pass array directly
            detector_backend='opencv',
            enforce_detection=False,
            align=False
        )
        face_img = cv2.resize(faces[0]['face'] if faces else img, TARGET_SIZE)
        
        # Track resources
        resource_mgr.add_temp_array(full_img)
        resource_mgr.add_temp_array(face_img)
        
        return full_img, face_img
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    try:
        # Configure TensorFlow for minimal resource usage
        tf.config.optimizer.set_experimental_options({
            'constant_folding': True,
            'shape_optimization': True,
            'arithmetic_optimization': True,
        })
        
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        # Load model with custom objects
        model_path = os.getenv('MODEL_PATH')
        custom_objects = {
            'EfficientChannelAttention': EfficientChannelAttention,
            'FixedSpatialAttention': FixedSpatialAttention,
            'FixedHybridBlock': FixedHybridBlock
        }
        
        app.state.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Warm up model
        dummy_input = np.zeros((1, *TARGET_SIZE, 3))
        app.state.model.predict([dummy_input, dummy_input], steps=1)
        
        logger.info("Service initialized with strict resource management")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Initialization error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'executor'):
        app.state.executor.shutdown(wait=False)
    tf.keras.backend.clear_session()
    logger.info("Service shutdown complete")

@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(413, "Maximum 10 files allowed")
    
    async def process_wrapper(file: UploadFile):
        # Create resource manager for this request
        resource_mgr = RequestResourceManager()
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
            
            # Process in memory
            full_img, face_img = process_image_in_memory(file_bytes, resource_mgr)
            
            # Predict
            prediction = app.state.model.predict([
                np.array([full_img]), 
                np.array([face_img])
            ], verbose=0)[0]
            
            result.update({
                "class": CLASS_NAMES[np.argmax(prediction)],
                "confidence": float(np.max(prediction)),
                "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, prediction)}
            })
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {file.filename}: {str(e)}")
        finally:
            # Ensure cleanup happens even if error occurs
            resource_mgr.cleanup()
            del resource_mgr
            
        return result
    
    try:
        return await asyncio.gather(*(process_wrapper(f) for f in files))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, "Processing failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        limit_concurrency=4,
    )
