import os
import sys
import asyncio
import logging
import numpy as np
import cv2
import imghdr
from concurrent.futures import ThreadPoolExecutor
import shutil
from typing import List, Dict, Optional

# Configure environment before any imports
os.environ.update({
    'DEEPFACE_HOME': '/app/.deepface',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'TF_ENABLE_ONEDNN_OPTS': '1',
    'DEEPFACE_CACHE': '0'
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from deepface import DeepFace
from pydantic import BaseModel, Field
from layers import (
    EfficientChannelAttention,
    FixedSpatialAttention,
    FixedHybridBlock
)

# Set mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake images with build-time initialization",
    version="1.0"
)

# Constants
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10485760))  # 10MB
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 2))

# Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]
    error: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    model_exists: bool

class RequestResourceManager:
    """Manages resources for a single request"""
    def __init__(self):
        self.temp_arrays = []
    
    def add_temp_array(self, arr):
        self.temp_arrays.append(arr)
    
    def cleanup(self):
        tf.keras.backend.clear_session()
        for arr in self.temp_arrays: del arr
        self.temp_arrays.clear()

def is_valid_image(file_bytes):
    """Validate image format"""
    try: return imghdr.what(None, h=file_bytes) in ['jpeg', 'jpg', 'png']
    except: return False

def process_image_in_memory(file_bytes, resource_mgr):
    """Process image without disk I/O"""
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: raise ValueError("Invalid image data")
        
        full_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        full_img = full_img.astype(np.float32) / 255.0
        full_img = cv2.resize(full_img, TARGET_SIZE)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend='opencv',
            enforce_detection=False,
            align=False
        )
        face_img = cv2.resize(faces[0]['face'] if faces else img, TARGET_SIZE)
        
        resource_mgr.add_temp_array(full_img)
        resource_mgr.add_temp_array(face_img)
        
        return full_img, face_img
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    try:
        # Configure TensorFlow
        tf.config.optimizer.set_experimental_options({
            'constant_folding': True,
            'shape_optimization': True,
            'arithmetic_optimization': True,
        })
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        # Load main model with custom layers
        custom_objects = {
            'EfficientChannelAttention': EfficientChannelAttention,
            'FixedSpatialAttention': FixedSpatialAttention,
            'FixedHybridBlock': FixedHybridBlock
        }
        app.state.model = tf.keras.models.load_model(
            os.environ['MODEL_PATH'],
            custom_objects=custom_objects
        )
        
        # Warm up model with correct input shape
        dummy_input = [
            np.zeros((1, *TARGET_SIZE, 3)),  # Full image
            np.zeros((1, *TARGET_SIZE, 3))    # Face image
        ]
        app.state.model.predict(dummy_input, verbose=0)
        
        # Initialize thread pool
        app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        logger.info("Service initialized and ready")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'executor'):
        app.state.executor.shutdown(wait=False)
    tf.keras.backend.clear_session()
    logger.info("Service shutdown complete")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    model_loaded = hasattr(app.state, 'model') and app.state.model is not None
    model_path = os.environ.get('MODEL_PATH')
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_path": model_path,
        "model_exists": os.path.exists(model_path) if model_path else False
    }

@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
    if len(files) > 10: raise HTTPException(413, "Maximum 10 files allowed")
    
    async def process_wrapper(file: UploadFile):
        resource_mgr = RequestResourceManager()
        result = {"filename": file.filename, "class": None, 
                 "confidence": None, "probabilities": None, "error": None}
        try:
            # Validate input
            file_bytes = await file.read()
            if len(file_bytes) > MAX_FILE_SIZE:
                raise ValueError("File size exceeds limit")
            if not is_valid_image(file_bytes):
                raise ValueError("Invalid image format")
            
            # Process image
            full_img, face_img = process_image_in_memory(file_bytes, resource_mgr)
            
            # Predict
            prediction = await asyncio.get_event_loop().run_in_executor(
                app.state.executor,
                lambda: app.state.model.predict(
                    [np.array([full_img]), np.array([face_img])],
                    verbose=0
                )[0]
            )
            
            # Format results
            result.update({
                "class": CLASS_NAMES[np.argmax(prediction)],
                "confidence": float(np.max(prediction)),
                "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, prediction)}
            })
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing {file.filename}: {str(e)}")
        finally:
            resource_mgr.cleanup()
        return result
    
    return await asyncio.gather(*(process_wrapper(f) for f in files))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        limit_concurrency=4,
    )
