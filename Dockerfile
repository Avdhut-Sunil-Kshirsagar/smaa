import sys
import os
import asyncio
os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
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

# Load configuration from environment variables
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/model/final_model_11_4_2025.keras')
TARGET_SIZE = tuple(map(int, os.environ.get('TARGET_SIZE', '224,224').split(',')))
MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', 10485760))  # 10MB
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 2))

app = FastAPI(
    title="Optimized Deepfake Detection API",
    description="CPU-optimized API for deepfake detection with strict resource constraints",
    version="2.0",
)

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

# Custom layers (keep identical but add memory cleanup)
class EfficientChannelAttention(layers.Layer):
    # Keep implementation same but add destructor
    def __del__(self):
        if hasattr(self, 'fc'):
            del self.fc
            del self.avg_pool
            del self.max_pool

# Similar destructors for other custom layers
class FixedSpatialAttention(layers.Layer):
    def __del__(self):
        if hasattr(self, 'conv'):
            del self.conv

class FixedHybridBlock(layers.Layer):
    def __del__(self):
        for attr in ['conv1', 'conv2', 'eca', 'sa', 'norm1', 'norm2']:
            if hasattr(self, attr):
                delattr(self, attr)

# Memory-optimized FaceCropper
class FaceCropper:
    def __init__(self):
        self.detector = DeepFace.build_model('Facenet')
        
    def safe_crop(self, img_array: np.ndarray) -> np.ndarray:
        try:
            faces = DeepFace.extract_faces(
                img_path=img_array,
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )
            if faces:
                face_img = faces[0]['face']
                resized = cv2.resize(face_img, TARGET_SIZE)
                del faces, face_img  # Manual cleanup
                return resized
            return cv2.resize(img_array, TARGET_SIZE)
        except Exception as e:
            return cv2.resize(img_array, TARGET_SIZE)
        finally:
            del img_array  # Force cleanup

# Memory-friendly image processing
def process_image(file: UploadFile):
    result = {"filename": file.filename, "error": None}
    try:
        if file.size > MAX_FILE_SIZE:
            raise ValueError("File size exceeded")
            
        file_bytes = file.file.read(MAX_FILE_SIZE)
        if not imghdr.what(None, file_bytes):
            raise ValueError("Invalid image")
            
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Process and immediately clear memory
        full_img = cv2.resize(img, TARGET_SIZE)
        face_img = app.state.cropper.safe_crop(img)
        del img, nparr, file_bytes  # Manual memory cleanup
        
        # Predict and format result
        prediction = app.state.model.predict([
            np.expand_dims(full_img, 0), 
            np.expand_dims(face_img, 0)
        ], verbose=0)[0]
        
        result.update({
            "class": ['AI', 'FAKE', 'REAL'][np.argmax(prediction)],
            "confidence": float(np.max(prediction)),
            "probabilities": dict(zip(['AI', 'FAKE', 'REAL'], prediction.astype(float)))
        })
        
    except Exception as e:
        result["error"] = str(e)
    finally:
        # Force garbage collection
        gc.collect()
        return result

@app.on_event("startup")
async def initialize():
    # Configure TensorFlow for low memory usage
    tf.config.optimizer.set_jit(True)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    
    # Load model once at startup
    if not hasattr(app.state, 'model'):
        logger.info("Loading optimized model...")
        app.state.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'EfficientChannelAttention': EfficientChannelAttention,
                'FixedSpatialAttention': FixedSpatialAttention,
                'FixedHybridBlock': FixedHybridBlock
            }
        )
        # Warmup with small tensor
        dummy = [np.zeros((1,*TARGET_SIZE,3)), np.zeros((1,*TARGET_SIZE,3))]
        app.state.model.predict(dummy, verbose=0)
        del dummy
        gc.collect()
        
    # Initialize face detector
    app.state.cropper = FaceCropper()
    app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

@app.on_event("shutdown")
def cleanup():
    if hasattr(app.state, 'model'):
        del app.state.model
        tf.keras.backend.clear_session()
    if hasattr(app.state, 'cropper'):
        del app.state.cropper
    gc.collect()

@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile]):
    if len(files) > 5:
        raise HTTPException(413, "Max 5 files allowed")
    
    futures = []
    for file in files:
        futures.append(
            asyncio.get_event_loop().run_in_executor(
                app.state.executor,
                process_image,
                file
            )
        )
    
    results = await asyncio.gather(*futures)
    
    # Aggressive cleanup after processing
    for f in files:
        if hasattr(f.file, 'close'):
            f.file.close()
    gc.collect()
    
    return results

@app.get("/health", response_model=HealthCheckResponse)
def health():
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
