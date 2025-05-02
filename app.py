import os
import sys
import asyncio
import logging
import numpy as np
import cv2
import imghdr
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Configure environment for maximum CPU efficiency
os.environ.update({
    'DEEPFACE_HOME': '/app/.deepface',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TF_CPP_MIN_LOG_LEVEL': '2',  # Suppress TensorFlow logs
    'TF_ENABLE_ONEDNN_OPTS': '1',
    'DEEPFACE_CACHE': '0',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'TF_NUM_INTEROP_THREADS': '1',
    'TF_NUM_INTRAOP_THREADS': '1'
})

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduced from INFO to WARNING
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
from deepface import DeepFace
from pydantic import BaseModel, Field

app = FastAPI(
    title="Deepfake Detection API",
    description="Ultra-optimized API for detecting deepfake images on low-CPU environments",
    version="3.0"
)

# Constants optimized for low-resource environment
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = 5 * 1024 * 1024  # Reduced to 5MB for memory safety
MAX_WORKERS = 2  # Conservative for 0.5 vCPU
MODEL_INPUT_SHAPE = (1, *TARGET_SIZE, 3)
BATCH_SIZE = 2  # Optimal for memory constraints

# Simplified Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]

class HealthCheckResponse(BaseModel):
    status: str
    model_ready: bool

# Memory-efficient image processing
def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Ultra-optimized image preprocessing"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0

async def process_single_image(file: UploadFile) -> PredictionResult:
    """Streamlined single image processing pipeline"""
    try:
        # Read and validate
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError("File too large")
        
        if not imghdr.what(None, h=file_bytes) in ['jpeg', 'jpg', 'png']:
            raise ValueError("Invalid image format")

        # Decode and preprocess
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")

        # Process full image
        full_img = preprocess_image(img)
        
        # Face extraction with immediate fallback
        face_img = full_img
        try:
            face_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = DeepFace.extract_faces(
                img_path=face_img_rgb,
                detector_backend='opencv',
                enforce_detection=False,
                align=False,
                grayscale=False
            )
            if faces and isinstance(faces[0].get('face'), np.ndarray):
                face_img = preprocess_image(faces[0]['face'])
        except Exception:
            pass

        # Predict
        prediction = app.state.model.predict(
            [np.expand_dims(full_img, axis=0), 
            np.expand_dims(face_img, axis=0)],
            verbose=0,
            batch_size=1
        )[0]

        return {
            "filename": file.filename,
            "class": CLASS_NAMES[np.argmax(prediction)],
            "confidence": float(np.max(prediction)),
            "probabilities": dict(zip(CLASS_NAMES, prediction.astype(float).tolist()))
        }

    except Exception as e:
        logger.warning(f"Processing failed for {file.filename}: {str(e)}")
        return {
            "filename": file.filename,
            "class": "ERROR",
            "confidence": 0.0,
            "probabilities": {name: 0.0 for name in CLASS_NAMES}
        }

@app.on_event("startup")
async def startup_event():
    """Ultra-lean startup configuration"""
    try:
        # Minimal TensorFlow configuration
        tf.config.optimizer.set_experimental_options({
            'constant_folding': True,
            'disable_meta_optimizer': True  # Faster startup
        })
        
        # Load model with absolute minimum overhead
        app.state.model = tf.keras.models.load_model(
            os.environ['MODEL_PATH'],
            compile=False
        )
        app.state.model.trainable = False

        # Warm up with tiny input
        dummy = np.zeros((1, *TARGET_SIZE, 3), dtype=np.float32)
        app.state.model.predict([dummy, dummy], verbose=0, batch_size=1)

        # Conservative resource pool
        app.state.executor = ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix='df_worker'
        )

    except Exception as e:
        logger.critical(f"Startup failed: {str(e)}")
        sys.exit(1)

@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
    """Optimized prediction endpoint"""
    if len(files) > 10:
        raise HTTPException(413, "Maximum 10 files allowed")

    # Process files with controlled concurrency
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    
    async def limited_process(file):
        async with semaphore:
            return await process_single_image(file)
    
    return await asyncio.gather(*[limited_process(file) for file in files])

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return {
        "status": "ready",
        "model_ready": hasattr(app.state, 'model')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for 0.5 vCPU
        limit_concurrency=MAX_WORKERS,
        timeout_keep_alive=30,  # Reduced from 60
        log_level="warning"  # Reduced logging overhead
    )
