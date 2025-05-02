import os
import sys
import asyncio
import logging
import numpy as np
import cv2
import imghdr
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf

# Environment configuration
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'CUDA_VISIBLE_DEVICES': '-1',
    'OMP_NUM_THREADS': '1',
    'TF_NUM_INTRAOP_THREADS': '1',
    'TF_NUM_INTEROP_THREADS': '1'
})

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="UltraFast Deepfake API", version="3.0")

# Constants
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MODEL_INPUT_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
BATCH_SIZE = 8  # Optimized for CPU cache
MAX_CONCURRENT_BATCHES = 2

# Pydantic models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionResult(BaseModel):
    results: List[PredictionResult]

# Optimized preprocessing
def optimized_preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0

# Face detector using OpenCV Haar cascades (faster than DeepFace)
class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return img[y:y+h, x:x+w]
        return img

# Core processing class
class DeepfakeProcessor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def warmup_model(self, model_path: str):
        """Warmup model with optimized settings"""
        self.model = tf.keras.models.load_model(
            model_path, compile=False
        )
        self.model.trainable = False
        
        # Initial warmup
        dummy = np.zeros((2, *MODEL_INPUT_SHAPE), dtype=np.float32)
        self.model.predict([dummy, dummy], batch_size=2, verbose=0)
        
    async def process_file(self, file: UploadFile):
        """Ultra-fast file processing pipeline"""
        try:
            # Read and validate
            img_data = await file.read()
            if len(img_data) > MAX_FILE_SIZE:
                return None, "File too large"
                
            # Decode and detect
            img = cv2.imdecode(
                np.frombuffer(img_data, np.uint8), 
                cv2.IMREAD_COLOR
            )
            if img is None:
                return None, "Invalid image"
                
            # Parallel processing
            full_img = await asyncio.get_event_loop().run_in_executor(
                self.executor, optimized_preprocess, img
            )
            face_img = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.face_detector.detect, img
            )
            face_img = await asyncio.get_event_loop().run_in_executor(
                self.executor, optimized_preprocess, face_img
            )
            
            return (full_img, face_img), None
            
        except Exception as e:
            return None, str(e)

# FastAPI endpoints
@app.on_event("startup")
async def startup():
    processor = DeepfakeProcessor()
    await processor.warmup_model(os.environ['MODEL_PATH'])
    app.state.processor = processor

@app.post("/predict", response_model=BatchPredictionResult)
async def predict(files: List[UploadFile] = File(...)):
        
    processor = app.state.processor
    results = []
    
    # Process files in parallel batches
    batch_tasks = []
    for i in range(0, len(files), BATCH_SIZE * MAX_CONCURRENT_BATCHES):
        batch_group = files[i:i+BATCH_SIZE * MAX_CONCURRENT_BATCHES]
        batch_tasks.append(
            process_batch(processor, batch_group)
        )
    
    # Collect and flatten results
    batch_results = await asyncio.gather(*batch_tasks)
    for br in batch_results:
        results.extend(br)
    
    return {"results": results}

async def process_batch(processor, files):
    """Process a batch of files with optimized pipeline"""
    processed = []
    predictions = []
    
    # Process all files in parallel
    process_tasks = [processor.process_file(f) for f in files]
    batch_results = await asyncio.gather(*process_tasks)
    
    # Prepare batch inputs
    full_imgs, face_imgs = [], []
    for result, error in batch_results:
        if error:
            predictions.append(create_error_result(result, error))
        else:
            full_imgs.append(result[0])
            face_imgs.append(result[1])
    
    # Batch predict if we have valid inputs
    if full_imgs:
        full_imgs = np.array(full_imgs)
        face_imgs = np.array(face_imgs)
        
        # Predict in parallel thread
        preds = await asyncio.get_event_loop().run_in_executor(
            processor.executor,
            lambda: processor.model.predict(
                [full_imgs, face_imgs],
                batch_size=BATCH_SIZE,
                verbose=0
            )
        )
        
        # Create results
        for i, pred in enumerate(preds):
            predictions.append(create_prediction(
                files[i].filename, pred
            ))
    
    return predictions

def create_prediction(filename: str, pred: np.ndarray) -> Dict:
    return {
        "filename": filename,
        "class": CLASS_NAMES[np.argmax(pred)],
        "confidence": float(np.max(pred)),
        "probabilities": dict(zip(CLASS_NAMES, pred.astype(float).tolist()))
    }

def create_error_result(filename: str, error: str) -> Dict:
    return {
        "filename": filename,
        "class": "ERROR",
        "confidence": 0.0,
        "probabilities": {c: 0.0 for c in CLASS_NAMES},
        "error": error
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=1000,
        timeout_keep_alive=30
    )
