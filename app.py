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

# Configure environment before any imports
os.environ.update({
    'DEEPFACE_HOME': '/app/.deepface',
    'CUDA_VISIBLE_DEVICES': '-1',
    'TF_CPP_MIN_LOG_LEVEL': '2',
    'TF_ENABLE_ONEDNN_OPTS': '1',
    'DEEPFACE_CACHE': '0',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1'
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from deepface import DeepFace
from pydantic import BaseModel, Field
from layers_helper import (
    EfficientChannelAttention,
    FixedSpatialAttention,
    FixedHybridBlock
)

# Disable TensorFlow warnings
tf.get_logger().setLevel('ERROR')

app = FastAPI(
    title="Deepfake Detection API",
    description="Optimized API for detecting deepfake images on CPU",
    version="2.1"
)

# Constants
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10485760))  # 10MB
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 4))
MODEL_INPUT_SHAPE = (1, *TARGET_SIZE, 3)

# Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]
    error: Optional[str] = None

# Update the HealthCheckResponse model
class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    directories: Dict[str, Dict]
    
    
class RequestResourceManager:
    """Manages resources for a single request"""
    def __init__(self):
        self.temp_arrays = []
        self.temp_files = []
    
    def add_temp_array(self, arr):
        self.temp_arrays.append(arr)
    
    def add_temp_file(self, filepath):
        self.temp_files.append(filepath)
    
    def cleanup(self):
        tf.keras.backend.clear_session()
        for arr in self.temp_arrays: 
            del arr
        for filepath in self.temp_files:
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
            except:
                pass
        self.temp_arrays.clear()
        self.temp_files.clear()

def is_valid_image(file_bytes) -> bool:
    """Validate image format"""
    try: 
        return imghdr.what(None, h=file_bytes) in ['jpeg', 'jpg', 'png']
    except: 
        return False

def ensure_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure image is in uint8 format"""
    if img.dtype != np.uint8:
        if img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Optimized image preprocessing pipeline"""
    try:
        # Ensure proper image format
        img = ensure_uint8(img)
        
        # Convert color space and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) * (1./255.0)
        img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        return img
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise ValueError("Image preprocessing failed")

def extract_face(img: np.ndarray) -> np.ndarray:
    """Optimized and robust face extraction"""
    try:
        # Ensure proper image format for DeepFace
        img_rgb = cv2.cvtColor(ensure_uint8(img), cv2.COLOR_BGR2RGB)
        
        faces = DeepFace.extract_faces(
            img_path=img_rgb,
            detector_backend='opencv',
            enforce_detection=False,
            align=False,
            grayscale=False
        )
        
        if faces and 'face' in faces[0]:
            face_img = faces[0]['face']
            # Ensure face image is in correct format
            if isinstance(face_img, np.ndarray):
                return face_img
        return img
    except Exception as e:
        logger.warning(f"Face extraction failed, using full image: {str(e)}")
        return img

def process_image_in_memory(file_bytes: bytes, resource_mgr: RequestResourceManager) -> Tuple[np.ndarray, np.ndarray]:
    """Process image with robust error handling"""
    try:
        # Decode image
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        
        # Process full image with error handling
        try:
            full_img = preprocess_image(img)
        except Exception as e:
            logger.error(f"Full image processing failed: {str(e)}")
            raise ValueError("Full image processing failed")
        
        # Process face image with fallback
        try:
            face_img = extract_face(img)
            face_img = preprocess_image(face_img)
        except Exception as e:
            logger.warning(f"Face processing failed, using full image: {str(e)}")
            face_img = full_img.copy()
        
        resource_mgr.add_temp_array(full_img)
        resource_mgr.add_temp_array(face_img)
        
        return full_img, face_img
        
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    try:
        # Configure TensorFlow for CPU optimization
        tf.config.optimizer.set_experimental_options({
            'constant_folding': True,
            'shape_optimization': True,
            'arithmetic_optimization': True,
            'disable_meta_optimizer': False,
            'remapping': True,
        })
        
        # Set thread configuration
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        # Disable TensorFlow features that consume memory
        tf.config.optimizer.set_jit(False)
        
        # Load main model with custom layers
        custom_objects = {
            'EfficientChannelAttention': EfficientChannelAttention,
            'FixedSpatialAttention': FixedSpatialAttention,
            'FixedHybridBlock': FixedHybridBlock
        }
        
        # Load model with optimized settings
        app.state.model = tf.keras.models.load_model(
            os.environ['MODEL_PATH'],
            custom_objects=custom_objects,
            compile=False
        )
        
        # Disable training mode
        app.state.model.trainable = False
        
        # Warm up model with correct input shape
        dummy_input = [
            np.zeros(MODEL_INPUT_SHAPE, dtype=np.float32),
            np.zeros(MODEL_INPUT_SHAPE, dtype=np.float32)
        ]
        app.state.model.predict(dummy_input, verbose=0, batch_size=1)
        
        # Initialize thread pool with limited workers
        app.state.executor = ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix='deepfake_worker'
        )
        
        logger.info(f"Service initialized. Model input shape: {MODEL_INPUT_SHAPE}")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'executor'):
        app.state.executor.shutdown(wait=False)
    tf.keras.backend.clear_session()
    logger.info("Service shutdown complete")



# Add this new function to scan directories
def scan_directory(path: str) -> Dict:
    """Recursively scan directory and return structure with sizes"""
    path = Path(path)
    if not path.exists():
        return {"error": f"Path {path} does not exist"}
    
    result = {
        "path": str(path),
        "type": "directory",
        "size_mb": 0,
        "contents": []
    }
    
    try:
        total_size = 0
        for item in path.iterdir():
            item_info = {
                "name": item.name,
                "path": str(item)
            }
            
            if item.is_dir():
                subdir = scan_directory(str(item))
                item_info.update({
                    "type": "directory",
                    "size_mb": subdir["size_mb"],
                    "contents": subdir["contents"]
                })
                total_size += subdir["size_mb"]
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                item_info.update({
                    "type": "file",
                    "size_mb": round(size_mb, 4)
                })
                total_size += size_mb
            
            result["contents"].append(item_info)
        
        result["size_mb"] = round(total_size, 4)
    except Exception as e:
        result["error"] = str(e)
    
    return result


# Update the health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    model_loaded = hasattr(app.state, 'model') and app.state.model is not None
    
    # Scan important directories
    directories = {
        "temp": scan_directory("/temp"),
        "app": scan_directory("/app"),
    }
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "directories": directories
    }



@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
  
    # Dynamic batch sizing based on available capacity
    BATCH_SIZE = min(10, len(files))  # Can adjust up to 4 based on your vCPU headroom
    results = [None] * len(files)
    
    async def process_batch(batch_files):
        batch_results = []
        resource_mgrs = []
        
        try:
            # Prepare batch data
            batch_inputs = [[], []]  # [full_imgs, face_imgs]
            
            for file in batch_files:
                resource_mgr = RequestResourceManager()
                resource_mgrs.append(resource_mgr)
                
                try:
                    file_bytes = await file.read()
                    if len(file_bytes) > MAX_FILE_SIZE:
                        raise ValueError("File size exceeds limit")
                    if not is_valid_image(file_bytes):
                        raise ValueError("Invalid image format")
                    
                    # Process image
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("Invalid image data")
                    
                    # Process full image
                    full_img = preprocess_image(ensure_uint8(img))
                    
                    # Try face extraction
                    face_img = full_img  # Default to full image
                    try:
                        face_img_rgb = cv2.cvtColor(ensure_uint8(img), cv2.COLOR_BGR2RGB)
                        faces = DeepFace.extract_faces(
                            img_path=face_img_rgb,
                            detector_backend='opencv',
                            enforce_detection=False,
                            align=False,
                            grayscale=False
                        )
                        if faces and 'face' in faces[0] and isinstance(faces[0]['face'], np.ndarray):
                            face_img = preprocess_image(faces[0]['face'])
                    except Exception:
                        pass  # Use full_img as fallback
                    
                    batch_inputs[0].append(full_img)
                    batch_inputs[1].append(face_img)
                    
                except Exception as e:
                    batch_results.append({
                        "filename": file.filename,
                        "class": "ERROR",
                        "confidence": 0.0,
                        "probabilities": dict.fromkeys(CLASS_NAMES, 0.0),
                        "error": str(e)
                    })
            
            # Batch prediction
            if batch_inputs[0]:  # If we have valid inputs
                predictions = await asyncio.get_event_loop().run_in_executor(
                    app.state.executor,
                    lambda: app.state.model.predict(
                        [np.array(batch_inputs[0]), np.array(batch_inputs[1])],
                        verbose=0,
                        batch_size=BATCH_SIZE
                    )
                )
                
                for i, prediction in enumerate(predictions):
                    if prediction is not None and len(prediction) == len(CLASS_NAMES):
                        batch_results.append({
                            "filename": batch_files[i].filename,
                            "class": CLASS_NAMES[np.argmax(prediction)],
                            "confidence": float(np.max(prediction)),
                            "probabilities": dict(zip(CLASS_NAMES, prediction.astype(float).tolist())),
                            "error": None
                        })
                    else:
                        batch_results.append({
                            "filename": batch_files[i].filename,
                            "class": "ERROR",
                            "confidence": 0.0,
                            "probabilities": dict.fromkeys(CLASS_NAMES, 0.0),
                            "error": "Invalid prediction result"
                        })
                        
        finally:
            for resource_mgr in resource_mgrs:
                resource_mgr.cleanup()
            tf.keras.backend.clear_session()
            
        return batch_results

    # Process in batches
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i+BATCH_SIZE]
        batch_results = await process_batch(batch)
        for j, result in enumerate(batch_results):
            results[i+j] = result
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        limit_concurrency=4,
        timeout_keep_alive=60,
        log_level="info"
    )
