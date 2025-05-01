import os
import sys
import asyncio
import logging
import numpy as np
import cv2
import imghdr
import shutil
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
    version="2.0"
)

# Constants
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 10485760))  # 10MB
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 2))
MODEL_INPUT_SHAPE = (1, *TARGET_SIZE, 3)

# Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float
    probabilities: Dict[str, float]
    error: Optional[str] = None
    disk_usage_before: Dict[str, int]
    disk_usage_after: Dict[str, int]
    processing_time_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    model_exists: bool
    system_info: Dict[str, str]

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

def get_disk_usage(path: str) -> int:
    """Get total size of directory in bytes"""
    try:
        return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
    except:
        return 0

def get_system_info() -> Dict[str, str]:
    """Get basic system information without psutil"""
    return {
        "cpu_count": str(os.cpu_count()),
        "total_memory": "Not available",
        "disk_usage": "Not available",
        "python_version": sys.version,
        "platform": sys.platform
    }

def is_valid_image(file_bytes) -> bool:
    """Validate image format"""
    try: 
        return imghdr.what(None, h=file_bytes) in ['jpeg', 'jpg', 'png']
    except: 
        return False

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Optimized image preprocessing pipeline"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) * (1./255.0)  # Faster than division
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)  # Faster interpolation
    return img

def extract_face(img: np.ndarray) -> np.ndarray:
    """Optimized face extraction"""
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend='opencv',
            enforce_detection=False,
            align=False,
            grayscale=False
        )
        return faces[0]['face'] if faces else img
    except:
        return img

def process_image_in_memory(file_bytes: bytes, resource_mgr: RequestResourceManager) -> Tuple[np.ndarray, np.ndarray]:
    """Process image without disk I/O with optimizations"""
    try:
        # Decode image with optimized parameters
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise ValueError("Invalid image data")
        
        # Process full image
        full_img = preprocess_image(img)
        
        # Extract and process face
        face_img = extract_face(img)
        face_img = preprocess_image(face_img)
        
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

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    model_loaded = hasattr(app.state, 'model') and app.state.model is not None
    model_path = os.environ.get('MODEL_PATH')
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_path": model_path,
        "model_exists": os.path.exists(model_path) if model_path else False,
        "system_info": get_system_info()
    }

@app.post("/predict", response_model=List[PredictionResult])
async def predict(files: List[UploadFile] = File(...)):
    if len(files) > 10:
        raise HTTPException(413, "Maximum 10 files allowed")
    
    async def process_wrapper(file: UploadFile):
        start_time = asyncio.get_event_loop().time()
        resource_mgr = RequestResourceManager()
        
        # Get disk usage before processing
        disk_before = {
            "app": get_disk_usage("/app"),
            "tmp": get_disk_usage("/tmp")
        }
        
        result = {
            "filename": file.filename,
            "class": None,
            "confidence": None,
            "probabilities": None,
            "error": None,
            "disk_usage_before": disk_before,
            "disk_usage_after": {},
            "processing_time_ms": 0
        }
        
        try:
            # Validate input
            file_bytes = await file.read()
            if len(file_bytes) > MAX_FILE_SIZE:
                raise ValueError("File size exceeds limit")
            if not is_valid_image(file_bytes):
                raise ValueError("Invalid image format")
            
            # Process image
            full_img, face_img = process_image_in_memory(file_bytes, resource_mgr)
            
            # Predict with batch size 1 for memory efficiency
            prediction = await asyncio.get_event_loop().run_in_executor(
                app.state.executor,
                lambda: app.state.model.predict(
                    [np.expand_dims(full_img, axis=0), 
                    np.expand_dims(face_img, axis=0)],
                    verbose=0,
                    batch_size=1
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
            logger.error(f"Error processing {file.filename}: {str(e)}", exc_info=True)
            
        finally:
            # Get disk usage after processing
            result["disk_usage_after"] = {
                "app": get_disk_usage("/app"),
                "tmp": get_disk_usage("/tmp")
            }
            
            # Calculate processing time
            end_time = asyncio.get_event_loop().time()
            result["processing_time_ms"] = round((end_time - start_time) * 1000, 2)
            
            # Cleanup resources
            resource_mgr.cleanup()
            
        return result
    
    # Process files sequentially to avoid memory spikes (but using async for each)
    results = []
    for file in files:
        results.append(await process_wrapper(file))
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        limit_concurrency=2,
        timeout_keep_alive=30,
        log_level="info"
    )
