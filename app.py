import os
import sys
import asyncio
import tempfile
import logging
import numpy as np
import cv2
import imghdr
from concurrent.futures import ThreadPoolExecutor
import psutil
import shutil
from datetime import datetime
from pathlib import Path

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
    description="API for detecting deepfake, real and AI-generated images with storage monitoring",
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
class StorageInfo(BaseModel):
    path: str
    size_mb: float
    files: List[str]
    directories: List[str]

class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class")
    confidence: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float]
    error: Optional[str] = None
    storage_before: Optional[Dict] = Field(None, description="Storage usage before processing")
    storage_after: Optional[Dict] = Field(None, description="Storage usage after processing")
    storage_diff: Optional[Dict] = Field(None, description="Storage differences")

    class Config:
        allow_population_by_field_name = True

class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str]
    model_exists: Optional[bool]
    storage: Dict[str, Union[int, float, str]]
    temp_dir_info: Optional[StorageInfo]
    deepface_dir_info: Optional[StorageInfo]

# Storage monitoring functions
def scan_directory(path: str) -> StorageInfo:
    """Scan a directory and return its contents and size"""
    path_obj = Path(path)
    if not path_obj.exists():
        return StorageInfo(path=path, size_mb=0, files=[], directories=[])
    
    total_size = 0
    files = []
    directories = []
    
    for item in path_obj.rglob('*'):
        try:
            if item.is_file():
                total_size += item.stat().st_size
                files.append(str(item.relative_to(path_obj)))
            elif item.is_dir():
                directories.append(str(item.relative_to(path_obj)))
        except Exception as e:
            logger.warning(f"Could not scan {item}: {str(e)}")
    
    return StorageInfo(
        path=path,
        size_mb=round(total_size / (1024 * 1024), 2),
        files=files,
        directories=directories
    )

def get_storage_info() -> Dict:
    """Get detailed storage information"""
    partitions = []
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            partitions.append({
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "percent_used": usage.percent
            })
        except Exception as e:
            logger.warning(f"Could not get partition info for {partition.mountpoint}: {str(e)}")
    
    return {
        "partitions": partitions,
        "timestamp": datetime.now().isoformat(),
        "temp_dir": scan_directory('/tmp'),
        "deepface_dir": scan_directory('/tmp/.deepface')
    }

def track_storage_changes(before: Dict, after: Dict) -> Dict:
    """Calculate storage changes between two states"""
    changes = {}
    
    # Track partition changes
    for before_part, after_part in zip(before['partitions'], after['partitions']):
        if before_part['mountpoint'] == after_part['mountpoint']:
            changes[before_part['mountpoint']] = {
                "used_diff_mb": round((after_part['used_gb'] - before_part['used_gb']) * 1024, 2),
                "free_diff_mb": round((after_part['free_gb'] - before_part['free_gb']) * 1024, 2)
            }
    
    # Track directory changes
    for dir_type in ['temp_dir', 'deepface_dir']:
        changes[f"{dir_type}_size_diff_mb"] = after[dir_type].size_mb - before[dir_type].size_mb
        changes[f"{dir_type}_files_diff"] = len(after[dir_type].files) - len(before[dir_type].files)
        changes[f"{dir_type}_dirs_diff"] = len(after[dir_type].directories) - len(before[dir_type].directories)
    
    return changes

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

# Resource management with enhanced storage tracking
class ResourceTracker:
    def __init__(self):
        self.temp_files = []
        self.temp_arrays = []
        self.temp_dirs = []
    
    def add_temp_file(self, path):
        self.temp_files.append(path)
    
    def add_temp_dir(self, path):
        self.temp_dirs.append(path)
    
    def add_temp_array(self, arr):
        self.temp_arrays.append(arr)
    
    def cleanup_tensors(self):
        """Explicitly clear TensorFlow tensors and Keras backend"""
        tf.keras.backend.clear_session()
        for arr in self.temp_arrays:
            del arr
        self.temp_arrays = []
    
    def cleanup_files(self):
        """Clean up any temporary files with storage tracking"""
        deleted_size = 0
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    deleted_size += os.path.getsize(f)
                    os.unlink(f)
            except Exception as e:
                logger.warning(f"Could not delete temp file {f}: {str(e)}")
        self.temp_files = []
        
        for d in self.temp_dirs:
            try:
                if os.path.exists(d):
                    deleted_size += sum(f.stat().st_size for f in Path(d).rglob('*') if f.is_file())
                    shutil.rmtree(d)
            except Exception as e:
                logger.warning(f"Could not delete temp dir {d}: {str(e)}")
        self.temp_dirs = []
        
        logger.info(f"Cleaned up {round(deleted_size / (1024 * 1024), 2)} MB of temporary files/dirs")
    
    def cleanup_deepface_cache(self):
        """Clean DeepFace cache with storage tracking"""
        cache_path = os.path.join(os.environ['DEEPFACE_HOME'], '.deepface')
        if os.path.exists(cache_path):
            before_size = sum(f.stat().st_size for f in Path(cache_path).rglob('*') if f.is_file()) / (1024 * 1024)
            try:
                shutil.rmtree(cache_path)
                logger.info(f"Cleaned up DeepFace cache (freed {round(before_size, 2)} MB)")
            except Exception as e:
                logger.error(f"Error cleaning DeepFace cache: {str(e)}")
    
    def cleanup_all(self):
        """Clean all temporary resources with storage tracking"""
        before_storage = get_storage_info()
        
        self.cleanup_tensors()
        self.cleanup_files()
        self.cleanup_deepface_cache()
        
        after_storage = get_storage_info()
        storage_changes = track_storage_changes(before_storage, after_storage)
        
        logger.info(f"Resource cleanup completed. Storage changes: {storage_changes}")

# FaceCropper with resource management
class FaceCropper:
    def __init__(self):
        logger.info("DeepFace detector initialized successfully.")
        self.resource_tracker = ResourceTracker()

    def safe_crop(self, img_array: np.ndarray) -> np.ndarray:
        """Process image with cleanup"""
        try:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Create temp file for DeepFace processing
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
            cv2.imwrite(temp_file, img_rgb)
            self.resource_tracker.add_temp_file(temp_file)
            
            faces = DeepFace.extract_faces(
                img_path=temp_file,
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
def process_single_file(file: UploadFile, resource_tracker: ResourceTracker):
    """Process file with explicit cleanup and storage monitoring"""
    # Get initial storage state
    initial_storage = get_storage_info()
    
    result = {
        "filename": file.filename,
        "class": None,
        "confidence": None,
        "probabilities": None,
        "error": None,
        "storage_before": initial_storage,
        "storage_after": None,
        "storage_diff": None
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
        
        # Track resources
        resource_tracker.add_temp_array(full_img)
        resource_tracker.add_temp_array(face_img)
        
        # Predict
        prediction = app.state.model.predict([
            np.array([full_img]), 
            np.array([face_img])
        ], verbose=0)[0]
        
        # Get storage usage after processing
        current_storage = get_storage_info()
        storage_changes = track_storage_changes(initial_storage, current_storage)
        
        # Update result with storage info
        result.update({
            "class": CLASS_NAMES[np.argmax(prediction)],
            "confidence": float(np.max(prediction)),
            "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, prediction)},
            "storage_after": current_storage,
            "storage_diff": storage_changes
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
        app.state.resource_tracker = ResourceTracker()
        
        # Warm up model
        dummy_input = [
            tf.convert_to_tensor(np.zeros((1, *TARGET_SIZE, 3))),
            tf.convert_to_tensor(np.zeros((1, *TARGET_SIZE, 3)))
        ]
        app.state.model.predict(dummy_input, steps=1)
        
        logger.info("Service initialized with resource tracking")
        logger.info(f"Initial storage state: {get_storage_info()}")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Initialization error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'resource_tracker'):
        app.state.resource_tracker.cleanup_all()
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
            file,
            app.state.resource_tracker
        )
    
    try:
        # Cleanup before processing new batch
        app.state.resource_tracker.cleanup_all()
        
        results = await asyncio.gather(*(process_wrapper(f) for f in files))
        
        # Cleanup after processing
        app.state.resource_tracker.cleanup_all()
        
        return results
    except Exception as e:
        app.state.resource_tracker.cleanup_all()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, "Processing failed")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    try:
        model_exists = os.path.exists(os.getenv('MODEL_PATH'))
        storage_info = get_storage_info()
        
        return {
            "status": "healthy",
            "model_loaded": hasattr(app.state, "model"),
            "model_path": os.getenv('MODEL_PATH'),
            "model_exists": model_exists,
            "storage": {
                "total_space_gb": sum(p['total_gb'] for p in storage_info['partitions']),
                "used_space_gb": sum(p['used_gb'] for p in storage_info['partitions']),
                "temp_dir_size_mb": storage_info['temp_dir'].size_mb,
                "deepface_dir_size_mb": storage_info['deepface_dir'].size_mb
            },
            "temp_dir_info": storage_info['temp_dir'],
            "deepface_dir_info": storage_info['deepface_dir']
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(503, detail=f"Service unavailable: {str(e)}")

@app.get("/storage")
async def get_storage():
    """Endpoint to get detailed storage information"""
    try:
        return JSONResponse(content=get_storage_info())
    except Exception as e:
        logger.error(f"Storage info failed: {e}")
        raise HTTPException(500, detail=f"Could not get storage info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,
        limit_concurrency=4,
    )
