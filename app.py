import sys
import os
import asyncio
os.environ['DEEPFACE_HOME'] = './tmp/.deepface'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import cv2
import imghdr
from deepface import DeepFace
from typing import List, Dict, Optional, Union
import tempfile
import requests
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set mixed precision policy
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

# Constants
MODEL_URL = "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng"
MODEL_DIR = os.path.join(os.getenv('MODEL_DIR', './tmp/model'))
MODEL_PATH = os.path.join(MODEL_DIR, "final_model_11_4_2025.keras")
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_WORKERS = 2  # Limit concurrent processing for CPU constraints

# Pydantic Models
class PredictionResult(BaseModel):
    filename: str
    class_: str = Field(..., alias="class", description="Classification result", example="REAL")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score", example=0.95)
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Probability distribution across classes",
        example={"AI": 0.02, "FAKE": 0.03, "REAL": 0.95}
    )
    error: Optional[str] = Field(None, description="Error message if processing failed")

    class Config:
        allow_population_by_field_name = True

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Service status", example="healthy")
    model_loaded: bool = Field(..., description="Whether model is loaded", example=True)
    model_path: Optional[str] = Field(None, description="Path to model file")
    model_exists: Optional[bool] = Field(None, description="Whether model file exists")

# Custom layers (simplified for CPU)
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

def download_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading model...")
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            # Stream download to avoid memory issues
            with requests.get(MODEL_URL, stream=True) as response:
                response.raise_for_status()
                
                temp_path = MODEL_PATH + '.tmp'
                with open(temp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                
                # Atomic rename to avoid partial downloads
                os.rename(temp_path, MODEL_PATH)
                
            logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    else:
        logger.info("Model already exists, skipping download.")



# Update the load_model function to handle warmup properly
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    # Configure TensorFlow for CPU optimization
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    custom_objects = {
        'EfficientChannelAttention': EfficientChannelAttention,
        'FixedSpatialAttention': FixedSpatialAttention,
        'FixedHybridBlock': FixedHybridBlock
    }
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        
        # Warm up the model with TensorFlow operations
        dummy_input = [
            tf.convert_to_tensor(np.zeros((1, *TARGET_SIZE, 3))),
            tf.convert_to_tensor(np.zeros((1, *TARGET_SIZE, 3)))
        ]
        model.predict(dummy_input, steps=1)
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    
    
# Modified FaceCropper class
class FaceCropper:
    def __init__(self):
        logger.info("DeepFace detector initialized successfully.")

    def safe_crop(self, img_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
        """Process image entirely in memory"""
        try:
            # Convert to RGB for DeepFace
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Direct face extraction from memory
            faces = DeepFace.extract_faces(
                img_path=img_rgb,
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )
            
            if len(faces) == 0:
                logger.debug("No face detected, using full image")
                return cv2.resize(img_array, target_size)
            
            face_img = faces[0]['face']
            return cv2.resize(face_img, target_size)
            
        except Exception as e:
            logger.error(f"Face processing error: {str(e)}")
            return cv2.resize(img_array, target_size)


def is_valid_image(file_bytes):
    try:
        # Check file size first
        if len(file_bytes) > MAX_FILE_SIZE:
            return False
            
        # Check image type
        image_type = imghdr.what(None, h=file_bytes)
        return image_type in ['jpeg', 'jpg', 'png', 'bmp']
    except Exception:
        return False

# Modified preprocessing function
def preprocess_image(img_array: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """Process image from memory buffer"""
    try:
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return cv2.resize(img, target_size)
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        return np.zeros((*target_size, 3), dtype=np.float32)

# Updated file processing
def process_single_file(file: UploadFile):
    """Process file entirely in memory"""
    result = {
        "filename": file.filename,
        "class": None,
        "confidence": None,
        "probabilities": None,
        "error": None
    }
    
    try:
        # Read and validate directly from memory
        file_bytes = file.file.read(MAX_FILE_SIZE + 1)
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError("File size exceeds 10MB limit")
            
        if not is_valid_image(file_bytes):
            raise ValueError("Invalid image format")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        
        # Process in memory
        full_img = preprocess_image(img)
        face_img = app.state.cropper.safe_crop(img)
        
        # Make prediction
        prediction = app.state.model.predict([
            np.array([full_img]), 
            np.array([face_img])
        ], verbose=0)[0]
        
        # Build result
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
        # CPU-specific optimizations
        tf.config.optimizer.set_experimental_options({
            'layout_optimizer': False,  # Disable for CPU
            'constant_folding': True,
            'shape_optimization': True,
            'remapping': False,  # Disable for CPU
            'arithmetic_optimization': True,
        })
        
        # Remove GPU-specific memory growth setting
        # Only configure thread pools for CPU
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        
        download_model()
        app.state.model = load_model()
        app.state.cropper = FaceCropper()
        app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logger.info("Service initialized with CPU optimizations.")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise RuntimeError(f"Initialization error: {str(e)}")
    
    

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'executor'):
        app.state.executor.shutdown(wait=False)
    logger.info("Service shutdown complete.")

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
        return await asyncio.gather(*(process_wrapper(f) for f in files))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, "Processing failed")
    
    

@app.get("/health", 
         response_model=HealthCheckResponse,
         responses={
             503: {"description": "Service unavailable"}
         })
async def health_check():
    try:
        return {
            "status": "healthy",
            "model_loaded": hasattr(app.state, "model"),
            "model_path": MODEL_PATH,
            "model_exists": os.path.exists(MODEL_PATH)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}"
        )

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        contact=app.contact,
        license_info=app.license_info
    )
    
    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Local development server"
        },
        {
            "url": "https://your-production-url.com",
            "description": "Production server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1,  # Single worker for limited CPU
        limit_concurrency=4,  # Limit total concurrent requests
    )
