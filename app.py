import os
os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_NUM_INTEROP_THREADS'] = '8'  # Optimize for CPU
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'  # Optimize for CPU
os.environ['OMP_NUM_THREADS'] = '8'  # Optimize for CPU

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
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure TensorFlow for optimal CPU performance
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.set_soft_device_placement(True)

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake, real and AI-generated images",
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
MODEL_DIR = os.path.join(os.getenv('MODEL_DIR', '/app/model'))
MODEL_PATH = os.path.join(MODEL_DIR, "final_model_11_4_2025.keras")
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']
MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)  # Optimal worker count

# Pydantic Models (unchanged from original)
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

# Custom layers (unchanged from original)
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
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Model downloaded successfully to {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    else:
        logger.info("Model already exists, skipping download.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    custom_objects = {
        'EfficientChannelAttention': EfficientChannelAttention,
        'FixedSpatialAttention': FixedSpatialAttention,
        'FixedHybridBlock': FixedHybridBlock
    }
    
    # Load with optimized settings for CPU
    model = tf.keras.models.load_model(
        MODEL_PATH, 
        custom_objects=custom_objects,
        compile=False
    )
    
    # Warm up the model
    dummy_input = [
        np.random.rand(1, *TARGET_SIZE, 3).astype(np.float32),
        np.random.rand(1, *TARGET_SIZE, 3).astype(np.float32)
    ]
    model.predict(dummy_input, batch_size=1)
    
    return model

class FaceCropper:
    def __init__(self):
        try:
            logger.info("DeepFace detector initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing DeepFace: {e}")

    def safe_crop(self, image_path, target_size=(224, 224)):
        try:
            # Use OpenCV directly for faster image loading
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to read image file")
            
            # Convert to RGB (what DeepFace expects)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Try face detection
            try:
                faces = DeepFace.extract_faces(
                    img_path=img_rgb,
                    detector_backend='opencv',
                    enforce_detection=False
                )
                
                if len(faces) == 0:
                    logger.debug("No face detected, using full image")
                    cropped_face = cv2.resize(img, target_size)
                else:
                    cropped_face = faces[0]['face']
                    cropped_face = cv2.resize(cropped_face, target_size)
            except Exception as e:
                logger.warning(f"Face detection failed, using full image: {e}")
                cropped_face = cv2.resize(img, target_size)
            
            return cropped_face
        except Exception as e:
            logger.error(f"Error during cropping: {e}")
            return np.zeros((*target_size, 3), dtype=np.float32)

def is_valid_image(file_bytes):
    try:
        image_type = imghdr.what(None, h=file_bytes)
        return image_type in ['jpeg', 'jpg', 'png', 'bmp']
    except Exception:
        return False

def preprocess_image(file_path, target_size=(224, 224)):
    try:
        # Use OpenCV for faster image loading
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError("Failed to read image file")
        
        # Convert to float32 and normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # Resize using OpenCV (faster than tf.image.resize)
        return cv2.resize(img, target_size)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return np.zeros((*target_size, 3), dtype=np.float32)

def process_single_file(file: UploadFile):
    result = {
        "filename": file.filename,
        "class": None,
        "confidence": None,
        "probabilities": None,
        "error": None
    }
    
    try:
        # Validate image
        file_bytes = file.file.read()
        if not is_valid_image(file_bytes):
            result["error"] = f"Invalid image format for {file.filename}. Supported: JPEG, PNG, BMP"
            return result
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=os.path.splitext(file.filename)[1]
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        
        try:
            # Preprocess images
            full_img = preprocess_image(temp_path, TARGET_SIZE)
            face_img = app.state.cropper.safe_crop(temp_path, TARGET_SIZE)
            
            # Make prediction
            prediction = app.state.model.predict([
                np.array([full_img]), 
                np.array([face_img])
            ], batch_size=1)[0]
            
            # Convert to Python native types
            result.update({
                "class": CLASS_NAMES[np.argmax(prediction)],
                "confidence": float(np.max(prediction)),
                "probabilities": {
                    name: float(prob) 
                    for name, prob in zip(CLASS_NAMES, prediction)
                }
            })
            
        except Exception as e:
            result["error"] = f"Processing error: {str(e)}"
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        
    return result

@app.on_event("startup")
async def startup_event():
    try:
        download_model()
        app.state.model = load_model()
        app.state.cropper = FaceCropper()
        app.state.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logger.info(f"Service initialized with {MAX_WORKERS} workers")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise RuntimeError(f"Failed to initialize service: {e}")

@app.post("/predict", 
          response_model=List[PredictionResult],
          responses={
              400: {"description": "Invalid input (no files or invalid format)"},
              500: {"description": "Internal server error"}
          })
async def predict(files: List[UploadFile] = File(..., description="Image files to analyze")):
    if not files:
        raise HTTPException(status_code=400, detail="At least one image file is required")
    
    # Process files in parallel
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(app.state.executor, process_single_file, file)
        for file in files
    ]
    
    results = await asyncio.gather(*tasks)
    return results

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
    
    openapi_schema["servers"] = [
        {
            "url": "http://localhost:8000",
            "description": "Local development server"
        },
        {
            "url": "https://dff47398-ad64-430a-956d-ee6c36ac85ea-dev.e1-us-east-azure.choreoapis.dev/default/smaa/v1.0",
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
        workers=1,  # We're using ThreadPoolExecutor for parallelism
        limit_concurrency=MAX_WORKERS * 2
    )
