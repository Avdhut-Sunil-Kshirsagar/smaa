import os
os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'

import logging
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from tensorflow.keras import layers, models
from fastapi.responses import JSONResponse
from deepface import DeepFace
from pathlib import Path
import imghdr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Image Classification API",
    description="API for detecting AI-generated, fake, or real images",
    version="1.0.0"
)

# Configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
UPLOAD_FOLDER = '/tmp/uploads'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model_11_4_2025.keras')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.environ['DEEPFACE_HOME'], exist_ok=True)

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Response Models
class PredictionResult(BaseModel):
    image: str
    predicted_class: str
    confidence: float
    probabilities: dict

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool

# 1. Image Validator
def is_valid_image_file(filepath: str) -> bool:
    try:
        ext = os.path.splitext(filepath)[1].lower()[1:]
        if ext not in ALLOWED_EXTENSIONS:
            return False
        img_type = imghdr.what(filepath)
        if img_type not in ['jpeg', 'png', 'bmp']:
            return False
        img = cv2.imread(filepath)
        return img is not None
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False

# 2. Face Cropper
class FaceCropper:
    def __init__(self):
        logger.info("DeepFace initialized")

    def safe_crop(self, image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
        try:
            faces = DeepFace.extract_faces(
                image_path, 
                detector_backend='opencv', 
                enforce_detection=False
            )
            if len(faces) == 0:
                logger.info("Using full image as fallback")
                img = cv2.imread(image_path)
                return cv2.resize(img, target_size)
            face = faces[0]['face']
            resized = cv2.resize(face, target_size)
            return resized if resized.shape == (*target_size, 3) else np.zeros((*target_size, 3), dtype=np.float32)
        except Exception as e:
            logger.error(f"Cropping error: {e}")
            return np.zeros((*target_size, 3), dtype=np.float32)

# 3. Attention Modules
class EfficientChannelAttention(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = models.Sequential([
            layers.Dense(channels // reduction, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])

    def call(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return tf.reshape(out, [-1, 1, 1, self.channels]) * x

    def get_config(self):
        return {"channels": self.channels, "reduction": self.reduction}

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
        residual = self.res_conv(inputs) if self.res_conv else inputs
        x = self.act(self.norm1(self.conv1(inputs)))
        x = self.eca(x)
        x = self.sa(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)

# 4. Load model
def load_custom_model(model_path: str) -> tf.keras.Model:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file missing at {model_path}")
    custom_objects = {
        'EfficientChannelAttention': EfficientChannelAttention,
        'FixedSpatialAttention': FixedSpatialAttention,
        'FixedHybridBlock': FixedHybridBlock
    }
    logger.info("Loading model...")
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Initialize
cropper = FaceCropper()
model = load_custom_model(MODEL_PATH)

# 5. Image processing
def preprocess_images(image_paths: List[str], target_size: tuple = (224, 224)) -> tuple:
    def process_image(path: str) -> np.ndarray:
        try:
            img = tf.io.read_file(path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, target_size).numpy()
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return np.zeros((*target_size, 3), dtype=np.float32)

    full_imgs = [process_image(p) for p in image_paths]
    face_imgs = [cropper.safe_crop(p, target_size) for p in image_paths]
    return np.array(full_imgs), np.array(face_imgs)

# 6. API Endpoints
@app.post("/predict", response_model=List[PredictionResult])
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
            raise HTTPException(status_code=400, detail="Invalid file type")

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if not is_valid_image_file(file_path):
            raise HTTPException(status_code=400, detail="Invalid image file")

        full_img, face_img = preprocess_images([file_path])
        prediction = model.predict([full_img, face_img])

        result = {
            "image": file.filename,
            "predicted_class": ['AI', 'FAKE', 'REAL'][np.argmax(prediction[0])],
            "confidence": float(np.max(prediction[0])),
            "probabilities": {
                'AI': float(prediction[0][0]),
                'FAKE': float(prediction[0][1]),
                'REAL': float(prediction[0][2])
            }
        }
        return [result]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/batch_predict", response_model=List[PredictionResult])
async def batch_predict(files: List[UploadFile] = File(...)):
    saved_paths = []
    try:
        for file in files:
            if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())

            if not is_valid_image_file(file_path):
                raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")

            saved_paths.append(file_path)

        full_imgs, face_imgs = preprocess_images(saved_paths)
        predictions = model.predict([full_imgs, face_imgs])

        results = []
        for i, path in enumerate(saved_paths):
            results.append({
                "image": os.path.basename(path),
                "predicted_class": ['AI', 'FAKE', 'REAL'][np.argmax(predictions[i])],
                "confidence": float(np.max(predictions[i])),
                "probabilities": {
                    'AI': float(predictions[i][0]),
                    'FAKE': float(predictions[i][1]),
                    'REAL': float(predictions[i][2])
                }
            })

        return results
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in saved_paths:
            if os.path.exists(path):
                os.remove(path) 

@app.get("/", response_model=HealthCheck)
async def health_check():
    return {
        "status": "API is running",
        "model_loaded": os.path.exists(MODEL_PATH)
    }

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
