import sys
import os
os.environ['DEEPFACE_HOME'] = '/tmp/.deepface'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add this before any other imports
try:
    import keras
    import tensorflow as tf
    sys.modules['tf_keras'] = tf.keras
except ImportError:
    pass
  
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import imghdr
from deepface import DeepFace
from typing import List
import tempfile
import requests

# Set mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

app = FastAPI(title="Deepfake Detection API",
              description="API for detecting deepfake, real and AI-generated images",
              version="1.0")

# Constants
MODEL_URL = "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "final_model_11_4_2025.keras")
TARGET_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'FAKE', 'REAL']

# Custom layers
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
        print("Downloading model...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print("Model already exists, skipping download.")

def load_model():
    custom_objects = {
        'EfficientChannelAttention': EfficientChannelAttention,
        'FixedSpatialAttention': FixedSpatialAttention,
        'FixedHybridBlock': FixedHybridBlock
    }
    return tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

class FaceCropper:
    def __init__(self):
        try:
            print("DeepFace detector initialized successfully.")
        except Exception as e:
            print(f"Error initializing DeepFace: {e}")

    def safe_crop(self, image_path, target_size=(224, 224)):
        try:
            faces = DeepFace.extract_faces(image_path, detector_backend='opencv', enforce_detection=False)
            
            if len(faces) == 0:
                print("No face detected, using full image as cropped face.")
                full_img = cv2.imread(image_path)
                cropped_face = cv2.resize(full_img, target_size)
            else:
                cropped_face = faces[0]['face']
                cropped_face = cv2.resize(cropped_face, target_size)
            
            if cropped_face.shape != (target_size[0], target_size[1], 3):
                print("Warning: Cropped face shape is not as expected.")
                cropped_face = np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)
            
            return cropped_face
        except Exception as e:
            print(f"Error during cropping: {e}")
            return np.zeros((*target_size, 3), dtype=np.float32)

def is_valid_image(file_bytes):
    try:
        image_type = imghdr.what(None, h=file_bytes)
        return image_type in ['jpeg', 'png', 'bmp']
    except:
        return False

def preprocess_image(file_path, target_size=(224, 224)):
    try:
        img = tf.io.read_file(file_path)
        try:
            img = tf.image.decode_jpeg(img, channels=3)
        except:
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, target_size).numpy()
    except:
        return np.zeros((*target_size, 3), dtype=np.float32)

@app.on_event("startup")
async def startup_event():
    try:
        download_model()
        app.state.model = load_model()
        app.state.cropper = FaceCropper()
        print("Model and face cropper initialized successfully.")
    except Exception as e:
        print(f"Error during startup: {e}")
        raise

@app.post("/predict", response_model=List[dict])
async def predict(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        try:
            # Validate image
            file_bytes = await file.read()
            if not is_valid_image(file_bytes):
                raise HTTPException(status_code=400, detail=f"Invalid image file: {file.filename}")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                # Preprocess images
                full_img = preprocess_image(temp_path, TARGET_SIZE)
                face_img = app.state.cropper.safe_crop(temp_path, TARGET_SIZE)
                
                # Make prediction
                prediction = app.state.model.predict([np.array([full_img]), np.array([face_img])])[0]
                
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = float(np.max(prediction))
                
                results.append({
                    "filename": file.filename,
                    "class": predicted_class,
                    "confidence": confidence,
                    "probabilities": {name: float(prob) for name, prob in zip(CLASS_NAMES, prediction)}
                })
            finally:
                os.unlink(temp_path)
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return results

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": hasattr(app.state, "model")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
