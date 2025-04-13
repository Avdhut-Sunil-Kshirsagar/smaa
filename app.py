import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras import layers, models
from deepface import DeepFace
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
UPLOAD_FOLDER = '/tmp/uploads'
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'final_model_11_4_2025.keras')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB limit
    MODEL_PATH=MODEL_PATH
)

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# 1. Image Validator
def is_valid_image_file(filepath):
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

# 2. Face Cropper Class
class FaceCropper:
    def __init__(self):
        logger.info("DeepFace initialized")

    def safe_crop(self, image_path, target_size=(224, 224)):
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

# 3. Attention Modules (unchanged from your original)   
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

# 4. Model Loading with Verification
def load_custom_model(model_path):
    # Verify model exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file missing at {model_path}")
    
    custom_objects = {
        'EfficientChannelAttention': EfficientChannelAttention,
        'FixedSpatialAttention': FixedSpatialAttention,
        'FixedHybridBlock': FixedHybridBlock
    }
    
    logger.info("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# Initialize components
cropper = FaceCropper()
model = load_custom_model(app.config['MODEL_PATH'])

# 5. Image Processing
def preprocess_images(image_paths, target_size=(224, 224)):
    def process_image(path):
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

# 6. Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'Empty file list'}), 400
    
    saved_paths = []
    try:
        # Save and validate files
        for file in files:
            if not (file and allowed_file(file.filename)):
                raise ValueError("Invalid file type")
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            saved_paths.append(file_path)
        
        # Process and predict
        results = []
        full_imgs, face_imgs = preprocess_images(saved_paths)
        predictions = model.predict([full_imgs, face_imgs])
        
        for i, path in enumerate(saved_paths):
            results.append({
                'image': os.path.basename(path),
                'class': ['AI', 'FAKE', 'REAL'][np.argmax(predictions[i])],
                'confidence': float(np.max(predictions[i])),
                'probabilities': {
                    'AI': float(predictions[i][0]),
                    'FAKE': float(predictions[i][1]),
                    'REAL': float(predictions[i][2])
                }
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.path.remove(path)
            except:
                pass

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': os.path.exists(app.config['MODEL_PATH'])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))