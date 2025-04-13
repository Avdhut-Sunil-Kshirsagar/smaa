FROM python:3.9-slim

# 1. Create non-root user and required directories
RUN useradd -u 10014 -m appuser && \
    mkdir -p /app/model && \
    mkdir -p /tmp/uploads && \
    mkdir -p /tmp/.deepface && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /tmp

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Install specific TensorFlow version (2.15.0 works best with Keras 3)
COPY --chown=10014:10014 requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install tensorflow==2.15.0 keras==2.15.0  # Downgrade to compatible versions

# 4. Download model
RUN wget --no-check-certificate \
    "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng" \
    -O /app/model/final_model_11_4_2025.keras && \
    chown 10014:10014 /app/model/final_model_11_4_2025.keras

# 5. Create verification script with fixed precision handling
RUN cat <<EOF > verify_model.py
import os
import tensorflow as tf
from tensorflow.keras import layers

# Disable mixed precision for loading
tf.keras.mixed_precision.set_global_policy('float32')

print("\n=== Starting Model Verification ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# FixedHybridBlock with proper precision handling
class FixedHybridBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
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
        
        # Ensure consistent precision
        x = tf.cast(inputs, tf.float32)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Cast residual if needed
        if residual.dtype != x.dtype:
            residual = tf.cast(residual, x.dtype)
            
        return self.act(x + residual)
    
    def get_config(self):
        return {'filters': self.filters}

# Other custom layers (simplified versions)
class EfficientChannelAttention(layers.Layer):
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.fc = tf.keras.Sequential([
            layers.Dense(channels//reduction, activation='relu'),
            layers.Dense(channels, activation='sigmoid')
        ])
        
    def call(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return tf.reshape(out, [-1, 1, 1, self.channels]) * x
    
    def get_config(self):
        return {'channels': self.channels, 'reduction': self.reduction}

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
        return {}

# Verification
model_path = "/app/model/final_model_11_4_2025.keras"
print(f"\nModel path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
print(f"File size: {os.path.getsize(model_path)/1e6:.2f} MB")

try:
    # Load with custom objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'FixedHybridBlock': FixedHybridBlock,
            'EfficientChannelAttention': EfficientChannelAttention,
            'FixedSpatialAttention': FixedSpatialAttention
        },
        compile=False
    )
    print("\n✅ Model loaded successfully!")
    model.summary()
except Exception as e:
    print(f"\n❌ Failed to load model: {str(e)}")
    raise

print("\n=== Verification completed ===")
EOF

# 6. Run verification
RUN python verify_model.py && rm verify_model.py

# 7. Copy application code
COPY --chown=10014:10014 . .

# 8. Set environment variables
ENV MODEL_PATH=/app/model/final_model_11_4_2025.keras
ENV DEEPFACE_HOME=/tmp/.deepface
ENV UPLOAD_FOLDER=/tmp/uploads

# 9. Use non-root user and expose port
USER 10014
EXPOSE 8000

# 10. Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
