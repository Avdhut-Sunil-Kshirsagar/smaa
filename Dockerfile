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

# 3. Copy requirements first for better caching
COPY --chown=10014:10014 requirements.txt .

# 4. Install Python dependencies with exact versions
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Download model using direct Google API link
RUN wget --no-check-certificate \
    "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng" \
    -O /app/model/final_model_11_4_2025.keras && \
    chown 10014:10014 /app/model/final_model_11_4_2025.keras && \
    echo "Downloaded file size: $(du -h /app/model/final_model_11_4_2025.keras | cut -f1)"

# 6. Create verification script with TF 2.17.1 compatibility
RUN cat <<EOF > verify_model.py
import os
import tensorflow as tf
from tensorflow.keras import layers

print("\n=== Starting Model Verification ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

# Define all custom layer classes
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
        config = super().get_config()
        config.update({'channels': self.channels, 'reduction': self.reduction})
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
        config.update({'filters': self.filters})
        return config

# Verification logic
print("\n[1/4] Verifying model file...")
model_path = "/app/model/final_model_11_4_2025.keras"
print(f"Model path: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")
file_size = os.path.getsize(model_path)/1e6
print(f"File size (MB): {file_size}")

if file_size < 100:
    raise ValueError(f"File too small ({file_size}MB), expected ~130MB")

# Verify file header
print("\n[2/4] Verifying file header...")
try:
    with open(model_path, "rb") as f:
        header = f.read(4)
        print(f"File header: {header}")
        assert header == b'PK\x03\x04', "Invalid Keras model file header"
        print("✅ File header verification passed")
except Exception as e:
    print(f"❌ File verification failed: {str(e)}")
    raise

# Prepare custom objects
print("\n[3/4] Preparing custom objects...")
custom_objects = {
    'EfficientChannelAttention': EfficientChannelAttention,
    'FixedSpatialAttention': FixedSpatialAttention,
    'FixedHybridBlock': FixedHybridBlock
}

# Load model
print("\n[4/4] Loading model...")
try:
    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False
    )
    print("✅ Model loaded successfully!")
    print("\nModel summary:")
    model.summary()
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise

print("\n=== Verification completed successfully ===")
EOF

# 7. Run model verification
RUN python verify_model.py && rm verify_model.py

# 8. Copy the rest of the application
COPY --chown=10014:10014 . .

# 9. Set environment variables
ENV MODEL_PATH=/app/model/final_model_11_4_2025.keras
ENV DEEPFACE_HOME=/tmp/.deepface
ENV UPLOAD_FOLDER=/tmp/uploads

# 10. Use non-root user and expose port
USER 10014
EXPOSE 8000

# 11. Run the FastAPI app with Gunicorn + Uvicorn worker
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
