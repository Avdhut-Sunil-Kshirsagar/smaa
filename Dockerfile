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

# 3. Upgrade pip and install Python dependencies
COPY --chown=10014:10014 requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Download model using direct Google API link
RUN wget --no-check-certificate \
    "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng" \
    -O /app/model/final_model_11_4_2025.keras && \
    chown 10014:10014 /app/model/final_model_11_4_2025.keras && \
    echo "Downloaded file size: $(du -h /app/model/final_model_11_4_2025.keras | cut -f1)"

# 5. Create and verify the model file
RUN cat <<EOF > verify_model.py
import os
import tensorflow as tf

print("\n=== Starting Model Verification ===")
print("Model path:", "/app/model/final_model_11_4_2025.keras")
print("File exists:", os.path.exists("/app/model/final_model_11_4_2025.keras"))
file_size = os.path.getsize("/app/model/final_model_11_4_2025.keras")/1e6
print(f"File size: {file_size:.2f} MB")

# Verify minimum expected size (at least 100MB)
assert file_size > 100, f"File too small ({file_size:.2f} MB), expected >100MB"

# Verify file header
try:
    with open("/app/model/final_model_11_4_2025.keras", "rb") as f:
        header = f.read(4)
        print("\nFile header:", header)
        assert header == b'PK\x03\x04', f"Invalid Keras model file header: {header}"
        print("✅ File header verification passed")
        
        # Quick load test without processing the full model
        print("\nTesting model load...")
        custom_objects = {
            'EfficientChannelAttention': None,
            'FixedSpatialAttention': None,
            'FixedHybridBlock': None
        }
        model = tf.keras.models.load_model(
            "/app/model/final_model_11_4_2025.keras",
            custom_objects=custom_objects,
            compile=False
        )
        print("✅ Model loaded successfully!")
        print("Model summary:")
        model.summary()
except Exception as e:
    print(f"\n❌ Verification failed: {str(e)}")
    raise

print("\n=== Verification completed successfully ===")
EOF

# 6. Run model verification
RUN python verify_model.py && rm verify_model.py

# 7. Copy the rest of the application
COPY --chown=10014:10014 . .

# 8. Set environment variables
ENV MODEL_PATH=/app/model/final_model_11_4_2025.keras
ENV DEEPFACE_HOME=/tmp/.deepface
ENV UPLOAD_FOLDER=/tmp/uploads

# 9. Use non-root user and expose port
USER 10014
EXPOSE 8000

# 10. Run the FastAPI app with Gunicorn + Uvicorn worker
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
