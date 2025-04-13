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

# 4. Download model from Google Drive
RUN wget --load-cookies /tmp/cookies.txt \
    "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sUNdQHfqKBCW44wGEi158W2DK71g0BZE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sUNdQHfqKBCW44wGEi158W2DK71g0BZE" \
    -O /app/model/final_model_11_4_2025.keras && \
    rm -rf /tmp/cookies.txt && \
    chown 10014:10014 /app/model/final_model_11_4_2025.keras

# 5. Create and verify the model file
RUN cat <<EOF > verify_model.py
import os
import tensorflow as tf

print("Verifying model file...")
print("Model file exists:", tf.io.gfile.exists("/app/model/final_model_11_4_2025.keras"))
print("File size:", os.path.getsize("/app/model/final_model_11_4_2025.keras"))

try:
    with open("/app/model/final_model_11_4_2025.keras", "rb") as f:
        header = f.read(4)
        print("File header:", header)
        assert header == b'PK\x03\x04', "Invalid Keras model file header"
    print("Basic file verification passed")
    
    # Skip actual model loading in build stage to keep image smaller
    print("Skipping full model load during build (will load at runtime)")
except Exception as e:
    print(f"File verification failed: {str(e)}")
    raise
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
