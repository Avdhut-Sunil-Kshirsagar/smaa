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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Upgrade pip and install Python dependencies (improves build caching)
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy model separately (allows Docker caching)
COPY --chown=appuser:appuser model/ /app/model/

# 5. Create and verify the model file
RUN cat <<EOF > verify_model.py
import tensorflow as tf
print("Model file exists:", tf.io.gfile.exists("/app/model/final_model_11_4_2025.keras"))
try:
    tf.keras.models.load_model("/app/model/final_model_11_4_2025.keras")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {str(e)}")
EOF

# 6. Run model verification
RUN python verify_model.py && rm verify_model.py

# 7. Copy the rest of the application
COPY --chown=appuser:appuser . .

# 8. Set environment variables
ENV MODEL_PATH=/app/model/final_model_11_4_2025.keras
ENV DEEPFACE_HOME=/tmp/.deepface
ENV UPLOAD_FOLDER=/tmp/uploads

# 9. Use non-root user and expose port
USER 10014
EXPOSE 8000

# 10. Run the FastAPI app with Gunicorn + Uvicorn worker
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
