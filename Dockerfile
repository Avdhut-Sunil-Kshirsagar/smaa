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

# 3. Copy requirements and install
COPY --chown=10014:10014 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy model and verify it
COPY --chown=10014:10014 model/ /app/model/
RUN python3 -c "
import tensorflow as tf
import os
print(f'Model file size: {os.path.getsize(\"/app/model/final_model_11_4_2025.keras\")} bytes')
try:
    model = tf.keras.models.load_model('/app/model/final_model_11_4_2025.keras')
    print('Model verification passed!')
    model.summary()
except Exception as e:
    print(f'Model verification failed: {str(e)}')
    raise
"

# 5. Copy application code
COPY --chown=10014:10014 . .

# 6. Environment variables
ENV MODEL_PATH=/app/model/final_model_11_4_2025.keras
ENV DEEPFACE_HOME=/tmp/.deepface
ENV UPLOAD_FOLDER=/tmp/uploads

# 7. Runtime configuration
USER 10014
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
