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

# 3. Install Python dependencies first (better caching)
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy model file separately (large file optimization)
COPY --chown=appuser:appuser model/ /app/model/

# 5. Create a verification script
RUN echo -e 'import tensorflow as tf\n\
print("Model file exists:", tf.io.gfile.exists("/app/model/final_model_11_4_2025.keras"))\n\
try:\n\
    tf.keras.models.load_model("/app/model/final_model_11_4_2025.keras")\n\
    print("Model loaded successfully")\n\
except Exception as e:\n\
    print(f"Model loading failed: {str(e)}")' > verify_model.py

# 6. Verify model file
RUN python verify_model.py && rm verify_model.py

# 7. Copy the rest of the application
COPY --chown=appuser:appuser . .

# 8. Set environment variables
ENV MODEL_PATH=/app/model/final_model_11_4_2025.keras
ENV DEEPFACE_HOME=/tmp/.deepface
ENV UPLOAD_FOLDER=/tmp/uploads

USER 10014
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
