FROM python:3.9-slim

# 1. Create non-root user and needed dirs
RUN useradd -u 10014 -m appuser && \
    mkdir -p /app /tmp/uploads /tmp/.deepface && \
    chown -R appuser:appuser /app /tmp

WORKDIR /app

# 2. Install system dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# 3. Copy code
COPY --chown=appuser:appuser . .

# 4. Install Python deps
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Env vars
ENV DEEPFACE_HOME=/tmp/.deepface
ENV UPLOAD_FOLDER=/tmp/uploads
ENV MODEL_PATH=/tmp/model/final_model_11_4_2025.keras


USER 10014
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
