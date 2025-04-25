FROM python:3.9-slim

# 1. Create non-root user and required directories
RUN useradd -u 10014 -m appuser && \
    mkdir -p /app && \
    mkdir -p /tmp/.deepface && \
    mkdir -p /tmp/model && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /tmp

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the application
COPY --chown=appuser:appuser . .

# 6. Set environment variables
ENV DEEPFACE_HOME=/tmp/.deepface
ENV MODEL_DIR=/tmp/model
ENV MODEL_PATH=/tmp/model/final_model_11_4_2025.keras

# 7. Verify directory permissions
RUN chmod -R a+rwx /tmp/model && \
    chmod -R a+rwx /tmp/.deepface

USER 10014
EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
