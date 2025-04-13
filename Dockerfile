FROM python:3.9-slim

# 1. Create non-root user and required directories
RUN useradd -u 10014 -m appuser && \
    mkdir -p /app && \
    mkdir -p /tmp/.deepface && \
    chown -R appuser:appuser /app /tmp/.deepface

# 2. Set environment variable for DeepFace
ENV DEEPFACE_HOME=/tmp/.deepface

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Copy requirements first for caching
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY --chown=appuser:appuser . .

# 6. Switch to non-root user
USER 10014

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
