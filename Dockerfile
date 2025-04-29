# Stage 1: Builder for model download
FROM alpine:3.18 as downloader

RUN apk add --no-cache wget
RUN mkdir -p /model && \
    wget -q -O /model/final_model_11_4_2025.keras \
    "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng"

# Stage 2: Final image
FROM python:3.10-slim

# Set environment variables
ENV DEEPFACE_HOME=/tmp/.deepface \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_NUM_INTEROP_THREADS=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MODEL_PATH=/app/model/final_model_11_4_2025.keras \
    MAX_WORKERS=2 \
    MAX_FILE_SIZE=10485760

# Create app directory
RUN mkdir -p /app/model

# Copy only the pre-downloaded model
COPY --from=downloader /model/final_model_11_4_2025.keras /app/model/

# Install runtime dependencies and clean cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip

# Copy requirements and install
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + && \
    find /opt/venv -name '*.so' -exec strip {} +

# Copy application code
COPY app.py .

# Clean up unnecessary files
RUN apt-get purge -y --auto-remove && \
    rm -rf /root/.cache /tmp/*

# Create non-root user with UID in 10000-20000 range
RUN useradd -mU -u 15000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

USER 15000

EXPOSE 8000

# Start command with memory limits
CMD ["/opt/venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--limit-concurrency", "4", "--timeout-keep-alive", "30"]
