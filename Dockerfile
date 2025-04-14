FROM python:3.9-slim

# Create non-root user
RUN useradd -u 15000 -m appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEEPFACE_HOME=/tmp/.deepface
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_USE_LEGACY_KERAS=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies with exact versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check

# Clean up Python cache
RUN find /usr/local/lib/python3.9 -type d -name '__pycache__' -exec rm -r {} + && \
    rm -rf /root/.cache/pip

COPY app.py .

# Set up temp directory
RUN mkdir -p /tmp/.deepface && \
    chmod -R 777 /tmp && \
    chown -R appuser:appuser /tmp

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
