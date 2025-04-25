# Use a more secure base image with minimal dependencies
FROM python:3.9-slim-bullseye

# 1. Create non-root user and required directories
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    mkdir -p /app && \
    mkdir -p /tmp/.deepface && \
    mkdir -p /tmp/model && \
    chown -R appuser:appgroup /app && \
    chown -R appuser:appgroup /tmp

# 2. Install system dependencies with no recommended packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy requirements first for better caching
COPY --chown=appuser:appgroup requirements.txt .

# 4. Install Python dependencies with security updates
RUN pip install --no-cache-dir --upgrade pip==23.3.2 && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the application
COPY --chown=appuser:appgroup . .

# 6. Set environment variables
ENV DEEPFACE_HOME=/tmp/.deepface \
    MODEL_DIR=/tmp/model \
    MODEL_PATH=/tmp/model/final_model_11_4_2025.keras \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 7. Set secure permissions
RUN chmod -R 750 /tmp/model && \
    chmod -R 750 /tmp/.deepface && \
    find /app -type d -exec chmod 750 {} \; && \
    find /app -type f -exec chmod 640 {} \; && \
    chmod 750 /app/app.py

# 8. Switch to non-root user
USER appuser

# 9. Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# 10. Use exec form for CMD
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--threads", "2", "--timeout", "120", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
