FROM python:3.9-slim

# 1. Create non-root user and required directories
RUN mkdir -p /app && \
    useradd -u 10014 -m appuser && \
    mkdir -p /tmp/model && \
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

# 4. Copy the application
COPY --chown=appuser:appuser . .

# 5. Set environment variables
ENV DEEPFACE_HOME=/tmp/.deepface
ENV UPLOAD_FOLDER=/tmp/uploads

USER 10014
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
