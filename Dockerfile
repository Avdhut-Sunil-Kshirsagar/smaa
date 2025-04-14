# Use the official Python slim image
FROM python:3.9-slim

# Create non-root user (UID between 10000-20000)
RUN useradd -u 15000 -m appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEEPFACE_HOME=/tmp/.deepface
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1

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

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies with specific versions
RUN pip install --no-cache-dir -r requirements.txt && \
    # Install tf_keras separately if needed
    pip install tf_keras==2.15.0 && \
    # Clean up
    find /usr/local/lib/python3.9 -type d -name '__pycache__' -exec rm -r {} + && \
    rm -rf /root/.cache/pip

# Copy application code
COPY app.py .

# Set up temp directory
RUN mkdir -p /tmp/.deepface && \
    chmod -R 777 /tmp && \
    chown -R appuser:appuser /tmp

# Switch to non-root user
USER 15000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
