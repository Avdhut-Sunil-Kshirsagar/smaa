# Stage 1: Build the application
FROM python:3.11-slim-bookworm AS builder

# Set environment variables
ENV DEEPFACE_HOME=/tmp/.deepface \
    UPLOAD_FOLDER=/tmp/uploads \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Stage 2: Create a minimal runtime image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV DEEPFACE_HOME=/tmp/.deepface \
    UPLOAD_FOLDER=/tmp/uploads \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -u 10014 -m appuser

# Set work directory
WORKDIR /app

# Copy installed Python packages and application code from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

# Set ownership and permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER 10014

# Expose the application's port
EXPOSE 8000

# Define the default command
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
