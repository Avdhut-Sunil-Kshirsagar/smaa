# Use the official Python image as a base with a specific version for reproducibility
FROM python:3.9-slim@sha256:2c8a6d6a4a5e5a5f5e5d5c5b5a5a595958585756555453525150494847464544

# Create a non-root user with UID in the 10000-20000 range (15000) and a fixed GID
RUN groupadd -g 15000 appgroup && \
    useradd -u 15000 -g appgroup -m appuser && \
    mkdir -p /app && \
    chown appuser:appgroup /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEEPFACE_HOME=/tmp/.deepface \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY --chown=appuser:appgroup requirements.txt .

# Install dependencies with security best practices
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    find /usr/local -depth \
        \( \
            \( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
            -o \
            \( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
        \) -exec rm -rf '{}' +

# Copy the rest of the application
COPY --chown=appuser:appgroup . .

# Ensure the temp directory exists and is writable
RUN mkdir -p ${DEEPFACE_HOME} && \
    chmod -R 770 /tmp && \
    chown -R appuser:appgroup /tmp

# Verify the user configuration
RUN id appuser && \
    [ "$(id -u appuser)" -ge 10000 ] && \
    [ "$(id -u appuser)" -le 20000 ] || (echo "User ID out of range" && exit 1)

# Switch to non-root user
USER 15000:15000

# Expose the port the app runs on
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
