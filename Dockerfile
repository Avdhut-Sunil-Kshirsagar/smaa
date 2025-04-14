# Use the official Python image as a base
FROM python:3.9-slim

# Create a non-root user with UID in the 10000-20000 range
RUN useradd -u 15000 -m appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEEPFACE_HOME=/tmp/.deepface
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc python3-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY app.py .

# Ensure the temp directory exists and is writable
RUN mkdir -p /tmp/.deepface && \
    chmod -R 777 /tmp && \
    chown -R appuser:appuser /tmp

# Switch to non-root user
USER 15000

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
