# Use the official Python 3.8 slim image as a base
FROM python:3.8-slim

# Set environment variables
ENV DEEPFACE_HOME="/tmp/.deepface"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV CUDA_VISIBLE_DEVICES="-1"
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root user with UID between 10000 and 20000
RUN useradd -m -u 15000 deepfakeuser

# Set the user to be the non-root user
USER deepfakeuser

# Set the working directory for the app
WORKDIR /app

# Copy the requirements.txt and install dependencies
COPY --chown=deepfakeuser:deepfakeuser requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container with the appropriate ownership
COPY --chown=deepfakeuser:deepfakeuser . /app

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
