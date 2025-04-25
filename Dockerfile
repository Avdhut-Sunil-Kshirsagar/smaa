# Use the official Python 3.10 slim image as a base
FROM python:3.10-slim

# Set environment variables
ENV DEEPFACE_HOME="/tmp/.deepface"
ENV TF_CPP_MIN_LOG_LEVEL="3"
ENV CUDA_VISIBLE_DEVICES="-1"
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR="/app/model"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create model directory and set permissions
RUN mkdir -p ${MODEL_DIR} && chmod a+rwx ${MODEL_DIR}

# Download the model during build
RUN wget -O "${MODEL_DIR}/final_model_11_4_2025.keras" \
    "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng"

# Add a non-root user with UID between 10000 and 20000
RUN useradd -m -u 15000 deepfakeuser

# Set the user to be the non-root user
USER 15000

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
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
