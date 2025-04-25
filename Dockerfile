FROM python:3.10-slim

# Set environment variables
ENV DEEPFACE_HOME=/tmp/.deepface \
    CUDA_VISIBLE_DEVICES=-1 \
    MODEL_DIR=/app/model \
    PATH="/home/deepfakeuser/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*
    
# Create model directory and download
RUN mkdir -p ${MODEL_DIR} && \
    wget -O "${MODEL_DIR}/final_model_11_4_2025.keras" \
    "https://www.googleapis.com/drive/v3/files/1sUNdQHfqKBCW44wGEi158W2DK71g0BZE?alt=media&key=AIzaSyAQWd9J7XainNo1hx3cUzJsklrK-wm9Sng" && \
    chmod a+r ${MODEL_DIR}/final_model_11_4_2025.keras

# Create non-root user
RUN useradd -m -u 15000 deepfakeuser && \
    mkdir -p /app && \
    chown -R deepfakeuser:deepfakeuser /app

USER 15000
WORKDIR /app

COPY --chown=deepfakeuser:deepfakeuser requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=deepfakeuser:deepfakeuser . .

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
