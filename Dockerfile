FROM python:3.9-slim

# 1. Create non-root user and required directories
RUN useradd -u 10014 -m appuser && \
    mkdir -p /app/model && \
    chown -R appuser:appuser /app

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Install Python dependencies first (better caching)
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy model file separately (large file optimization)
COPY --chown=appuser:appuser model/final_model_11_4_2025.keras /app/model/

# 5. Copy the rest of the application
COPY --chown=appuser:appuser . .

# 6. Set environment variables
ENV MODEL_PATH=/app/model/final_model_11_4_2025.keras

USER 10014
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
