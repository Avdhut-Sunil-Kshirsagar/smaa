FROM python:3.9-slim

# 1. Create non-root user with UID in Choreo's range and app directory
RUN useradd -u 10014 -m appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt ./

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install gunicorn uvicorn

# 6. Copy application code
COPY --chown=appuser:appuser . .

# 7. Environment variables
ENV PYTHONPATH=/app \
    PORT=8000 \
    MODEL_PATH=/app/model/final_model_11_4_2025.keras \
    UPLOAD_FOLDER=/tmp/uploads

# 8. Create upload directory
RUN mkdir -p ${UPLOAD_FOLDER} && \
    chown appuser:appuser ${UPLOAD_FOLDER}

# 9. Switch to non-root user
USER 10014

# 10. Expose port (matches FastAPI default)
EXPOSE 8000

# 11. Run with uvicorn through gunicorn (async workers for FastAPI)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "120", "-k", "uvicorn.workers.UvicornWorker", "app:app"]
