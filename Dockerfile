FROM python:3.9-slim

# 1. Create non-root user and /app directory
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

# 4. Copy requirements.txt first to leverage Docker layer caching
COPY --chown=appuser:appuser requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the app (including app.py, model folder, etc.)
COPY --chown=appuser:appuser . .

# 6. Switch to non-root user
USER 10014

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
