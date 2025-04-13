FROM python:3.9-slim

# 1. Create non-root user with UID in Choreo's required range (10000-20000)
RUN useradd -u 10014 -m appuser && \
    chown -R appuser:appuser /app

# 2. Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy requirements with proper permissions
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY --chown=appuser:appuser . .

# 5. Switch to non-root user (with approved UID)
USER 10014

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
