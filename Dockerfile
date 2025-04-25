# Stage 1: Builder
FROM python:3.11-slim-bookworm AS builder

# Set environment variables
ENV VIRTUAL_ENV=/opt/venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install required system packages
RUN apt-get update && \
    apt-get install -y build-essential libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Stage 2: Final runtime image
FROM python:3.11-slim-bookworm

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user
RUN useradd -u 10014 -m appuser
USER appuser

WORKDIR /app

# Copy application code
COPY . /app

# Expose the port your app runs on
EXPOSE 8000

# Run your app (edit if using Flask, FastAPI, etc.)
CMD ["python", "app.py"]
