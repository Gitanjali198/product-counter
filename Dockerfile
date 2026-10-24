# Production Dockerfile for React + FastAPI AI Product Counter
FROM python:3.11-slim

WORKDIR /app

# Install minimal OS dependencies for computer vision
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-built React frontend
COPY . .

# Ensure data directories exist
RUN mkdir -p data/batches

# Default port
EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000

# Run FastAPI backend + React SPA
CMD ["python", "api.py"]
