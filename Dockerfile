# Dockerfile for Adhan SLM Inference Server
# Build: docker build -t adhan-slm:latest .
# Run: docker run -p 8000:8000 adhan-slm:latest

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies (including serving extra: fastapi, uvicorn, httpx)
RUN pip install --upgrade pip && \
    pip install -e ".[serving]"

# Create non-root user for security
RUN useradd -m -u 1000 adhan && \
    chown -R adhan:adhan /app

USER adhan

# Expose API port
EXPOSE 8000

# Health check: Requires a loaded model. If no model is loaded, /health returns HTTP 503
# causing urllib.request to raise HTTPError and exit 1 (reporting unhealthy container).
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Note: The inference server requires a trained model checkpoint and tokenizer (e.g. mounted to /app/checkpoints).
# If no model is available, the server stays in standby mode where /health returns 503 and inference endpoints fail honestly.
CMD ["python", "scripts/run_api_server.py", "--host", "0.0.0.0", "--port", "8000"]

