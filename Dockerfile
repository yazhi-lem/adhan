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

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev,jax,tamil-nlp]" && \
    pip install fastapi uvicorn pydantic

# Create non-root user for security
RUN useradd -m -u 1000 adhan && \
    chown -R adhan:adhan /app

USER adhan

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start API server
CMD ["python", "scripts/run_api_server.py", "--host", "0.0.0.0", "--port", "8000"]
