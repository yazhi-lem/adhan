# Adhan SLM Deployment Guide

This guide covers deploying Adhan SLM models to production via the **yazhi-api** platform.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [yazhi-api Integration](#yazhi-api-integration)
6. [API Reference](#api-reference)
7. [Monitoring & Logging](#monitoring--logging)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Running the API Locally

```bash
# Install dependencies
pip install -e ".[dev,jax,tamil-nlp]"
pip install fastapi uvicorn pydantic

# Start the API server
python scripts/run_api_server.py --model adhan-nano --port 8000

# Test the API
curl -X POST http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "தமிழ் மொழி"}'
```

The API will be available at `http://localhost:8000`

### Documentation

- **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)

---

## Local Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yazhi-lem/adhan.git
cd adhan

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev,jax,tamil-nlp]"

# Install API dependencies
pip install fastapi uvicorn
```

### Running Tests

```bash
# Run unit tests
pytest src/adhan_slm/ -v

# Run integration tests
pytest tests/integration/ -v -m integration

# Run with coverage
pytest src/adhan_slm/ --cov=src/adhan_slm --cov-report=html
```

### API Server Options

```bash
# Help
python scripts/run_api_server.py --help

# Development (with auto-reload)
python scripts/run_api_server.py --reload --log-level DEBUG

# Production (single worker)
python scripts/run_api_server.py --log-level INFO

# Custom model and port
python scripts/run_api_server.py --model adhan-tiny --port 8080
```

---

## Docker Deployment

### Build Docker Image

```bash
# Build image
docker build -t adhan-slm:latest .

# Build with custom tag
docker build -t adhan-slm:v0.1.0 .

# List images
docker images | grep adhan
```

### Run Container

```bash
# Basic run
docker run -p 8000:8000 adhan-slm:latest

# With environment variables
docker run \
  -p 8000:8000 \
  -e ADHAN_LOG_LEVEL=INFO \
  -e ADHAN_JSON_LOGS=true \
  adhan-slm:latest

# With volume mounts (for models/data)
docker run \
  -p 8000:8000 \
  -v /path/to/data:/app/data \
  -v /path/to/checkpoints:/app/checkpoints \
  adhan-slm:latest

# Background mode
docker run -d \
  --name adhan-api \
  -p 8000:8000 \
  adhan-slm:latest

# View logs
docker logs adhan-api

# Stop container
docker stop adhan-api
```

### Docker Compose (Local Testing)

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f adhan-api

# Stop services
docker-compose down

# Clean up volumes
docker-compose down -v
```

### Docker Health Checks

```bash
# Check health
docker inspect adhan-api | jq '.[] | .State.Health'

# Manual health check
curl http://localhost:8000/health
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.20+)
- kubectl configured
- Docker image pushed to registry

### Create Namespace

```bash
# Create namespace
kubectl create namespace adhan

# Set as default
kubectl config set-context --current --namespace=adhan
```

### Deploy via Kubernetes Manifests

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get deployments -n adhan
kubectl get services -n adhan
kubectl get pods -n adhan

# View logs
kubectl logs -f deployment/adhan-api -n adhan

# Port forward for local testing
kubectl port-forward svc/adhan-api 8000:8000 -n adhan
```

### Kubernetes Manifests

**`k8s/deployment.yaml`**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adhan-api
  namespace: adhan
spec:
  replicas: 2
  selector:
    matchLabels:
      app: adhan-api
  template:
    metadata:
      labels:
        app: adhan-api
    spec:
      containers:
      - name: api
        image: adhan-slm:latest
        ports:
        - containerPort: 8000
        env:
        - name: ADHAN_LOG_LEVEL
          value: INFO
        - name: ADHAN_JSON_LOGS
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

**`k8s/service.yaml`**:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: adhan-api
  namespace: adhan
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: adhan-api
```

---

## yazhi-api Integration

Adhan is designed to deploy via the **yazhi-api** platform.

### Deployment Flow

```
Adhan Checkpoint
    ↓ (Register Model)
    ↓
MLflow Model Registry
    ↓ (Push to HF/S3)
    ↓
Container Registry
    ↓ (Deploy)
    ↓
yazhi-api Platform
    ↓ (REST Endpoint)
    ↓
/models/adhan-nano/infer (Live)
```

### Register Model

```python
import mlflow

# Log model to MLflow
mlflow.pytorch.log_model(model, "adhan-model")

# Register in model registry
client = mlflow.tracking.MlflowClient()
client.register_model(model_uri, "adhan-nano")
```

### Push to Registry

```bash
# Tag and push Docker image
docker tag adhan-slm:latest registry.example.com/yazhi/adhan-slm:latest
docker push registry.example.com/yazhi/adhan-slm:latest

# Push to HuggingFace Hub
huggingface-cli upload yazhi-lem/adhan-nano \
  checkpoints/adhan-nano-v0.1.0 .
```

### Deploy to yazhi-api

```bash
# Authenticate with yazhi-api
yazhi-api auth login

# Deploy model
yazhi-api models deploy \
  --model adhan-nano \
  --checkpoint ./checkpoints/adhan-nano-v0.1.0 \
  --api-version v1

# Check deployment status
yazhi-api models status adhan-nano

# View endpoints
yazhi-api models endpoints adhan-nano
```

---

## API Reference

### Authentication

Requests can include an API key header (header: `X-API-Key`) if your deployment enforces it:

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/tokenize
```

### Endpoints

#### `/health` (GET)

Health check endpoint.

```bash
curl http://localhost:8000/health

# Response
{
  "status": "ok",
  "model": "adhan-nano"
}
```

#### `/tokenize` (POST)

Tokenize Tamil text to token IDs.

```bash
curl -X POST http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d {
    "text": "தமிழ் மொழி"
  }

# Response
{
  "tokens": [1, 2, 3, 4],
  "token_ids": [1, 2, 3, 4],
  "num_tokens": 4,
  "text": "தமிழ் மொழி"
}
```

#### `/decode` (POST)

Decode token IDs back to Tamil text.

```bash
curl -X POST http://localhost:8000/decode \
  -H "Content-Type: application/json" \
  -d {
    "text": "1 2 3 4"
  }

# Response
{
  "text": "தமிழ் மொழி",
  "num_tokens": 4
}
```

#### `/generate` (POST)

Generate Tamil text from a prompt.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d {
    "text": "சொல், உனக்கு பிடித்த உணவு என்ன?",
    "max_length": 100,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.0
  }

# Response
{
  "text": "சொல், உனக்கு பிடித்த உணவு என்ன? பொங்கல்...",
  "num_tokens": 15
}
```

### Request Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Input Tamil text or space-separated token IDs |
| `max_length` | integer | No | Maximum generation length (default: 100) |
| `temperature` | float | No | Sampling temperature 0.0-2.0 (default: 0.7) |
| `top_k` | integer | No | Top-k sampling parameter (default: 50) |
| `top_p` | float | No | Nucleus sampling parameter 0.0-1.0 (default: 0.9) |
| `repetition_penalty` | float | No | Penalize repetitions ≥1.0 (default: 1.0) |

---

## Monitoring & Logging

### Structured Logging

Logs are output in JSON format for production (configurable):

```bash
# Enable JSON logging
export ADHAN_JSON_LOGS=true
export ADHAN_LOG_LEVEL=INFO

# Disable JSON logging (colored output for dev)
export ADHAN_JSON_LOGS=false
export ADHAN_LOG_LEVEL=DEBUG
```

### Log Aggregation

Send logs to external service (e.g., ELK, Datadog):

```bash
# Via environment variable
export ADHAN_LOG_DESTINATION=https://logs.example.com/ingest
```

### Metrics

Metrics are exported to MLflow:

```bash
# Start MLflow UI
mlflow ui

# View at http://localhost:5000
```

### Prometheus Metrics

Add Prometheus scraping (optional):

```python
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
request_latency = Histogram('request_latency_seconds', 'Request latency')
```

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs <container_id>

# Rebuild image
docker build --no-cache -t adhan-slm:latest .

# Check file permissions
docker run -it adhan-slm:latest bash
```

### API returning 500 errors

```bash
# Enable debug logging
export ADHAN_LOG_LEVEL=DEBUG

# Check logs
curl http://localhost:8000/health
docker logs <container_id>
```

### Out of memory

```bash
# Increase container memory
docker run -m 4g adhan-slm:latest

# Or in docker-compose
services:
  adhan-api:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Model loading fails

```bash
# Verify checkpoint exists
ls -la checkpoints/adhan-nano

# Check permissions
chmod -R 755 checkpoints/

# Mount checkpoint volume
docker run -v /path/to/checkpoints:/app/checkpoints adhan-slm:latest
```

---

## Performance Tuning

### GPU Acceleration

```bash
# Use GPU in container
docker run --gpus all adhan-slm:latest

# In kubernetes
resources:
  limits:
    nvidia.com/gpu: 1
```

### Concurrent Requests

```bash
# Run with multiple workers (via Gunicorn)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  scripts.run_api_server:app
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_tokens(text):
    return tokenizer.encode(text)
```

---

## Security

### API Key Authentication

```python
from fastapi import Depends, Header, HTTPException

async def verify_api_key(x_token: str = Header(...)):
    if x_token != os.getenv("API_KEY"):
        raise HTTPException(status_code=403)
```

### Rate Limiting

```bash
pip install slowapi

# Add rate limiting to endpoints
```

### HTTPS/TLS

```bash
# In Kubernetes, use cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/.../cert-manager.yaml
```

---

## References

- **API Docs**: http://localhost:8000/docs
- **GitHub**: https://github.com/yazhi-lem/adhan
- **Roadmap**: ROADMAP_JAX_SLM.md
- **Architecture**: docs/ARCHITECTURE_SWARAM_SLM.md
- **yazhi-api Docs**: https://yazhi-api.example.com/docs
