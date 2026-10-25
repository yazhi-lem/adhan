# Adhan SLM — SRE / MLOps Production Runbook

> **Audience**: Platform engineers, SREs, and ML infrastructure owners operating Adhan SLM in production (cloud TPU/GPU, on-prem, or sovereign air-gapped environments).
> **Version**: 1.0.0 | **Owner**: Adhan Infrastructure Team

---

## 1. Operational Overview

### 1.1 Service Tiers

| Tier | Component | SLO | Latency Target | Critical Path |
|------|-----------|-----|----------------|---------------|
| P0 | Training controller (train_jax / train_sft) | 99.5% success rate | N/A (batch) | Yes |
| P0 | Checkpoint storage (Orbax + GCS/S3) | 99.99% durability | <5s save latency | Yes |
| P1 | Inference API (serving/api.py) | 99.9% availability | p99 <200ms/token | Yes |
| P1 | MLflow tracking | 99.5% availability | <1s metric ingest | No |
| P2 | Corpus ingestion cron | 95% daily success | <4h per run | No |

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ADHAN SLM PRODUCTION MESH                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Ingress                                                                  │
│    ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│    │ Cloud LB     │────▶│ FastAPI      │────▶│ Rate Limiter         │   │
│    │ (HTTPS)      │     │ (Uvicorn)    │     │ (Token Bucket/IP)    │   │
│    └──────────────┘     └──────────────┘     └──────────────────────┘   │
│                                                       │                   │
│  Inference Layer                                      ▼                   │
│    ┌──────────────────────────────────────────────────────────────┐     │
│    │  AdhanInferenceAPI                                           │     │
│    │    • Async model loading (loading → ok → error states)       │     │
│    │    • Semaphore-bounded concurrent generation (default 8)     │     │
│    │    • Per-request timeout (default 60s)                       │     │
│    └──────────────────────────────────────────────────────────────┘     │
│                              │                                            │
│  Model Artifact Store        ▼                                            │
│    ┌──────────────────────────────────────────────────────────────┐     │
│    │  Orbax CheckpointManager                                     │     │
│    │    • params/  (inference)                                    │     │
│    │    • state/   (training resume)                              │     │
│    │    • metadata/ (config hash, model_config, tokenizer_dir)    │     │
│    └──────────────────────────────────────────────────────────────┘     │
│                              │                                            │
│  Observability               ▼                                            │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐    │
│    │ MLflow       │  │ Structured   │  │ Prometheus / CloudWatch  │    │
    │ (sqlite/GCS) │  │ JSON Logs    │  │ (optional metrics export)│    │
│    └──────────────┘  └──────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deployment

### 2.1 Docker Image

```dockerfile
# Dockerfile.prod
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV ADHAN_LOG_DIR=/var/log/adhan
ENV ADHAN_JSON_LOGS=true

# Install JAX with the correct backend (CPU / CUDA / TPU)
ARG JAX_BACKEND=cpu
COPY requirements-jax.txt .
RUN pip install --no-cache-dir -r requirements-jax.txt \
    && pip install --no-cache-dir "jax[${JAX_BACKEND}]"

COPY pyproject.toml .
COPY src/ ./src/
RUN pip install -e .

# Non-root user for security
RUN useradd -m -u 1000 adhan && mkdir -p /var/log/adhan && chown -R adhan:adhan /var/log/adhan
USER adhan

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

ENTRYPOINT ["uvicorn", "adhan_slm.serving.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.2 Kubernetes Deployment

```yaml
# k8s/adhan-inference.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: adhan-inference
  labels:
    app: adhan-inference
spec:
  replicas: 2
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: adhan-inference
  template:
    metadata:
      labels:
        app: adhan-inference
    spec:
      containers:
        - name: adhan
          image: ghcr.io/yazhi-lem/adhan-slm:latest
          ports:
            - containerPort: 8000
          env:
            - name: ADHAN_CHECKPOINT_DIR
              value: "/mnt/checkpoints/adhan-nano-sft"
            - name: ADHAN_TOKENIZER_DIR
              value: "/mnt/checkpoints/adhan-nano-sft"
            - name: ADHAN_LOG_DIR
              value: "/var/log/adhan"
            - name: ADHAN_JSON_LOGS
              value: "true"
          resources:
            requests:
              memory: "2Gi"
              cpu: "1000m"
            limits:
              memory: "4Gi"
              cpu: "2000m"
          volumeMounts:
            - name: checkpoints
              mountPath: /mnt/checkpoints
              readOnly: true
            - name: logs
              mountPath: /var/log/adhan
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
      volumes:
        - name: checkpoints
          persistentVolumeClaim:
            claimName: adhan-checkpoints-pvc
        - name: logs
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: adhan-inference
spec:
  selector:
    app: adhan-inference
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

### 2.3 MLflow Database Migration (Critical)

MLflow SQLite schema must be upgraded before every deployment where the MLflow version changes.

```bash
# In Dockerfile ENTRYPOINT or init container
mlflow db upgrade sqlite:///mlflow.db
```

**If you skip this**: All tracking calls raise `MlflowException: Detected out-of-date database schema`. Training runs become untrackable.

---

## 3. Checkpoint Resume Protocol

### 3.1 Automatic Resume

The training loop now supports **automatic resume** with config-hash verification:

```python
# On startup, the loop checks:
# 1. Is there a latest checkpoint in checkpoint_dir?
# 2. Does the saved metadata.config_hash match the current YAML hash?
# 3. If yes → restore optimizer state + params, continue from completed steps.
# 4. If no  → warn and start fresh (prevents mixing incompatible runs).
```

### 3.2 Resume Safety Rules

| Scenario | Behavior |
|----------|----------|
| Same config, new run (preempted VM) | Auto-resume from latest step |
| Changed LR / batch size / architecture | **Refuses resume** — hash mismatch warning |
| Explicit warm restart | Set `warm_restart_on_resume: true` in config |
| New checkpoint_dir | Fresh start regardless of other dirs |

### 3.3 Manual Recovery Procedure

```bash
# 1. Identify the last good checkpoint
ls -lt checkpoints/adhan-nano-sft/
# 2. If corrupted, restore from backup (you have backups, right?)
# 3. Resume with explicit step override if needed
python -m adhan_slm.training.train_jax \
    --config src/adhan_slm/configs/adhan_slm_nano_cpu.yaml \
    --checkpoint-dir checkpoints/adhan-nano-sft
```

---

## 4. Observability

### 4.1 Structured Logging

All logs are structured JSON when `ADHAN_JSON_LOGS=true`:

```json
{
  "timestamp": "2025-01-15T08:23:14.123456",
  "level": "INFO",
  "logger": "adhan_slm.training.train_jax",
  "message": "step  1234  loss 2.3412  ppl 10.39  tok/s 12,345",
  "module": "train_jax",
  "function": "_flush",
  "line": 512,
  "correlation_id": "req-7a3f9e2b"
}
```

**File rotation**: 10 MB per file, 5 backups retained.

### 4.2 Key Metrics to Alert On

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| `train_loss` spike >3× median | 5 min | 15 min | Check data pipeline |
| `val_perplexity` increase >20% vs best | 1 eval | 3 evals | Early stopping trigger |
| `tokens_per_sec` <50% of baseline | 10 min | 30 min | Check VM preemption / CPU throttling |
| `checkpoint_save_latency_sec` >30s | 1 save | 3 saves | Check storage I/O |
| Inference p99 latency >500ms | 5 min | 15 min | Scale replicas |
| Inference 5xx rate >1% | 1 min | 5 min | Rollback / investigate |

### 4.3 Health Check States

```
/healthz returns:

  { "status": "loading", "model": "adhan-nano" }
    → Return 503 from load balancer (not ready yet)

  { "status": "ok", "model": "adhan-nano" }
    → Return 200, route traffic

  { "status": "error", "model": "adhan-nano", "error": "..." }
    → Return 503, page on-call (terminal load failure)
```

---

## 5. Scaling & Sharding

### 5.1 Data Parallel (Current Priority)

Add to your training config:

```yaml
# In config YAML or CLI override
train:
  mesh_shape: [4]          # 4 devices data-parallel
  mesh_axis_names: ["dp"]
```

The training loop uses `jax.sharding` to shard batches across devices:

```python
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils

# P1: Data-parallel training path
devices = mesh_utils.create_device_mesh((4,))
mesh = Mesh(devices, axis_names=("dp",))
sharding = NamedSharding(mesh, PartitionSpec("dp", None))

# Shard the batch across devices
batch = jax.device_put(batch, sharding)
```

### 5.2 FSDP / Tensor Parallel (Future)

For `adhan-mini` (110M params) on multi-GPU, consider:
- **FSDP**: Shard optimizer states + params across data-parallel replicas
- **Tensor parallel**: Only needed if model weights exceed single-device HBM (not yet true for Adhan sizes)

---

## 6. Security & Compliance

### 6.1 COPPA / Children’s Privacy (yazh-unity Integration)

- Never log child-generated prompts or PII
- Use `ADHAN_LOG_DIR` with 7-day retention for inference logs
- Disable MLflow artifact logging for child-facing endpoints

### 6.2 Model Provenance

Every checkpoint carries self-describing metadata:

```json
{
  "model_config": { "vocab_size": 8000, "d_model": 256, ... },
  "tokenizer_dir": "data/final/nano_300m_s512",
  "base_checkpoint": "checkpoints/adhan-nano-ta",
  "config_hash": "a1b2c3d4e5f67890"
}
```

**Verify before deployment**:

```bash
python -c "
from adhan_slm.inference import load_checkpoint_metadata
meta = load_checkpoint_metadata('checkpoints/adhan-nano-sft')
print(meta['config_hash'])
"
```

---

## 7. Disaster Recovery

| Event | RPO | RTO | Procedure |
|-------|-----|-----|-----------|
| VM preemption | 0 (checkpoint every N steps) | <5 min | Auto-resume from latest Orbax step |
| Checkpoint corruption | Last good step | <30 min | Restore from GCS/S3 backup, resume |
| MLflow DB loss | Last metric flush | <1h | Recreate DB; training continues unmonitored |
| Full region outage | Last GCS cross-region replication | <4h | Fail over to DR region, restore checkpoints |

---

## 8. Runbooks

### 8.1 "Training Loss is NaN"

1. Check learning rate — reduce by 10×
2. Check `grad_accum_steps` — too large with bf16 can overflow
3. Check input data — corrupted shard? Run `scripts/phase2_validate.py`
4. Enable `remat: true` in model config (memory pressure → numerical instability)

### 8.2 "Inference API Returns 503"

1. Check `/healthz` — is status `loading` or `error`?
2. If `loading`: wait (cold start on large checkpoint can take 2–5 min)
3. If `error`: check logs for `FileNotFoundError` on checkpoint/tokenizer paths
4. Verify `ADHAN_CHECKPOINT_DIR` env var points to existing Orbax directory

### 8.3 "MLflow Not Tracking"

1. Check `mlflow.db` schema version: `mlflow db status sqlite:///mlflow.db`
2. If out of date: `mlflow db upgrade sqlite:///mlflow.db`
3. Check disk space — SQLite can fail silently when partition is full
4. Verify `mlflow_uri` in config matches the DB path

---

## 9. Contact

- **On-call**: #adhan-sre-alerts (Slack)
- **Escalation**: infra@yazhi.dev
- **Runbook updates**: PR to `docs/MLOPS_SRE_RUNBOOK.md`
