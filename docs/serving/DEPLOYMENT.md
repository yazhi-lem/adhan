# Adhan SLM Deployment Guide

How to serve Adhan locally, in a container, and through **yazhi-api**. The production checklist and the release
process are in `factory/RUNWAY.md` and `archives/2026-10-27-docs-digest.md` §4.

## Run the API

```bash
pip install -e ".[dev,jax,tamil-nlp,serving]"
python scripts/run_api_server.py --model adhan-nano --port 8000      # --reload --log-level DEBUG for dev
# or: adhan serve --port 8000 --model adhan-nano
curl -s localhost:8000/health
```

The server needs a trained checkpoint + frozen tokenizer. With none loaded, `/health` and the inference endpoints
return **503** rather than fabricated output. Interactive docs are at `/docs` (Swagger) and `/redoc`. An
OpenAI-compatible `/v1/chat/completions` lets `bench/tamil-bench` point at it with `--endpoint`.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok" \| "loading" \| "error", "model"}` |
| POST | `/tokenize` | `{"text"}` (1–16,384 chars) | `{"tokens", "token_ids", "num_tokens", "text"}` |
| POST | `/decode` | `{"text": "1 2 3"}` | `{"text", "num_tokens"}` |
| POST | `/generate` | `{"text", "max_length", "temperature", "top_k", "top_p", "repetition_penalty"}` | generated text + token counts |

Per-IP token-bucket rate limiting and request timeouts are built in. API-key checks belong at the gateway
(yazhi-api), not in this server.

## Container

```bash
docker build -t adhan-slm:v0.1.0 .
docker run -p 8000:8000 -v $PWD/checkpoints:/app/checkpoints -v $PWD/data/final:/app/data/final adhan-slm:v0.1.0
docker compose up -d && docker compose logs -f          # local stack with health checks
```

On GPU hosts, add `--gpus all` and install the `jax-cuda` extra in the image. Scale out with uvicorn/Gunicorn
workers ≈ cores; the nano model is small enough to keep several hot.

## Through yazhi-api

checkpoint → MLflow registry (code SHA, corpus + tokenizer hashes) → image in the registry → deploy through the
yazhi-api service template → endpoint `/models/adhan-nano/infer` and the Python SDK
(`from yazhi_api import AdhanClient`). Rollout: internal canary → bounded beta → public, keeping the previous
artifact served for one-command rollback.

## Operate

- **Logs:** `ADHAN_JSON_LOGS=true` for JSON, `ADHAN_LOG_LEVEL` for verbosity; never log raw user text by default.
- **Metrics:** prompt length, tokens/s, refusal rate, p50/p95 latency. Alert on 5xx > 1%, p99 > 500 ms, and a
  bits/akshara regression > 5% in the nightly eval.
- **Troubleshooting:** a container that won't start is usually a missing checkpoint mount or permissions; a 500 means
  a debug-level log is needed; OOM means more memory or fewer workers.
- **Security:** API keys at the gateway, TLS at the edge (cert-manager on K8s), rate limits on every endpoint.
