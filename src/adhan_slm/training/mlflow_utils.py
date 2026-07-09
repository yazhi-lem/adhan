"""MLflow experiment-tracking helpers for Adhan SLM training.

Enforces the reproducibility contract from docs/ARCHITECTURE_SWARAM_SLM.md §4:
every run logs the full config, the code git SHA, and the dataset version, so any
run can be re-derived. If mlflow is not installed, a no-op tracker is returned so
smoke runs still work.
"""
from __future__ import annotations

import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class _NoOpRun:
    def log_params(self, *_a, **_k): ...
    def log_metric(self, *_a, **_k): ...
    def log_artifact(self, *_a, **_k): ...
    def set_tag(self, *_a, **_k): ...


@contextmanager
def track_run(experiment: str = "adhan-slm",
              run_name: Optional[str] = None,
              params: Optional[Dict[str, Any]] = None,
              data_version: Optional[str] = None,
              tracking_uri: Optional[str] = None):
    """Context manager yielding a tracker with .log_metric / .log_artifact.

    Usage:
        with track_run("adhan-slm", "nano-run", params=cfg, data_version="v1") as run:
            run.log_metric("train_loss", loss, step=i)
    """
    try:
        import mlflow
    except ImportError:
        print("[mlflow-utils] mlflow not installed — metrics will not be tracked.")
        yield _NoOpRun()
        return

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("code_git_sha", _git_sha())
        if data_version:
            mlflow.set_tag("data_version", data_version)
        if params:
            # flatten one level for MLflow param table
            flat = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
                    for k, v in params.items()}
            mlflow.log_params(flat)

        class _Run:
            def log_metric(self, key, value, step=None):
                mlflow.log_metric(key, float(value), step=step)

            def log_artifact(self, path, artifact_path=None):
                mlflow.log_artifact(path, artifact_path)

            def set_tag(self, k, v):
                mlflow.set_tag(k, v)

        yield _Run()
