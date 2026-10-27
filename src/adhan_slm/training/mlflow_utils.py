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

from adhan_slm.core.logging import get_logger

logger = get_logger(__name__)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def resolve_tracking_uri(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the MLflow tracking URI with a clear precedence order.

    Precedence: explicit arg > ``MLFLOW_TRACKING_URI`` env > ``None`` (MLflow's
    own default, which honours ``mlflow.set_tracking_uri``/``file:./mlruns``).

    Why an env override matters: a local ``sqlite:///mlflow.db`` cannot be
    shared by multiple training hosts and corrupts if a process dies mid-write.
    In distributed/cloud runs, point ``MLFLOW_TRACKING_URI`` at Postgres or a
    shared file store without editing every YAML, e.g.

        export MLFLOW_TRACKING_URI=postgresql://user:pass@host/mlflow
        export MLFLOW_TRACKING_URI=file:/mnt/shared/mlruns
    """
    if explicit:
        return explicit
    import os

    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return env_uri or None


def _ensure_sqlite_schema(uri: str) -> None:
    """Best-effort schema check for sqlite backends.

    An out-of-date SQLite schema makes every ``log_metric`` raise, which turns
    tracking failures into training-loop failures. We detect the common case and
    emit an actionable message instead of a stack trace deep inside MLflow.
    """
    if not uri.startswith("sqlite:"):
        return
    try:
        from mlflow.store.db.utils import _get_schema_version, _get_alembic_config
    except Exception:  # private API moved between mlflow versions
        return
    try:
        import sqlalchemy

        db_path = uri.split("///", 1)[-1]
        engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
        current = _get_schema_version(engine)
        expected = _get_alembic_config().get_main_option("revision")
        if current != expected:
            logger.error(
                "MLflow SQLite schema is out of date (found %s, expected %s). "
                "Run: mlflow db upgrade %s — or set MLFLOW_TRACKING_URI to a "
                "Postgres/file store for multi-host runs.",
                current,
                expected,
                uri,
            )
    except Exception:
        # Never let a diagnostics helper break the training run.
        return


class _NoOpRun:
    """Tracker stand-in when mlflow is absent, so training still runs unchanged."""

    def log_params(self, *_a, **_k): ...
    def log_metric(self, *_a, **_k): ...
    def log_artifact(self, *_a, **_k): ...
    def set_tag(self, *_a, **_k): ...


@contextmanager
def track_run(
    experiment: str = "adhan-slm",
    run_name: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    data_version: Optional[str] = None,
    tracking_uri: Optional[str] = None,
):
    """Context manager yielding a tracker with .log_metric / .log_artifact.

    Usage:
        with track_run("adhan-slm", "nano-run", params=cfg, data_version="v1") as run:
            run.log_metric("train_loss", loss, step=i)
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow not installed — metrics will not be tracked (pip install mlflow)")
        yield _NoOpRun()
        return

    uri = resolve_tracking_uri(tracking_uri)
    if uri:
        mlflow.set_tracking_uri(uri)
    _ensure_sqlite_schema(mlflow.get_tracking_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as active:
        logger.info(
            "mlflow run %s in experiment %r (uri %s)",
            active.info.run_id,
            experiment,
            mlflow.get_tracking_uri(),
        )
        mlflow.set_tag("code_git_sha", _git_sha())
        if data_version:
            mlflow.set_tag("data_version", data_version)
        if params:
            # flatten one level for MLflow param table
            flat = {
                k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
                for k, v in params.items()
            }
            mlflow.log_params(flat)

        class _Run:
            def log_params(self, params):
                mlflow.log_params(params)

            def log_metric(self, key, value, step=None):
                mlflow.log_metric(key, float(value), step=step)

            def log_artifact(self, path, artifact_path=None):
                mlflow.log_artifact(path, artifact_path)

            def set_tag(self, k, v):
                mlflow.set_tag(k, v)

        yield _Run()
