#!/usr/bin/env python3
"""Quick MLflow status checker for Adhan SLM training.

Usage:
    .venv/bin/python scripts/mlflow_status.py              # latest run summary
    .venv/bin/python scripts/mlflow_status.py --live       # poll every 30s
    .venv/bin/python scripts/mlflow_status.py --compare    # compare all runs
"""

import argparse
import time
from pathlib import Path


def _client(uri: str = "sqlite:///mlflow.db"):
    import mlflow

    mlflow.set_tracking_uri(uri)
    from mlflow.tracking import MlflowClient

    return MlflowClient()


def latest_run(uri: str = "sqlite:///mlflow.db"):
    client = _client(uri)
    exp = client.get_experiment_by_name("adhan-slm-cpu")
    if not exp:
        print("No 'adhan-slm-cpu' experiment found.")
        return None
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=5)
    if not runs:
        print("No runs in experiment.")
        return None
    for run in runs:
        info = run.info
        print("=" * 60)
        print(f"Run ID : {info.run_id}")
        print(f"Name   : {info.run_name or '(unnamed)'}")
        print(f"Status : {info.status}")
        print(
            f"Start  : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.start_time/1000))}"
        )
        if info.end_time:
            dur = (info.end_time - info.start_time) / 1000
            print(
                f"End    : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info.end_time/1000))}"
            )
            print(f"Duration: {dur/60:.1f} min")
        print("Params :")
        for k, v in sorted(run.data.params.items()):
            print(f"    {k:40s} = {v}")
        print("Metrics (latest):")
        metrics = run.data.metrics
        for k in sorted(metrics):
            print(f"    {k:30s} = {metrics[k]:.6f}")
        # show tags
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        if tags:
            print(f"Tags   : {tags}")
    return runs[0]


def plot_loss(uri: str = "sqlite:///mlflow.db"):
    """Load metric history and print a simple ASCII loss curve."""
    client = _client(uri)
    exp = client.get_experiment_by_name("adhan-slm-cpu")
    if not exp:
        print("No experiment found.")
        return
    run = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=1)[0]
    rid = run.info.run_id
    hist = client.get_metric_history(rid, "train_loss")
    if not hist:
        print("No train_loss metrics yet.")
        return
    vals = [h.value for h in hist]
    steps = [h.step for h in hist]
    print(f"\nLoss history ({len(vals)} points):")
    print(f"  First: {vals[0]:.4f} @ step {steps[0]}")
    print(f"  Last : {vals[-1]:.4f} @ step {steps[-1]}")
    if len(vals) > 1:
        n = len(vals)
        mx, mi = max(vals), min(vals)
        for i in range(min(40, n)):
            idx = int(i * (n - 1) / 40)
            v = vals[idx]
            bar = int((v - mi) / (mx - mi + 1e-9) * 40) if mx > mi else 0
            print(f"  step {steps[idx]:6d}  {'█' * (40 - bar)}{'░' * bar}  {v:.4f}")


def check_process():
    import glob
    import subprocess

    # find the train_jax process
    result = subprocess.run(
        ["pgrep", "-f", "train_jax.*nano_existing"], capture_output=True, text=True
    )
    pid = result.stdout.strip()
    if not pid:
        print("⚠️ Training process not running.")
        return False
    ps = subprocess.run(
        ["ps", "-p", pid, "-o", "pid,pcpu,etime,vsz"], capture_output=True, text=True
    )
    print("🟢 Training process running:")
    print(ps.stdout)
    # tail log
    logfiles = glob.glob("training.log*")
    if logfiles:
        log = Path(sorted(logfiles)[-1])
        print(f"\n📄 Last 3 log lines from {log.name}:")
        for line in log.read_text().splitlines()[-3:]:
            print(f"   {line}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Check MLflow runs for Adhan SLM")
    ap.add_argument("--live", action="store_true", help="poll every 30 seconds")
    ap.add_argument("--compare", action="store_true", help="show all runs (not just latest)")
    ap.add_argument("--plot", action="store_true", help="ASCII loss curve")
    ap.add_argument("--uri", default="sqlite:///mlflow.db", help="MLflow tracking URI")
    args = ap.parse_args()

    if args.live:
        while True:
            import os

            os.system("clear")
            check_process()
            latest_run(args.uri)
            if args.plot:
                plot_loss(args.uri)
            time.sleep(30)
    else:
        check_process()
        latest_run(args.uri)
        if args.plot:
            plot_loss(args.uri)


if __name__ == "__main__":
    main()
