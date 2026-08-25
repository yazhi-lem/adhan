#!/usr/bin/env python3
"""Quick monitor for the autonomous Adhan SLM training run."""

import subprocess
import sys
import time


def status():
    try:
        from pathlib import Path

        log = Path("training.log")
        if not log.exists():
            print("No training.log found.")
            return
        lines = log.read_text().splitlines()
        # last few log lines
        for line in lines[-8:]:
            print(line)
        # check process
        ps = subprocess.run(
            ["pgrep", "-f", "train_jax.*nano_existing"], capture_output=True, text=True
        )
        if ps.stdout.strip():
            print(f"\nTraining process running (PID {ps.stdout.strip()})")
        else:
            print("\nTraining process not found.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        while True:
            subprocess.run(["clear"])
            status()
            time.sleep(30)
    else:
        status()
