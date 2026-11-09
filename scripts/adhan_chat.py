#!/usr/bin/env python3
"""Thin wrapper: run the packaged chat CLI from a source checkout.

The implementation lives in :mod:`adhan_slm.chat_cli` so it can also be installed
as the ``adhan-chat`` console script (see ``[project.scripts]`` in pyproject.toml).

    python scripts/adhan_chat.py --checkpoint checkpoints/adhan-nano-sft \
        --tokenizer data/final/tamil_slm --prompt "வணக்கம்! நீ யார்?"
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adhan_slm.chat_cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
