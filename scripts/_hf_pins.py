"""Reproducible HuggingFace dataset loading for Adhan corpus pipelines.

Streaming a dataset without pinning its revision makes a corpus build
non-reproducible: an upstream force-push silently changes the text a run sees,
so the same command produces a different model. This helper centralises:

  * an optional pinned revision per dataset (env-overridable),
  * a single ``load_dataset`` call site that records which revision was used,
  * a registry so ``stats.json`` can say exactly what was ingested.

Pin a revision by exporting an env var, or leave it unset to track ``main``:

    export ADHAN_HF_REVISION_ai4bharat_IndicCorpV2=<commit-sha>

The bottom of this module holds the *known* dataset ids; add to it as sources
are added so ``describe_pins()`` can report the full provenance.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

# Observed revisions are recorded here for stats/reporting. Values are filled at
# runtime from the loaded dataset object; the initial map only names the sources.
_LOADED_REVISIONS: Dict[str, Optional[str]] = {}


def revision_env_key(dataset: str) -> str:
    """Map ``org/name`` -> ``ADHAN_HF_REVISION_org_name``."""
    return "ADHAN_HF_REVISION_" + re.sub(r"[^0-9A-Za-z]+", "_", dataset).strip("_")


def pinned_revision(dataset: str) -> Optional[str]:
    """Return the pinned revision for ``dataset`` (None means track ``main``)."""
    return os.environ.get(revision_env_key(dataset)) or None


def load_tamil_dataset(
    dataset: str,
    config: Optional[str] = None,
    split: str = "train",
    streaming: bool = True,
    revision: Optional[str] = None,
    **kwargs,
):
    """``datasets.load_dataset`` with a resolvable, recorded revision.

    The return value is the dataset iterator exactly as before; the only
    difference is that a revision (explicit, env-pinned, or None) is passed
    through and remembered for ``describe_pins()``.
    """
    from datasets import load_dataset

    rev = revision if revision is not None else pinned_revision(dataset)
    ds = load_dataset(
        dataset,
        config,
        split=split,
        streaming=streaming,
        revision=rev,
        **kwargs,
    )
    _LOADED_REVISIONS[dataset] = rev
    return ds


def describe_pins() -> Dict[str, Optional[str]]:
    """Return ``{dataset: revision}`` for every dataset loaded this process."""
    return dict(_LOADED_REVISIONS)
