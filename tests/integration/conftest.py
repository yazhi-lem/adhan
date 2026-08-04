"""Shared fixtures for the CPU training integration tests.

The `train_cpu_*_tests.py` modules all need the same expensive setup: a small Tamil
corpus, a frozen swaram tokenizer, and packed `train.bin` / `val.bin` shards. Building
that once per session (rather than per module) is what keeps the whole CPU suite near
three minutes rather than ten.

Everything here is jax-free on purpose, so collection still works in an environment
without the JAX stack — each test module guards itself with `pytest.importorskip`.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# A closed word list: the corpus is learnable within a handful of steps but still real
# Tamil, with aksharas the swaram tokenizer actually has to segment and merge.
WORDS: List[str] = [
    "அம்மா",
    "அப்பா",
    "பாட்டி",
    "தாத்தா",
    "வீடு",
    "பள்ளி",
    "புத்தகம்",
    "மரம்",
    "நாய்",
    "பூனை",
    "யானை",
    "மாடு",
    "சாதம்",
    "பால்",
    "தண்ணீர்",
    "பழம்",
    "நல்ல",
    "பெரிய",
    "சிறிய",
    "அழகான",
    "ஒன்று",
    "இரண்டு",
    "மூன்று",
    "படித்தேன்",
    "சென்றேன்",
    "வந்தான்",
    "விளையாடினோம்",
    "இருக்கிறது",
]

SEQ_LEN = 64
VOCAB_TARGET = 512
N_VAL_DOCS = 40


@dataclass(frozen=True)
class CpuCorpus:
    """A frozen tokenizer + packed shards on disk, ready for `train_jax.train()`."""

    shards: Path
    seq_len: int
    vocab_size: int


def write_corpus(path: Path, n_lines: int = 1200, seed: int = 7) -> Path:
    rng = random.Random(seed)
    lines = [
        " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 12))) + "." for _ in range(n_lines)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def cpu_corpus(tmp_path_factory) -> CpuCorpus:
    """Corpus -> frozen swaram tokenizer -> train.bin / val.bin.

    Mirrors `scripts/prepare_slm_corpus.py` exactly, so these tests exercise the same
    on-disk contract a real run does. Session-scoped: built once for every CPU module.
    """
    from adhan_slm.data import packing
    from adhan_slm.tokenizer import SwaramTokenizer

    out = tmp_path_factory.mktemp("slm_shards")
    docs = write_corpus(out / "corpus.txt").read_text(encoding="utf-8").splitlines()
    val_docs, train_docs = docs[:N_VAL_DOCS], docs[N_VAL_DOCS:]

    tok = SwaramTokenizer.train(train_docs, vocab_size=VOCAB_TARGET, min_freq=2)
    tok.save(str(out / "vocab.json"), str(out / "merges.txt"))

    for name, subset in (("train", train_docs), ("val", val_docs)):
        seqs = packing.pack_documents(subset, tok, seq_len=SEQ_LEN)
        assert seqs, f"{name} split too small to fill one packed sequence"
        packing.write_shard(seqs, out / f"{name}.bin", seq_len=SEQ_LEN, vocab_size=len(tok))

    return CpuCorpus(shards=out, seq_len=SEQ_LEN, vocab_size=len(tok))


@pytest.fixture(scope="session")
def cpu_config(tmp_path_factory, cpu_corpus: CpuCorpus) -> Callable[..., Path]:
    """Factory: write a CPU training config YAML and return its path.

    Call as ``cpu_config(max_steps=20, batch_size=2)`` for train overrides, or
    ``cpu_config(name="resume", model={"max_seq_len": 32})`` to reach the other
    sections. Pass a distinct ``name`` per config.

    Session-scoped so module-scoped fixtures can use it — an expensive training run
    shared across a module's assertions cannot depend on a per-test ``tmp_path``.
    """
    config_dir = tmp_path_factory.mktemp("cpu_configs")

    def _make(
        name: str = "config",
        model: Dict[str, Any] | None = None,
        checkpoint_dir: str | None = None,
        **train_overrides: Any,
    ) -> Path:
        cfg = {
            "experiment": "adhan-slm-test",
            "mlflow_uri": None,
            "checkpoint_dir": checkpoint_dir,
            "model": {
                "size": "nano",
                "vocab_size": VOCAB_TARGET,
                "max_seq_len": cpu_corpus.seq_len,
                "dtype": "bfloat16",
                **(model or {}),
            },
            "train": {
                "batch_size": 4,
                "learning_rate": 3.0e-3,
                "warmup_steps": 10,
                "max_steps": 40,
                "log_every": 20,
                "eval_every": 20,
                "checkpoint_every": 20,
                **train_overrides,
            },
            "data": {"version": "test-v0", "shards": str(cpu_corpus.shards) + "/"},
        }
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path

    return _make
