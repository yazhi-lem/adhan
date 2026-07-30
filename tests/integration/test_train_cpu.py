"""End-to-end CPU training tests (roadmap Phase 3 + Phase C4 integration tests).

These exercise the whole CPU path with no GPU and no network: build a small Tamil
corpus, freeze the swaram tokenizer, pack shards, then train. The centrepiece is
**overfit-a-batch** — the roadmap's sanity gate before any long pretraining run. A
correctly wired causal LM memorises a single batch almost completely, so a loss that
refuses to collapse means the model, optimizer or token ids are wrong, and no amount
of hyperparameter tuning will save the real run.

Everything is sized to run in well under a minute on a CI-class 4-core CPU.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

jax = pytest.importorskip("jax", reason="CPU training tests need the JAX stack")
pytest.importorskip("flax", reason="CPU training tests need flax")
pytest.importorskip("optax", reason="CPU training tests need optax")

from adhan_slm.core.exceptions import ConfigError  # noqa: E402
from adhan_slm.data import packing  # noqa: E402
from adhan_slm.tokenizer import SwaramTokenizer  # noqa: E402
from adhan_slm.training import train_jax  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.jax, pytest.mark.slow]

# A closed word list so the corpus is learnable within a handful of steps but still
# real Tamil (aksharas the swaram tokenizer actually has to segment and merge).
_WORDS = [
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


def _write_corpus(path: Path, n_lines: int = 1200, seed: int = 7) -> Path:
    rng = random.Random(seed)
    lines = [
        " ".join(rng.choice(_WORDS) for _ in range(rng.randint(6, 12))) + "."
        for _ in range(n_lines)
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def packed_shards(tmp_path_factory) -> Path:
    """Corpus -> frozen swaram tokenizer -> train.bin / val.bin, exactly as the
    `scripts/prepare_slm_corpus.py` pipeline does it."""
    out = tmp_path_factory.mktemp("slm_shards")
    docs = _write_corpus(out / "corpus.txt").read_text(encoding="utf-8").splitlines()
    val_docs, train_docs = docs[:40], docs[40:]

    tok = SwaramTokenizer.train(train_docs, vocab_size=VOCAB_TARGET, min_freq=2)
    tok.save(str(out / "vocab.json"), str(out / "merges.txt"))

    for name, subset in (("train", train_docs), ("val", val_docs)):
        seqs = packing.pack_documents(subset, tok, seq_len=SEQ_LEN)
        assert seqs, f"{name} split too small to fill one packed sequence"
        packing.write_shard(seqs, out / f"{name}.bin", seq_len=SEQ_LEN, vocab_size=len(tok))
    return out


def _write_config(path: Path, shards: Path, **train_overrides) -> Path:
    cfg = {
        "experiment": "adhan-slm-test",
        "mlflow_uri": None,
        "checkpoint_dir": None,
        "model": {
            "size": "nano",
            "vocab_size": VOCAB_TARGET,
            "max_seq_len": SEQ_LEN,
            "dtype": "bfloat16",
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
        "data": {"version": "test-v0", "shards": str(shards) + "/"},
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_backend_is_cpu():
    """These tests are meaningless if something quietly grabbed a GPU."""
    assert jax.default_backend() == "cpu", f"expected CPU backend, got {jax.default_backend()}"


def test_smoke_run_trains_on_synthetic_data(tmp_path, packed_shards):
    """`--smoke` must work with no corpus at all — the zero-setup entry point."""
    cfg = _write_config(tmp_path / "smoke.yaml", packed_shards)
    result = train_jax.train(str(cfg), smoke=True)

    assert result["status"] == "ok"
    assert result["mode"] == "smoke"
    assert result["backend"] == "cpu"
    assert result["final_train_loss"] == result["final_train_loss"]  # not NaN


def test_overfit_a_batch_collapses_loss(tmp_path, packed_shards):
    """Roadmap Phase 3 sanity gate: one batch, repeated, must be memorised."""
    cfg = _write_config(tmp_path / "overfit.yaml", packed_shards)
    result = train_jax.train(str(cfg), overfit_batch=True, overrides={"max_steps": 120})

    assert result["mode"] == "overfit-batch"
    assert result["initial_train_loss"] > result["final_train_loss"]
    assert result["overfit_passed"], (
        "model failed to memorise a single batch "
        f"({result['initial_train_loss']:.3f} -> {result['final_train_loss']:.3f}); "
        "the model/optimizer/data wiring is broken, not the hyperparameters"
    )


def test_real_run_uses_frozen_vocab_and_validates(tmp_path, packed_shards):
    """A non-smoke run must read the shards, honour the frozen vocab, and eval."""
    cfg = _write_config(tmp_path / "real.yaml", packed_shards, max_steps=20, eval_every=10)
    result = train_jax.train(str(cfg))

    assert result["mode"] == "pretrain"
    assert result["seq_len"] == SEQ_LEN
    assert result["best_val_loss"] is not None, "val.bin present but never evaluated"
    assert result["tokens_per_sec"] > 0


def test_gradient_accumulation_scales_effective_batch(tmp_path, packed_shards):
    """grad_accum_steps is how a CPU box reaches a usable effective batch size."""
    cfg = _write_config(
        tmp_path / "accum.yaml", packed_shards, batch_size=2, grad_accum_steps=4, max_steps=10
    )
    result = train_jax.train(str(cfg))

    assert result["effective_batch"] == 8
    assert result["final_train_loss"] == result["final_train_loss"]  # not NaN


def test_checkpoint_resume_continues_step_budget(tmp_path, packed_shards):
    """Resuming must continue the global budget, not restart it from step 0.

    Long CPU runs get interrupted; a resume that silently re-ran `max_steps` from
    scratch would double the wall clock and corrupt the metric history.
    """
    pytest.importorskip("orbax.checkpoint", reason="resume test needs orbax-checkpoint")
    ckpt = tmp_path / "ckpt"

    first = tmp_path / "resume_a.yaml"
    _write_config(first, packed_shards, max_steps=10, checkpoint_every=5, eval_every=10)
    cfg = yaml.safe_load(first.read_text(encoding="utf-8"))
    cfg["checkpoint_dir"] = str(ckpt)
    first.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert train_jax.train(str(first))["status"] == "ok"

    cfg["train"]["max_steps"] = 16
    second = tmp_path / "resume_b.yaml"
    second.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = train_jax.train(str(second))

    # 10 updates were already done, so only 6 remain — not another full 16.
    assert result["status"] == "ok"
    assert result["resumed_from_step"] == 10
    assert result["updates_this_run"] == 6


def test_seq_len_longer_than_context_is_rejected(tmp_path, packed_shards):
    """A shard packed longer than model.max_seq_len is a config error, not a silent run."""
    cfg = _write_config(tmp_path / "toolong.yaml", packed_shards)
    raw = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    raw["model"]["max_seq_len"] = SEQ_LEN // 2
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="exceeds model.max_seq_len"):
        train_jax.train(str(cfg))
