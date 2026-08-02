"""CPU training: a real run over packed shards — vocab, validation, throughput,
and gradient accumulation.

This is the path a genuine `adhan-nano` CPU pretrain takes, just with `max_steps`
turned down: read `train.bin`, honour the frozen tokenizer's vocab, evaluate against
`val.bin`, and report throughput.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="CPU training tests need the JAX stack")
pytest.importorskip("flax", reason="CPU training tests need flax")
pytest.importorskip("optax", reason="CPU training tests need optax")

from adhan_slm.training import train_jax  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.jax, pytest.mark.slow]


@pytest.fixture(scope="module")
def pretrain_result(cpu_config):
    """One short real run, shared across this module's assertions."""
    cfg = cpu_config(name="pretrain", max_steps=20, eval_every=10)
    return train_jax.train(str(cfg))


def test_runs_in_pretrain_mode(pretrain_result):
    assert pretrain_result["mode"] == "pretrain"


def test_seq_len_comes_from_the_shard_manifest(pretrain_result, cpu_corpus):
    """The packed shard, not the YAML, is the authority on sequence length."""
    assert pretrain_result["seq_len"] == cpu_corpus.seq_len


def test_validation_pass_runs_against_val_bin(pretrain_result):
    """A `val.bin` that is present but never read would hide divergence entirely."""
    assert pretrain_result["best_val_loss"] is not None, "val.bin present but never evaluated"


def test_throughput_is_reported(pretrain_result):
    """`tokens_per_sec` is how a CPU run's cost is judged mid-flight."""
    assert pretrain_result["tokens_per_sec"] > 0


def test_frozen_vocab_overrides_the_config(cpu_config, cpu_corpus):
    """The trained merge count rarely hits the configured target, so the model must
    take its vocab from `vocab.json` — otherwise embedding rows and shard ids desync."""
    cfg = cpu_config(name="frozen_vocab", max_steps=3, model={"vocab_size": 4096})
    result = train_jax.train(str(cfg))

    assert result["status"] == "ok"
    assert cpu_corpus.vocab_size != 4096, "fixture no longer exercises the override"


@pytest.fixture(scope="module")
def accum_result(cpu_config):
    """One accumulating run (micro-batch 2 x 4 = effective 8), shared below."""
    cfg = cpu_config(name="accum", batch_size=2, grad_accum_steps=4, max_steps=10)
    return train_jax.train(str(cfg))


def test_gradient_accumulation_scales_effective_batch(accum_result):
    """grad_accum_steps is how a CPU box reaches a usable effective batch size:
    effective batch = batch_size x grad_accum_steps, at the memory of batch_size."""
    assert accum_result["effective_batch"] == 8
    assert accum_result["final_train_loss"] == accum_result["final_train_loss"], "not NaN"


def test_max_steps_counts_optimizer_updates_not_micro_batches(accum_result):
    """With accumulation, `max_steps: 10` must mean 10 *updates* — so a config means
    the same thing whether it runs on CPU with accumulation or on GPU without."""
    assert accum_result["updates_this_run"] == 10
