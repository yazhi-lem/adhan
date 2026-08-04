"""CPU training: configs that must be rejected up front rather than mis-trained.

Each of these fails *before* the first step. The alternative — a run that starts, looks
healthy, and produces a subtly wrong model — is the expensive failure mode on CPU,
where a bad config costs hours instead of minutes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("jax", reason="CPU training tests need the JAX stack")
pytest.importorskip("flax", reason="CPU training tests need flax")
pytest.importorskip("optax", reason="CPU training tests need optax")

from adhan_slm.core.exceptions import ConfigError  # noqa: E402
from adhan_slm.training import train_jax  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.jax]


def test_shard_longer_than_model_context_is_rejected(cpu_config, cpu_corpus):
    """Training past the declared context would silently exceed what the config —
    and any later export or serving stack — promises."""
    cfg = cpu_config(name="ctx_too_short", model={"max_seq_len": cpu_corpus.seq_len // 2})

    with pytest.raises(ConfigError, match="exceeds model.max_seq_len"):
        train_jax.train(str(cfg))


def test_float16_is_rejected_on_cpu(cpu_config):
    """XLA:CPU has no float16 kernels. Silently upcasting would make the run's logged
    precision a lie, and a run that logs the wrong precision cannot be reproduced."""
    cfg = cpu_config(name="fp16", model={"dtype": "float16"})

    with pytest.raises(ConfigError, match="not supported on cpu"):
        train_jax.train(str(cfg))


def test_float32_is_accepted_on_cpu(cpu_config):
    """bf16 is the faster default, but fp32 must stay available for debugging."""
    cfg = cpu_config(name="fp32", model={"dtype": "float32"}, max_steps=3)
    result = train_jax.train(str(cfg), smoke=True)

    assert result["compute_dtype"] == "float32"


def test_dtype_aliases_are_normalised(cpu_config):
    """`bf16` in a hand-written YAML must mean bfloat16, not an unknown dtype error."""
    cfg = cpu_config(name="bf16_alias", model={"dtype": "bf16"})
    result = train_jax.train(str(cfg), smoke=True)

    assert result["compute_dtype"] == "bfloat16"


def test_missing_shards_gives_an_actionable_error(cpu_config, tmp_path):
    """The message must name the prep command — this is the most common first-run trip."""
    cfg_path = cpu_config(name="no_shards")
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["data"]["shards"] = str(tmp_path / "does_not_exist") + "/"
    broken = tmp_path / "no_shards.yaml"
    broken.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="prepare_slm_corpus"):
        train_jax.train(str(broken))
