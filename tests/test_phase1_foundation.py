"""Tests for Phase 1 foundation hardening.

Covers:
  * reasoning-aware loss masking (only answer tokens are supervised)
  * sharding helpers degrade to no-ops on a single device
"""

from __future__ import annotations

import numpy as np
import pytest

from adhan_slm.training.sharding import (
    ShardingInfo,
    global_batch_size,
    make_data_parallel_mesh,
)


# --------------------------------------------------------------------------- #
# Reasoning-aware mask
# --------------------------------------------------------------------------- #

class FakeTokenizer:
    """Minimal tokenizer good enough for mask tests.

    Maps characters to ids so multi-char Tamil markers become multiple tokens,
    forcing the mask search to do real subsequence matching.
    """

    def __init__(self):
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.unk_id = 3
        self._next = 4
        self._table = {}

    def _tok(self, ch: str) -> int:
        if ch not in self._table:
            self._table[ch] = self._next
            self._next += 1
        return self._table[ch]

    def encode(self, text: str, add_special: bool = False):
        return [self._tok(ch) for ch in text]


@pytest.fixture()
def proto_tok():
    return FakeTokenizer()


def test_reasoning_mask_supervises_only_answer(proto_tok):
    from adhan_slm import protocol as p
    from adhan_slm.training.train_sft import encode_examples

    target = p.render_target(
        answer="மருதம்",
        thought="இது மருதத் திணை.",
        rationalize="வயல் குறிப்பு உள்ளது.",
        reflect="எனினும் சரிபார்க்கிறேன்.",
        citations=["[மேற்கோள்: ஐங்குறுநூறு 2]"],
    )
    row = {"id": "t1", "task": "x", "prompt": p.USER + " திணை என்ன?", "target": target}
    examples, stats = encode_examples([row], proto_tok, max_seq_len=512)
    assert stats["kept"] == 1
    ex = examples[0]
    ids = ex["input_ids"].tolist()
    mask = ex["mask"].tolist()

    # Answer marker token ids
    ans_ids = proto_tok.encode(p.ANSWER)
    start = None
    for i in range(len(ids) - len(ans_ids) + 1):
        if ids[i : i + len(ans_ids)] == ans_ids:
            start = i
            break
    assert start is not None, "answer marker must be present in the encoded target"

    # Everything before the answer marker is masked out (loss = 0).
    assert all(m == 0.0 for m in mask[:start])
    # Answer marker + answer + eos are supervised.
    assert any(m == 1.0 for m in mask[start:])


def test_reasoning_mask_ignores_answer_word_in_prompt(proto_tok):
    """A prompt containing 'பதில்' must not shift where supervision starts."""
    from adhan_slm import protocol as p
    from adhan_slm.training.train_sft import encode_examples

    # Prompt deliberately contains the answer marker word.
    prompt = f"{p.USER} 'பதில்:' என்ற சொல்லின் பொருள் என்ன?"
    target = p.render_target(answer="பதில்", thought="சொற்பொருள் கேள்வி.", reflect="உறுதி.")
    row = {"id": "t2", "task": "x", "prompt": prompt, "target": target}
    examples, _ = encode_examples([row], proto_tok, max_seq_len=512)
    ex = examples[0]
    ids = ex["input_ids"].tolist()
    mask = ex["mask"].tolist()

    # First supervised index must be inside the *target*, i.e. after the prompt.
    prompt_ids = proto_tok.encode(prompt)
    target_start = 1 + len(prompt_ids)
    first_supervised = next(i for i, m in enumerate(mask) if m == 1.0)
    assert first_supervised >= target_start


# --------------------------------------------------------------------------- #
# Sharding helpers
# --------------------------------------------------------------------------- #

def test_single_device_mesh_is_disabled():
    mesh, info = make_data_parallel_mesh(1)
    assert mesh is None
    assert info.enabled is False
    assert info.n_devices == 1


def test_global_batch_size_scales_down_for_devices():
    info = ShardingInfo(enabled=True, n_devices=4)
    assert global_batch_size(32, info) == 8
    single = ShardingInfo(enabled=False, n_devices=1)
    assert global_batch_size(32, single) == 32


def test_sharding_info_params():
    info = ShardingInfo(enabled=True, n_devices=4)
    params = info.as_params()
    assert params["device.n_devices"] == 4
    assert params["device.data_parallel"] is True


# --------------------------------------------------------------------------- #
# KV cache correctness
# --------------------------------------------------------------------------- #

def test_kv_cache_matches_full_forward():
    """Cached single-token decoding must equal a full-sequence forward pass."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    from adhan_slm.model import AdhanConfig, AdhanSLM

    cfg = AdhanConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64,
                      max_seq_len=32, dtype="float32", dropout=0.0)
    model = AdhanSLM(cfg)
    rng = jax.random.PRNGKey(0)
    ids = jax.random.randint(rng, (1, 6), 1, cfg.vocab_size, dtype=jnp.int32)
    params = model.init(rng, ids)["params"]

    full = model.apply({"params": params}, ids, deterministic=True)  # [1,6,V]

    # Prefill first 5 tokens, then decode token index 5 one step at a time.
    logits_prefill, cache = model.apply(
        {"params": params}, ids[:, :5], cache=None, offset=0, return_cache=True
    )
    assert len(cache) == cfg.n_layers
    logits_step, _ = model.apply(
        {"params": params}, ids[:, 5:6], cache=cache, offset=5, return_cache=True
    )

    # Prefill last position predicts token 5; cached step predicts token 6.
    # full[0, 4] predicts token 5; full[0, 5] predicts token 6.
    import numpy as np
    np.testing.assert_allclose(
        np.asarray(logits_prefill[0, -1]), np.asarray(full[0, 4]), rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(logits_step[0, -1]), np.asarray(full[0, 5]), rtol=1e-3, atol=1e-3
    )
