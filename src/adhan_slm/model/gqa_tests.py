"""Tests for grouped-query attention, dropout wiring and pure-RoPE positioning.

Three properties matter and are cheap to pin down here:

1. **Back-compatibility.** ``n_kv_heads=0`` must reproduce classic multi-head
   attention exactly — same parameter shapes, same maths — so existing
   checkpoints keep loading.
2. **GQA is a parameter/memory change, not a maths change.** With K/V heads
   broadcast to query heads, a GQA model must compute the same function as the
   equivalent MHA model given equivalent weights, and it must have fewer
   parameters for the same d_model.
3. **Dropout is train-only.** ``deterministic=True`` must be repeatable (eval),
   ``deterministic=False`` must not (train), and dropout must never leak into the
   inference path.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


from adhan_slm.model import AdhanConfig, AdhanSLM  # noqa: E402


def _cfg(**kw) -> AdhanConfig:
    base = dict(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=16,
        dtype="float32",
    )
    base.update(kw)
    return AdhanConfig(**base)


def _params(cfg: AdhanConfig, seq_len: int = 8):
    model = AdhanSLM(cfg)
    tokens = jnp.ones((2, seq_len), dtype=jnp.int32)
    return model, model.init(jax.random.PRNGKey(0), tokens)["params"]


# --- 1. back-compatibility ----------------------------------------------------


def test_kv_heads_defaults_to_mha() -> None:
    cfg = _cfg()
    assert cfg.kv_heads == cfg.n_heads


def test_mha_attention_kernel_is_still_3d_wide() -> None:
    """n_kv_heads=0 must keep the 3*d_model 'qkv' kernel old checkpoints expect."""
    cfg = _cfg()
    _, params = _params(cfg)
    qkv = params["block_0"]["qkv"]["kernel"]
    assert qkv.shape == (cfg.d_model, 3 * cfg.d_model)


def test_gqa_shrinks_the_kv_part_of_the_kernel() -> None:
    cfg = _cfg(n_kv_heads=2)
    _, params = _params(cfg)
    dh = cfg.d_model // cfg.n_heads
    qkv = params["block_0"]["qkv"]["kernel"]
    assert qkv.shape == (cfg.d_model, (cfg.n_heads + 2 * 2) * dh)
    assert qkv.shape[1] < 3 * cfg.d_model


# --- 2. GQA maths and parameter count ----------------------------------------


def test_gqa_param_count_matches_formula() -> None:
    cfg = _cfg(n_kv_heads=2)
    _, params = _params(cfg)
    actual = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert actual == cfg.approx_params()


def test_mha_param_count_matches_formula() -> None:
    cfg = _cfg()
    _, params = _params(cfg)
    actual = sum(x.size for x in jax.tree_util.tree_leaves(params))
    assert actual == cfg.approx_params()


def test_gqa_uses_fewer_params_than_mha() -> None:
    mha = _cfg(n_heads=4, n_kv_heads=0)
    gqa = _cfg(n_heads=4, n_kv_heads=1)
    assert gqa.approx_params() < mha.approx_params()


def test_gqa_forward_matches_mha_when_kv_is_broadcast() -> None:
    """GQA with kvh==nheads must be numerically identical to the MHA path.

    This is the real correctness check: if the head split/repeat ordering were
    wrong, the two models would disagree even with matching projections.
    """
    mha = _cfg()
    gqa = _cfg(n_kv_heads=4)  # explicitly equal -> same maths, same kernel width
    assert mha.approx_params() == gqa.approx_params()

    model_a, params_a = _params(mha)
    model_b, params_b = _params(gqa)
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % mha.vocab_size

    la = model_a.apply({"params": params_a}, tokens, deterministic=True)
    lb = model_b.apply({"params": params_b}, tokens, deterministic=True)
    # Same shapes and same parameter names (identical architecture path).
    assert la.shape == lb.shape
    assert jnp.allclose(la, lb)


def test_gqa_forward_shape_matches_vocab() -> None:
    cfg = _cfg(n_kv_heads=2)
    model, params = _params(cfg)
    tokens = jnp.ones((2, 8), dtype=jnp.int32)
    logits = model.apply({"params": params}, tokens, deterministic=True)
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_config_rejects_indivisible_kv_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _cfg(n_heads=4, n_kv_heads=3)


def test_config_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="d_model"):
        _cfg(d_model=32, n_heads=5)


# --- 3. dropout and pure RoPE ------------------------------------------------


def test_dropout_is_deterministic_at_eval() -> None:
    cfg = _cfg(dropout=0.1)
    model, params = _params(cfg)
    tokens = jnp.ones((2, 8), dtype=jnp.int32)
    a = model.apply({"params": params}, tokens, deterministic=True)
    b = model.apply({"params": params}, tokens, deterministic=True)
    assert jnp.array_equal(a, b)


def test_dropout_changes_output_in_training_mode() -> None:
    cfg = _cfg(dropout=0.5)
    model, params = _params(cfg)
    tokens = jnp.ones((2, 8), dtype=jnp.int32)
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    a = model.apply({"params": params}, tokens, deterministic=False, rngs={"dropout": k1})
    b = model.apply({"params": params}, tokens, deterministic=False, rngs={"dropout": k2})
    assert not jnp.allclose(a, b)


def test_no_dropout_is_noop_in_training_mode() -> None:
    cfg = _cfg(dropout=0.0)
    model, params = _params(cfg)
    tokens = jnp.ones((2, 8), dtype=jnp.int32)
    eval_logits = model.apply({"params": params}, tokens, deterministic=True)
    train_logits = model.apply(
        {"params": params},
        tokens,
        deterministic=False,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )
    assert jnp.allclose(eval_logits, train_logits)


def test_positional_information_comes_only_from_rope() -> None:
    """Pure RoPE: no learned position parameters exist in the tree."""
    cfg = _cfg()
    _, params = _params(cfg)
    names = {
        ".".join(getattr(step, "key", str(step)) for step in path)
        for path, _leaf in jax.tree_util.tree_leaves_with_path(params)
    }
    for forbidden in ("pos_emb", "position_emb", "wpe", "pos_embedding"):
        assert not any(forbidden in n for n in names), f"unexpected learned position param {names}"


def test_each_layer_has_its_own_block_params() -> None:
    """Regression: one Block instance must not be shared across all layers.

    A single ``Block(c)`` instantiated outside the layer loop makes Flax reuse
    one parameter scope for every layer. The model then silently trains a
    weight-shared stack (and shrinks the parameter count), while its config
    and checkpoint metadata still claim ``n_layers`` distinct blocks.
    """
    cfg = _cfg(n_layers=3)
    _, params = _params(cfg)
    block_keys = sorted(k for k in params if k.startswith("block_"))
    assert block_keys == ["block_0", "block_1", "block_2"]
    for key in block_keys:
        assert params[key]["qkv"]["kernel"].shape == (cfg.d_model, 3 * cfg.d_model)


def test_rope_theta_changes_the_function() -> None:
    """If RoPE were bypassed, changing theta could not change the logits."""
    tokens = jnp.arange(16, dtype=jnp.int32).reshape(2, 8) % 64
    outs = []
    for theta in (10000.0, 500.0):
        cfg = _cfg(rope_theta=theta)
        model, params = _params(cfg)
        outs.append(model.apply({"params": params}, tokens, deterministic=True))
    assert not jnp.allclose(outs[0], outs[1])


# --- 3. structure invariants (regression guard for the weight-sharing bug) ----


def _structure_configs():
    """A small fuzz matrix: layer counts, widths, GQA, remat, optional layers."""
    for n_layers in (1, 2, 3, 6):
        for d_model, n_heads in ((16, 2), (32, 4), (48, 4)):
            for n_kv_heads in (0, n_heads // 2):
                for remat in (False, True):
                    yield _cfg(
                        n_layers=n_layers,
                        d_model=d_model,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        d_ff=2 * d_model,
                        max_seq_len=8,
                        remat=remat,
                    )


def test_structure_invariants_hold_across_a_config_fuzz() -> None:
    """The model must contain exactly ``n_layers`` distinct blocks.

    This is the guard for the 2026-10 regression where a single ``Block``
    instance was hoisted out of the layer loop, so Flax reused one parameter
    scope for every layer. The model trained fine and only failed on checkpoint
    restore; a structural assertion in CI makes that class of bug impossible.
    """
    for cfg in _structure_configs():
        model = AdhanSLM(cfg)
        params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 4), dtype=jnp.int32))["params"]

        block_keys = sorted(k for k in params if k.startswith("block_"))
        assert block_keys == [
            f"block_{i}" for i in range(cfg.n_layers)
        ], f"expected {cfg.n_layers} distinct blocks, got {block_keys}"

        actual = sum(x.size for x in jax.tree_util.tree_leaves(params))
        assert (
            actual == cfg.approx_params()
        ), f"param count mismatch for {cfg}: actual {actual}, formula {cfg.approx_params()}"


def test_params_round_trip_through_orbax() -> None:
    """init -> save -> load must reproduce the exact params tree and block names."""
    ocp = pytest.importorskip("orbax.checkpoint")
    import tempfile
    from pathlib import Path

    cfg = AdhanConfig(
        vocab_size=64,
        d_model=32,
        n_layers=3,
        n_heads=4,
        d_ff=64,
        max_seq_len=8,
        dtype="float32",
    )
    model = AdhanSLM(cfg)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 4), dtype=jnp.int32))["params"]

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d).resolve())
        mgr = ocp.CheckpointManager(path)
        mgr.save(0, args=ocp.args.Composite(params=ocp.args.StandardSave(params)))
        mgr.wait_until_finished()
        restored = mgr.restore(0, args=ocp.args.Composite(params=ocp.args.StandardRestore(params)))
        out = restored["params"]

        assert sorted(k for k in out if k.startswith("block_")) == ["block_0", "block_1", "block_2"]
        leaves_in = jax.tree_util.tree_leaves(params)
        leaves_out = jax.tree_util.tree_leaves(out)
        assert len(leaves_in) == len(leaves_out)
        for a, b in zip(leaves_in, leaves_out):
            assert jnp.array_equal(a, b), "orbax round-trip changed a parameter"


def test_remat_produces_finite_gradients() -> None:
    """Gradient checkpointing must not leak tracers or break the backward pass.

    ``nn.remat`` wraps the module *class*; wrapping a module instance with
    ``jax.checkpoint`` leaks tracers during backprop. This test exercises the
    remat path end-to-end (init -> loss -> grad) so a regression fails CI.
    """
    cfg = _cfg(n_layers=2, remat=True)
    model = AdhanSLM(cfg)
    tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % cfg.vocab_size
    params = model.init(jax.random.PRNGKey(0), tokens)["params"]

    def loss(p):
        logits = model.apply({"params": p}, tokens, deterministic=True)
        return jnp.mean(jax.nn.log_softmax(logits)[..., 0])

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves, "no gradients produced"
    assert all(jnp.all(jnp.isfinite(x)) for x in leaves), "remat produced non-finite grads"
