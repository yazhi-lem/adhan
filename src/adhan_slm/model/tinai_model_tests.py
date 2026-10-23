"""Tests for the Tiṇai conditioning layer inside AdhanSLM (needs jax + flax)."""

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")
import jax.numpy as jnp  # noqa: E402

from adhan_slm.model import AdhanConfig, AdhanSLM, generate, tinai_aux_loss  # noqa: E402

KW = dict(
    vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_ff=64, max_seq_len=16, dtype="float32"
)
TOKENS = (jnp.arange(24).reshape(3, 8) * 5) % 64


def _init(use_tinai):
    model = AdhanSLM(AdhanConfig(**KW, use_tinai=use_tinai))
    return model, model.init(jax.random.PRNGKey(0), TOKENS)["params"]


def _activate(params):
    """Make the conditioning non-trivial (init is the identity by design)."""
    params = jax.tree_util.tree_map(lambda x: x, params)
    params["tinai_gate"] = jnp.asarray(1.0)
    for name, block in params.items():
        if name.startswith("block_"):
            block["film"]["kernel"] = 0.3 * jax.random.normal(
                jax.random.PRNGKey(7), block["film"]["kernel"].shape
            )
    return params


def test_flag_off_adds_no_parameters():
    _, plain = _init(False)
    assert not any(k.startswith("tinai") for k in plain)
    assert not any("film" in block for k, block in plain.items() if k.startswith("block_"))


def test_enabled_model_starts_identical_to_the_plain_model():
    off, p_off = _init(False)
    on, p_on = _init(True)
    assert {"tinai_emb", "tinai_gate", "tinai_router"} <= set(p_on)
    a = off.apply({"params": p_off}, TOKENS)
    b = on.apply({"params": p_on}, TOKENS)
    assert jnp.allclose(a, b, atol=1e-6)


def test_router_and_conditioning_are_causal():
    on, params = _init(True)
    params = _activate(params)
    changed = TOKENS.at[:, -1].set((TOKENS[:, -1] + 1) % 64)  # perturb only the last token
    a, ra = on.apply({"params": params}, TOKENS, return_router=True)
    b, rb = on.apply({"params": params}, changed, return_router=True)
    assert jnp.allclose(ra[:, :-1], rb[:, :-1], atol=1e-5)
    assert jnp.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_explicit_label_steers_output_and_minus_one_defers_to_router():
    on, params = _init(True)
    params = _activate(params)
    out = {t: on.apply({"params": params}, TOKENS, tinai=t) for t in (0, 3)}
    assert not jnp.allclose(out[0], out[3], atol=1e-4)
    auto = on.apply({"params": params}, TOKENS)
    deferred = on.apply({"params": params}, TOKENS, tinai=jnp.array([-1, -1, -1]))
    assert jnp.allclose(auto, deferred, atol=1e-6)


def test_aux_loss_ignores_unlabelled_examples():
    on, params = _init(True)
    _, rl = on.apply({"params": params}, TOKENS, return_router=True)
    assert float(tinai_aux_loss(rl, jnp.array([-1, -1, -1]))) == 0.0
    assert float(tinai_aux_loss(rl, jnp.array([0, 2, 4]))) > 0.0


def test_router_learns_weak_labels():
    """Toy data: tiṇai k draws tokens from its own vocabulary band. Supervising the
    router with the aux loss must make it predict the right class."""
    model, params = _init(True)
    rng = jax.random.PRNGKey(1)
    labels = jnp.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    band = jax.random.randint(rng, (10, 8), 0, 12)
    toks = band + 12 * labels[:, None]

    def loss_fn(p):
        logits, rl = model.apply({"params": p}, toks, return_router=True)
        logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
        lm = -jnp.take_along_axis(logp, toks[:, 1:, None], axis=-1).mean()
        return lm + tinai_aux_loss(rl, labels)

    grad = jax.jit(jax.grad(loss_fn))
    for _ in range(120):
        g = grad(params)
        params = jax.tree_util.tree_map(lambda w, d: w - 0.3 * d, params, g)
    _, rl = model.apply({"params": params}, toks, return_router=True)
    assert float((rl[:, -1].argmax(-1) == labels).mean()) >= 0.9


def test_generate_accepts_tinai_and_validates_it():
    on, params = _init(True)
    out = generate(on, params, [1, 2, 3], max_new_tokens=4, tinai="mullai", seed=0)
    assert len(out) == 7
    with pytest.raises(ValueError):
        generate(on, params, [1, 2, 3], max_new_tokens=1, tinai=9)
    off, p_off = _init(False)
    with pytest.raises(ValueError):
        generate(off, p_off, [1, 2, 3], max_new_tokens=1, tinai=0)
