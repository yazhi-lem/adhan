from .aksharam_tokenizer import (
    AksharamTokenizer,
    default_aksharam_inventory,
    segment_devanagari,
)
from .jax_encode import encode_batch_jax, has_jax
from .punarchi import (
    PunarchiSplit,
    apply_rules,
    best_split,
    join,
    join_units,
    punarchi_class,
    reverse_all,
    reverse_keduthal,
    reverse_miguthal,
    reverse_thirithal,
    reverse_thondral,
)
from .swaram_tokenizer import (
    SPECIAL_TOKENS,
    WORD_MARK,
    SwaramTokenizer,
    default_akshara_inventory,
    segment_aksharas,
)

__all__ = [
    # Swaram — Dravidian (Tamil) prototype
    "SwaramTokenizer",
    "segment_aksharas",
    "default_akshara_inventory",
    # Aksharam — Indic (Hindi/Devanagari) prototype
    "AksharamTokenizer",
    "segment_devanagari",
    "default_aksharam_inventory",
    # JAX-accelerated batch encoding (shared)
    "encode_batch_jax",
    "has_jax",
    # shared constants
    "SPECIAL_TOKENS",
    "WORD_MARK",
    # Puṇarcci — classical sandhi, both directions (Layer B's grammar)
    "PunarchiSplit",
    "apply_rules",
    "best_split",
    "join",
    "join_units",
    "punarchi_class",
    "reverse_all",
    "reverse_keduthal",
    "reverse_miguthal",
    "reverse_thirithal",
    "reverse_thondral",
]
