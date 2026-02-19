# -*- coding: utf-8 -*-
"""
Core constants for Adhan project.
Unified keys, hashtags, sources, and values across all modules.
"""
from typing import Dict, List

# ============================================================
# SOCIAL MEDIA - Hashtags
# ============================================================

TAMIL_HASHTAGS: List[str] = [
    # Core Tamil hashtags
    "tamil",
    "tamilculture",
    "tamilpoetry",
    "tamilnadu",
    "tamilcinema",
    "tamilmusic",
    "tamilnews",
    "tamil literature",
    "tamillanguage",
    # Extended
    "thirukkural",
    "sangam",
    "tamilheritage",
    "taamil",
    "madrasi",
    # Trending
    "tamiltrending",
    "viral",
]

ENGLISH_HASHTAGS: List[str] = [
    "tamil",
    "tamilnadu",
    "india",
    "language",
    "ai",
    "lLM",
]

# ============================================================
# DATA SOURCES
# ============================================================

SOURCES: List[str] = [
    "wikipedia",
    "literature",
    "news",
    "social",
    "local",
]

SOURCE_URLS: Dict[str, str] = {
    "wikipedia": "https://ta.wikipedia.org",
    "projectmadurai": "https://www.projectmadurai.org",
    "tamilwire": "https://tamilwire.com",
}

# ============================================================
# REDDIT SUBREDDITS
# ============================================================

REDDIT_SUBREDDITS: List[str] = [
    "Tamil",
    "tamilnadu",
    "Thiruvalluvar",
    "Thirukkural",
    "Chennai",
    "TamilCinema",
    "Kerala",
]

# ============================================================
# TELEGRAM CHANNELS (Public)
# ============================================================

TELEGRAM_CHANNELS: List[str] = [
    "TamilNews",
    "TamilCinemaUpdates",
    "TamilViral",
]

# ============================================================
# COLLOQUIAL DETECTION PATTERNS
# ============================================================

COLLOQUIAL_PATTERNS: List[str] = [
    "ன்னு",
    "ச்சு",
    "வச்சேன்",
    "வச்சி",
    "போயிட்டோன்",
    "போயிட்டேன்",
    "ஏங்க",
    "யானா",
    "யாரு",
    "என்ன",
    "எந்த",
    "டா",
    "யே",
    "யா",
    "யோ",
    "ஹெய்",
    "வா",
    "வாங்க",
    "சார்",
    "சின்ன",
]

# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_CONFIG: Dict = {
    "base_model": "xlm-roberta-base",
    "vocab_size": 250_000,
    "max_length": 512,
    "learning_rate": 5e-5,
    "batch_size": 32,
    "epochs": 10,
    "warmup_steps": 100,
    "masking_prob": 0.15,
}

# ============================================================
# DATA PATHS
# ============================================================

PATHS: Dict[str, str] = {
    "raw": "data/raw",
    "intermediate": "data/intermediate",
    "final": "data/final",
    "models": "models",
    "checkpoints": "models/checkpoints",
    "tokenized": "models/tokenized_datasets",
    "logs": "logs",
}
