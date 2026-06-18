#!/usr/bin/env python3
"""
tamil_tokenizer.py — Production Tamil-Aware Tokenizer (Swaram + BPE)
ARIVU + Hermes | Rotation 26 Cycle 3 | Jun 18, 2026

A production-quality Tamil tokenizer with:
  - Swaram-based decomposition (primary, per Tamil First Doctrine)
  - Morphology awareness (suffixes, verb inflections, honorifics)
  - Sandhi rules (word-boundary sound changes)
  - Colloquial variant handling (spoken → literary normalization)
  - BPE fallback (secondary, statistical tokenization)

Design: Each Tamil word decomposes into atomic swaram tokens + morphology
markers + sandhi markers + register flags. BPE provides a comparison baseline.

Compatible with train_adhan_real.py — provides encode()/decode() interface.

Reference: docs/TAMIL_FIRST_DOCTRINE.md | docs/research/SWARAM_VS_TOKEN_RESEARCH.md
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# ===========================================================================
# TAMIL UNICODE KNOWLEDGE
# ===========================================================================

# Unicode blocks
TAMIL_BLOCK_START = 0x0B80
TAMIL_BLOCK_END = 0x0BFF

# 12 Uyir (vowels)
TAMIL_UYIR = [
    'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ',
    'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ',
]

# 18 Mei (consonants) — base forms
TAMIL_MEI = [
    'க்', 'ங்', 'ச்', 'ஞ்', 'ட்', 'ண்',
    'த்', 'ந்', 'ப்', 'ம்', 'ய்', 'ர்',
    'ல்', 'வ்', 'ழ்', 'ள்', 'ற்', 'ன்',
]

# Mei letters without pulli (for decomposition)
MEI_BASE = {
    'க': 'க்', 'ங': 'ங்', 'ச': 'ச்', 'ஞ': 'ஞ்',
    'ட': 'ட்', 'ண': 'ண்', 'த': 'த்', 'ந': 'ந்',
    'ப': 'ப்', 'ம': 'ம்', 'ய': 'ய்', 'ர': 'ர்',
    'ல': 'ல்', 'வ': 'வ்', 'ழ': 'ள்', 'ள': 'ள்',
    'ற': 'ற்', 'ந': 'ன்',
}

# Special
TAMIL_AYTAM = 'ஃ'
TAMIL_PULLI = '்'

# ===========================================================================
# SWARAM CATEGORIES — Musical-atom classification
# ===========================================================================

# 12 uyir → swaram mapping (Carnatic music theory basis)
SWARAM_VOWEL = {
    # Sa (Shadjam) — foundational
    'அ': 'sa',      # short a — the root
    'ஆ': 'sa_long', # long aa

    # Ri (Rishabham) — rising
    'இ': 'ri1',     # short i
    'ஈ': 'ri_long', # long ii
    'உ': 'ri2',     # short u
    'ஊ': 'ri_long2',# long uu

    # Ga (Gandharam) — middle
    'எ': 'ga1',     # short e
    'ஏ': 'ga_long', # long ee
    'ஐ': 'ga_diph', # diphthong ai

    # Pa (Panchamam) — fixed
    'ஒ': 'pa1',     # short o
    'ஓ': 'pa_long', # long oo
    'ஔ': 'pa_diph', # diphthong au
}

# 18 mei → consonant class mapping (based on place of articulation)
CONSONANT_CLASS = {
    # Vallinam (hard/plosive) — 6
    'க்': 'C_velar_plosive',       # ka-series
    'ச்': 'C_palatal_plosive',     # ca-series
    'ட்': 'C_retroflex_plosive',   # ta-series
    'த்': 'C_dental_plosive',      # tha-series
    'ப்': 'C_labial_plosive',      # pa-series
    'ற்': 'C_alveolar_plosive',    # rra-series (unique to Tamil)

    # Mellinam (soft/nasal) — 6
    'ங்': 'C_velar_nasal',         # nga
    'ஞ்': 'C_palatal_nasal',       # nja
    'ண்': 'C_retroflex_nasal',     # nna
    'ந்': 'C_dental_nasal',        # na
    'ம்': 'C_labial_nasal',        # ma
    'ன்': 'C_alveolar_nasal',      # na (alveolar)

    # Idaiyinam (median/approximant) — 6
    'ய்': 'C_palatal_approx',      # ya
    'ர்': 'C_alveolar_trill',      # ra
    'ல்': 'C_alveolar_lateral',    # la
    'வ்': 'C_labial_approx',       # va
    'ழ்': 'C_retroflex_approx',    # zha (unique to Tamil!)
    'ள்': 'C_retroflex_lateral',   # lla
}

# Aytam
SPECIAL_MAP = {
    'ஃ': 'AYTAM',
    '்': 'PULLI',
}

# Grantha letters (Sanskrit borrowings in Tamil script)
GRANTHA_MAP = {
    'ஜ': 'C_palatal_voiced',      # ja
    'ஷ': 'C_retroflex_sibilant',  # sha
    'ஸ': 'C_dental_sibilant',     # sa
    'ஹ': 'C_glottal_fricative',   # ha
    'க்ஷ': 'C_conjunct_ksha',     # ksha
    'ஶ': 'C_palatal_sanskrit',    # sha (Sanskrit)
}

# ===========================================================================
# MORPHOLOGY DETECTION — Tamil suffixes & inflections
# ===========================================================================

# Common Tamil suffixes (atomic tokens — not char-by-char)
MORPHOLOGY_SUFFIXES = [
    # Plural markers
    'கள்', 'க்கள்',        # -kal (plural)
    # Case markers
    'ஐ',                     # accusative (-ai)
    'ஆல்', 'ஆனால்',         # instrumental / ablative (-aal)
    'இல்',                   # locative (-il)
    'உமையான்',               # agentive (-umaiyaan)
    'இடமிருந்து',            # ablative (-idamirundhu)
    'ஓடு', 'உடன்',          # sociative (-odu, -udan)
    'இன்',                   # genitive (-in)
    'க்கு',                  # dative (-kku)
    # Postpositions
    'மேல்', 'கீழ்',          # above, below
    'முன்', 'பின்',          # before, after
    'உள்ளே', 'வெளியே',      # inside, outside
]

# Verb inflection suffixes
VERB_SUFFIXES = [
    # Present tense
    'கின்றான்', 'கின்றாள்', 'கின்றார்',  # male/female/honorific present
    'கிறான்', 'கிறாள்', 'கிறார்',        # colloquial present
    'கின்றன', 'கின்றது',                    # neuter present
    # Past tense
    'ந்தான்', 'ந்தாள்', 'ந்தார்',         # male/female/honorific past
    'த்தான்', 'த்தாள்', 'த்தார்',         # dental past
    'ய்தான்', 'ய்தாள்', 'ய்தார்',         # y-past
    'டான்', 'டாள்', 'டார்',               # retroflex past
    # Future tense
    'வான்', 'வாள்', 'வார்',               # male/female/honorific future
    'ப்பான்', 'ப்பாள்', 'ப்பார்',         # pp-future
    # Infinitive / Gerund
    'கின்ற',                          # gerund present
    'வதற்கு',                         # for the purpose of
    'வதற்காக',                        # for the sake of
    # Imperative / Hortative
    'வும்', 'உம்',                    # also / inclusive
    'ஆம்',                            # yes / possibility
    'லாம்', 'வில்லை',                 # can / not
]

# Honorific markers
HONORIFIC_MARKERS = [
    'அவர்கள்',      # they (honorific)
    'கிறார்',       # present honorific
    'வார்',         # future honorific
    'ந்தார்',       # past honorific
    'ங்கள்',        # honorific plural imperative
]

# All detectable suffixes in order of length (longest first)
ALL_MORPH_SUFFIXES = sorted(
    set(MORPHOLOGY_SUFFIXES + VERB_SUFFIXES + HONORIFIC_MARKERS),
    key=len, reverse=True
)

# ===========================================================================
# SANDHI RULES — Tamil word-boundary sound changes
# ===========================================================================

# Sandhi patterns: (trigger, replacement, description)
# When tokenizing, we detect and split these
SANDHI_PATTERNS = [
    # Final m → t (makara eRu) — most common sandhi
    # மரம் + இலை → மரத்திலை
    # Pattern: word ending in ம் + next word = m→tt
    (r'(ம்)([\u0B85-\u0B89\u0B8E-\u0B8F\u0B92-\u0B93])', 'M_ERU'),  # m → tt before vowels

    # Nasal assimilation — n → n homorganic
    # பொன் + வண்டு → பொன்வண்டு (final n drops before v)
    (r'(ன்)([வய])', 'N_MELLINAM'),  # n assimilates

    # Doublet (iyal maaru) — consonant doubling
    # கல் + வடிவம் → கல்வடிவம் (ll → l)
    (r'(ள்)([\u0B85-\u0B94])', 'LL_DOUBLET'),
    (r'(ல்)([\u0B85-\u0B94])', 'L_DOUBLET'),

    # T-variants — m→nt before k/ch
    (r'(ம்)([கச])', 'M_KASHA'),

    # Vowel + vowel sandhi
    (r'([\u0B85-\u0B94])([\u0B85-\u0B94])', 'VOWEL_JOIN'),
]

def detect_sandhi(word):
    """
    Detect sandhi in a word and return (base_form, sandhi_type).
    If no sandhi detected, returns (word, None).
    This is a pre-processing step — tokens extracted separately.
    """
    for pattern, sandhi_type in SANDHI_PATTERNS:
        if re.search(pattern, word):
            return sandhi_type
    return None

# ===========================================================================
# COLLOQUIAL NORMALIZATION — spoken → literary Tamil
# ===========================================================================

# Colloquial → Literary mapping (common spoken Tamil digressions)
COLLOQUIAL_MAP = {
    # Phonological reductions
    'இல்ல': 'இல்லை',         # illai (not) — dropped final -ai
    'பண்ண': 'செய்',           # panna → sey (do) — different root
    'போடு': 'வை',             # podu → vai (put) — different root
    'கிட்ட': 'அருகில்',       # kitta → arugil (near)
    'கூட': 'உடன்',            # kooda → udan (with)
    'வெளங்க': 'புரிய',         # velanga → puriya (understand)
    'கொஞ்சம்': 'சிறிது',      # konjam → siridhu (little)
    'ரொம்ப': 'மிகவும்',       # romba → migavum (very)
    'பேசல': 'பேசவில்லை',      # pesala → pesavillai (didn't speak)
    'சொல்ற': 'சொல்கிற',       # solra → solgira (saying)

    # Grammatical simplifications
    'போறேன்': 'போகிறேன்',     # poren → pogiren (i'm going)
    'வர்றேன்': 'வருகிறேன்',   # varren → varugiren (i'm coming)
    'சாப்ட': 'சாப்பிட்ட',      # saapta → saappitta (ate)
    'இருந்துச்சு': 'இருந்தது', # irundhuchu → irundhadhu (was)
    'போச்சு': 'போய்விட்டது',   # pochu → poivittadhu (went)

    # Vocabulary differences (spoken prefers different roots)
    'வீணா': 'விருதா',          # veenaa → virudhaa (wasted)
    'ஜாலி': 'மகிழ்ச்சியான',    # jaali → magizhchiyana (fun)
    'ஃப்ரீ': 'இலவச',           # free (English) → ilavasa

    # Sentence-final particles (spoken only)
    'ண்ணா': 'அண்ணா',          # nna → anna (brother, vocative)
    'டா': '',                   # da (informal male address, removed)
    'டி': '',                   # di (informal female address, removed)
    'யா': '',                   # yaa (informal address, removed)
    'ல': 'இல்லை',              # la → illai (colloquial negation)
    'மா': '',                   # maa (informal particle, removed)
}

# Source words that, when present, flag the entire token as colloquial
COLLOQUIAL_FLAG_WORDS = set(COLLOQUIAL_MAP.keys())

def normalize_colloquial(text):
    """
    Normalize colloquial Tamil → literary Tamil.
    Returns (normalized_text, colloquial_flags) where flags is a list of
    (position, original_word, literary_word) for each substitution.
    """
    flags = []
    result_words = []
    words = text.split()
    for w in words:
        if w in COLLOQUIAL_MAP:
            lit = COLLOQUIAL_MAP[w]
            flags.append((len(result_words), w, lit))
            if lit:
                result_words.append(lit)
            # If lit is empty, the word is removed (like particles)
        else:
            result_words.append(w)
    return ' '.join(result_words), flags

# ===========================================================================
# CORE TOKENIZER — Swaram-based decomposition
# ===========================================================================

def is_tamil_char(c):
    """Check if character is in Tamil Unicode block."""
    cp = ord(c)
    return TAMIL_BLOCK_START <= cp <= TAMIL_BLOCK_END

def is_tamil_vowel_sign(c):
    """Check if character is a Tamil vowel sign (maatra)."""
    cp = ord(c)
    # Vowel signs: U+0BBE..U+0BCD (combining marks)
    return 0x0BBE <= cp <= 0x0BCD

def decompose_uyirmei(char):
    """
    Decompose a uyirmei (consonant-vowel ligature) into:
    (consonant_class, vowel_swaram, has_pulli)

    Tamil Unicode normalization: NFC decomposition gives base consonant +
    vowel sign. We then map to swaram categories.

    Returns list of token strings.
    """
    # Normalize to NFD to separate base consonant from vowel sign
    decomposed = unicodedata.normalize('NFD', char)

    tokens = []
    consonant_found = None
    vowel_sign_found = None

    for c in decomposed:
        cp = ord(c)
        if c == TAMIL_PULLI:
            tokens.append('PULLI')
        elif c == TAMIL_AYTAM:
            tokens.append('AYTAM')
        elif 0x0B95 <= cp <= 0x0BB9:
            # Base consonant (including grantha)
            # Convert to mei form with pulli for class lookup
            mei_form = c + TAMIL_PULLI
            if mei_form in CONSONANT_CLASS:
                tokens.append(CONSONANT_CLASS[mei_form])
                consonant_found = mei_form
            elif c in GRANTHA_MAP:
                tokens.append(GRANTHA_MAP[c])
                consonant_found = c
            else:
                # Unknown consonant — use raw codepoint
                tokens.append(f'C_UNK_{cp:04X}')
                consonant_found = c
        elif 0x0BBE <= cp <= 0x0BCD:
            # Vowel sign (maatra)
            vowel_sign_found = c
        elif c in GRANTHA_MAP:
            tokens.append(GRANTHA_MAP[c])
        elif c in SPECIAL_MAP:
            tokens.append(SPECIAL_MAP[c])

    # If no vowel sign, it's the inherent 'a' (sa)
    if consonant_found and not vowel_sign_found:
        tokens.append('sa')  # Inherent vowel

    return tokens

def swaram_tokenize(text, detect_morphology=True, sandhi_detection=True,
                    normalize_colloquial_flag=True, tokenize_non_tamil=False):
    """
    Primary tokenizer: Swaram-based decomposition of Tamil text.

    Args:
        text: Input Tamil text (string)
        detect_morphology: If True, detect and tokenize morphological suffixes
        detect_sandhi: If True, detect and flag sandhi patterns
        normalize_colloquial_flag: If True, normalize colloquial → literary
        tokenize_non_tamil: If True, tokenize non-Tamil instead of dropping

    Returns:
        dict with:
            'tokens': list of token strings
            'morphology': list of detected suffix markers
            'sandhi': list of detected sandhi patterns
            'colloquial': list of colloquial flags
            'original_text': the input text
    """
    result = {
        'tokens': [],
        'morphology': [],
        'sandhi': [],
        'colloquial': [],
        'original_text': text,
    }

    # Step 0: Normalize colloquial
    original_words = text.split()
    if normalize_colloquial_flag:
        normalized_text, c_flags = normalize_colloquial(text)
        result['colloquial'] = c_flags
    else:
        normalized_text = text

    # Step 1: Unicode normalization (NFC for reading, NFD for decomposition)
    normalized_text = unicodedata.normalize('NFC', normalized_text)

    # Step 2: Word-level processing
    words = re.findall(r'[\u0B80-\u0BFF]+|[^\u0B80-\u0BFF\s]+|\s+',
                       normalized_text)

    for word in words:
        if not word.strip():
            # Whitespace token
            result['tokens'].append('<SP>')
            continue

        if not is_tamil_char(word[0]):
            # Non-Tamil word
            if tokenize_non_tamil:
                if word.isdigit():
                    result['tokens'].append(f'<NUM:{word}>')
                else:
                    result['tokens'].append(f'<NT:{word}>')
            continue

        # Step 2a: Sandhi detection
        sandhi_type = None
        if detect_sandhi and len(word) > 2:
            sandhi_type = detect_sandhi(word)
            if sandhi_type:
                result['sandhi'].append((word, sandhi_type))

        # Step 2b: Morphology detection — try to strip known suffixes
        base_word = word
        suffix_found = None
        if detect_morphology and len(word) > 2:
            for suffix in ALL_MORPH_SUFFIXES:
                if len(word) > len(suffix) and word.endswith(suffix):
                    base_word = word[:-len(suffix)]
                    suffix_found = suffix
                    break

        if suffix_found:
            # Add morphology marker as separate token
            result['morphology'].append((word, suffix_found, base_word))
            result['tokens'].append(f'<MORPH:{suffix_found}>')

        # Step 2c: Character-level swaram decomposition of base word
        for char in base_word:
            if char in SWARAM_VOWEL:
                result['tokens'].append(SWARAM_VOWEL[char])
            elif char == TAMIL_PULLI:
                result['tokens'].append('PULLI')
            elif char == TAMIL_AYTAM:
                result['tokens'].append('AYTAM')
            elif char in GRANTHA_MAP:
                result['tokens'].append(GRANTHA_MAP[char])
            elif is_tamil_char(char):
                # Might be uyirmei — decompose
                decomposed = decompose_uyirmei(char)
                if decomposed:
                    result['tokens'].extend(decomposed)
                else:
                    result['tokens'].append(f'<UNK_T:{ord(char):04X}>')
            else:
                if tokenize_non_tamil:
                    result['tokens'].append(f'<NT_C:{char}>')

        # Step 2d: Add sandhi marker if detected
        if sandhi_type:
            result['tokens'].append(f'<SANDHI:{sandhi_type}>')

    return result

# ===========================================================================
# BPE TOKENIZER (Secondary / Fallback)
# ===========================================================================

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer for Tamil.
    Trained from scratch on Tamil corpus (not adapted from English).

    This implementation uses a byte-level BPE approach that works well
    with Tamil Unicode text. Uses frequency-based merge operations.
    """

    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.merges = {}           # (token1, token2) → new_token_id
        self.vocab = {}            # token → id
        self.inv_vocab = {}        # id → token
        self.special_tokens = {
            '<PAD>': 0,
            '<EOS>': 1,
            '<UNK>': 2,
            '<BOS>': 3,
        }
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.inv_vocab[idx] = token

    def _get_stats(self, corpus):
        """Count frequency of adjacent token pairs."""
        pairs = Counter()
        for tokens in corpus:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs

    def _merge_tokens(self, tokens, pair):
        """Merge all occurrences of a pair into a new single token."""
        new_token = pair[0] + pair[1]
        merged = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and
                    tokens[i] == pair[0] and
                    tokens[i + 1] == pair[1]):
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    def train(self, texts, num_merges=None):
        """
        Train BPE on a list of Tamil texts.

        Args:
            texts: List of Tamil text strings
            num_merges: Number of merge operations (default: vocab_size - 256)
        """
        if num_merges is None:
            num_merges = min(self.vocab_size - 256, 30000)

        # Step 1: Initialize with UTF-8 bytes
        corpus = []
        for text in texts:
            # Encode each character as bytes, then as individual tokens
            text = unicodedata.normalize('NFC', text)
            tokens = []
            for ch in text:
                # Tamil characters stay as-is; ASCII as bytes
                if is_tamil_char(ch):
                    tokens.append(ch)
                else:
                    # Encode non-Tamil as byte tokens
                    for byte in ch.encode('utf-8'):
                        tokens.append(f'<B:{byte:02x}>')
            if tokens:
                corpus.append(tokens)

        # Step 2: Calculate merges
        print(f"[BPE] Training on {len(corpus)} sequences, "
              f"{num_merges} merges...")

        for merge_step in range(num_merges):
            pairs = self._get_stats(corpus)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]

            # Store the merge
            new_token = best_pair[0] + best_pair[1]
            merge_id = merge_step + len(self.special_tokens)
            self.merges[best_pair] = merge_id
            self.vocab[new_token] = merge_id
            self.inv_vocab[merge_id] = new_token

            # Apply merge to all sequences
            new_corpus = []
            for tokens in corpus:
                merged = self._merge_tokens(tokens, best_pair)
                if merged:
                    new_corpus.append(merged)
            corpus = new_corpus

            if (merge_step + 1) % 5000 == 0:
                print(f"  ... {merge_step + 1}/{num_merges} merges, "
                      f"vocab={len(self.vocab)}")

        print(f"[BPE] Training complete: {len(self.vocab)} vocabulary entries")

    def encode(self, text):
        """
        Encode text to BPE token IDs.

        Args:
            text: Input text string

        Returns:
            List of integer token IDs
        """
        text = unicodedata.normalize('NFC', text)

        # Start with character-level tokens
        tokens = []
        for ch in text:
            if is_tamil_char(ch):
                tokens.append(ch)
            else:
                for byte in ch.encode('utf-8'):
                    tokens.append(f'<B:{byte:02x}>')

        if not tokens:
            return [self.special_tokens['<EOS>']]

        # Apply learned merges
        while True:
            # Find the best merge to apply
            best_pair = None
            best_rank = float('inf')

            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    rank = self.merges[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            # Apply the merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                        tokens[i] == best_pair[0] and
                        tokens[i + 1] == best_pair[1]):
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.special_tokens['<UNK>'])
        ids.append(self.special_tokens['<EOS>'])
        return ids

    def decode(self, ids):
        """
        Decode token IDs back to text.

        Args:
            ids: List of integer token IDs

        Returns:
            Decoded text string
        """
        chars = []
        for tid in ids:
            if tid == self.special_tokens['<PAD>']:
                break
            if tid == self.special_tokens['<EOS>']:
                break
            if tid == self.special_tokens['<UNK>']:
                chars.append('�')
                continue
            if tid in self.inv_vocab:
                token = self.inv_vocab[tid]
                chars.append(token)
        return ''.join(chars)

    def save(self, path):
        """Save BPE tokenizer to JSON file."""
        data = {
            'version': '1.0',
            'model_type': 'BPE',
            'language': 'Tamil',
            'vocab_size': len(self.vocab),
            'special_tokens': self.special_tokens,
            'vocab': self.vocab,
            'merges': {f'{k[0]}|||{k[1]}': v for k, v in self.merges.items()},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[BPE] Saved to {path}")

    def load(self, path):
        """Load BPE tokenizer from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data.get('vocab', {})
        self.special_tokens = data.get('special_tokens', self.special_tokens)
        # Convert string keys back to tuple keys
        self.merges = {}
        for k, v in data.get('merges', {}).items():
            parts = k.split('|||')
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = v
        # Rebuild inv_vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        print(f"[BPE] Loaded from {path} (vocab_size={len(self.vocab)})")


# ===========================================================================
# Unified Tokenizer Interface (compatible with train_adhan_real.py)
# ===========================================================================

class TamilSwaramTokenizer:
    """
    Production Tamil tokenizer with swaram + morphology + sandhi + colloquial.

    Compatible interface with train_adhan_real.py's TamilTokenizer:
    - encode(text) → list of int token IDs
    - decode(ids) → text string
    - vocab_size_actual() → int

    Also provides rich tokenization with swaram_tokenize() for analysis.
    """

    def __init__(self, max_vocab=65000):
        self.max_vocab = max_vocab
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.bos_token_id = 3

        # Build vocabulary from all swaram categories
        self._build_vocab()

    def _build_vocab(self):
        """Build the swaram vocabulary with fixed IDs."""
        vocab = {}

        # Special tokens
        vocab['<PAD>'] = 0
        vocab['<EOS>'] = 1
        vocab['<UNK>'] = 2
        vocab['<BOS>'] = 3
        vocab['<SP>'] = 4

        next_id = 5

        # Vowel swarams
        for swaram_name in sorted(set(SWARAM_VOWEL.values())):
            vocab[swaram_name] = next_id
            next_id += 1

        # Consonant classes
        for cons_class in sorted(set(CONSONANT_CLASS.values())):
            vocab[cons_class] = next_id
            next_id += 1

        # Special markers
        for marker in sorted(set(SPECIAL_MAP.values())):
            vocab[marker] = next_id
            next_id += 1

        # Grantha
        for grantha_class in sorted(set(GRANTHA_MAP.values())):
            vocab[grantha_class] = next_id
            next_id += 1

        # Morphology suffixes
        for suffix in sorted(ALL_MORPH_SUFFIXES):
            vocab[f'<MORPH:{suffix}>'] = next_id
            next_id += 1

        # Sandhi markers
        sandhi_types = sorted(set(st for _, st in SANDHI_PATTERNS))
        for s_type in sandhi_types:
            vocab[f'<SANDHI:{s_type}>'] = next_id
            next_id += 1

        # Colloquial flags
        vocab['<COLLOQ>'] = next_id
        next_id += 1

        # Non-Tamil markers
        vocab['<NUM>'] = next_id
        next_id += 1
        vocab['<NT>'] = next_id
        next_id += 1

        self._vocab = vocab
        self._inv_vocab = {v: k for k, v in vocab.items()}
        self._vocab_size = next_id

    def encode(self, text):
        """
        Encode text to token IDs (compatible with train_adhan_real.py).

        Args:
            text: Tamil text string

        Returns:
            List of integer token IDs
        """
        result = swaram_tokenize(text)
        ids = []
        for token in result['tokens']:
            if token in self._vocab:
                ids.append(self._vocab[token])
            elif token.startswith('<NUM:'):
                ids.append(self._vocab['<NUM>'])
            elif token.startswith('<NT'):
                ids.append(self._vocab['<NT>'])
            else:
                ids.append(self._vocab['<UNK>'])
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids):
        """
        Decode token IDs back to text (best-effort for swaram tokens).

        Args:
            ids: List of integer token IDs

        Returns:
            Decoded text string
        """
        parts = []
        for tid in ids:
            if tid == self.pad_token_id:
                break
            if tid == self.eos_token_id:
                break
            if tid in self._inv_vocab:
                token = self._inv_vocab[tid]
                # Only include actual tamil tokens, skip meta tokens
                if not token.startswith('<'):
                    parts.append(token)
                elif token in SWARAM_VOWEL.values():
                    parts.append(token)
                elif token in CONSONANT_CLASS.values():
                    parts.append(token)
        return ' '.join(parts)

    def vocab_size_actual(self):
        """Return the actual vocabulary size."""
        return self._vocab_size

    def rich_tokenize(self, text):
        """
        Rich tokenization returning full analysis.

        Returns:
            dict with tokens, morphology, sandhi, colloquial info
        """
        return swaram_tokenize(text)


# ===========================================================================
# ANALYSIS & COMPARISON FUNCTIONS
# ===========================================================================

def analyze_corpus(corpus_file, max_entries=1000, tokenizer_type='swaram'):
    """
    Analyze a JSONL corpus with the chosen tokenizer.

    Args:
        corpus_file: Path to JSONL file (OpenSangam format)
        max_entries: Number of entries to process
        tokenizer_type: 'swaram' or 'bpe'
    """
    print(f"[ANALYZE] Loading {corpus_file}...")
    entries = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_entries:
                break
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"[ANALYZE] Loaded {len(entries)} entries")
    print(f"[ANALYZE] Tokenizer: {tokenizer_type}")
    print()

    # Statistics
    token_counts = Counter()
    morphology_found = Counter()
    sandhi_found = Counter()
    colloquial_found = Counter()
    total_chars = 0
    total_tokens = 0
    total_words = 0
    total_morph = 0
    total_sandhi = 0
    total_colloquial = 0
    sources = Counter()

    # Track unique token types
    unique_tokens = set()

    for entry in entries:
        text = entry.get('text', '')
        source = entry.get('source', 'unknown')
        sources[source] += 1
        total_chars += len(text)

        if tokenizer_type == 'swaram':
            result = swaram_tokenize(text)
            total_tokens += len(result['tokens'])
            total_morph += len(result['morphology'])
            total_sandhi += len(result['sandhi'])
            total_colloquial += len(result['colloquial'])

            for token in result['tokens']:
                token_counts[token] += 1
                unique_tokens.add(token)

            for _, suffix, _ in result['morphology']:
                morphology_found[suffix] += 1
            for _, stype in result['sandhi']:
                sandhi_found[stype] += 1
            for _, orig, lit in result['colloquial']:
                colloquial_found[f'{orig}→{lit}'] += 1

            total_words += len(text.split())
        else:
            # For BPE, just approximate
            words = text.split()
            total_words += len(words)
            for w in words:
                token_counts[w] += 1
                unique_tokens.add(w)
            total_tokens += len(words)

    # Print statistics
    print("=" * 75)
    print("CORPUS STATISTICS")
    print("=" * 75)
    print(f"{'Metric':<40} {'Value':>15}")
    print("-" * 55)
    print(f"{'Total entries':<40} {len(entries):>15,}")
    print(f"{'Total characters':<40} {total_chars:>15,}")
    print(f"{'Total words (split)':<40} {total_words:>15,}")
    print(f"{'Total tokens':<40} {total_tokens:>15,}")
    print(f"{'Unique token types':<40} {len(unique_tokens):>15,}")
    if total_chars > 0:
        print(f"{'Tokens per char':<40} {total_tokens/total_chars:>15.3f}")
        print(f"{'Chars per token':<40} {total_chars/total_tokens:>15.2f}")

    if tokenizer_type == 'swaram':
        print(f"{'Morphology detections':<40} {total_morph:>15,}")
        print(f"{'Sandhi detections':<40} {total_sandhi:>15,}")
        print(f"{'Colloquial flags':<40} {total_colloquial:>15,}")

    print()

    # Top tokens
    print(f"Top 20 tokens ({tokenizer_type}):")
    for token, count in token_counts.most_common(20):
        token_display = token[:40] if len(token) > 40 else token
        print(f"  {token_display:<45} {count:>10,}")

    if tokenizer_type == 'swaram':
        if morphology_found:
            print(f"\nTop 10 morphology suffixes:")
            for suffix, count in morphology_found.most_common(10):
                print(f"  {suffix:<30} {count:>10,}")

        if sandhi_found:
            print(f"\nSandhi patterns detected:")
            for stype, count in sandhi_found.most_common(10):
                print(f"  {stype:<30} {count:>10,}")

        if colloquial_found:
            print(f"\nColloquial normalizations:")
            for cform, count in colloquial_found.most_common(10):
                print(f"  {cform:<40} {count:>10,}")

        # Vocabulary efficiency
        print(f"\n{'='*75}")
        print(f"VOCABULARY EFFICIENCY ANALYSIS")
        print(f"{'='*75}")

        # Count swaram vocabulary used
        swaram_vocab = sum(1 for t in unique_tokens
                          if any(t == v for v in SWARAM_VOWEL.values()))
        cons_vocab = sum(1 for t in unique_tokens
                        if any(t == v for v in CONSONANT_CLASS.values()))
        morph_vocab = sum(1 for t in unique_tokens if t.startswith('<MORPH:'))
        sandhi_vocab = sum(1 for t in unique_tokens if t.startswith('<SANDHI:'))

        print(f"Vocabulary composition:")
        print(f"  Swaram (vowel):      {swaram_vocab}")
        print(f"  Swaram (consonant):  {cons_vocab}")
        print(f"  Morphology markers:  {morph_vocab}")
        print(f"  Sandhi markers:      {sandhi_vocab}")
        print(f"  Special/meta:        {sum(1 for t in unique_tokens if t.startswith('<'))}")
        print(f"  Total unique:        {len(unique_tokens)}")

        # Source distribution
        print(f"\nSource distribution:")
        for src, count in sources.most_common():
            print(f"  {src:<30} {count:>10,}")

    print()

    return {
        'entries': len(entries),
        'total_chars': total_chars,
        'total_tokens': total_tokens,
        'unique_tokens': len(unique_tokens),
        'morphology': total_morph,
        'sandhi': total_sandhi,
        'colloquial': total_colloquial,
        'top_tokens': token_counts.most_common(20),
    }


def demo_tokenization():
    """Demonstrate tokenization on Tamil words with morphology and sandhi."""
    test_cases = [
        # Basic words
        ("வணக்கம்", "Hello (formal)"),
        ("தமிழ்", "Tamil"),
        ("யாழ்", "Yazh (the harp)"),

        # Morphology-heavy
        ("படிக்கிறான்", "He reads (present, agglutinative)"),
        ("படிக்கின்றார்கள்", "They read (present, honorific plural)"),
        ("மரங்களில்", "In the trees (sandhi: maram + kal + il)"),
        ("வீடுகளுக்கு", "To the houses (dative plural)"),
        ("பள்ளிக்கூடத்தில்", "In the school (locative)"),
        ("திருக்குறளில்", "In the Thirukkural"),

        # Sandhi examples
        ("மரத்திலை", "Tree leaf (maram + ilai → marathilai)"),
        ("பொன்வண்டு", "Golden beetle (pon + vandu → ponvandu)"),
        ("மணற்பரப்பு", "Sand surface (manal + parappu)"),

        # Colloquial
        ("கிட்ட", "Near (colloquial)"),
        ("ரொம்ப நல்லா இருக்கு", "Very good (colloquial)"),
        ("சாப்டாச்சா", "Ate? (colloquial)"),
        ("போறேன்", "I'm going (colloquial)"),

        # Mixed Sanskrit/Tamil
        ("சங்கீதம்", "Music (Sanskrit loan)"),
        ("க்ஷத்திரியன்", "Kshatriya (grantha conjunct)"),
    ]

    print("=" * 75)
    print("TAMIL SWARAM TOKENIZER — DEMONSTRATION")
    print("=" * 75)
    print(f"{'Word':<25} {'Meaning':<25} {'Tokens'}")
    print("-" * 75)

    for word, meaning in test_cases:
        result = swaram_tokenize(word)
        token_str = ' '.join(result['tokens'])
        # Truncate for display
        if len(token_str) > 55:
            token_str = token_str[:52] + '...'
        print(f"{word:<25} {meaning:<25} {token_str}")

        if result['morphology']:
            for orig, suffix, base in result['morphology']:
                print(f"  {'':>50} MORPH: {orig} = {base} + [{suffix}]")
        if result['sandhi']:
            for w, stype in result['sandhi']:
                print(f"  {'':>50} SANDHI: [{stype}]")
        if result['colloquial']:
            for pos, orig, lit in result['colloquial']:
                print(f"  {'':>50} COLLOQ: {orig} → {lit}")

    print()


def train_and_save_bpe(corpus_file, output_path, max_entries=5000):
    """Train BPE tokenizer on corpus and save to file."""
    entries = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_entries:
                break
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    texts = [e.get('text', '') for e in entries if e.get('text', '').strip()]
    print(f"[BPE] Training on {len(texts)} texts from corpus")

    bpe = BPETokenizer(vocab_size=30000)
    bpe.train(texts, num_merges=10000)
    bpe.save(output_path)

    return bpe


# ===========================================================================
# MAIN CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Tamil-Aware Production Tokenizer (Swaram + BPE)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze corpus with swaram tokenizer
  python3 tamil_tokenizer.py --analyze --corpus train.jsonl --max-entries 1000

  # Demo tokenization on sample Tamil text
  python3 tamil_tokenizer.py --demo

  # Tokenize a single text
  python3 tamil_tokenizer.py --tokenize "வணக்கம் தமிழ்"

  # Train BPE tokenizer
  python3 tamil_tokenizer.py --train-bpe --corpus train.jsonl \\
      --bpe-output models/adhan/tokenizers/bpe_merges.json

  # Analyze with BPE comparison
  python3 tamil_tokenizer.py --analyze --corpus train.jsonl \\
      --tokenizer bpe --max-entries 1000
        """
    )
    parser.add_argument('--analyze', action='store_true',
                        help='Run corpus analysis')
    parser.add_argument('--demo', action='store_true',
                        help='Run tokenization demo')
    parser.add_argument('--tokenize', type=str,
                        help='Tokenize a single text string')
    parser.add_argument('--train-bpe', action='store_true',
                        help='Train BPE tokenizer on corpus')
    parser.add_argument('--corpus', type=str,
                        default='models/sangam/release/v1.0.0/data/train.jsonl',
                        help='Path to JSONL corpus file')
    parser.add_argument('--max-entries', type=int, default=1000,
                        help='Maximum entries to process')
    parser.add_argument('--tokenizer', type=str, default='swaram',
                        choices=['swaram', 'bpe'],
                        help='Tokenizer type for analysis')
    parser.add_argument('--bpe-output', type=str,
                        default='models/adhan/tokenizers/bpe_merges.json',
                        help='Output path for trained BPE tokenizer')
    parser.add_argument('--output-stats', type=str,
                        help='Save statistics to JSON file')

    args = parser.parse_args()

    print("=" * 75)
    print("TAMIL-AWARE PRODUCTION TOKENIZER")
    print("Swaram Primary + BPE Fallback | ARIVU + Hermes")
    print("Rotation 26 Cycle 3 | Jun 18, 2026")
    print("=" * 75)
    print()

    if args.demo:
        demo_tokenization()
        return 0

    if args.tokenize:
        result = swaram_tokenize(args.tokenize)
        print(f"Input:   {args.tokenize}")
        print(f"Tokens:  {' '.join(result['tokens'])}")
        if result['morphology']:
            for orig, suffix, base in result['morphology']:
                print(f"Morph:   {orig} = {base} + [{suffix}]")
        if result['sandhi']:
            for w, stype in result['sandhi']:
                print(f"Sandhi:  [{stype}] in '{w}'")
        if result['colloquial']:
            for pos, orig, lit in result['colloquial']:
                print(f"Colloq:  {orig} → {lit}")
        return 0

    if args.train_bpe:
        if not Path(args.corpus).exists():
            print(f"ERROR: Corpus not found at {args.corpus}")
            return 1
        bpe = train_and_save_bpe(args.corpus, args.bpe_output, args.max_entries)
        print(f"\nBPE tokenizer saved to {args.bpe_output}")
        print(f"Vocabulary size: {len(bpe.vocab)}")
        return 0

    if args.analyze:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"ERROR: Corpus not found at {args.corpus}")
            print(f"Looking for: {corpus_path.absolute()}")
            # Try relative from Yazhi root
            alt_path = Path('/home/neutron/Yazhi') / args.corpus
            if alt_path.exists():
                corpus_path = alt_path
                print(f"Found at: {corpus_path}")
            else:
                return 1

        stats = analyze_corpus(
            str(corpus_path),
            max_entries=args.max_entries,
            tokenizer_type=args.tokenizer
        )

        if args.output_stats:
            output_path = Path(args.output_stats)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"Statistics saved to {args.output_stats}")

        return 0

    # Default: show help
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
