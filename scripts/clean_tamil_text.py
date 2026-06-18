#!/usr/bin/env python3
"""
clean_tamil_text.py — Tamil text cleaning utility for Adhan training data
ARIVU + Hermes | Rotation 26 Cycle 2 | Jun 18, 2026

Functions:
- Unicode normalization (NFC) for Tamil
- Common Tamil OCR error correction
- Remove URLs, emails, phone numbers
- Normalize whitespace (Tamil-aware)
- Sentence boundary detection (Tamil delimiters: । ॥ . ! ? + newline)
- Chunk long articles (max 512 tokens, word-aware splitting)
- Quality filter: reject if < 50% Tamil characters

Usage:
    from clean_tamil_text import clean_tamil, split_into_chunks, is_quality_text
    cleaned = clean_tamil(raw_text)
    chunks = split_into_chunks(cleaned, max_tokens=512)
"""

import re
import unicodedata

# ── Tamil Unicode constants ──────────────────────────────────────────────────

TAMIL_RANGE = (0x0B80, 0x0BFF)       # Core Tamil block
TAMIL_SUPPLEMENT = (0x11FC0, 0x11FFF)  # Tamil Supplement (rare)

# Common Tamil OCR / encoding error mappings
# These map frequently-misrecognized characters to their correct forms
OCR_CORRECTIONS = {
    # Grantha characters that appear in Tamil text (should be Tamil equivalents)
    '\u0B95\u0BCD\u0BB7': '\u0B95\u0BCD\u0BB7',  # க்ஷ (keep as-is, it's valid)
    # Common confusions in legacy encodings (TSCII → UTF-8 mojibake)
    'à®': '\u0B80',   # Common TSCII mojibake prefix
    'à¯': '\u0B80',   # Variant
    'à®¾': '\u0BBE',  # ா (aa vowel sign)
    'à®¿': '\u0BBF',  # ி (i vowel sign)
    'à¯€': '\u0BC0',  # ீ (ii vowel sign)
    'à¯': '\u0BC0',   # Variant
    'à®²': '\u0BB2',  # ல
    'à®©': '\u0BA9',  # ன
    'à®£': '\u0BA3',  # ண
    'à®¤': '\u0BA4',  # த
    'à®®': '\u0BAE',  # ம
    'à®°': '\u0BB0',  # ர
    'à®µ': '\u0BB5',  # வ
    'à®³': '\u0BB3',  # ள
    'à®±': '\u0BB1',  # ற
    'à®ª': '\u0BAA',  # ப
    'à®¨': '\u0BA8',  # ந
    'à®™': '\u0B99',  # ங
    'à®š': '\u0B9A',  # ச
    'à®ž': '\u0B9E',  # ஞ
    'à®Ÿ': '\u0B9F',  # ட
    'à®£': '\u0BA3',  # ண
    'à®•': '\u0B95',  # க
    'à®¤': '\u0BA4',  # த
    'à®ª': '\u0BAA',  # ப
    'à®®': '\u0BAE',  # ம
    'à®¯': '\u0BAF',  # ய
    'à®°': '\u0BB0',  # ர
    'à®²': '\u0BB2',  # ல
    'à®µ': '\u0BB5',  # வ
    'à®´': '\u0BB4',  # ழ
    'à®³': '\u0BB3',  # ள
    'à®±': '\u0BB1',  # ற
    'à®©': '\u0BA9',  # ன
}

# Sentence boundary delimiters for Tamil
# । (U+0964) = Devanagari danda (sometimes used in Tamil)
# ॥ (U+0965) = Devanagari double danda
# . ! ? = Standard punctuation
# \n = Newline (paragraph break)
SENTENCE_DELIMITERS = r'[।॥.!?\n]+'

# URL pattern
URL_PATTERN = re.compile(
    r'https?://[^\s<>"\')\]]+'
    r'|www\.[^\s<>"\')\]]+',
    re.IGNORECASE
)

# Email pattern
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')

# Phone number pattern (Indian formats)
PHONE_PATTERN = re.compile(
    r'(?:\+?91[\-\s]?)?[6-9]\d{9}'     # Mobile
    r'|\d{3,4}[\-\s]?\d{6,8}'           # Landline
)

# HTML tag pattern
HTML_PATTERN = re.compile(r'<[^>]+>')

# Zero-width characters
ZW_PATTERN = re.compile(r'[\u200b\u200c\u200d\ufeff]')

# Multiple whitespace (preserve newlines)
MULTI_SPACE_PATTERN = re.compile(r'[ \t]+')

# Multiple newlines
MULTI_NEWLINE_PATTERN = re.compile(r'\n{3,}')

# Common Tamil news site noise patterns (ads, navigation, etc.)
NOISE_PATTERNS = [
    re.compile(r'Share\s+(?:on|via)\s+(?:Facebook|Twitter|WhatsApp|Telegram)', re.IGNORECASE),
    re.compile(r'Click\s+here\s+to\s+', re.IGNORECASE),
    re.compile(r'Read\s+More\s*:', re.IGNORECASE),
    re.compile(r'Also\s+Read\s*:', re.IGNORECASE),
    re.compile(r'Related\s+(?:Articles?|News|Stories?)', re.IGNORECASE),
    re.compile(r'Advertisement\s*:?', re.IGNORECASE),
    re.compile(r'Sponsored\s+(?:Content|Link)', re.IGNORECASE),
    re.compile(r'©\s*\d{4}.*', re.IGNORECASE),
    re.compile(r'All\s+rights\s+reserved', re.IGNORECASE),
    re.compile(r'Follow\s+us\s+on', re.IGNORECASE),
    re.compile(r'Subscribe\s+(?:to|for)', re.IGNORECASE),
    re.compile(r'Comments?\s*:?\s*\d*', re.IGNORECASE),
    re.compile(r'பகிர்வு\s*:?', re.IGNORECASE),  # "Share:"
    re.compile(r'மேலும்\s+படிக்க\s*:?', re.IGNORECASE),  # "Read more:"
]


# ── Core functions ────────────────────────────────────────────────────────────

def is_tamil_char(c):
    """Check if character is in Tamil Unicode block."""
    cp = ord(c)
    return (TAMIL_RANGE[0] <= cp <= TAMIL_RANGE[1]) or \
           (TAMIL_SUPPLEMENT[0] <= cp <= TAMIL_SUPPLEMENT[1])


def tamil_ratio(text):
    """Calculate ratio of Tamil characters among all alphabetic characters."""
    if not text:
        return 0.0
    tamil_count = sum(1 for c in text if is_tamil_char(c))
    alpha_count = sum(1 for c in text if c.isalpha())
    return tamil_count / alpha_count if alpha_count > 0 else 0.0


def fix_ocr_errors(text):
    """Fix common Tamil OCR/encoding errors."""
    for wrong, right in OCR_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def clean_tamil(text):
    """
    Full cleaning pipeline for Tamil text.
    
    Steps:
    1. Unicode NFC normalization
    2. Remove zero-width characters
    3. Fix OCR errors
    4. Remove HTML tags
    5. Remove URLs
    6. Remove email addresses
    7. Remove phone numbers
    8. Remove noise patterns (ads, navigation)
    9. Normalize whitespace
    10. Strip
    
    Args:
        text: Raw text (may contain HTML, URLs, etc.)
    
    Returns:
        Cleaned Tamil text string
    """
    if not text:
        return ""

    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize('NFC', text)

    # Step 2: Remove zero-width characters
    text = ZW_PATTERN.sub('', text)

    # Step 3: Fix OCR errors
    text = fix_ocr_errors(text)

    # Step 4: Remove HTML tags
    text = HTML_PATTERN.sub(' ', text)

    # Step 5: Remove URLs
    text = URL_PATTERN.sub('', text)

    # Step 6: Remove email addresses
    text = EMAIL_PATTERN.sub('', text)

    # Step 7: Remove phone numbers
    text = PHONE_PATTERN.sub('', text)

    # Step 8: Remove noise patterns
    for pattern in NOISE_PATTERNS:
        text = pattern.sub('', text)

    # Step 9: Normalize whitespace
    text = MULTI_SPACE_PATTERN.sub(' ', text)
    text = MULTI_NEWLINE_PATTERN.sub('\n\n', text)

    # Step 10: Strip
    text = text.strip()

    return text


def split_sentences(text):
    """
    Split Tamil text into sentences.
    
    Uses Tamil-appropriate delimiters:
    - । (danda) and ॥ (double danda) — traditional
    - . ! ? — standard punctuation
    - \n — paragraph breaks
    
    Args:
        text: Cleaned Tamil text
    
    Returns:
        List of sentence strings
    """
    if not text:
        return []

    # Split on delimiters
    parts = re.split(SENTENCE_DELIMITERS, text)
    
    # Filter empty and very short segments
    sentences = []
    for part in parts:
        part = part.strip()
        if len(part) >= 5:  # Minimum 5 chars for a valid sentence
            sentences.append(part)
    
    return sentences


def split_into_chunks(text, max_tokens=512):
    """
    Split long text into chunks of approximately max_tokens.
    
    Uses word-aware splitting (Tamil words are space-delimited).
    Approximates tokens as words * 1.3 (Tamil BPE typically has ~1.3 tokens/word).
    
    Args:
        text: Cleaned Tamil text
        max_tokens: Maximum tokens per chunk (default 512)
    
    Returns:
        List of text chunks
    """
    if not text:
        return []

    words = text.split()
    if not words:
        return []

    # Approximate: 1 Tamil word ≈ 1.3 BPE tokens
    max_words = int(max_tokens / 1.3)
    
    chunks = []
    current_chunk = []
    current_word_count = 0

    for word in words:
        current_chunk.append(word)
        current_word_count += 1

        if current_word_count >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_word_count = 0

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def is_quality_text(text, min_tamil_ratio=0.5, min_length=100):
    """
    Quality filter for Tamil text.
    
    Args:
        text: Text to evaluate
        min_tamil_ratio: Minimum ratio of Tamil characters (default 0.5)
        min_length: Minimum character length (default 100)
    
    Returns:
        True if text passes quality filter
    """
    if not text:
        return False
    
    if len(text) < min_length:
        return False
    
    if tamil_ratio(text) < min_tamil_ratio:
        return False
    
    return True


def estimate_tokens(text):
    """
    Estimate token count for Tamil text.
    Uses character-based heuristic: ~4 chars per BPE token for Tamil.
    """
    if not text:
        return 0
    return len(text) // 4


# ── CLI interface ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python clean_tamil_text.py <command> [args]")
        print("Commands:")
        print("  clean <text>          — Clean Tamil text")
        print("  sentences <text>      — Split into sentences")
        print("  chunks <text> [max]   — Split into chunks (max tokens)")
        print("  quality <text>        — Check quality (exit 0=pass, 1=fail)")
        print("  stats <text>          — Show text statistics")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == 'clean':
        text = ' '.join(sys.argv[2:])
        print(clean_tamil(text))
    
    elif command == 'sentences':
        text = ' '.join(sys.argv[2:])
        cleaned = clean_tamil(text)
        sentences = split_sentences(cleaned)
        for i, s in enumerate(sentences):
            print(f"[{i+1}] {s}")
    
    elif command == 'chunks':
        text = ' '.join(sys.argv[2:-1]) if len(sys.argv) > 3 else ' '.join(sys.argv[2:])
        max_tokens = int(sys.argv[-1]) if len(sys.argv) > 3 and sys.argv[-1].isdigit() else 512
        cleaned = clean_tamil(text)
        chunks = split_into_chunks(cleaned, max_tokens)
        for i, chunk in enumerate(chunks):
            print(f"--- Chunk {i+1} (~{estimate_tokens(chunk)} tokens) ---")
            print(chunk)
            print()
    
    elif command == 'quality':
        text = ' '.join(sys.argv[2:])
        cleaned = clean_tamil(text)
        result = is_quality_text(cleaned)
        print(f"Quality check: {'PASS' if result else 'FAIL'}")
        print(f"  Length: {len(cleaned)} chars")
        print(f"  Tamil ratio: {tamil_ratio(cleaned):.2%}")
        sys.exit(0 if result else 1)
    
    elif command == 'stats':
        text = ' '.join(sys.argv[2:])
        cleaned = clean_tamil(text)
        sentences = split_sentences(cleaned)
        chunks = split_into_chunks(cleaned)
        print(f"Text Statistics:")
        print(f"  Raw length:     {len(text)} chars")
        print(f"  Cleaned length: {len(cleaned)} chars")
        print(f"  Tamil ratio:    {tamil_ratio(cleaned):.2%}")
        print(f"  Words:          {len(cleaned.split())}")
        print(f"  Sentences:      {len(sentences)}")
        print(f"  Chunks (512):   {len(chunks)}")
        print(f"  Est. tokens:    {estimate_tokens(cleaned)}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
