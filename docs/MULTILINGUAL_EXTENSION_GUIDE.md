# Multilingual Adhan SLM — Extension Guide for Indian & Dravidian Languages

> **Audience**: NLP researchers, linguists, and engineers extending Adhan's swaram architecture to Kannada, Telugu, Malayalam, Hindi, Bengali, and other Indic scripts.
> **Prerequisite**: Understanding of `docs/ARCHITECTURE_SWARAM_SLM.md` and the swaram tokenizer design.

---

## 1. Design Philosophy: One Architecture, Many Scripts

Adhan's core insight is **script-native tokenization** — rather than forcing all languages through a byte-level or Latin-centric BPE, we build a **deterministic grapheme-cluster layer** (Layer A) specific to each script's Unicode combining behavior, then share the **morpheme-merge layer** (Layer B) across related languages.

### 1.1 Why This Works for India

India's language landscape is uniquely suited to this approach:

| Language Family | Scripts | Shared Features | Tokenizer Strategy |
|-----------------|---------|-----------------|-------------------|
| Dravidian | Tamil, Kannada, Telugu, Malayalam | Agglutination, sandhi, matra combining | Common Layer A pattern, script-specific inventories |
| Indo-Aryan | Devanagari (Hindi, Marathi), Bengali, Gujarati, Odia, Punjabi | Conjunct consonants (kuṭṭiyaluttu/যুক্তাক্ষর), matras | Conjunct-aware segmentation + BPE |
| Tibeto-Burman | Manipuri (Meitei Mayek) | Smaller corpus, endangered | Transfer from Bengali/Devanagari Layer B |
| Austroasiatic | Santali (Ol Chiki) | Non-Brahmic, syllabic | Character-level Layer A, aggressive BPE |

---

## 2. Extending the Tokenizer

### 2.1 Add a New Script: The 5-Step Recipe

Let's walk through adding **Kannada** as an example.

#### Step 1: Define the Script's Unicode Geometry

Create `src/adhan_slm/tokenizer/kannada_akshara.py`:

```python
"""Kannada akshara segmentation — Layer A for ಕನ್ನಡ."""

# Kannada Unicode Block: U+0C80–U+0CFF
_KND_MATRAS = set(range(0x0CBE, 0x0CD5))  # ಾ ಿ ೀ ು ೂ ೃ ೄ ೆ ೇ ೈ ೊ ೋ ೌ
_KND_HALANT = 0x0CCD  # ್ (virama)
_KND_COMBINING = _KND_MATRAS | {_KND_HALANT}

# Base consonants (34) + vowels (16)
_KND_VOWELS = list("ಅಆಇಈಉಊಋಎಏಐಒಓಔಅಂಅಃ")  # Simplified; use actual Unicode
_KND_CONSONANTS = list("ಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹ")

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>"]
WORD_MARK = "▁"
```

#### Step 2: Implement Akshara Segmentation

```python
def segment_kannada_aksharas(text: str) -> list[str]:
    """Lossless segmentation: ''.join(out) == text."""
    out = []
    i = 0
    while i < len(text):
        cp = ord(text[i])
        # Start of a new akshara?
        if 0x0C80 <= cp <= 0x0CFF and cp not in _KND_COMBINING:
            cluster = text[i]
            i += 1
            # Absorb all following combining marks
            while i < len(text) and ord(text[i]) in _KND_COMBINING:
                cluster += text[i]
                i += 1
            out.append(cluster)
        else:
            out.append(text[i])
            i += 1
    return out
```

#### Step 3: Build the Closed Inventory

```python
def default_kannada_inventory() -> list[str]:
    """Complete base akshara set for Kannada."""
    inv = list(_KND_VOWELS)
    matras = [chr(cp) for cp in range(0x0CBE, 0x0CD5)]
    for c in _KND_CONSONANTS:
        inv.append(c)  # inherent 'a' form
        inv.append(c + chr(_KND_HALANT))  # halant form
        for m in matras:
            inv.append(c + m)  # consonant + matra
    # Deduplicate
    seen, out = set(), []
    for a in inv:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out
```

#### Step 4: Train Layer B (BPE) on Multilingual Corpus

```bash
# Combine Kannada corpus with Tamil (transfer learning for morphological patterns)
python scripts/freeze_fast_tokenizer.py \
    --corpus data/raw/kannada_expanded \
    --out data/final/kannada_slm \
    --vocab-size 28000 \
    --seq-len 512 \
    --base-inventory src/adhan_slm/tokenizer/kannada_akshara.py
```

**Key insight**: Because Kannada and Tamil share agglutinative morphology, the BPE merge table from Tamil **bootstraps** Kannada training. Start with Tamil's `merges.txt` and continue training on Kannada text — this cuts required corpus size by ~40%.

#### Step 5: Register in the Tokenizer Factory

```python
# In src/adhan_slm/tokenizer/__init__.py
from .kannada_akshara import KannadaTokenizer, segment_kannada_aksharas

_REGISTRY = {
    "swaram": SwaramTokenizer,
    "aksharam": AksharamTokenizer,
    "kannada": KannadaTokenizer,
}
```

---

## 3. Script-Specific Considerations

### 3.1 Devanagari (Hindi, Marathi, Sanskrit)

**Challenge**: Conjunct consonants (क्ष, ज्ञ, त्र, श्र) are **not** decomposable in Unicode. They have their own codepoints.

**Solution**: Add them explicitly to the closed inventory:

```python
_DEVA_CONJUNCTS = ["क्ष", "ज्ञ", "त्र", "श्र", "द्य", "द्व"]  # etc.
```

**Challenge**: Half-forms (क्‍,द्‍) used in words like स्‍वागत. Unicode uses ZWNJ (U+200C) which complicates segmentation.

**Solution**: Treat ZWNJ as an **invisible combining mark** that stays attached to the preceding consonant:

```python
_DEVA_ZWNJ = 0x200C
# In segmenter: absorb ZWNJ into the cluster
```

### 3.2 Bengali (Bangla)

**Challenge**: The hasant (্) behaves differently from Devanagari virama. Some conjuncts have **arbitrary ligature forms** (e.g., ক্ষ = ক + ্ + ষ, but displayed as a single glyph).

**Solution**: Normalize at Layer A by always decomposing ligatures to their consonant+hasant+consonant sequence. This makes the tokenizer **orthography-agnostic** — it doesn't care about display glyphs, only about the underlying phonological structure.

```python
def normalize_bengali_ligatures(text: str) -> str:
    """Expand visual ligatures to phonological sequences."""
    # Use ICU transliterator or hardcoded mapping
    LIGATURE_MAP = {
        "ক্ষ": "ক্‌ষ",  # ksha
        "জ্ঞ": "জ্‌ঞ",  # gyana
        # ... etc
    }
    for lig, expanded in LIGATURE_MAP.items():
        text = text.replace(lig, expanded)
    return text
```

### 3.3 Telugu & Malayalam

**Challenge**: These scripts have **vowel signs that attach to the left** of the consonant (pre-base matras), breaking simple left-to-right cluster logic.

**Solution**: Use the Unicode grapheme cluster algorithm (`unicodedata.grapheme` in Python 3.13+, or the `regex` module with `\X`) as the **first pass**, then apply script-specific rules as the second pass.

```python
import regex

def segment_telugu_graphemes(text: str) -> list[str]:
    """Use Unicode grapheme clusters as the foundation."""
    return regex.findall(r'\X', text)
```

---

## 4. Multilingual Training Strategies

### 4.1 Vocabulary Sharing

| Strategy | Vocab Size | Pros | Cons |
|----------|-----------|------|------|
| **Script-specific** | 8K–16K/script | Best fertility per language | No cross-lingual transfer |
| **Shared Dravidian** | 24K | Tamil-Kannada-Telugu-Malayalam share subwords | Slightly worse fertility per language |
| **Pan-Indic** | 48K | One model serves all 22 languages | High fertility for smaller scripts |
| **Script-agnostic byte** | 256 | Universal, simple | 3–6 tokens per akshara (terrible) |

**Recommendation**: Start with **Shared Dravidian** for the four southern languages, then add a **Devanagari** cluster. Use separate output heads (or language IDs) only if evidence shows interference.

### 4.2 Data Mixing Ratios

For a multilingual pretraining corpus:

```yaml
# config: multilingual_mix.yaml
data:
  sources:
    - name: tamil
      path: data/raw/tamil_expanded
      weight: 0.30
      script: dravidian
    - name: kannada
      path: data/raw/kannada_expanded
      weight: 0.20
      script: dravidian
    - name: telugu
      path: data/raw/telugu_expanded
      weight: 0.20
      script: dravidian
    - name: malayalam
      path: data/raw/malayalam_expanded
      weight: 0.15
      script: dravidian
    - name: hindi
      path: data/raw/hindi_expanded
      weight: 0.15
      script: devanagari
```

**Rule**: Upsample lower-resource languages (Malayalam, Kannada) to match Tamil's token count. This prevents the model from becoming "Tamil with a Kannada accent."

### 4.3 Language ID Conditioning

Extend Tiṇai conditioning to a general **Language/Router conditioning**:

```python
# In model/transformer.py
class AdhanConfig:
    # ... existing fields ...
    n_languages: int = 1  # 1 = monolingual, >1 = multilingual
    use_language_id: bool = False
```

Add a language embedding table:

```python
if c.use_language_id and language_ids is not None:
    lang_emb = self.param(
        "lang_emb", nn.initializers.normal(0.02), (c.n_languages, c.d_model)
    )
    x = x + lang_emb[language_ids][:, None, :]
```

This is **zero-initialized** like Tiṇai, so a monolingual checkpoint loads unchanged into a multilingual model.

---

## 5. Evaluation Across Languages

### 5.1 Fertility Benchmark

Every script extension must report:

```bash
python -m adhan_slm.eval.run_eval \
    --tokenizer-dir data/final/kannada_slm \
    --eval-text data/raw/kannada_expanded/val.txt
```

**Target**: `< 1.15 tokens/akshara` for Dravidian scripts, `< 1.25` for Devanagari (conjunct penalty).

### 5.2 Cross-Lingual Transfer Benchmark

Test if Tamil pretraining helps Kannada:

```bash
# 1. Pretrain on Tamil only
# 2. Fine-tune on Kannada SFT with frozen vs unfrozen embeddings
# 3. Compare perplexity on Kannada held-out set
```

**Metric**: `transfer_gain = ppl_scratch - ppl_transfer`  
Positive = transfer helped. Negative = interference hurt.

### 5.3 Script-Specific Probes

| Language | Probe | Implementation |
|----------|-------|----------------|
| Tamil | Sandhi correctness | `eval/morphology.py` |
| Kannada | Alaṅkāra (prosody) pattern | New: `eval/kannada_prosody.py` |
| Telugu | CVCV syllable stress | New: `eval/telugu_phonology.py` |
| Hindi | Vibhakti (case ending) agreement | New: `eval/hindi_vibhakti.py` |
| Bengali | Compound verb splitting | New: `eval/bengali_compound.py` |

---

## 6. ONNX Export for On-Device Deployment

### 6.1 Multilingual ONNX

Export a multilingual checkpoint to ONNX for `illakiya` and `yazh-unity`:

```bash
python scripts/export_onnx.py \
    --checkpoint checkpoints/adhan-dravidian-mini \
    --output models/adhan-dravidian-mini.onnx \
    --opset 18 \
    --language-ids 4  # Tamil, Kannada, Telugu, Malayalam
```

### 6.2 Quantization-Aware Calibration

For INT8/INT4 deployment, calibrate on a **balanced multilingual sample**:

```python
# calibration_data.py
CALIBRATION_SAMPLE = {
    "tamil": "அறிவியல் மற்றும் தொழில்நுட்பம் ...",
    "kannada": "ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ...",
    "telugu": "విజ్ఞానం మరియు సాంకేతికత ...",
    "malayalam": "ശാസ്ത്രവും സാങ്കേതികവിദ്യയും ...",
}
```

**Critical**: If you calibrate on Tamil-only, INT8 quantization will have high error on Kannada rare aksharas. Balance your calibration set by script frequency.

---

## 7. Roadmap: From 1 Language to 22

| Phase | Languages | Tokenizer | Model Size | Notes |
|-------|-----------|-----------|------------|-------|
| **Now** | Tamil | Swaram | nano/tiny | Baseline proven |
| **Q1** | Tamil + Kannada | Shared Dravidian (24K) | tiny | Test transfer |
| **Q2** | Tamil + Kannada + Telugu + Malayalam | Shared Dravidian (32K) | mini | Full southern coverage |
| **Q3** | + Hindi (Devanagari) | Pan-Indic (48K) | mini | First Indo-Aryan |
| **Q4** | + Bengali + Gujarati + Marathi | Pan-Indic (56K) | mini | Full constitution-schedule coverage |
| **2026** | + Punjabi + Odia + Assamese | Pan-Indic (64K) | small (~200M) | Requires scaling team |

---

## 8. Contributing a New Language

1. **Open an issue**: `language-request: [Language Name]`
2. **Provide corpus**: Minimum 10M characters of clean text (CC-BY or public domain)
3. **Provide script inventory**: List all base characters + combining marks
4. **Implement tokenizer**: Follow the 5-step recipe above
5. **Add evaluation probe**: Script-specific linguistic test
6. **Run fertility gate**: Must be < 1.25 tokens/akshara
7. **Submit PR**: Include `docs/LANGUAGE_[CODE].md` with script details

---

*ஒன்றே குலம் ஒருவனே தேவன் — One world, one family of languages.*
