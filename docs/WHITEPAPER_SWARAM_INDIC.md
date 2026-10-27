# Swaram: Script-Native Tokenization for Indian Languages

**Removing the Indic token tax, from Tamil to ten languages of India**

*Yazhi / Adhan SLM — Technical Whitepaper v2.0 (Indic edition)*
*Supersedes the scope of [`WHITEPAPER_SWARAM.md`](WHITEPAPER_SWARAM.md) (Tamil, v1.0); all Tamil results are carried over unchanged.*

---

## How to read this paper

This paper separates three kinds of claims and labels each one:

| label | meaning |
|---|---|
| **[measured]** | produced by a script in this repository on a real corpus; reproducible from §11 |
| **[audited]** | a deterministic property of the code or of Unicode, checked in this repository (Appendix B) |
| **[target]** | a goal or hypothesis with a stated rationale and a pass/fail gate; *not* a result |

The Tamil results are measured. For the nine expansion languages, the results
are targets until the benchmark in §7 runs on each one. We keep the labels
because the token-tax argument only works if the numbers can be trusted.

---

## Executive summary

**The problem.** The tokenizers in widely deployed language models were trained
mostly on English. They split Indian-language text into byte fragments, so the
same sentence costs several times more tokens than it should. We call that
multiplier the **token tax**. Indian-language users pay it three times: in
training compute, in per-token inference cost and latency, and in how much text
fits in the model's context window.

**What Swaram is.** Swaram is a tokenizer built on the *akshara*, the syllabic
unit that every Brahmic script in India is organised around, rather than on
byte pairs learned from English. It has three layers: a deterministic, lossless
akshara segmenter (Layer A); a BPE merge layer learned over aksharas, which
never splits inside an akshara (Layer B); and a UTF-8 byte fallback (Layer C),
so any input round-trips exactly.

**What is proven (Tamil, 538k-document corpus) [measured].**

| | Swaram v3 | GPT-4o `o200k` | GPT-4 `cl100k` | UTF-8 bytes |
|---|---|---|---|---|
| token tax | **1.00×** | 1.21× | **4.21×** | 8.56× |
| Tamil words per 512-token window | **178** | 148 | 42 | 21 |
| round-trip | 100% | 100% | 100% | 100% |

Swaram encodes at about 2.5 Mchar/s on one CPU core and packed 258M tokens in
129 s.

**What this paper adds.**

1. **Ten languages, two script clusters, one design.** We extend Swaram to
   Telugu, Kannada, Malayalam and Tulu (*Swaram-Drāviḍa*), and to Hindi,
   Haryanvi, Punjabi, Odia and Bengali (*Swaram-Ārya*). Together with Tamil,
   these languages have about **927 million first-language speakers in India**,
   roughly three quarters of the population (Census of India 2011).
2. **A readiness audit [audited].** Running the generic segmenter against the
   Unicode database turned up concrete defects that would bias any benchmark
   until they are fixed. The most serious: Punjabi *tippi* (ੰ) and *addak* (ੱ)
   are split off as separate units, so ਪੰਜਾਬੀ is cut inside an akshara. The
   runtime also cannot see the Tulu-Tigalari script, which was added in
   Unicode 16.0.
3. **Three design contributions:**
   - **fertility-parity vocabulary allocation**, which shares a multilingual
     vocabulary by equalising cost per language rather than by corpus size;
   - **akshara-slot alignment**, which uses the parallel ISCII-derived layout of
     the Unicode Indic blocks to share embeddings across scripts;
   - a **Layer 0 normalisation contract** covering legacy Malayalam chillus,
     nukta forms, ZWJ/ZWNJ and Unicode 16.
4. **A measurement protocol** built on parallel text (FLORES-200 where it covers
   a language), so token tax and cross-language parity are measured on
   identical content.
5. **A phased roadmap** with explicit release gates. No language ships until its
   own measurements pass.

**Why it matters strategically.** No Indian deployment can out-spend frontier
labs on pretraining, but any deployment can stop paying the token tax. A
tokenizer that encodes Indian languages at parity, losslessly and with
open provenance, is foundational national infrastructure: it lowers the cost of
every Indian-language model, API call and on-device keyboard built on top of
it.

---

## Abstract

General-purpose subword tokenizers impose a large, uneven token tax on Indian
languages. We present **Swaram**, a script-native tokenizer family whose atomic
unit is the Brahmic akshara. On Tamil, Swaram reaches 0.506 tokens per akshara
with 100% byte-exact round-trip, and it removes 76% of GPT-4's token overhead
and 88% of the raw-byte overhead [measured].

We generalise the design to nine more languages across two families:
Dravidian (Telugu, Kannada, Malayalam, Tulu) and Indo-Aryan (Hindi, Haryanvi,
Punjabi, Odia, Bengali). The two families need different akshara rules: Tamil
*pulli* splits consonant clusters, while the other scripts join them into
conjuncts. Eight Unicode blocks are covered (Tamil plus seven), and
Tulu-Tigalari is added as a supplementary-plane target. We audit the existing multi-script segmenter
against Unicode 15.1, report the defects that block fair measurement, and
propose:

- a normalisation layer;
- a shared-vocabulary allocation rule that targets equal fertility rather than
  proportional data;
- a cross-script embedding scheme based on akshara-slot alignment;
- an evaluation protocol on parallel text.

Results for the expansion languages are stated as targets with release gates.

---

## 1. Introduction

### 1.1 The Indic token tax

A tokenizer maps text to the integer sequence a model actually processes. For a
tokenizer `T` and a reference tokenizer `R`, define

```
tax_T(x) = |T(x)| / |R(x)|
```

over the same text `x`. Throughout this paper `R` is the Swaram tokenizer
trained for the language of `x`. With a fixed compute budget, a tax of `k`
means the model:

1. needs **k× the training tokens** to see the same text;
2. pays **k× the inference tokens and decode steps** for every response;
3. fits only **1/k as much text** in its context window.

English-centric byte-level BPE vocabularies contain few merges for Indian
scripts, so most Indic text falls back to fragments. The resulting premium has
been documented across many languages (Petrov et al., 2023; Ahia et al., 2023).
The premium is largest for scripts that are poorly represented in the
tokenizer's training data, which includes most Indian scripts. Our Tamil
measurement puts a number on it: **4.21×** for `cl100k_base` and **1.21×** for
`o200k_base` [measured, §6.1]. Newer, larger vocabularies narrow the gap but do
not close it, and none of them make their per-language trade-offs explicit.

### 1.2 Who pays

First-language speakers in India, Census of India 2011 (language and
mother-tongue returns):

| language | ISO 639-3 | script(s) | first-language speakers (India, 2011) |
|---|---|---|---|
| Hindi | `hin` | Devanagari | 528.3 M *(includes grouped mother tongues, among them Haryanvi)* |
| Bengali | `ben` | Bengali–Assamese | 97.2 M |
| Telugu | `tel` | Telugu | 81.1 M |
| Tamil | `tam` | Tamil | 69.0 M |
| Kannada | `kan` | Kannada | 43.7 M |
| Odia | `ory` | Odia | 37.5 M |
| Malayalam | `mal` | Malayalam | 34.8 M |
| Punjabi | `pan` | Gurmukhi (India); Shahmukhi (Pakistan) | 33.1 M |
| Haryanvi | `bgc` | Devanagari | 9.8 M *(returned as a mother tongue under Hindi)* |
| Tulu | `tcy` | Kannada (current practice); Tulu-Tigalari | 1.8 M |

The total, counting Haryanvi only inside Hindi, is about **927 M**. Bengali and
Punjabi also have very large speaker populations outside India, in Bangladesh
and Pakistan respectively. For this population, the token tax is not an
academic curiosity. It is a per-token surcharge on every Indian-language
product built on general-purpose models.

### 1.3 Contributions

1. A three-layer tokenizer whose Layer A is one generic Brahmic segmenter,
   parameterised per script by a small table.
2. A per-language linguistic and Unicode profile for the ten languages (§3).
3. A readiness audit of the current code against the Unicode database, with
   the defects that must be fixed before measuring (§5) [audited].
4. Fertility-parity allocation, akshara-slot alignment and a Layer 0
   normalisation contract (§4).
5. A parallel-text evaluation protocol and per-language release gates (§7, §9).

---

## 2. Background: the Brahmic akshara

### 2.1 One structure, many scripts

Every script in scope descends from Brahmi and is an **abugida**:

- a consonant letter carries an **inherent vowel**;
- a dependent vowel sign (**matra**) replaces that vowel;
- a **virama** (called *pulli* in Tamil, *halant* in Hindi, *hasanta* in Bengali
  and Odia, *chandrakkala* in Malayalam, *pollu* in Telugu) suppresses the
  vowel;
- signs such as **anusvara**, **chandrabindu**, **visarga** and **nukta**
  modify the cluster.

An **akshara** is a base letter plus every mark attached to it. It is close to
a written syllable and is what a native reader perceives as one character.

### 2.2 The Unicode Indic blocks are aligned

The Unicode blocks for the major Indic scripts inherit the layout of the Indian
standard **ISCII (IS 13194:1991)**. The same phonological unit therefore sits at
the same offset in each block:

| offset | role | Devanagari | Bengali | Gurmukhi | Odia | Tamil | Telugu | Kannada | Malayalam |
|---|---|---|---|---|---|---|---|---|---|
| `+0x05` | vowel *a* | अ | অ | ਅ | ଅ | அ | అ | ಅ | അ |
| `+0x15` | consonant *ka* | क | ক | ਕ | କ | க | క | ಕ | ക |
| `+0x3F` | matra *i* | ि | ি | ਿ | ି | ி | ి | ಿ | ി |
| `+0x4D` | virama | ् | ্ | ੍ | ୍ | ் | ్ | ್ | ് |

Alignment is not total: each script leaves different offsets unassigned and
has its own extensions. Still, *ka + i* is `base + 0x15, base + 0x3F` in every
column. Swaram uses this in two places: one generic Layer A (§4.2), and a shared
cross-script embedding initialisation (§4.5).

### 2.3 Two families, two akshara rules

| | Tamil | the other nine languages in scope |
|---|---|---|
| virama + consonant | **splits** (க் + ஷ) — Tamil script does not form conjunct graphemes | **joins** into one conjunct akshara (क्ष, ಕ್ಕ, ক্ষ) |
| consequence | closed akshara inventory (404 in Swaram v3) | **open** set: conjuncts of 2–4 consonants are productive |
| implemented as | `Script.join_conjuncts = False` | `Script.join_conjuncts = True` |

Telugu, Kannada and Malayalam are Dravidian but their scripts join conjuncts
(Telugu *vattulu*, Kannada *ottakshara*). So the akshara rule follows the
**script**, not the language family. This is the single most important
difference between the Tamil paper and this one: from Telugu onward, the base
inventory is no longer closed by construction. We cover that in §4.4.

### 2.4 Morphology

Dravidian languages (Tamil, Telugu, Kannada, Malayalam, Tulu) are
**agglutinative**: one written word can hold a stem and a chain of suffixes,
and sandhi fuses words at their boundaries. Malayalam takes this furthest, with
very long written compounds. Indo-Aryan languages in scope are mainly
**fusional with separately written postpositions** (Hindi *का, में, से*;
Bengali case suffixes are attached). As a result, merges that capture whole
suffixes are worth more in Dravidian text, and words are shorter in Indo-Aryan
text. Hindi and Bengali tokens per *word* will be lower than Tamil, and tokens
per *akshara* is the fair cross-language measure (§7.2).

---

## 3. Language profiles

Each profile lists what a tokenizer must get right for that language. Every
code point named here is checked against the Unicode database in Appendix B.

### 3.1 Telugu (తెలుగు) — Swaram-Drāviḍa

- **Script / block:** Telugu, U+0C00–U+0C7F.
- **Akshara:** conjuncts via virama (*vattulu*, subscript consonants), often
  chains of two or three consonants (స్త్ర).
- **Marks to attach:** U+0C00 combining candrabindu above, U+0C04 combining
  anusvara above, U+0C55/U+0C56 length marks, U+0C62/U+0C63 vocalic L/LL
  matras. *None of these are attached today (§5).*
- **Morphology:** agglutinative; most native words end in a vowel; productive
  sandhi.

### 3.2 Kannada (ಕನ್ನಡ) — Swaram-Drāviḍa

- **Script / block:** Kannada, U+0C80–U+0CFF.
- **Akshara:** *ottakshara* conjuncts (ನ್ನ in ಕನ್ನಡ is one akshara, as the
  current segmenter correctly produces).
- **Marks to attach:** U+0CD5/U+0CD6 length marks, U+0CE2/U+0CE3, U+0CF3
  (Unicode 15). *Currently unattached.*
- **Strategic note:** Kannada's Layer A is also **Tulu's** Layer A (§3.4), so
  fixing Kannada fixes the majority of written Tulu as well.

### 3.3 Malayalam (മലയാളം) — Swaram-Drāviḍa

- **Script / block:** Malayalam, U+0D00–U+0D7F.
- **Akshara:** dense conjunct set. **Chillu letters** (ൺ ൻ ർ ൽ ൾ ൿ, U+0D7A–U+0D7F)
  are atomic code points since Unicode 5.1, but older text encodes them as
  *consonant + virama + ZWJ*. Both spellings must normalise to the atomic form
  so they share one token (§4.3).
- **Orthography:** the 1971 reformed and the traditional script share code
  points and differ only in font rendering. This does not affect tokenization,
  but readers sometimes mistake it for an encoding issue.
- **Marks to attach:** U+0D00, U+0D3B (vertical-bar virama), U+0D57 (au length
  mark, which modern text often writes on its own), U+0D62/U+0D63. *Currently
  unattached.*
- **Morphology:** the most agglutinative language in scope. Expect the largest
  gain from suffix-capturing merges, and the hardest boundary-agreement problem.

### 3.4 Tulu (ತುಳು) — Swaram-Drāviḍa, low-resource

- **Script:** today mostly **Kannada script**. The historical
  **Tulu-Tigalari** script was encoded in **Unicode 16.0 (2024)** at
  U+11380–U+113FF. That block is in the supplementary plane, so each code point
  is 4 bytes in UTF-8, and it has its own virama and conjunct behaviour.
- **Runtime gap [audited]:** the Python runtime in this repository ships
  Unicode **15.1**, so `unicodedata` does not know any Tulu-Tigalari character.
  NFC normalisation and category lookups are undefined for that script until
  the runtime (or a pinned Unicode data table) is upgraded.
- **Strategy:** (a) tokenize Kannada-script Tulu with the Kannada Layer A and a
  Tulu-specific merge budget; (b) add a Tigalari `Script` table once Unicode 16
  data is pinned; (c) keep a deterministic Kannada↔Tigalari transliteration so
  the two scripts can share aksharas and embeddings through slot alignment.
- **Data:** about 1.8 M speakers and a small written corpus, so Tulu is a
  transfer case: it borrows Kannada merges and Dravidian embeddings, and its
  success criterion is cross-lingual gain, not standalone scale (§9).

### 3.5 Hindi (हिन्दी) — Swaram-Ārya

- **Script / block:** Devanagari, U+0900–U+097F. This is the existing
  `AksharamTokenizer` prototype.
- **Akshara:** conjuncts (क्ष, त्र, ज्ञ, श्र); *reph* (र्क) and *rakar*
  (क्र) are conjunct spellings handled by the same virama rule.
- **Nukta:** precomposed letters U+0958–U+095F (क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़) are
  *composition exclusions*. NFC turns them into base + U+093C nukta, so the
  segmenter must always run after NFC, which it does.
- **Marks to attach:** U+093A, U+093B, U+094E, U+094F, U+0955–U+0957.
  *Currently unattached*; all are rare in modern Hindi.
- **Usage:** heavy code-mixing with English, and widely written in Roman script
  online ("Hinglish"). Byte fallback keeps Roman text lossless, but that text
  pays the byte price. Romanised Hindi is a separate tokenization problem and
  is out of scope for v2.0 (§8).

### 3.6 Haryanvi (हरियाणवी) — Swaram-Ārya, low-resource

- **Script:** Devanagari, so the Layer A is identical to Hindi's.
- **Language:** part of a dialect continuum with Hindi, and returned under
  Hindi in the census. It differs in lexicon and morphology (for example the
  copula *सै* where Hindi has *है*). It is primarily spoken; written text is
  scarce and spread across folk literature (*ragni*), cinema, music and social
  media.
- **Strategy:** share the Hindi tokenizer and add a **Haryanvi merge budget**
  learned from a curated corpus. Haryanvi is the clearest test of whether
  fertility-parity allocation (§4.4) protects a low-resource variety from being
  absorbed by its high-resource neighbour.
- **Data note:** much of the available text is transcribed speech and user
  content. Consent, licensing and community partnership are prerequisites, not
  afterthoughts (§8).

### 3.7 Punjabi (ਪੰਜਾਬੀ) — Swaram-Ārya

- **Script / block:** Gurmukhi, U+0A00–U+0A7F. Shahmukhi (Perso-Arabic,
  Pakistan) is out of scope for v2.0 and round-trips via byte fallback.
- **Akshara:** Gurmukhi rarely shows a visible virama. It is used for the
  subjoined (*pairin*) forms of ਹ ਰ ਵ (ਪ੍ਰ). Nasalisation is written with
  **tippi** ੰ (U+0A70) and **bindi** ਂ (U+0A02), and gemination with **addak** ੱ
  (U+0A71).
- **Defect [audited]:** tippi and addak are **not** treated as combining marks
  today, so they become separate units:

  ```
  segment("ਪੰਜਾਬੀ ਪੱਕਾ")  →  ['ਪ', 'ੰ', 'ਜਾ', 'ਬੀ', ' ', 'ਪ', 'ੱ', 'ਕਾ']
  expected             →  ['ਪੰ', 'ਜਾ', 'ਬੀ', ' ', 'ਪੱ', 'ਕਾ']
  ```

  Because both marks are very frequent in Punjabi, any Punjabi fertility number
  measured before this fix would be inflated and would violate the
  akshara-atomicity contract. **This is a release-blocking bug for Punjabi.**

### 3.8 Odia (ଓଡ଼ିଆ) — Swaram-Ārya

- **Script / block:** Odia, U+0B00–U+0B7F.
- **Akshara:** conjuncts, many with subscript forms (*phala*).
- **Nukta:** ଡ଼ ଢ଼ (U+0B5C/U+0B5D) are composition exclusions, so NFC
  decomposes them. ଓଡ଼ିଆ segments correctly today as ଓ | ଡ଼ି | ଆ.
- **Marks to attach:** U+0B55 (overline), U+0B56/U+0B57 (ai/au length marks),
  U+0B62/U+0B63. *Currently unattached.*

### 3.9 Bengali (বাংলা) — Swaram-Ārya

- **Script / block:** Bengali–Assamese, U+0980–U+09FF. The block is shared
  with Assamese (ৰ ৱ), so language ID, not script ID, decides which merges
  apply.
- **Akshara:** *juktakkhor* conjuncts; **reph** (র্ক), **ya-phala** (্য) and
  **ba-phala** (্ব) all follow the virama rule. *Khanda ta* ৎ (U+09CE) is a
  standalone letter. ZWJ/ZWNJ select between ligature and explicit-hasanta
  rendering of the same letters, which the segmenter handles by attaching
  joiners.
- **Nukta:** ড় ঢ় য় (U+09DC, U+09DD, U+09DF) are composition exclusions and
  decompose under NFC.
- **Marks to attach:** U+09D7 (au length mark), U+09E2/U+09E3, U+09FE.
  *Currently unattached.*

---

## 4. Swaram for Indian languages: design

```
          raw text
             │
   Layer 0   │  normalise: NFC, legacy chillu → atomic, ZWJ/ZWNJ policy,
             │  Unicode-16 tables, script + language ID per run
             ▼
   Layer A   │  generic Brahmic segmenter, one Script table per script
             │  (lossless: "".join(segment(x)) == NFC(x))
             ▼
   Layer B   │  BPE over aksharas, per-language merge budgets drawn from a
             │  shared vocabulary (fertility-parity allocation)
             ▼
   Layer C   │  UTF-8 byte fallback <0xNN>: total, never <unk>
             ▼
          token ids
```

### 4.1 What carries over unchanged from Tamil

- the **two-layer contract**: merges never cross an akshara boundary;
- **byte fallback**, which keeps the tokenizer total and lossless on any input,
  including Latin, digits, emoji and unsupported scripts;
- the **Rust fast path** (`fast_bpe.py`). It already takes an
  `indic_akshara.Script` parameter, so training and encoding for a new script
  is a parameter change, not a new tokenizer [audited];
- the **optional morphology guard**, which needs per-language suffix tables
  (§9).

### 4.2 Layer A: one segmenter, a table per script

`src/adhan_slm/tokenizer/indic_akshara.py` defines a `Script` record with five
fields:

- the script's block range;
- whether virama joins conjuncts;
- which block offsets are combining signs;
- the consonant range;
- the virama.

It also defines a single `segment()` that is lossless by construction. Adding a
script means adding one table row. Tamil keeps its hand-curated v3 segmenter
and inventory, so shipped Tamil behaviour does not change.

**Required change (P0, §9):** derive the combining set from the Unicode
general category (`Mn`/`Mc`) for each block, instead of the current shared
offset list, and keep explicit per-script exceptions (Tamil U+0B83 *aytham*,
Malayalam chillus). Appendix B lists every code point the current offset list
misses. Each miss is a place where an akshara is cut today.

### 4.3 Layer 0: the normalisation contract

Layer A is only as good as its input. Layer 0 is the explicit, versioned
contract for what gets normalised:

| rule | why |
|---|---|
| Unicode **NFC** | composes split matras (Kannada ೀ = ಿ + ೕ, Odia ୈ = େ + ୖ, Malayalam ൌ = െ + ൗ) and decomposes nukta composition exclusions (क़ → क + ़), so one spelling gives one token sequence |
| **legacy chillu → atomic chillu** (Malayalam ന + ് + ZWJ → ൻ) | one chillu, one token |
| **ZWJ/ZWNJ** attach to the preceding unit and are never dropped | they change rendering (Bengali র্য, explicit hasanta), so dropping them would be lossy |
| **pinned Unicode version** (target: 16.0) | Tulu-Tigalari must be a known script; normalisation must not depend on the host Python's Unicode version |
| **script run + language ID** emitted as side information | Bengali/Assamese and Hindi/Haryanvi share a script; language-specific merge budgets need the language |

Normalisation must stay *canonical-equivalence only*. Swaram never folds
visually similar but distinct letters (for example Bengali র and Assamese ৰ), because that would break losslessness.

### 4.4 Layer B: merges over an open akshara set

**The open-inventory problem.** Tamil's base inventory is closed (404
aksharas), because pulli never creates conjunct graphemes. With
`join_conjuncts=True`, every observed conjunct (क्ष, স্ত্র, ಕ್ಷ್ಮ) is a
distinct akshara, and the set grows with the corpus. Swaram handles this as
follows:

1. **Derived base inventory:** consonant, consonant + virama, and consonant +
   each matra, over *assigned* code points only. Today
   `base_inventory()` iterates the full consonant offset range, including
   unassigned code points, and returns 629 entries for every script whether it
   has 32 consonants or 37 [audited]. P0 filters to assigned letters.
2. **Frequency-admitted conjuncts:** conjunct aksharas above a minimum count in
   the training sample become base units. Rarer ones decompose into their
   consonant + virama parts, which are still whole aksharas from the
   derived inventory, so they never fall back to bytes.
3. **PUA budget.** The Rust trainer maps each akshara to one Private Use Area
   code point (BMP PUA: **6,400 slots**, U+E000–U+F8FF) [audited]. A per-script
   tokenizer fits easily. A unified ten-language tokenizer with admitted
   conjuncts may not, so the unified build (§9, P3) moves to the
   supplementary PUA planes (U+F0000–U+10FFFD, about 131k slots).

**Fertility-parity allocation.** If a multilingual vocabulary is learned by
plain frequency, Hindi, with by far the most data, takes most of the merges,
and Tulu and Haryanvi are left near character level. That reproduces the token
tax *inside* India. Swaram instead gives each language `ℓ` a merge budget
`m_ℓ`, chosen to minimise the spread of fertility:

```
choose { m_ℓ }  to minimise   max_ℓ f_ℓ(m_ℓ) / min_ℓ f_ℓ(m_ℓ)
subject to      Σ_ℓ m_ℓ ≤ V − |base| − |specials| − 256
```

Here `f_ℓ(m)` is held-out fertility of language `ℓ` with `m` merges. Each
`f_ℓ` falls as merges are added, with diminishing returns, so a simple greedy
allocator works well in practice: it keeps giving the next block of merges to
the language that is currently worst. The ratio being minimised, the **parity ratio**, is reported as a
first-class metric (§7.2). Merges shared between languages of the same
script (Hindi/Haryanvi, Kannada/Kannada-script Tulu) count toward every
language that uses them.

### 4.5 Cross-script embeddings: akshara-slot alignment

ISCII-derived alignment (§2.2) gives each Brahmic akshara a script-independent
**slot**, for example (consonant offset 0x15, matra offset 0x3F) = *ki*. Swaram
exposes the slot for every base akshara, so a model can initialise or tie
embeddings as

```
E[token] = E_slot[slot(token)] + E_script[script(token)]  (+ E_merge for merged units)
```

Byte-level BPE vocabularies have no such structure: क and க have no relation in
their vocabularies. With slot alignment, Telugu *ki* can start from Tamil *ki*,
which is the transfer path for Tulu and the other lower-resource languages.
This is a **[target]**: the mechanism is implementable from the tables today,
and whether it helps is an empirical question with its own gate (§9, P1).

### 4.6 Layer C: byte fallback

This layer is unchanged from Tamil. Anything outside the enabled scripts,
including Shahmukhi, Romanised text, emoji and unsupported scripts,
round-trips through `<0xNN>`. This keeps Swaram **total**: no input can produce
`<unk>`. The byte share of a document is also a useful data-quality signal,
because a high share means the wrong language or script.

---

## 5. Readiness audit [audited]

We ran the generic segmenter against the Unicode 15.1 database shipped with the
repository's Python runtime. For each script block we listed every code point
with general category `Mn` or `Mc` (a combining mark) that `Script.is_combining`
does **not** attach. Full output is in Appendix B.

| script | languages | marks not attached today | impact | severity |
|---|---|---|---|---|
| Gurmukhi | Punjabi | **U+0A70 tippi, U+0A71 addak**, U+0A75 yakash, U+0A51 udaat | very frequent marks split off; ਪੰਜਾਬੀ is cut inside an akshara | **blocker** |
| Malayalam | Malayalam | U+0D00, U+0D3B, **U+0D57 au length mark**, U+0D62/63 | standalone ൗ is common in modern spelling | high |
| Telugu | Telugu | U+0C00, U+0C04, U+0C55/56, U+0C62/63 | candrabindu-above appears in some orthographies | medium |
| Kannada | Kannada, Tulu | U+0CD5/D6, U+0CE2/E3, U+0CF3 | length marks usually compose under NFC; rare standalone | medium |
| Odia | Odia | U+0B55, U+0B56/57, U+0B62/63 | as Kannada | medium |
| Bengali | Bengali | U+09D7, U+09E2/E3, U+09FE | as Kannada | medium |
| Devanagari | Hindi, Haryanvi | U+093A, U+093B, U+094E, U+094F, U+0955–57 | rare in modern Hindi | low |
| Tulu-Tigalari | Tulu | *entire block unknown to runtime (Unicode 16.0)* | script cannot be normalised or segmented | blocker for Tigalari only |

There are also two inventory findings:

- `base_inventory()` includes unassigned code points, giving 629 entries for
  every script. It should cover assigned consonants only: Bengali has 32 in its
  consonant range, Devanagari 37.
- The shared offset list marks some *unassigned* offsets as combining (for
  example Gurmukhi U+0A43–U+0A46). This has no runtime effect, because
  unassigned code points do not occur in valid text, but it shows the offset
  list is an approximation and should be replaced by Unicode properties.

Correct segmentation today [audited]:

```
Kannada    ಕನ್ನಡ ತುಳು   → ಕ | ನ್ನ | ಡ  ·  ತು | ಳು
Devanagari हरियाणवी छोरा → ह | रि | या | ण | वी  ·  छो | रा
Odia       ଓଡ଼ିଆ          → ଓ | ଡ଼ି | ଆ
Bengali    বাংলা         → বাং | লা
Telugu     తెలుగు        → తె | లు | గు
Malayalam  മലയാളം        → മ | ല | യാ | ളം
```

**Consequence.** The architecture is ready: one segmenter, a script-parameterised
Rust trainer, and byte fallback. The tables are not. No expansion-language
benchmark is published until the P0 fixes land with a test for every row
above.

---

## 6. Results to date

### 6.1 Tamil token tax (355 held-out documents) [measured]

| tokenizer | fertility (tok/akshara) | tokens/word | tax vs Swaram | Tamil words in 512 ctx | round-trip |
|---|---|---|---|---|---|
| **Swaram v3** | **0.554** | **2.874** | **1.00×** | **178** | 100% |
| Swaram v2 | 0.564 | 2.926 | 1.02× | 175 | 100% |
| GPT-4o (`o200k`) | 0.667 | 3.463 | 1.21× | 148 | 100% |
| GPT-4 (`cl100k`) | 2.328 | 12.085 | **4.21×** | 42 | 100% |
| code points | 1.808 | 9.383 | 3.26× | 55 | 100% |
| UTF-8 bytes | 4.740 | 24.604 | **8.56×** | 21 | 100% |

Source: `docs/TOKEN_TAX_BENCHMARK_V3.md`. Corpus `data/raw/expanded_v2`
(538,562 documents, 901.6 M characters). Fertility for the v3 artifact at
freeze time was 0.506 [measured, artifact card in Appendix D].

### 6.2 Tamil tokenizer quality (300 held-out documents) [measured]

| artifact | vocab | fertility* | lossless | `<unk>` docs | throughput | morph. agreement |
|---|---|---|---|---|---|---|
| Swaram v2 (32k) | 31,964 | 0.339 | 100% | 0 | 2.48 Mchar/s | 5.6% |
| **Swaram v3 (32k, expanded)** | 31,964 | **0.334** | **100%** | **0** | **2.58 Mchar/s** | 5.6% |

\* Measured on a different sample from §6.1, so compare within the column
only.

### 6.3 What Tamil tells us about the other languages

- **Lossless and total behaviour transfers by construction.** Layers A and C
  are deterministic, and the round-trip property is proven per script by
  tests, not estimated.
- **The cost floor transfers.** All in-scope Indic scripts sit in the Basic
  Multilingual Plane, so each code point is 3 bytes in UTF-8 (Tulu-Tigalari is
  4). A byte-level tokenizer that fails to merge pays at least 3 tokens per
  code point. Code-point tokenization pays one token per code point, which is
  2–3 per akshara once matras and viramas are counted. These floors bound the
  tax from below for any tokenizer that does not merge the script.
- **The compression level does not transfer automatically.** Conjunct joining
  changes the akshara count (one conjunct akshara replaces two to three Tamil
  aksharas), and fusional morphology changes how much a merge can capture.
  Fertility must be measured per language, which is the purpose of §7.

---

## 7. Evaluation protocol for Indian languages

### 7.1 Corpora

| purpose | source | coverage of the ten languages |
|---|---|---|
| tokenizer training | IndicCorp v2, Sangraha (AI4Bharat), FineWeb-2, mC4, Wikipedia (incl. `tcy` Tulu Wikipedia), digitised libraries | all except Haryanvi at useful scale |
| **parallel token tax** | **FLORES-200 devtest** | `hin_Deva`, `ben_Beng`, `pan_Guru`, `ory_Orya`, `tam_Taml`, `tel_Telu`, `kan_Knda`, `mal_Mlym` (8 of 10) |
| low-resource held-out | curated Tulu (Kannada-script and Tigalari) and Haryanvi sets built with community partners | Tulu, Haryanvi |
| code-mix robustness | Hinglish / Tanglish samples (romanised + native) | hi, ta first |

Parallel text is the key methodological choice. On FLORES every language
expresses the **same content**, so cross-language comparisons (parity, tax
versus English) are free of topic and genre effects that a monolingual sample
cannot control.

### 7.2 Metrics

| metric | definition | why |
|---|---|---|
| fertility | tokens / akshara (akshara counted by the language's own Swaram Layer A) | primary compression measure; comparable within a language |
| **token tax** | tokens_T / tokens_Swaram on identical text | the headline cost multiplier |
| **tax vs English** | tokens_T(x_ℓ) / tokens_T(x_en) on FLORES-parallel text | shows the premium a given tokenizer charges language `ℓ` |
| **parity ratio** | max_ℓ fertility / min_ℓ fertility across a shared vocabulary | fairness *inside* the tokenizer family (§4.4) |
| bits per akshara | (NLL per token / ln 2) × (tokens / akshara) | the only cross-tokenizer comparable LM quality measure |
| round-trip | share of documents with `decode(encode(x)) == NFC(x)` | must be 100% |
| `<unk>` documents | count | must be 0 |
| byte share | share of tokens that are `<0xNN>` | detects script gaps and dirty data |
| morphology agreement | share of token boundaries that coincide with gold stem/suffix boundaries | Dravidian quality signal; needs suffix tables |
| throughput | characters / second, single core | deployability |

### 7.3 Baselines

Each language is compared against:

- `cl100k_base` (GPT-4) and `o200k_base` (GPT-4o);
- the tokenizers of open multilingual and Indic models. These are run from
  their published tokenizer files, with names and versions recorded per run;
- code-point and UTF-8 byte tokenization, as floors;
- a **monolingual byte-level BPE trained on the same corpus at the same
  vocabulary size**. This is the strongest fair baseline and the one the Tamil
  paper lacked (Tamil v1.0, §7, item 5).

### 7.4 Harness

The Tamil benchmarks (`scripts/benchmark_token_tax.py`,
`scripts/benchmark_tokenizer_v2.py`) generalise by adding a `--lang` /
`--script` argument that selects the Layer A table for the akshara count, plus
a FLORES-parallel mode. Every run writes a datasheet with corpus, code SHA,
Unicode version and tokenizer versions.

---

## 8. Data plan

| tier | languages | situation | plan |
|---|---|---|---|
| high-resource | Hindi, Bengali | large web corpora; risk of dominating shared vocabularies | sample down for tokenizer training; enforce fertility parity |
| mid-resource | Telugu, Tamil, Kannada, Malayalam, Odia, Punjabi | adequate web + Wikipedia + news; classical and literary text in digital libraries | reuse the Tamil pipeline: exact + LSH dedup, script-ratio and quality gates per script |
| low-resource | Tulu | Tulu Wikipedia, Yakshagana and literary publications, academy publications; Tigalari manuscripts | community partnership; Kannada-script first, Tigalari as digitisation matures |
| very low written resource | Haryanvi | mostly spoken; folk literature, cinema/music, social media | consent-first curation; transcribed speech; small, high-quality held-out sets |

Quality gates are per script. The Tamil gate "Tamil-character ratio" becomes
"target-script ratio" with per-language thresholds. Language ID replaces script
ID for script-sharing pairs (Bengali/Assamese, Hindi/Haryanvi/Marathi). Every
source gets a datasheet entry with licence, provenance and collection method;
a sovereign tokenizer is only as defensible as its data provenance.

**Out of scope for v2.0:** Romanised Indian languages, Shahmukhi Punjabi,
Assamese, Marathi, Gujarati and the remaining scheduled languages. The design
covers them (Gujarati and Assamese are already in the script tables), but they
are not release targets here.

---

## 9. Roadmap and release gates

Each phase ships only when **all** of its gates pass on held-out data. Targets
are set from the Tamil results and from §6.3, and they will be revised in the
open once they are measured.

### P0 — Layer A hardening (prerequisite for every expansion language)

- Combining set derived from Unicode `Mn`/`Mc` per block, with explicit
  exceptions; tests for every row of §5.
- `base_inventory()` restricted to assigned code points.
- Layer 0 normalisation contract implemented and versioned; Unicode 16.0 data
  pinned (Tulu-Tigalari).
- Rust/Python Layer A parity tests extended to all enabled scripts.
- **Gate:** 100% round-trip, zero `<unk>`, and an akshara-atomicity test (no
  token boundary inside any akshara) on a fixed per-script fixture corpus.
  Tamil v3 outputs unchanged, byte for byte.

### P1 — Swaram-Drāviḍa: Telugu, Kannada, Malayalam, Tulu

- Per-language 32k tokenizers first, then a shared Dravidian tokenizer with
  fertility-parity allocation; Tulu on the Kannada Layer A.
- Akshara-slot alignment exposed in the artifact.
- **Gates [target]:**
  - fertility ≤ 0.60 tokens/akshara per language at 32k;
  - tax vs `cl100k_base` ≥ 3× per language;
  - shared-vocabulary parity ratio ≤ 1.25;
  - shared tokenizer within 10% of each monolingual tokenizer's fertility;
  - Tulu: Dravidian-transfer model beats a Tulu-only model on bits/akshara.

### P2 — Swaram-Ārya: Hindi, Haryanvi, Punjabi, Odia, Bengali

- Conjunct admission thresholds per script; Hindi/Haryanvi share Layer A with
  separate merge budgets; Punjabi only after the P0 tippi/addak fix.
- **Gates [target]:**
  - fertility within 1.5× of the Dravidian cluster's median (Indo-Aryan
    aksharas are fewer and longer, so a higher value is expected and must be
    explained, not hidden);
  - tax vs `cl100k_base` ≥ 2× per language;
  - Haryanvi fertility within 15% of Hindi on its own held-out set (the
    fertility-parity test).

### P3 — Swaram-Bhārat: one tokenizer, ten languages

- Unified vocabulary (planning range 96k–128k, set by the parity curve rather
  than fixed in advance); supplementary-plane PUA training view; language-ID
  side channel.
- **Gates [target]:**
  - parity ratio ≤ 1.30 across all ten;
  - no language more than 10% worse than its cluster tokenizer;
  - FLORES-parallel tax vs English reported for every tokenizer in §7.3.

### P4 — Downstream proof

- Train matched-compute Adhan models with Swaram and with a byte-level BPE
  baseline. Compare bits per akshara, sample efficiency, long-context recall
  and sandhi/morphology probes.
- Integrate with on-device runtimes (Illakiya IME, Yazh Unity) and measure
  latency and memory.
- **Gate:** at equal compute, Swaram models are no worse on bits/akshara and
  better on effective context. This is the claim the tokenizer exists to make
  true.

---

## 10. Strategic implications

**Cost.** For a workload of `N` Swaram tokens, a tokenizer with tax `k` costs
`k × N` tokens in training, `k × N` in billing and decode steps, and leaves
`window / k` of effective context (Appendix C). With Tamil measured at `k = 4.21`
for `cl100k_base`, a product serving Tamil on that tokenizer pays roughly four
times the token bill for the same text. P1–P3 will put the corresponding number
on each of the nine other languages.

**Sovereignty.** Swaram is built in the open from documented corpora, with
datasheets, code SHAs and reproducible scripts. A tokenizer is the one model
component that every downstream model inherits and cannot easily change later.
Owning it, and making it fair across India's languages by construction, is a
durable advantage: for public services, education, the classical and literary
record, and every Indian-language product that has to be cheap enough to run
at population scale.

**Ecosystem.** Swaram is a component, not a competitor, to national language
efforts such as Bhashini and to open Indic research corpora. Its outputs
(tokenizer files, Layer A tables, the normalisation contract and benchmarks)
are designed to be adopted by any model, not only Adhan.

**On device.** Deterministic segmentation plus a 32k vocabulary keeps the
tokenizer small and fast enough for keyboards and offline assistants, where
general-purpose tokenizers with multi-hundred-thousand-entry vocabularies are a
poor fit.

---

## 11. Reproducibility

### 11.1 Tamil (shipped)

```bash
# Freeze the tokenizer and pack shards
python scripts/freeze_fast_tokenizer.py \
  --corpus data/raw/expanded_v2 --out data/final/tamil_slm_v3 \
  --vocab-size 32000 --seq-len 512 --train-chars 300000000 --max-tokens 0

# Token tax and quality benchmarks
python scripts/benchmark_token_tax.py \
  --corpus data/raw/expanded_v2 --docs 400 \
  --swaram data/final/tamil_slm_v3 --swaram-v1 data/final/tamil_slm_v2 \
  --out docs/TOKEN_TAX_BENCHMARK_V3.md
python scripts/benchmark_tokenizer_v2.py --corpus data/raw/expanded_v2 \
  --tokenizer data/final/tamil_slm_v2:swaram-v2-32k \
  --tokenizer data/final/tamil_slm_v3:swaram-v3-32k-expanded \
  --out docs/TOKENIZER_BENCHMARK_V3.md
```

### 11.2 Readiness audit (Appendix B)

```bash
PYTHONPATH=src python - <<'PY'
import unicodedata as u
from adhan_slm.tokenizer import indic_akshara as ia
print("Unicode", u.unidata_version)
for s in ia.SCRIPTS:
    miss = [cp for cp in range(s.block_start, s.block_end + 1)
            if u.category(chr(cp)) in ("Mn", "Mc") and not s.is_combining(cp)]
    print(s.name, " ".join(f"U+{cp:04X}" for cp in miss) or "-")
PY
```

### 11.3 Expansion languages (planned interface)

```bash
# P1/P2: per-language freeze (adds --script / --lang to the existing CLI)
python scripts/freeze_fast_tokenizer.py --script telugu --lang tel \
  --corpus data/raw/te --out data/final/swaram_te_v1 --vocab-size 32000

# FLORES-parallel token tax across languages
python scripts/benchmark_token_tax.py --flores data/eval/flores200 \
  --langs tam_Taml,tel_Telu,kan_Knda,mal_Mlym,hin_Deva,ben_Beng,pan_Guru,ory_Orya \
  --out docs/TOKEN_TAX_BENCHMARK_INDIC.md
```

The interface in §11.3 is a specification. The flags do not exist yet and
land with P1.

---

## 12. Limitations and threats to validity

1. **Expansion results are targets, not measurements.** Only Tamil is measured.
   The gates in §9 are falsifiable on purpose.
2. **Conjunct-joined aksharas change the denominator.** Fertility per akshara
   is comparable within a language, but less directly across the two
   families. Cross-family claims should use FLORES-parallel tokens per
   sentence and bits per akshara.
3. **Low-resource evaluation is fragile.** Tulu and Haryanvi held-out sets will
   be small. Report confidence intervals, and do not generalise from them to
   the rest of the language.
4. **Dialect and register.** Haryanvi spans a continuum with Hindi, and much of
   Tulu's written record is literary or ritual. A tokenizer tuned on one
   register may underperform on another.
5. **Morphology tables exist only for Tamil.** Morphology agreement and the
   morphology guard need per-language suffix inventories. Tamil agreement
   (5.6% shipped; 37.5% with the guard on an earlier artifact) shows this is
   still the main open quality problem.
6. **Romanised text is out of scope.** A large share of real Hindi and Punjabi
   (and Tanglish) usage is Roman-script. Byte fallback keeps it lossless but
   not cheap.
7. **Tokenizer efficiency is not model quality.** Lower fertility buys context
   and cost. It does not by itself buy reasoning. P4 exists to test the link,
   not to assume it.

---

## 13. Conclusion

The Indic token tax comes from tokenizers designed for English. It is not a
property of Indian languages. Every script in this paper shares one
structure, the akshara, and Unicode lays the scripts out in parallel. That lets
one segmenter, one training path and one fallback serve all of them, with a
table per script.

On Tamil this design is measured: a 1.00× tax against GPT-4's 4.21×, 4.2× more
text per context window, and 100% lossless. For Telugu, Kannada, Malayalam,
Tulu, Hindi, Haryanvi, Punjabi, Odia and Bengali, the architecture is in place,
the remaining defects are named, and each language has a gate it must pass
before we claim anything. The next step is P0: make every akshara in every
script atomic, then measure.

---

## References

- Ahia, O. et al. (2023). *Do All Languages Cost the Same? Tokenization in the
  Era of Commercial Language Models.* EMNLP.
- Petrov, A. et al. (2023). *Language Model Tokenizers Introduce Unfairness
  Between Languages.* NeurIPS.
- Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of
  Rare Words with Subword Units.* ACL.
- NLLB Team (2022). *No Language Left Behind: Scaling Human-Centered Machine
  Translation* (FLORES-200).
- Bureau of Indian Standards. *IS 13194:1991, Indian Script Code for
  Information Interchange (ISCII).*
- The Unicode Consortium. *The Unicode Standard*, Version 15.1 (2023) and
  Version 16.0 (2024), chapter 12 (South and Central Asia-I) and the
  Tulu-Tigalari block.
- Office of the Registrar General & Census Commissioner, India. *Census of
  India 2011, Language: Paper 1 of 2018.*
- Yazhi / Adhan. *Swaram: A Script-Native Tokenizer for Tamil*, Technical
  Whitepaper v1.0 (`docs/WHITEPAPER_SWARAM.md`).

---

## Appendix A — Script reference

| script | block | `join_conjuncts` | virama name | notable units | languages in scope |
|---|---|---|---|---|---|
| Tamil | U+0B80–U+0BFF | no | pulli | aytham ஃ, Grantha ஜ ஷ ஸ ஹ, ஸ்ரீ | Tamil |
| Telugu | U+0C00–U+0C7F | yes | pollu (virama) | vattulu (subscript consonants) | Telugu |
| Kannada | U+0C80–U+0CFF | yes | halant (virama) | ottakshara | Kannada, Tulu |
| Malayalam | U+0D00–U+0D7F | yes | chandrakkala | chillu ൺ ൻ ർ ൽ ൾ ൿ | Malayalam |
| Tulu-Tigalari | U+11380–U+113FF | *to define (Unicode 16)* | virama | — | Tulu |
| Devanagari | U+0900–U+097F | yes | halant | reph, rakar, nukta | Hindi, Haryanvi |
| Gurmukhi | U+0A00–U+0A7F | yes | halant (rarely visible) | tippi ੰ, bindi ਂ, addak ੱ, pairin | Punjabi |
| Odia | U+0B00–U+0B7F | yes | hasanta | phala forms, ଡ଼ ଢ଼ | Odia |
| Bengali | U+0980–U+09FF | yes | hasanta | reph, ya-phala, khanda ta ৎ | Bengali |

## Appendix B — Readiness audit output

Unicode 15.1.0, `src/adhan_slm/tokenizer/indic_akshara.py` at the commit of
this paper. "Missing" means a combining mark (`Mn`/`Mc`) in the block that
`is_combining` does not attach. "Assigned consonants" counts `Lo` code points
in the script's consonant offset range.

| script | assigned code points | assigned consonants | `base_inventory()` size | missing combining marks |
|---|---|---|---|---|
| Devanagari | 128 | 37 | 629 | U+093A U+093B U+094E U+094F U+0955 U+0956 U+0957 |
| Bengali | 96 | 32 | 629 | U+09D7 U+09E2 U+09E3 U+09FE |
| Gurmukhi | 80 | 33 | 629 | U+0A51 **U+0A70** **U+0A71** U+0A75 |
| Gujarati* | 91 | 34 | 629 | U+0AE2 U+0AE3 U+0AFA–U+0AFF |
| Odia | 91 | 34 | 629 | U+0B55 U+0B56 U+0B57 U+0B62 U+0B63 |
| Tamil† | 72 | 23 | 629 | U+0BD7 |
| Telugu | 100 | 36 | 629 | U+0C00 U+0C04 U+0C55 U+0C56 U+0C62 U+0C63 |
| Kannada | 91 | 35 | 629 | U+0CD5 U+0CD6 U+0CE2 U+0CE3 U+0CF3 |
| Malayalam | 118 | 37 | 629 | U+0D00 U+0D3B U+0D57 U+0D62 U+0D63 |
| Sinhala* | 91 | 41 | 765 | the dependent vowel signs U+0DCF–U+0DDF, U+0DF2, U+0DF3 (Sinhala does not follow the ISCII layout, so it needs its own table) |

\* Supported by the tables but not a v2.0 release target.
† Shipped Tamil uses the curated segmenter in `swaram_tokenizer.py`, not this
generic path, so this row does not affect Swaram v3.

## Appendix C — Calculating the tax

For a fixed text `x` in language `ℓ`:

```
tax_T(x)      = |T(x)| / |Swaram_ℓ(x)|
fertility_T   = |T(x)| / aksharas_ℓ(x)
parity        = max_ℓ fertility_Swaram(ℓ) / min_ℓ fertility_Swaram(ℓ)
bits/akshara  = (NLL_per_token / ln 2) × fertility
```

| quantity | cost under tokenizer with tax `k` |
|---|---|
| training tokens | `k × N` |
| inference tokens / decode steps | `k × N` |
| effective context | `window / k` |
| KV-cache memory for the same text | `k ×` |

## Appendix D — Tamil artifact card (Swaram v3)

| field | value |
|---|---|
| tokenizer | swaram (Rust backend) |
| script | Tamil |
| vocab size | 31,964 |
| merges | 28,549 |
| base inventory | 404 |
| byte fallback | yes |
| morphology guard | no |
| merge sample | 300 M chars, stride 3 |
| corpus | `data/raw/expanded_v2` (538,562 docs, 901.6 M chars) |
| packed tokens | 258,338,989 |
| train / val sequences | 338,758 / 6,878 |
| seq len | 512 |
| eval fertility | 0.5061 |
| eval lossless | 1.0 |
| encode throughput | 473.7 kchar/s (freeze harness); ~2.5 Mchar/s (benchmark) |
| code SHA | `dad6f47` |

## Appendix E — Glossary

| term | meaning |
|---|---|
| akshara (अक्षर / அக்ஷரம்) | a base letter plus its attached marks; the written syllable |
| matra | dependent vowel sign |
| virama / halant / pulli / hasanta / chandrakkala / pollu | the sign that removes a consonant's inherent vowel |
| conjunct (*samyuktākṣara*, *ottakshara*, *juktakkhor*, *vattu*) | two or more consonants joined via virama into one akshara |
| chillu (ചില്ല്) | Malayalam pure-consonant letter, atomic since Unicode 5.1 |
| tippi / bindi / addak | Gurmukhi nasalisation (tippi, bindi) and gemination (addak) marks |
| reph / phala | Bengali/Odia/Devanagari r-initial and subscript conjunct forms |
| nukta | dot below that creates new consonants (क़, ড়, ଡ଼) |
| sandhi (புணர்ச்சி, ಸಂಧಿ, సంధి) | sound change at morpheme or word boundaries |
| fertility | tokens per akshara (lower is more compression) |
| token tax | tokens_T / tokens_Swaram on identical text |
| parity ratio | worst-to-best fertility across languages in one vocabulary |
| akshara slot | script-independent position of an akshara in the ISCII-aligned layout |

---

*Adhan SLM — Yazhi. Script-native language technology for India.*
