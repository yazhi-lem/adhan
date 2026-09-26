# Swaram Tokenizer Benchmark: vs BPE and SentencePiece

One-page comparison of the Swaram (akshara-native) tokenizer against a byte-level BPE tokenizer and a SentencePiece (BPE mode) tokenizer, trained and evaluated on the same frozen corpus.

## Methodology

- **Corpus:** Thirukkural (1330 documents: couplet + 3 classical
  commentaries each), fetched from `https://raw.githubusercontent.com/tk120404/thirukkural/master/thirukkural.json` via `src/data_scraper/raw_extractors/thirukkural_extractor.py`.
- **Split:** deterministic by document index -- every 10th document held out for eval (133 docs), the rest for training (1197 docs). No randomness, no seed needed.
- **Target vocab size:** 3000 for all three tokenizers. BPE's *actual* vocab (see table) can plateau below target -- the `tokenizers` library stops merging once no more frequent adjacent pairs exist in the training data, which can happen before the target on a small corpus. Swaram and SentencePiece both fill the full requested budget by falling back to lower-frequency merges/pieces, so they typically reach the target exactly.
- **Fertility** = tokens / akshara, using Swaram's own rule-based akshara segmenter (`segment_aksharas`) as the shared denominator for all three tokenizers -- akshara is the linguistically correct atomic unit for Tamil, so this keeps the comparison apples-to-apples regardless of how each tokenizer internally splits text.
- **Vocabulary coverage** = % of eval-set tokens that are NOT the tokenizer's `<unk>` token. Byte-level BPE has no `<unk>` by construction (any byte sequence is representable), so it always reports 100% here -- this is a real property of that approach, not a benchmark artifact. SentencePiece was trained with `byte_fallback=False` deliberately, so its coverage number reflects what its *learned* vocabulary actually covers, not a byte-level safety net.

## Results

| Tokenizer | Vocab size | Tokens/word | Vocab coverage | Fertility (mean) | Fertility (median) |
|---|---|---|---|---|---|
| Swaram | 3,000 | 2.768 | 99.99% | 0.606 | 0.605 |
| BPE (tokenizers, byte-level) | 1,677 | 6.338 | 100.00% | 1.388 | 1.391 |
| SentencePiece (BPE) | 3,000 | 2.053 | 99.89% | 0.450 | 0.441 |

## Interpretation

On this run, SentencePiece (BPE) achieves lower mean fertility (0.450 tokens/akshara) than Swaram (0.606). This is a genuine result, not a bug in the benchmark -- see the caveat below for the most likely reason, and re-running at a larger vocab size / on a less repetitive corpus is the natural next step to check whether it holds up.

A fertility near or below 1.0 means the tokenizer spends about one token (or fewer, once merges kick in) per akshara -- the theoretical floor for a lossless akshara-aware scheme -- rather than fragmenting Tamil script into multiple sub-akshara pieces the way byte-level tokenizers with no Tamil-specific prior tend to.

## Round-trip Losslessness

Fertility and coverage above only measure *efficiency*. Before committing to a real training run, the more important question is whether each tokenizer can *losslessly* reconstruct real Tamil -- `decode(encode(text)) == text` -- across the full eval corpus. A tokenizer that silently loses information here would silently corrupt every training run built on it.

| Tokenizer | Exact round-trip | Round-trip after NFC normalization | Docs checked |
|---|---|---|---|
| Swaram | 98.50% | 98.50% | 133 |
| BPE (tokenizers, byte-level) | 100.00% | 100.00% | 133 |
| SentencePiece (BPE) | 0.00% | 0.00% | 133 |

"Exact" is byte-for-byte identity with the original text -- what actually matters for training-data fidelity. "After NFC" separates genuine information loss from a tokenizer's Unicode normalization policy (e.g. SentencePiece normalizes input by default, so a document containing non-NFC sequences can legitimately decode to its NFC-normalized form rather than its original byte sequence -- that's a policy difference, not data corruption).

**Example round-trip mismatches found (first few per tokenizer):**

- `Swaram`:
  - input:  `'திருக்குறள் 870:\nகல்லான் வெகுளும் சிறுபொருள் எஞ்ஞான்றும்\nஒல்லானை ஒல்லா தொளி.\n\nமு. வரதராசனார் உரை: கல்வி கற்காதவனைப் பகைத்துக்கொள்ளும் எளிய செயலைச் செய்ய இயலாத ஒருவனிடம் எக்காலத்திலும் புகழ் வந்து பொருந்தாது.\n\nசாலமன் பாப்பையா உரை: நீதி நூல்களைக் கல்லாதவனைப் பகைப்பதால் கிடைக்கும் பொருள் சிறிது எனினும், அதை விரும்பாத அரசுக்கு ஒருபோது் புகழ் சேராது.\n\nகலைஞர் உரை: போர்முறை கற்றிடாத பகைவர்களைக்கூட எதிர்ப்பதற்குத் தயக்கம் காட்டுகிறவர்கள், உண்மையான வீரர்களை எப்படி எதிர்கொள்வார்கள் எனக் கேலி புரிந்து, புகழ் அவர்களை அணுகாமலே விலகிப் போய்விடும்'`
  - output: `'திருக்குறள் 870:\nகல்லான் வெகுளும் சிறுபொருள் எஞ்ஞான்றும்\nஒல்லானை ஒல்லா தொளி.\n\nமு. வரதராசனார் உரை: கல்வி கற்காதவனைப் பகைத்துக்கொள்ளும் எளிய செயலைச் செய்ய இயலாத ஒருவனிடம் எக்காலத்திலும் புகழ் வந்து பொருந்தாது.\n\nசாலமன் பாப்பையா உரை: நீதி நூல்களைக் கல்லாதவனைப் பகைப்பதால் கிடைக்கும் பொருள் சிறிது எனினும், அதை விரும்பாத அரசுக்கு ஒருபோ புகழ் சேராது.\n\nகலைஞர் உரை: போர்முறை கற்றிடாத பகைவர்களைக்கூட எதிர்ப்பதற்குத் தயக்கம் காட்டுகிறவர்கள், உண்மையான வீரர்களை எப்படி எதிர்கொள்வார்கள் எனக் கேலி புரிந்து, புகழ் அவர்களை அணுகாமலே விலகிப் போய்விடும்'`
  - likely cause: decoded output is shorter (535 vs 538 chars) with no visible unknown-token marker in the output -- consistent with the same silent unk-drop pattern above, just harder to spot here because nothing marks where the loss happened (unlike a tokenizer that leaves a visible <unk>/⁇ placeholder, which at least tells you where to look)
  - input:  `'திருக்குறள் 1210:\nவிடாஅது சென்றாரைக் கண்ணினால் காணப்\nபடாஅதி வாழி மதி.\n\nமு. வரதராசனார் உரை: தி்ங்களே! பிரியாமல் இருந்து இறுதியில் பிரிந்து சென்ற காதலரை என் கண்ணால் தேடிக் காணும்படியாக நீ மறைந்து விடாமல் இருப்பாயாக!\n\nசாலமன் பாப்பையா உரை: திங்களே! பிரியாமலிருந்து இறுதியில் பிரிந்து சென்ற காதலரை என் கண்ணால் தேடிக் காணும்படியாக நீ மறைந்து விடாமல் இருப்பாயாக!\n\nகலைஞர் உரை: நிலவே! நீ வாழ்க; இணைபிரியாமலிருந்து, பிரிந்து சென்றுள்ள காதலரை நான் என் கண்களால் தேடிக் கண்டுபிடித்திடத் துணையாக நீ மறையாமல் இருப்பாயாக'`
  - output: `'திருக்குறள் 1210:\nவிடாஅது சென்றாரைக் கண்ணினால் காணப்\nபடாஅதி வாழி மதி.\n\nமு. வரதராசனார் உரை: ங்களே! பிரியாமல் இருந்து இறுதியில் பிரிந்து சென்ற காதலரை என் கண்ணால் தேடிக் காணும்படியாக நீ மறைந்து விடாமல் இருப்பாயாக!\n\nசாலமன் பாப்பையா உரை: திங்களே! பிரியாமலிருந்து இறுதியில் பிரிந்து சென்ற காதலரை என் கண்ணால் தேடிக் காணும்படியாக நீ மறைந்து விடாமல் இருப்பாயாக!\n\nகலைஞர் உரை: நிலவே! நீ வாழ்க; இணைபிரியாமலிருந்து, பிரிந்து சென்றுள்ள காதலரை நான் என் கண்களால் தேடிக் கண்டுபிடித்திடத் துணையாக நீ மறையாமல் இருப்பாயாக'`
  - likely cause: decoded output is shorter (501 vs 504 chars) with no visible unknown-token marker in the output -- consistent with the same silent unk-drop pattern above, just harder to spot here because nothing marks where the loss happened (unlike a tokenizer that leaves a visible <unk>/⁇ placeholder, which at least tells you where to look)
- `SentencePiece (BPE)`:
  - input:  `'திருக்குறள் 10:\nபிறவிப் பெருங்கடல் நீந்துவர் நீந்தார்\nஇறைவன் அடிசேரா தார்.\n\nமு. வரதராசனார் உரை: இறைவனுடைய திருவடிகளை பொருந்தி நினைக்கின்றவர் பிறவியாகிய பெரிய கடலைக் கடக்க முடியும். மற்றவர் கடக்க முடியாது\n\nசாலமன் பாப்பையா உரை: கடவுளின் திருவடிகளைச் சேர்ந்தவர் பிறவியாகிய பெருங்கடலை நீந்திக் கடப்பர்; மற்றவர் நீந்தவும் மாட்டார்\n\nகலைஞர் உரை: வாழ்க்கை எனும் பெருங்கடலை நீந்திக் கடக்க முனைவோர், தலையானவனாக இருப்பவனின் அடி தொடர்ந்து செல்லாவிடில் நீந்த முடியாமல் தவிக்க நேரிடும்'`
  - output: `'திருக்குறள் 10: பிறவிப் பெருங்கடல் நீந்துவர் நீந்தார் இறைவன் அடிசேரா தார். மு. வரதராசனார் உரை: இறைவனுடைய திருவடிகளை பொருந்தி நினைக்கின்றவர் பிறவியாகிய பெரிய கடலைக் கடக்க முடியும். மற்றவர் கடக்க முடியாது சாலமன் பாப்பையா உரை: கடவுளின் திருவடிகளைச் சேர்ந்தவர் பிறவியாகிய பெருங்கடலை நீந்திக் கடப்பர்; மற்றவர் நீந்தவும் மாட்டார் கலைஞர் உரை: வாழ்க்கை எனும் பெருங்கடலை நீந்திக் கடக்க முனைவோர், தலையானவனாக இருப்பவனின் அடி தொடர்ந்து செல்லாவிடில் நீந்த முடியாமல் தவிக்க நேரிடும்'`
  - likely cause: differs only in whitespace/newlines -- tokenizer's normalizer collapsed whitespace or newlines (a real loss of document structure, not just Unicode form)
  - input:  `'திருக்குறள் 20:\nநீர்இன்று அமையாது உலகெனின் யார்யார்க்கும்\nவான்இன்று அமையாது ஒழுக்கு.\n\nமு. வரதராசனார் உரை: எப்படிப்பட்டவர்க்கும் நீர் இல்லாமல் உலக வாழ்க்கை நடைபெறாது என்றால், மழை இல்லையானால் ஒழுக்கமும் நிலைபெறாமல் போகும்\n\nசாலமன் பாப்பையா உரை: எத்தனை பெரியவரானாலும் நீர் இல்லாமல் வாழமுடியாது; அந்த நீரோ மழை இல்லாமல் கிடைக்காது\n\nகலைஞர் உரை: உலகில் மழையே இல்லையென்றால் ஒழுக்கமே கெடக்கூடும் என்ற நிலை இருப்பதால், நீரின் இன்றியமையாமையை உணர்ந்து செயல்பட வேண்டும்'`
  - output: `'திருக்குறள் 20: நீர்இன்று அமையாது உலகெனின் யார்யார்க்கும் வான்இன்று அமையாது ஒழுக்கு. மு. வரதராசனார் உரை: எப்படிப்பட்டவர்க்கும் நீர் இல்லாமல் உலக வாழ்க்கை நடைபெறாது என்றால், மழை இல்லையானால் ஒழுக்கமும் நிலைபெறாமல் போகும் சாலமன் பாப்பையா உரை: எத்தனை பெரியவரானாலும் நீர் இல்லாமல் வாழமுடியாது; அந்த நீரோ மழை இல்லாமல் கிடைக்காது கலைஞர் உரை: உலகில் மழையே இல்லையென்றால் ஒழுக்கமே கெடக்கூடும் என்ற நிலை இருப்பதால், நீரின் இன்றியமையாமையை உணர்ந்து செயல்பட வேண்டும்'`
  - likely cause: differs only in whitespace/newlines -- tokenizer's normalizer collapsed whitespace or newlines (a real loss of document structure, not just Unicode form)
  - input:  `'திருக்குறள் 30:\nஅந்தணர் என்போர் அறவோர்மற் றெவ்வுயிர்க்கும்\nசெந்தண்மை பூண்டொழுக லான்.\n\nமு. வரதராசனார் உரை: எல்லா உயிர்களிடத்திலும் செம்மையான அருளை மேற்கொண்டு ஒழுகுவதால், அறவோரே அந்தணர் எனப்படுவோர் ஆவர்.\n\nசாலமன் பாப்பையா உரை: எல்லா உயிர்களிடத்திலும் இரக்கம் கொண்டு வாழ்பவரே அறவோர்; அவரே அந்தணர்.\n\nகலைஞர் உரை: அனைத்து உயிர்களிடத்திலும் அன்புகொண்டு அருள் பொழியும் சான்றோர் எவராயினும் அவர் அந்தணர் எனப்படுவார்'`
  - output: `'திருக்குறள் 30: அந்தணர் என்போர் அறவோர்மற் றெவ்வுயிர்க்கும் செந்தண்மை பூண்டொழுக லான். மு. வரதராசனார் உரை: எல்லா உயிர்களிடத்திலும் செம்மையான அருளை மேற்கொண்டு ஒழுகுவதால், அறவோரே அந்தணர் எனப்படுவோர் ஆவர். சாலமன் பாப்பையா உரை: எல்லா உயிர்களிடத்திலும் இரக்கம் கொண்டு வாழ்பவரே அறவோர்; அவரே அந்தணர். கலைஞர் உரை: அனைத்து உயிர்களிடத்திலும் அன்புகொண்டு அருள் பொழியும் சான்றோர் எவராயினும் அவர் அந்தணர் எனப்படுவார்'`
  - likely cause: differs only in whitespace/newlines -- tokenizer's normalizer collapsed whitespace or newlines (a real loss of document structure, not just Unicode form)

## Tamil Script Stress Test

20 curated cases covering Tamil script edge cases that a tokenizer must handle correctly before it's safe to build a training pipeline on: the full vowel and consonant inventory, Grantha/borrowed letters (ஜ ஷ ஸ ஹ), complex conjuncts (க்ஷ், ஸ்ரீ), the rare ஆய்தம் (ஃ), Tamil numerals, code-switched English, punctuation, whitespace edge cases, and -- notably -- the *same visual word* encoded two different but both-valid ways in Unicode (composed `கொ` vs. its canonically-decomposed form), since real scraped text can contain either inconsistently.

| Tokenizer | Passed (NFC-tolerant) | Passed (byte-exact) |
|---|---|---|
| Swaram | 18/20 | 17/20 |
| BPE (tokenizers, byte-level) | 20/20 | 20/20 |
| SentencePiece (BPE) | 8/20 | 7/20 |

**Failing cases (NFC-tolerant), with likely cause:**

- `Swaram`:
  - `tamil_numerals`: `'௦௧௨௩௪௫௬௭௮௯'` -> `''`
    likely cause: entire input vanished on decode -- likely every token mapped to <unk> and got silently suppressed on decode (default skip_special behavior), rather than an encoding failure
  - `code_switched_english`: `'இது ஒரு AI model தான்'` -> `'இது ஒரு   தான்'`
    likely cause: decoded output is shorter (14 vs 21 chars) with no visible unknown-token marker in the output -- consistent with the same silent unk-drop pattern above, just harder to spot here because nothing marks where the loss happened (unlike a tokenizer that leaves a visible <unk>/⁇ placeholder, which at least tells you where to look)
- `SentencePiece (BPE)`:
  - `all_12_vowels`: `'அஆஇஈஉஊஎஏஐஒஓஔ'` -> `'அஆஇஈஉஊஎஏ ⁇ ஒஓ ⁇ '`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `uyirmei_full_row_ka`: `'கககாகிகீகுகூகெகேகைகொகோகௌ'` -> `'கககாகிகீகுகூகெகேகைகொகோக ⁇ '`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `aytham`: `'எஃகு'` -> `'எ ⁇ கு'`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `grantha_ja`: `'ஜனநாயகம்'` -> `' ⁇ னநாயகம்'`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `grantha_sha`: `'விஷயம்'` -> `'வி ⁇ யம்'`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `grantha_sa`: `'ஸ்ரீ'` -> `' ⁇ ்ரீ'`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `grantha_ha`: `'வாஹனம்'` -> `'வா ⁇ னம்'`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `ksha_conjunct`: `'லக்ஷ்மி'` -> `'லக் ⁇ ்மி'`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `tamil_numerals`: `'௦௧௨௩௪௫௬௭௮௯'` -> `' ⁇ '`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `code_switched_english`: `'இது ஒரு AI model தான்'` -> `'இது ஒரு  ⁇   ⁇  தான்'`
    likely cause: decoded output contains an explicit unknown-token marker -- input has character(s) outside this tokenizer's trained vocabulary (out-of-vocab / low character coverage for this character)
  - `repeated_whitespace`: `'  தமிழ்   மொழி  '` -> `'தமிழ் மொழி'`
    likely cause: differs only in whitespace/newlines -- tokenizer's normalizer collapsed whitespace or newlines (a real loss of document structure, not just Unicode form)
  - `newline_in_text`: `'வணக்கம்\nதமிழ் மொழி'` -> `'வணக்கம் தமிழ் மொழி'`
    likely cause: differs only in whitespace/newlines -- tokenizer's normalizer collapsed whitespace or newlines (a real loss of document structure, not just Unicode form)


Every Thirukkural document repeats the exact same boilerplate labels (`"மு. வரதராசனார் உரை:"`, `"சாலமன் பாப்பையா உரை:"`, `"கலைஞர் உரை:"`, `"திருக்குறள் <N>:"`) 1,330 times. A frequency-driven learner (BPE, SentencePiece) can dedicate a large slice of a *small* vocabulary budget to memorizing these exact recurring phrases as single efficient merges, which flatters tokens-per-word and fertility in a way that may not hold on more varied, less boilerplate-heavy Tamil text. Swaram's vocabulary is built akshara-first (247 base units) before merges, so a larger share of its budget goes to general coverage rather than corpus-specific memorization -- expected to matter more at larger vocab sizes and on more varied corpora, but not tested here. Treat this report as a first, reproducible data point, not a final verdict; a follow-up with a more varied corpus and multiple vocab sizes is the natural next step.

## Recommendations before starting real training

- Swaram failed 3/20 stress cases byte-exact. Check the failure diagnoses above -- if any show "entire input vanished on decode" or an unknown-token marker, that means `SwaramTokenizer.decode()`'s default `skip_special=True` is silently deleting out-of-vocabulary characters (English words, digits, rare symbols) rather than showing them. That's invisible unless you specifically decode with `skip_special=False` -- worth fixing (or at minimum documenting loudly) before trusting Swaram on real scraped corpora, which will contain code-switched English and digits far more often than Thirukkural does.
- Swaram's corpus-wide exact round-trip is 98.50%, not 100%. Given the akshara segmenter itself is documented as lossless (`segment_aksharas`), any gap here traces back to the same vocab-coverage/decode issue above, not the segmentation layer.
- SentencePiece (BPE)'s corpus-wide exact round-trip is only 0.00% (and 0.00% even after NFC normalization, so this is real information loss, not just a Unicode form difference). Do not use this configuration for real training data preparation without fixing the underlying cause (see failure diagnoses above).

Numbers above are from a single run on the frozen split described in Methodology; re-running `scripts/benchmark_tokenizers.py` reproduces them exactly (same corpus source, same split rule, same vocab size).
