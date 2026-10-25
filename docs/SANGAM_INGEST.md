# Sangam ingestion: never merge a song, label every dimension

The goal is Tamil *intelligence*, not Tamil *volume*. For classical material the two
diverge sharply: the Sangam corpus is small (~0.5M tokens) and its value is entirely in
structure — which landscape, which season, which emotion, which imagery. A corpus that
concatenates poems into blocks throws that structure away and keeps only the words.

Tool: `scripts/ingest_open_sangam.py` · canon: `src/adhan_slm/data/sangam_tinai_canon.json`
· loader: `src/adhan_slm/data/sangam_tinai.py`

---

## 1. What was extracted (and what the archive already had)

| layer | docs | chars | tokens (est) | already in archive | rejected |
|---|---|---|---|---|---|
| `verse` (`sangamTamil`) | 674 | 134,558 | ~41.8k | 674 | 1,995 |
| `urai` (traditional commentary) | 1,847 | 1,407,221 | **~437.6k** | 1,846 | 804 |
| `yazhi-urai` (model-generated commentary) | 24 | 8,610 | ~2.7k | 24 | 8 |
| **total** | **2,545** | **1,550,389** | **~482k** | 2,544 | 2,807 |

Two findings worth stating plainly:

1. **The archive already contains essentially all Sangam verse and commentary** — 2,544
   of 2,545 documents hash-match text already on disk. The generic JSON walker in
   `expand_tamil_corpus.py` had pulled every string field of every poem record, so this
   material was ingested long ago — *as fragments*: each `lines` item, each field, each
   commentary section as its own "document".
2. **So the gain is not tokens, it is coherence + labels.** The extractor re-emits the
   same text as **one document per (song, layer)** and attaches four poetics dimensions
   to each. That is what the model can actually learn from.

Excluded on purpose: `english`, `title`, `heading`, `lines`, `culturalNotes`,
`audioUrl`, `licenses`, `keywords`, `resources`. These are metadata, English glosses, or
sub-document fragments — the exact material that made the old `local:*` sources noisy.
(`agent_artifacts.jsonl` was also inspected and excluded: 56 records, **0% predominantly
Tamil** — it is an English QA log, not corpus.)

## 2. Songs are never merged

One document == one (song, layer). The extractor raises if two songs would share an id:

```python
if doc_id in seen_ids:
    raise SystemExit(f"duplicate song id {doc_id}: songs must not be merged")
```

A verse is never concatenated with its commentary, and two verses are never joined. The
reason is the label space below: a merged document would span tiṇais, seasons and
emotions, and every one of its labels would then be wrong.

With `--split-dirs` the corpus is also **partitioned on disk** by tiṇai and layer:

```
data/raw/sangam/
  kurinji/{verse,urai}.jsonl      neytal/{verse,urai}.jsonl
  mullai/{verse,urai}.jsonl       palai/{verse,urai}.jsonl
  marutam/{verse,urai}.jsonl      unknown/{verse,urai,yazhi-urai}.jsonl
```

so a mixture can sample kurinji-urai and palai-verse independently, and no document
ever crosses a landscape boundary.

## 3. The four dimensions

| dimension | what it is | example (kurinji) |
|---|---|---|
| **tiṇai** | one of the five akam landscapes | குறிஞ்சி — mountains |
| **kalam** | the season the landscape implies | கூதிர், முன்பனி |
| **uripporul** | the *essential action/emotion* | புணர்தல் — union |
| **karupporul** | the *essential substances/imagery* the diction is built from | முருகன், குறிஞ்சி, மயில், யானை, தண்ணுமை, அருவி, குறவர் |

The akam mapping used (from the Naṉṉūl / Tholkappiyam commentary tradition):

| tiṇai | kalam | uripporul |
|---|---|---|
| குறிஞ்சி kurinji | கூதிர், முன்பனி | புணர்தல் union |
| முல்லை mullai | கார் | இருத்தல் waiting |
| மருதம் marutam | பொது (any) | ஊடல் quarrel |
| நெய்தல் neytal | வேனில் | இரங்கல் pining |
| பாலை palai | வேனில், முன்பனி | பிரிதல் separation |

**Provenance and caveat, stated up front:** the tiṇai set and the uripporul mapping are
the stable part of the tradition; kalam assignments and especially the karupporul item
lists **vary between anthologies and commentaries**. The canon is therefore a JSON data
file, not code, with `attested: true/false` per imagery item and `note` fields for known
disagreements. `--attested-only` restricts emission to items a second source also lists.
Do not quote this table as scholarship without checking a printed edition.

## 4. Labels as training signal

`data/final/tinai_labels.jsonl`, one row per song (2,687 songs, **2,364 labelled = 88%**
after the knowledge graph's `HAS_TINAI` edges filled in gaps):

```json
{"id": "ainkurunooru_001", "work": "ainkurunooru", "layer": "song",
 "tinai": "marutam", "tinai_tamil": "மருதம்",
 "kalam": "podhu", "kalam_all": ["podhu"],
 "uripporul": "oodal", "uripporul_tamil": "ஊடல்",
 "karupporul": ["எருமை", "அன்னம்", "இந்திரன்", "தாமரை", "உழவர்", "மருதம்", "ஆறு"],
 "kalam_primary": "podhu", "kalam_raw": null, "poet": "…", "number": 1}
```

Unknown tiṇai yields **nulls, not a guess** — a wrong landscape label is worse than a
missing one for a conditioning signal, and the test suite enforces that contract.

Distribution in the corpus: neytal 744 · palai 596 · kurinji 511 · marutam 257 ·
mullai 256 · unlabelled 323. Seasons: வேனில் 1,340 · கூதிர் 511 · பொது 257 · கார் 256 ·
unknown 323.

This is what the model's Tiṇai conditioning layer (`docs/TINAI_CONDITIONING.md`,
`use_tinai`) needs. Wiring it into `train_jax` (reading the sidecar, passing labels,
adding `tinai_aux_loss`) is the next commit — the labels exist now, the layer exists,
they are just not connected yet.

## 5. Reproduce

```bash
python scripts/ingest_open_sangam.py --split-dirs \
    --sangam-root /path/to/open-sangam \
    --out data/raw/sangam \
    --labels data/final/tinai_labels.jsonl
```

`--sangam-root` is required (or set `SANGAM_ROOT=/path/to/open-sangam`); there is no
machine-specific default.

Then treat `data/raw/sangam` as its own source in the mixture
(`docs/DATA_WEIGHTING.md`) rather than merging it into `data/raw/expanded`:

```bash
python scripts/build_mixture.py --corpus data/raw/expanded \
    --out data/final/mix_v1 --max-tokens 30000000 \
    --weights "ai4bharat-indiccorp-v2:0.40,fineweb-2-ta:0.25,wikipedia-ta:0.13,local:0.12,sangam:0.10"
```

10% for ~0.5M tokens of the purest, most morphology-dense Tamil in existence is a
deliberate over-weight — that is the point of weighting, and `docs/DATA_WEIGHTING.md`
says how to test whether it pays off.
