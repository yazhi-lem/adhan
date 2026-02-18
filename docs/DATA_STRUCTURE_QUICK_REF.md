# Data Structure - Quick Reference

## 🎯 For Training: Use This Path

```
data/final/tamil_texts/hf/
├── train.jsonl           ← Training data (1,220 records)
├── validation.jsonl      ← Validation data (152 records)
├── test.jsonl            ← Test data (154 records)
└── README.md
```

**Copy this path for notebooks:**
```
data/final/tamil_texts/hf/
```

---

## 📊 Complete Data Directory Map

```
data/
├── README.md                          ← READ THIS FIRST

├── raw/                               (Source data ~268MB)
│   ├── tamil_social_sample.jsonl     (social media)
│   ├── tamil_corpus.txt              (generic corpus)
│   ├── projectmadurai_manifests/     (classical literature)
│   ├── pdf_books_manifests/          (books)
│   ├── tamilvu_manifests/            (TamilVu content)
│   ├── raw_html/, raw_pdf/           (extracted HTML/PDF)
│   └── ... (other raw sources)

├── intermediate/                      (Pipeline working files ~4MB)
│   ├── sentences/
│   │   ├── wiki_sentences.jsonl      (1,493 from local)
│   │   └── wiki_api_sentences.jsonl  (1,427 from Wikipedia)
│   └── rebalancing/
│       ├── v1_original.jsonl         (2,918 - merged)
│       ├── v2_rebalanced.jsonl       (2,900 - quality filtered)
│       └── v3_modern_enhanced.jsonl  (3,066 - modern sources)

└── final/                             (Training-ready ~2.7MB) ✅ USE THIS
    └── tamil_texts/
        └── hf/
            ├── train.jsonl            (80%)
            ├── validation.jsonl       (10%)
            ├── test.jsonl             (10%)
            └── README.md
```

---

## 🔍 Which File Should I Use?

| Task | Location | Size |
|------|----------|------|
| **Train model** | `data/final/tamil_texts/hf/` | 2.7 MB |
| Analyze original | `data/intermediate/rebalancing/v1_*.jsonl` | 1.2 MB |
| Analyze quality | `data/intermediate/rebalancing/v2_*.jsonl` | 1.1 MB |
| Analyze modern | `data/intermediate/rebalancing/v3_*.jsonl` | 1.2 MB |
| View raw sources | `data/raw/` | 268 MB |

---

## 📝 Record Format (All JSONL files)

Each line is one JSON record:
```json
{
  "id": "sha256_hash",
  "text": "தமிழ் text here...",
  "source": "wikipedia|news|social|local|literature|modern_conversational",
  "quality_score": 0.524,
  "tamil_fraction": 0.95,
  "url": "source_url_or_null"
}
```

---

## 🚀 Quick Commands

```bash
# Check training data
wc -l data/final/tamil_texts/hf/*.jsonl

# Peek at records
head -1 data/final/tamil_texts/hf/train.jsonl | python -m json.tool

# Count records per source
python -c "
import json
counts = {}
with open('data/final/tamil_texts/hf/train.jsonl') as f:
    for line in f:
        src = json.loads(line).get('source', 'unknown')
        counts[src] = counts.get(src, 0) + 1
for src, cnt in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'{src}: {cnt}')
"

# Show file sizes
du -sh data/*/* data/*
```

---

## ✨ Key Points

1. **Always use `data/final/Tamil_texts/hf/` for training**
2. **Don't manually edit JSONL files**
3. **Intermediate files can be regenerated if needed**
4. **Raw files are kept for reproducibility**
5. **See `data/README.md` for detailed explanation**

---

**Status**: ✅ Refactored Feb 19, 2026  
**Training Data**: `data/final/tamil_texts/hf/`
