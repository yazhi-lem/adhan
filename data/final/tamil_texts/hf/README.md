# Tamil Pretraining Dataset (Modern-Enhanced)

**Version**: 2.0 (Modern Tamil Focused)

## Dataset Overview

**Total Records**: 1,526 (80% train, 10% validation, 10% test)

### Version Changes (v2.0 - Modern Tamil Focus)

This version rebalances the corpus toward contemporary Tamil language:

- ✅ **Modern sources increased**: 3.1% → 9.8% (news, social, conversational)
- ✅ **Quality boosted**: Modern source quality avg: 0.531 (vs original 0.486)
- ✅ **Archaic reduction**: Classical/literary prioritized for quality over quantity
- ✅ **News integration**: News sources now 18.8% of training data (vs original 2.8%)

### New vs. Original Composition

| Category | Original | Modern-Enhanced | Change |
|----------|----------|-----------------|--------|
| Wikipedia | 48.8% | 46.5% | -2.3% |
| Project Madurai (local) | 43.4% | 41.3% | -2.1% |
| News | 2.8% | 8.8% | +6.0% |
| Literature | 4.7% | 2.4% | -2.3% |
| Social / Conversational | 0.3% | 1.0% | +0.7% |

## Train/Val/Test Split Details

**Training Set** (1,220 records):
- Wikipedia: 79.8% (formal contemporary)
- News: 17.5% (modern colloquial)
- Social: 1.5% (conversational)
- Modern Phrases: 0.6%
- Other: 0.6%

**Validation Set** (152 records): 10% stratified sample
**Test Set** (154 records): 10% stratified sample

## Quality Metrics

- **Avg Quality Score**: 0.524/1.0
- **Tamil Coverage**: ~85% Tamil characters
- **Modern Markers**: 10.0% of corpus
- **Deduplication**: Lowercased, normalized, SHA256 hashed
- **Source Weighting**: News/social boosted, classical literature reduced

## Record Format

Each record in JSONL format:
```json
{
  "id": "unique_hash",
  "text": "Tamil text content...",
  "source": "wikipedia|news|social|local|literature|modern_conversational",
  "quality_score": 0.524,
  "tamil_fraction": 0.95,
  "url": "source_url_if_available"
}
```

## Usage Examples

### Basic Load
```python
import json
records = []
with open('train.jsonl') as f:
    for line in f:
        records.append(json.loads(line))
```

### With Source-Based Sampling
```python
import json
from collections import defaultdict

records_by_source = defaultdict(list)
with open('train.jsonl') as f:
    for line in f:
        r = json.loads(line)
        records_by_source[r['source']].append(r)

# Sample with modern source weighting (3x for news/social)
modern_sources = ['news', 'social', 'modern_conversational']
weights = {s: 3 if s in modern_sources else 1 for s in records_by_source}
```

## Recommendations for Training

### For General Pretraining
Use all records with standard uniform sampling. This provides balanced coverage.

### For Modern Tamil Emphasis
- **Curriculum Learning**: Start with 100% modern sources (news/social) for first N epochs, gradually shift to full distribution
- **Weighted Sampling**: Apply 2-3x weight to news/social records during training
- **Fine-tune on News**: After pretraining, fine-tune 1-2 epochs on news corpus only
- **Filter by Quality**: Only use records with quality_score > 0.5 for critical applications

### To Further Enhance Modern Tamil
1. **Twitter/Social API**: Integrate live Tamil tweets (twitter/X API with ta language filter)
2. **News Scraping**: Scrape tamil.samayam.com, tamil.oneindia.com, manorama.com daily
3. **YouTube/Transcripts**: Extract Tamil YouTube video captions (music, vlogs, education)
4. **Reddit/Discord**: Archive Tamil community discussions from r/tamil, Discord servers
5. **Modern Literature**: Add contemporary Tamil novels, blogs, bloggers
6. **Speech Recognition**: Add ASR output from Tamil movies/shows (captures colloquial style)

## Version History

- **v1.0** (2025-02-16): Initial 2,918-record corpus (Project Madurai + Wikipedia)
- **v2.0** (2025-02-16): Modern-enhanced 3,066 → 1,526 deduplicated (9.8% modern)

## Next Steps

1. Train models on this dataset with recommended weighting
2. Evaluate on downstream Tamil tasks (sentiment, NER, MRC)
3. Add additional modern sources for v3.0
4. Collect human preference data for instruction-tuning

