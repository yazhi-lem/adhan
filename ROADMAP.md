# Adhan Roadmap

## Project Completion: ~60%

### ✅ Completed
- [x] Data pipeline (scrapers, cleaners, processors)
- [x] Modern Tamil corpus (1,220 train, 152 val, 154 test samples)
- [x] Training notebooks (setup, exploration, training)
- [x] Requirements & environment setup
- [x] HuggingFace-style dataset export

### 🚧 In Progress
- [ ] Model training (XLM-RoBERTa-base, 10 epochs)

### 📋 Planned

#### Phase 1: Model Training
- [ ] Run `02_model_training.ipynb` on GPU
- [ ] Evaluate on validation/test sets
- [ ] Generate training metrics

#### Phase 2: Social Media Integration
- [x] Add Reddit (r/Tamil, r/tamilnadu, r/kollywood, r/Chennai) scraper → `src/data_scraper/raw_extractors/reddit_scraper.py`
- [x] Add Twitter/X collector (via Nitter, no API key) → `src/data_scraper/raw_extractors/twitter_scraper.py`
- [x] Integrate social scrapers into `scripts/run_scraper.py` (`--social reddit|twitter|all`)
- [ ] Add Instagram/Telegram collectors
- [ ] Build colloquial Tamil corpus expansion

#### Phase 3: Optimization (ONNX/Quantization)
- [x] Convert to ONNX format → `scripts/export_onnx.py` (supports optimum & torch fallback)
- [x] INT8 dynamic and static post-training quantization → `scripts/quantize_model.py`
- [x] INT4 weight-only quantization for edge deployment → `scripts/quantize_model.py --mode int4`
- [ ] Benchmark on Raspberry Pi 5

#### Phase 4: Downstream Tasks
- [ ] Fine-tune for NER
- [x] Fine-tune for Sentiment Analysis → `src/models/sentiment/train_sentiment.py`
- [ ] Build instruction-tuning dataset
- [ ] Create chat/inference demo

---

## Data Stats
- **Train**: 1,220 samples (~550KB)
- **Validation**: 152 samples
- **Test**: 154 samples
- **Target Model**: XLM-RoBERTa-base (124M params)

## Next Action
Run training notebook or expand social media data.

## Recent completions (Phase 2–4)
- Social media scrapers (Reddit + Twitter/X) added and integrated into `run_scraper.py`
- ONNX export (`scripts/export_onnx.py`) and INT8/INT4 quantization (`scripts/quantize_model.py`) added
- Tamil sentiment fine-tuning trainer added (`src/models/sentiment/train_sentiment.py`)
