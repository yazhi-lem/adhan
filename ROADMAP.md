# Adhan Roadmap

## Project Completion: ~35%

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
- [ ] Add Twitter/X scraper for Tamil content
- [ ] Add Reddit (r/Tamil, r/tamilnadu) scraper
- [ ] Add Instagram/Telegram collectors
- [ ] Build colloquial Tamil corpus expansion

#### Phase 3: Optimization (ONNX/Quantization)
- [ ] Convert to ONNX format
- [ ] INT8 post-training quantization
- [ ] INT4 quantization for edge deployment
- [ ] Benchmark on Raspberry Pi 5

#### Phase 4: Downstream Tasks
- [ ] Fine-tune for NER
- [ ] Fine-tune for Sentiment Analysis
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
