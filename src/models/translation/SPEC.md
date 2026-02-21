# Adhan Translation Model - Specification

## Goal
Small, fast Tamil ↔ English translation model quantized for all devices.

## Architecture

| Model | Params | Use Case |
|-------|--------|----------|
| **Adhan-Trans-S** | 135M | Mobile/IoT (Q4/Q2) |
| **Adhan-Trans-M** | 500M | Laptop (Q8/Q4) |

Base: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) or [google/mt5-base](https://huggingface.co/google/mt5-base)

## Quantization Targets

| Format | Size | Devices |
|--------|------|---------|
| FP16 | ~1GB | Desktop |
| Q8 | ~600MB | Laptop |
| Q4 | ~300MB | Mobile |
| Q2 | ~150MB | IoT/Embedded |

## Training Strategy

1. **Base**: Fine-tune NLLB-200-distilled-600M on Tamil-English pairs
2. **Data**: Use existing Tamil corpus + English translations
3. **Task**: Seq2seq translation (Tamil → English, English → Tamil)

## Next Steps

- [ ] Acquire parallel Tamil-English dataset
- [ ] Create fine-tuning script
- [ ] Run training (1-3 epochs)
- [ ] Quantize with bitsandbytes
- [ ] Push to HF Hub

## Commands (Draft)

```bash
# Fine-tune
python scripts/finetune_translation.py --base_model facebook/nllb-200-distilled-600M \
    --data data/final/tamil_texts/hf \
    --lang_pair ta-en \
    --epochs 3

# Quantize
python scripts/quantize.py --model adhan-trans --bits 4
```
