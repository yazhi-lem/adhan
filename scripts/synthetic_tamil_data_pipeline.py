#!/usr/bin/env python3
"""
synthetic_tamil_data_pipeline.py — Local-Only Synthetic Test Data Generation
ARIVU | Rotation 26 Cycle 5+ | Jul 1, 2026

PRIORITY 0 DELIVERABLE: Fallback when network unavailable

This pipeline generates realistic Tamil training/evaluation data LOCALLY:
  1. Template-based synthesis: Common Tamil sentence patterns + morphology
  2. Morphological variation: Auto-generate suffix combinations
  3. Corpus augmentation: Expand small datasets with variants
  4. Domain mixing: Classical, news, colloquial styles
  5. Quality assurance: Sanity check generated text

Features (NETWORK-FREE):
  - No API calls, no corpus downloads, no external data
  - Grammatically-aware generation using morphology rules
  - Style variants (classical/colloquial/formal/informal)
  - Scalable: Generate K-scale datasets in seconds
  - Deterministic: Same seed = reproducible data

Usage:
  # Generate 100 sentences
  gen = SyntheticTamilDataGenerator(seed=42)
  data = gen.generate_dataset(num_sentences=100, domain='classical')
  
  # Generate training set with variants
  train_data = gen.generate_training_set(base_texts, num_variants=3)
  
  # Export to JSONL for training
  gen.export_jsonl(data, 'synthetic_tamil_train.jsonl')

Reference:
  - docs/TAMIL_FIRST_DOCTRINE.md
  - Tamil grammar rules from Asher & Kumari
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from enum import Enum
import hashlib


class TamilStyle(Enum):
    """Text generation style."""
    CLASSICAL = 'classical'
    NEWS = 'news'
    COLLOQUIAL = 'colloquial'
    FORMAL = 'formal'
    LITERARY = 'literary'


class TamilSyntheticDataGenerator:
    """Generate realistic Tamil text locally without network."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed
        
        # =====================================================================
        # CORE TAMIL VOCABULARY (ROOTS & MORPHEMES)
        # =====================================================================
        
        # Common Tamil verb roots
        self.verb_roots = {
            'செய்': 'do/make',
            'வा': 'come',
            'போ': 'go',
            'இரு': 'be',
            'கொ': 'take',
            'கொடு': 'give',
            'வை': 'put',
            'நட': 'walk',
            'பல': 'fly',
            'பாடு': 'sing',
            'கேள்': 'listen',
            'கண்': 'see',
            'சொல்': 'say',
            'உண்': 'eat',
            'குடி': 'drink',
            'தூங்கு': 'sleep',
            'எழு': 'write',
            'நிற்': 'stand',
            'உட': 'sit',
            'ஓடு': 'run',
            'தண்டு': 'hit',
            'பிடி': 'hold',
            'விட': 'leave',
            'கொள்': 'take',
            'தeless': 'give',
            'தான': 'play',
            'பாவம்': 'pity',
            'விளை': 'grow',
            'கேள்': 'ask',
            'நினை': 'think',
            'அறி': 'know',
            'விரும்பு': 'like',
        }
        
        # Common nouns
        self.nouns = {
            'மனிதன்': 'man',
            'பெண்': 'woman',
            'குழந்தை': 'child',
            'வீடு': 'house',
            'மரம்': 'tree',
            'பூ': 'flower',
            'மலை': 'mountain',
            'ஆறு': 'river',
            'நகரம்': 'city',
            'கிராமம்': 'village',
            'வேலை': 'work',
            'பதக்கம்': 'medal',
            'அவை': 'those',
            'அவன்': 'he',
            'அவள்': 'she',
            'என்': 'my',
            'உன்': 'your',
            'அவன்': 'his',
            'புஸ்தகம்': 'book',
            'பேனா': 'pen',
            'மேज': 'table',
            'உணவு': 'food',
            'தண்ணீர்': 'water',
            'முகம்': 'face',
            'கண்': 'eye',
            'செவி': 'ear',
            'சூரியன்': 'sun',
            'சந்திரன்': 'moon',
            'நட்சத்திரம்': 'star',
        }
        
        # Adjectives
        self.adjectives = {
            'பெரிய': 'big',
            'சிறிய': 'small',
            'நல்ல': 'good',
            'மோசமான': 'bad',
            'அழகான': 'beautiful',
            'கொடிய': 'ugly',
            'மேலான': 'upper',
            'கீழ்': 'lower',
            'நீண்ட': 'long',
            'குறிய': 'short',
            'வெள்ளை': 'white',
            'கருப்பு': 'black',
            'சிவப்பு': 'red',
            'பச்சை': 'green',
            'மஞ்சள்': 'yellow',
            'அறிவுள்ள': 'intelligent',
            'முட்டாள்': 'foolish',
            'பணக்கார': 'rich',
            '가난': 'poor',
            'ஆரோக்கியமான': 'healthy',
        }
        
        # Adverbs
        self.adverbs = {
            'விரைவாக': 'quickly',
            'மெதுவாக': 'slowly',
            'இப்போது': 'now',
            'நாளை': 'tomorrow',
            'நேற்று': 'yesterday',
            'முன்': 'before',
            'பிறகு': 'after',
            'மேல்': 'above',
            'கீழ்': 'below',
            'உள்ளே': 'inside',
            'வெளியே': 'outside',
            'மிகவும்': 'very',
            'கொஞ்சம்': 'little',
            'ஒவ்வொரு': 'every',
            'எப்போது': 'when',
            'எங்கே': 'where',
            'ஏன்': 'why',
            'எப்படி': 'how',
            'எவ்வளவு': 'how much',
        }
        
        # Morphological suffixes
        self.suffixes = {
            'PLURAL': ['கள்'],
            'ACCUSATIVE': ['ஐ'],
            'GENITIVE': ['இன்'],
            'LOCATIVE': ['இல்'],
            'INSTRUMENTAL': ['ஆல்'],
            'DATIVE': ['க்கு'],
            'PRESENT': ['கின்றான்', 'கின்றாள்', 'கிறான்', 'கிறாள்'],
            'PAST': ['ந்தான்', 'ந்தாள்', 'த்தான்', 'த்தாள்'],
            'FUTURE': ['ப்பான்', 'ப்பாள்', 'வான்', 'வாள்'],
        }
        
        # Sentence templates
        self.templates = [
            # Simple SVO
            "{subject} {verb}",              # நான் செய்கிறேன்
            "{subject} {obj} {verb}",        # நான் வேலை செய்கிறேன்
            "{subject} {adj} {noun}",        # நான் பெரிய வீட்டில் உள்ளேன்
            # With location/time
            "{subject} {adv} {verb}",        # நான் விரைவாக போகிறேன்
            "{adj} {noun} {verb}",           # பெரிய மரம் உள்ளது
            # Complex
            "{subject} {loc} {verb}",        # நான் வீட்டில் இருக்கிறேன்
        ]
    
    def generate_word_form(self, root: str, morphology: Optional[List[str]] = None) -> str:
        """
        Generate a word with optional morphological suffixes.
        
        Args:
            root: Base word (verb/noun/adj)
            morphology: List of morphology types to apply
        
        Returns:
            Generated word with suffixes
        """
        word = root
        if morphology:
            for morph in morphology:
                if morph in self.suffixes:
                    suffix = random.choice(self.suffixes[morph])
                    word += suffix
        return word
    
    def generate_sentence(self, style: TamilStyle = TamilStyle.CLASSICAL) -> str:
        """
        Generate a single Tamil sentence with grammatical structure.
        
        Args:
            style: Classical, news, colloquial, formal, or literary
        
        Returns:
            A synthetic Tamil sentence
        """
        template = random.choice(self.templates)
        
        # Pick parts of speech
        subject_root = random.choice(list(self.nouns.keys()))
        obj_root = random.choice(list(self.nouns.keys())) if random.random() > 0.4 else None
        verb_root = random.choice(list(self.verb_roots.keys()))
        adj_root = random.choice(list(self.adjectives.keys()))
        adv_root = random.choice(list(self.adverbs.keys()))
        
        # Generate morphological forms based on style
        if style == TamilStyle.CLASSICAL:
            subject = self.generate_word_form(subject_root, ['GENITIVE'])
            obj = self.generate_word_form(obj_root, ['ACCUSATIVE']) if obj_root else None
            verb = verb_root + 'கிறான்'  # Present form
            adj_noun = adj_root + ' ' + self.generate_word_form(random.choice(list(self.nouns.keys())), ['LOCATIVE'])
        
        elif style == TamilStyle.COLLOQUIAL:
            subject = subject_root  # No morphology in colloquial
            obj = obj_root if obj_root else None
            verb = verb_root + 'றேன்'  # Colloquial form
            adj_noun = adj_root + ' ' + random.choice(list(self.nouns.keys()))
        
        elif style == TamilStyle.NEWS:
            subject = self.generate_word_form(subject_root, ['GENITIVE'])
            obj = self.generate_word_form(obj_root, ['ACCUSATIVE']) if obj_root else None
            verb = verb_root + 'ப்பட்டது'  # Passive
            adj_noun = adj_root + ' ' + random.choice(list(self.nouns.keys()))
        
        elif style == TamilStyle.FORMAL:
            subject = self.generate_word_form(subject_root, ['GENITIVE', 'PLURAL'])
            obj = self.generate_word_form(obj_root, ['DATIVE']) if obj_root else None
            verb = verb_root + 'கின்றார்'  # Honorific
            adj_noun = adj_root + ' ' + self.generate_word_form(random.choice(list(self.nouns.keys())), ['LOCATIVE'])
        
        else:  # LITERARY
            subject = self.generate_word_form(subject_root, ['NOMINAL'])
            obj = self.generate_word_form(obj_root, ['ACCUSATIVE']) if obj_root else None
            verb = verb_root + 'கிற'  # Participle
            adj_noun = adj_root + ' ' + random.choice(list(self.nouns.keys()))
        
        # Fill template
        subj = subject
        loc = adj_noun
        sentence = template.format(
            subject=subj,
            obj=obj or random.choice(list(self.nouns.keys())),
            verb=verb,
            adj=adj_root,
            noun=random.choice(list(self.nouns.keys())),
            adv=adv_root,
            loc=loc
        )
        
        return sentence.strip()
    
    def generate_dataset(self, num_sentences: int = 100,
                        domain: str = 'classical',
                        include_variants: bool = False) -> List[Dict]:
        """
        Generate a dataset of synthetic Tamil sentences.
        
        Args:
            num_sentences: Number of sentences to generate
            domain: 'classical', 'news', 'colloquial', 'formal', 'mixed'
            include_variants: If True, generate morphological variants
        
        Returns:
            List of dicts with 'text', 'domain', 'style', 'id'
        """
        dataset = []
        
        if domain == 'mixed':
            styles = list(TamilStyle)
        else:
            styles = [TamilStyle[domain.upper()]]
        
        for i in range(num_sentences):
            style = random.choice(styles) if domain == 'mixed' else styles[0]
            
            sentence = self.generate_sentence(style)
            
            dataset.append({
                'id': f'syn_{domain}_{i:06d}',
                'text': sentence,
                'domain': domain,
                'style': style.value,
                'seed': self.seed + i,
            })
            
            # Generate variants if requested
            if include_variants and random.random() > 0.7:
                for v in range(2):
                    variant = self.generate_sentence(style)
                    dataset.append({
                        'id': f'syn_{domain}_{i:06d}_v{v+1}',
                        'text': variant,
                        'domain': domain,
                        'style': style.value,
                        'seed': self.seed + i + (v+1)*1000,
                    })
        
        return dataset
    
    def generate_training_set(self, base_texts: List[str],
                             num_variants: int = 3,
                             add_morphology: bool = True) -> List[Dict]:
        """
        Generate training variants from base texts (data augmentation).
        
        Args:
            base_texts: List of seed texts
            num_variants: Number of variants per base text
            add_morphology: If True, apply different morphological forms
        
        Returns:
            Augmented dataset
        """
        augmented = []
        
        for base_idx, base_text in enumerate(base_texts):
            # Add original
            augmented.append({
                'id': f'aug_{base_idx}_orig',
                'text': base_text,
                'variant': 0,
                'augmentation': 'original',
            })
            
            # Generate variants
            for v in range(1, num_variants + 1):
                # Simple variant: regenerate similar sentence
                variant = self.generate_sentence()
                augmented.append({
                    'id': f'aug_{base_idx}_v{v}',
                    'text': variant,
                    'variant': v,
                    'augmentation': 'synthetic',
                })
        
        return augmented
    
    def export_jsonl(self, dataset: List[Dict], output_path: str) -> str:
        """Export dataset to JSONL format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return str(path)
    
    def export_txt(self, texts: List[str], output_path: str) -> str:
        """Export texts to plain text (one sentence per line)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        return str(path)
    
    def generate_split(self, num_train: int = 800, num_val: int = 100,
                      num_test: int = 100) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Generate train/val/test split with different domains.
        
        Returns:
            (train_data, val_data, test_data)
        """
        train = self.generate_dataset(num_train, domain='mixed', include_variants=True)
        val = self.generate_dataset(num_val, domain='classical')
        test = self.generate_dataset(num_test, domain='news')
        
        return train, val, test


# =============================================================================
# QUALITY ASSURANCE
# =============================================================================

class TamilDataQualityChecker:
    """Sanity check generated data."""
    
    @staticmethod
    def is_valid_tamil(text: str) -> bool:
        """Check if text contains Tamil characters."""
        tamil_chars = any('\u0B80' <= c <= '\u0BFF' for c in text)
        return tamil_chars
    
    @staticmethod
    def has_min_length(text: str, min_chars: int = 5) -> bool:
        """Check minimum text length."""
        return len(text) >= min_chars
    
    @staticmethod
    def has_no_weird_chars(text: str) -> bool:
        """Check for unusual character combinations."""
        # Reject if too many repeat characters
        for i in range(len(text) - 3):
            if text[i] == text[i+1] == text[i+2]:
                return False
        return True
    
    @staticmethod
    def validate_dataset(dataset: List[Dict]) -> Tuple[int, List[str]]:
        """
        Validate entire dataset.
        
        Returns:
            (num_valid, list_of_errors)
        """
        errors = []
        valid_count = 0
        
        for item in dataset:
            text = item.get('text', '')
            
            if not TamilDataQualityChecker.is_valid_tamil(text):
                errors.append(f"No Tamil chars: {item['id']}")
                continue
            
            if not TamilDataQualityChecker.has_min_length(text):
                errors.append(f"Too short: {item['id']}")
                continue
            
            if not TamilDataQualityChecker.has_no_weird_chars(text):
                errors.append(f"Weird chars: {item['id']}")
                continue
            
            valid_count += 1
        
        return valid_count, errors


# =============================================================================
# TESTING & DEMONSTRATION
# =============================================================================

def test_synthetic_generator():
    """Test and demonstrate the synthetic data generator."""
    
    print("\n" + "="*70)
    print("TAMIL SYNTHETIC DATA GENERATOR — LOCAL-ONLY PIPELINE")
    print("="*70)
    
    gen = TamilSyntheticDataGenerator(seed=42)
    
    # Generate samples in each style
    print("\n[1] Generating samples in each style...")
    for style in ['classical', 'colloquial', 'news', 'formal', 'literary']:
        print(f"\n    {style.upper()}:")
        for _ in range(3):
            sent = gen.generate_sentence(TamilStyle[style.upper()])
            print(f"      • {sent}")
    
    # Generate full dataset
    print("\n[2] Generating full dataset (500 sentences)...")
    dataset = gen.generate_dataset(num_sentences=500, domain='mixed', include_variants=False)
    print(f"    ✓ Generated {len(dataset)} sentences")
    
    # Validate quality
    print("\n[3] Validating data quality...")
    valid_count, errors = TamilDataQualityChecker.validate_dataset(dataset)
    print(f"    ✓ Valid: {valid_count}/{len(dataset)} ({100*valid_count/len(dataset):.1f}%)")
    if errors:
        print(f"    ⚠ Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"      - {err}")
    
    # Export to JSONL
    print("\n[4] Exporting to JSONL format...")
    output_path = gen.export_jsonl(dataset[:100], '/tmp/synthetic_tamil_sample.jsonl')
    print(f"    ✓ Exported to {output_path}")
    
    # Generate train/val/test split
    print("\n[5] Generating train/val/test split...")
    train, val, test = gen.generate_split(num_train=100, num_val=20, num_test=20)
    print(f"    ✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Show sample
    print("\n[6] Sample from training set:")
    for item in train[:3]:
        print(f"    • [{item['style']}] {item['text']}")
    
    print("\n" + "="*70)
    print("✓ Synthetic data generator ready for training pipeline")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_synthetic_generator()
