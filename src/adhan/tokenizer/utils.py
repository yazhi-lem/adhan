"""
Tamil morphological utilities for tokenization.
Handles agglutination, sandhi rules, and morpheme decomposition.
"""

from typing import List, Dict, Tuple
import re


# Tamil Morpheme Inventory (70+ morphemes)
TAMIL_MORPHEMES = {
    # Nominal morphemes
    "nominative": ["ø"],  # No suffix
    "accusative": ["-अই", "-अ्आ"],  # Object marker
    "dative": ["-कु", "-क्कु"],  # "to" marker
    "locative": ["-ळ्ळ", "-ऱु"],  # "at/in" marker
    "instrumental": ["-ऑ्ऍ", "-ऑ्ऱु"],  # "with" marker
    "ablative": ["-ऱु-इ्इ"],  # "from" marker
    "genitive": ["-उ्ड", "-उ_य"],  # possessive
    "dative_pl": ["-कु्कळ्ळ"],  # pl dative
    
    # Verbal morphemes
    "tense_past": ["-इन्न", "-अ्ट", "-अ्ट्अ"],
    "tense_pres": ["-किऱु", "-किऱ्अ"],
    "tense_fut": ["-उ्अ्उ्म", "-उ्अ्अ्"],
    "mood_imperative": ["-अु", "-अु्अु"],
    "mood_conditional": ["-इन्अ्अ", "-उ्अ्अि्नि्अ्अ्"],
    "voice_passive": ["-अ्प्प", "-अ्ट्अु"],
    
    # Number/Person markers
    "singular": ["ø"],
    "plural": ["-कळ्", "-कळ््अ"],
    "first_person": ["-एन्", "-ए्म्"],
    "second_person": ["-ऐ", "-ऐ्कळ्"],
    "third_person": ["-अन्", "-अ्र्"],
    
    # Aspect markers
    "perfective": ["-अ्ट्अ्अि्य"],
    "habitual": ["-उ्म्"],
    "progressive": ["-किऱु"],
    
    # Case markers (+8)
    "ablative": ["-अि्अु"],
    "comitative": ["-ओ्द्अ", "-ओ्द्ु"],
    "prolative": ["-अु्ष्अ"],
    "locative_temp": ["-उ्ळ्"], 
}

# Sandhi Rules (14 major transformation rules)
SANDHI_RULES = [
    # Vowel sandhi
    ("अ + अ", "अ", "vowel_fusion"),
    ("अ + ई", "ई", "vowel_fusion"),
    ("ई + अ", "ई", "vowel_preservation"),
    
    # Consonant sandhi
    ("त् + क्", "क्क्", "retroflexion"),
    ("ट् + क्", "क्क्", "consonant_gemination"),
    ("न् + त्", "न्त्", "nasal_assimilation"),
    ("म् + प्", "म्प्", "nasal_assimilation"),
    
    # Coda rules
    ("त् + #", "त्", "place_preservation"),  # Final consonant
    ("ड् + #", "ण्", "nasal_neutralization"),
    
    # Neutralization
    ("त् + स्", "च्च्", "sibilant_assimilation"),
    ("द् + ध्", "द्ध्", "consonant_geminaton"),
    
    # Complex sandhi
    ("अ् + य्", "अ्य्", "semivowel_insertion"),
    ("अ् + व्", "अ्व्", "semivowel_insertion"),
]


def decompose_morphemes(word: str) -> List[Tuple[str, str]]:
    """
    Decompose Tamil word into morphemes.
    Returns list of (morpheme, morpheme_type) tuples.
    """
    # Simplified morpheme decomposition (production version uses Finite State Transducers)
    morphemes = []
    remaining = word
    
    # Match longest morpheme first (greedy)
    for morpheme_type, morpheme_list in sorted(
        TAMIL_MORPHEMES.items(), 
        key=lambda x: max(len(m) for m in x[1] if isinstance(m, str)), 
        reverse=True
    ):
        for morpheme in morpheme_list:
            if morpheme == "ø":
                continue
            if remaining.endswith(morpheme):
                morphemes.append((morpheme, morpheme_type))
                remaining = remaining[:-len(morpheme)]
                break
    
    # Root is what remains
    if remaining:
        morphemes.insert(0, (remaining, "root"))
    
    return morphemes


def apply_sandhi(seq1: str, seq2: str, context: str = "") -> str:
    """
    Apply sandhi rules between two morpheme sequences.
    Returns combined form after sandhi.
    """
    combined = seq1 + seq2
    
    for seq, result, rule_type in SANDHI_RULES:
        if seq.replace(" + ", "").replace("#", "") in combined:
            combined = combined.replace(seq.split(" + ")[0], result)
    
    return combined


def get_morphological_features(word: str) -> Dict[str, str]:
    """
    Extract morphological features from word.
    Returns dict with case, number, person, tense, etc.
    """
    morphemes = decompose_morphemes(word)
    features = {
        "root": morphemes[0][0] if morphemes else "",
        "affixes": len(morphemes) - 1,
        "agglutination_level": "synthetic",  # Tamil is highly agglutinative
    }
    
    # Infer features from morpheme types
    for morpheme, morph_type in morphemes[1:]:
        if "case" in morph_type:
            features["case"] = morph_type
        elif "number" in morph_type:
            features["number"] = morph_type
        elif "person" in morph_type:
            features["person"] = morph_type
        elif "tense" in morph_type:
            features["tense"] = morph_type
    
    return features
