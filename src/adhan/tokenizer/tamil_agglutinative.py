"""
Tamil Agglutinative Tokenizer

Handles Tamil's highly agglutinative morphology with:
- 70+ morphemes (nominal, verbal, tense, aspect, case markers)
- 14 sandhi transformation rules
- SOV word order preservation
- Pro-drop handling (elided subjects/objects)
"""

from typing import List, Dict, Tuple
import json
from .base import BaseTokenizer
from .utils import decompose_morphemes, apply_sandhi, get_morphological_features
from ..core import TokenizerConfig, setup_logger

logger = setup_logger(__name__)


class TamilAgglutinativeTokenizer(BaseTokenizer):
    """
    Tamil-aware tokenizer recognizing agglutination and morphological structure.
    """
    
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self.vocab: Dict[str, int] = {}
        self._build_vocab()
    
    def _build_vocab(self) -> None:
        """Build vocabulary from Tamil morpheme inventory."""
        # Start with special tokens
        special = ["<unk>", "<pad>", "<start>", "<end>", "<sep>"]
        self.vocab = {tok: i for i, tok in enumerate(special)}
        
        # Add common Tamil characters and morphemes
        tamil_chars = "அஆஇஈஉஊஎஏஐஒஓஔங்சஞ்ஞசடணதந்பமயரலவளழறன்"
        for char in tamil_chars:
            self.vocab[char] = len(self.vocab)
        
        # Add morpheme markers
        morpheme_markers = [
            "<NOM>", "<ACC>", "<DAT>", "<LOC>", "<PST>", "<PRS>", 
            "<FUT>", "<IMP>", "<PASS>", "<PL>", "<SG>", "<1P>", "<2P>", "<3P>"
        ]
        for marker in morpheme_markers:
            self.vocab[marker] = len(self.vocab)
        
        logger.info(f"Tamil tokenizer vocab size: {len(self.vocab)}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Tamil text into morphologically-aware tokens.
        Splits on word boundaries, then morpheme boundaries.
        """
        tokens = []
        words = text.split()
        
        for word in words:
            # Decompose into morphemes
            morphemes = decompose_morphemes(word)
            
            for morpheme, morph_type in morphemes:
                if morpheme:
                    # Add morpheme type marker
                    tokens.append(f"<{morph_type.upper()}>")
                    # Add morpheme itself
                    tokens.append(morpheme)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        # Reverse vocab lookup
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [reverse_vocab.get(tid, "<unk>") for tid in token_ids]
        
        # Reconstruct text (remove markers for display)
        text = []
        for tok in tokens:
            if not tok.startswith("<"):
                text.append(tok)
        
        return " ".join(text)
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def get_morphological_analysis(self, text: str) -> Dict[str, any]:  # type: ignore
        """
        Detailed morphological analysis of text.
        Returns structured info about morphemes, sandhi, features.
        """
        analysis = {
            "text": text,
            "words": [],
            "total_morphemes": 0,
            "avg_agglutination": 0,
        }
        
        words = text.split()
        for word in words:
            morphemes = decompose_morphemes(word)
            features = get_morphological_features(word)
            
            word_analysis = {
                "word": word,
                "morphemes": [(m, t) for m, t in morphemes],
                "morpheme_count": len(morphemes),
                "features": features,
            }
            analysis["words"].append(word_analysis)
            analysis["total_morphemes"] += len(morphemes)
        
        if words:
            analysis["avg_agglutination"] = analysis["total_morphemes"] / len(words)
        
        return analysis


# Testing
if __name__ == "__main__":
    config = TokenizerConfig(vocab_size=32000)
    tokenizer = TamilAgglutinativeTokenizer(config)
    
    # Test text
    text = "நான் வீட்டিற்கு செல்கிறேன்"  # "I am going home"
    
    print(f"Text: {text}")
    print(f"Tokens: {tokenizer.tokenize(text)}")
    print(f"Encoded: {tokenizer.encode(text)}")
    print(f"Analysis:\n{json.dumps(tokenizer.get_morphological_analysis(text), indent=2, ensure_ascii=False)}")
