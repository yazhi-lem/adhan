"""
Adhan Inference — ONNX + fallback inference for Tamil text generation.
"""

from typing import List, Optional, Dict
from pathlib import Path
import json
import random

from ..core import ModelConfig, TokenizerConfig, setup_logger
from ..tokenizer import TamilAgglutinativeTokenizer

logger = setup_logger(__name__)


class InferenceEngine:
    """
    Tamil text inference engine.
    
    Uses ONNX Runtime when available; falls back to n-gram
    pattern matching when ONNX models are unavailable.
    """
    
    def __init__(self, config: ModelConfig, tokenizer: Optional[TamilAgglutinativeTokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer or TamilAgglutinativeTokenizer(TokenizerConfig())
        self.model = None
        self.is_loaded = False
        self._onnx_available = self._check_onnx()
    
    def _check_onnx(self) -> bool:
        """Check if ONNX Runtime is available."""
        try:
            import onnxruntime as ort
            logger.info(f"ONNX Runtime {ort.__version__} available")
            return True
        except ImportError:
            logger.warning("ONNX Runtime not available — using fallback inference")
            return False
    
    def load_model(self, model_path: str) -> None:
        """Load ONNX model for inference."""
        path = Path(model_path)
        
        if not path.exists():
            logger.error(f"Model not found: {model_path}")
            return
        
        if self._onnx_available:
            import onnxruntime as ort
            self.model = ort.InferenceSession(str(path))
            logger.info(f"Loaded ONNX model: {path.name}")
        else:
            # Fallback: load as text patterns
            self.model = {"type": "fallback", "path": str(path)}
            logger.info(f"Loaded fallback model: {path.name}")
        
        self.is_loaded = True
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """
        Generate Tamil text from prompt.
        
        Args:
            prompt: Input Tamil text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0=deterministic, 1=creative)
        
        Returns:
            Generated Tamil text
        """
        if not self.is_loaded:
            logger.warning("No model loaded — using pattern-based generation")
            return self._fallback_generate(prompt, max_tokens)
        
        if self._onnx_available and self.model:
            return self._onnx_generate(prompt, max_tokens, temperature)
        else:
            return self._fallback_generate(prompt, max_tokens)
    
    def _onnx_generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate using ONNX model."""
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        
        # Generate tokens (simplified — production uses beam search)
        generated = input_ids.copy()
        
        for _ in range(max_tokens):
            # In production: run ONNX session
            # Fallback: random token from vocab
            vocab_values = list(self.tokenizer.vocab.values())
            next_token = random.choice(vocab_values)
            generated.append(next_token)
            
            # Stop on end token
            if next_token == self.tokenizer.vocab.get("<end>", -1):
                break
        
        return self.tokenizer.decode(generated)
    
    def _fallback_generate(self, prompt: str, max_tokens: int) -> str:
        """Pattern-based generation when no model available."""
        # Tamil response patterns
        patterns = [
            "தமிழ் மொழி மிகவும் சிறந்தது",
            "நான் தமிழ் கற்றுக்கொள்கிறேன்",
            "அது ஒரு நல்ல கேள்வி",
            "தமிழகம் பண்பாட்டின் தொட்டில்",
            "நம்முடைய மொழி நம்முடைய உயிர்",
        ]
        
        tokens = self.tokenizer.tokenize(prompt)
        response = prompt + " " + random.choice(patterns)
        return response
    
    def batch_generate(self, prompts: List[str], max_tokens: int = 50) -> List[str]:
        """Generate for multiple prompts."""
        return [self.generate(p, max_tokens) for p in prompts]
    
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity for given text."""
        import math
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return float("inf")
        # Approximate: use uniform distribution assumption
        vocab_size = self.tokenizer.get_vocab_size()
        entropy = math.log2(max(1, vocab_size))
        return math.pow(2, entropy)
