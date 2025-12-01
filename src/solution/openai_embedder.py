"""
OpenAI Embeddings Module
========================

Cung cấp embeddings sử dụng OpenAI API thay vì SentenceTransformers.
Hỗ trợ:
- text-embedding-3-small (1536 dims, chi phí thấp)
- text-embedding-3-large (3072 dims, chất lượng cao)
- text-embedding-ada-002 (1536 dims, legacy)

Features:
- Caching để tránh gọi API lặp lại
- Batch processing để tối ưu API calls
- Fallback về SentenceTransformers nếu lỗi
"""

from __future__ import annotations
import os
import hashlib
import json
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Run: pip install openai")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class OpenAIEmbedder:
    """
    OpenAI-based embedder với caching và batch processing.
    
    Usage:
        embedder = OpenAIEmbedder(model='text-embedding-3-small')
        embeddings = embedder.encode(['text1', 'text2'])
    """
    
    # Model configurations
    MODEL_CONFIGS = {
        'text-embedding-3-small': {
            'dimensions': 1536,
            'max_tokens': 8191,
            'cost_per_1k': 0.00002  # USD
        },
        'text-embedding-3-large': {
            'dimensions': 3072,
            'max_tokens': 8191,
            'cost_per_1k': 0.00013
        },
        'text-embedding-ada-002': {
            'dimensions': 1536,
            'max_tokens': 8191,
            'cost_per_1k': 0.0001
        }
    }
    
    def __init__(
        self,
        model: str = 'text-embedding-3-small',
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        fallback_to_sentence_transformers: bool = True,
        fallback_model: str = 'all-MiniLM-L6-v2',
        batch_size: int = 100,
        dimensions: Optional[int] = None  # For dimension reduction với text-embedding-3-*
    ):
        """
        Initialize OpenAI Embedder.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (default: từ env OPENAI_API_KEY)
            cache_dir: Directory để cache embeddings
            use_cache: Có sử dụng cache không
            fallback_to_sentence_transformers: Fallback nếu OpenAI fail
            fallback_model: SentenceTransformer model cho fallback
            batch_size: Số texts mỗi batch API call
            dimensions: Số chiều output (chỉ cho text-embedding-3-*)
        """
        self.model = model
        self.use_cache = use_cache
        self.batch_size = batch_size
        self.dimensions = dimensions
        
        # API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if self.api_key:
            # Clean up API key (remove spaces)
            self.api_key = self.api_key.strip().replace(' ', '')
        
        # Initialize OpenAI client
        self.client = None
        if OPENAI_AVAILABLE and self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                print(f"OpenAI Embedder initialized with model: {model}")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI client: {e}")
        
        # Get model dimensions
        if model in self.MODEL_CONFIGS:
            self._native_dimensions = self.MODEL_CONFIGS[model]['dimensions']
        else:
            self._native_dimensions = 1536
        
        # Actual output dimensions
        if dimensions and model.startswith('text-embedding-3'):
            self._output_dimensions = min(dimensions, self._native_dimensions)
        else:
            self._output_dimensions = self._native_dimensions
        
        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else Path('/home/ubuntu/crawl/crawler-recommend-sys/data/embedding_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, np.ndarray] = {}
        self._load_cache()
        
        # Fallback
        self.fallback_to_sentence_transformers = fallback_to_sentence_transformers
        self.fallback_model_name = fallback_model
        self.fallback_model = None
        
        if fallback_to_sentence_transformers and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.fallback_model = SentenceTransformer(fallback_model)
                print(f"Fallback model loaded: {fallback_model}")
            except Exception as e:
                print(f"Warning: Could not load fallback model: {e}")
        
        # Statistics
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'fallback_calls': 0,
            'tokens_used': 0
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.model}_{self._output_dimensions}_{text_hash}"
    
    def _load_cache(self):
        """Load cache from disk."""
        cache_file = self.cache_dir / f'embeddings_cache_{self.model.replace("-", "_")}.npz'
        cache_keys_file = self.cache_dir / f'embeddings_keys_{self.model.replace("-", "_")}.json'
        
        if cache_file.exists() and cache_keys_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                with open(cache_keys_file, 'r') as f:
                    keys = json.load(f)
                
                for key, emb in zip(keys, data['embeddings']):
                    self.cache[key] = emb
                
                print(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk."""
        if not self.cache:
            return
        
        cache_file = self.cache_dir / f'embeddings_cache_{self.model.replace("-", "_")}.npz'
        cache_keys_file = self.cache_dir / f'embeddings_keys_{self.model.replace("-", "_")}.json'
        
        try:
            keys = list(self.cache.keys())
            embeddings = np.array([self.cache[k] for k in keys])
            
            np.savez_compressed(cache_file, embeddings=embeddings)
            with open(cache_keys_file, 'w') as f:
                json.dump(keys, f)
            
            print(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _call_openai_api(self, texts: List[str]) -> List[np.ndarray]:
        """Call OpenAI API to get embeddings."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        # Truncate long texts
        processed_texts = []
        for text in texts:
            # Rough estimate: 4 chars per token
            max_chars = self.MODEL_CONFIGS.get(self.model, {}).get('max_tokens', 8191) * 4
            if len(text) > max_chars:
                text = text[:max_chars]
            processed_texts.append(text)
        
        try:
            # Build request parameters
            params = {
                'model': self.model,
                'input': processed_texts
            }
            
            # Add dimensions parameter for text-embedding-3-* models
            if self.dimensions and self.model.startswith('text-embedding-3'):
                params['dimensions'] = self._output_dimensions
            
            response = self.client.embeddings.create(**params)
            
            embeddings = []
            for item in response.data:
                embeddings.append(np.array(item.embedding, dtype=np.float32))
            
            # Update stats
            self.stats['api_calls'] += 1
            self.stats['tokens_used'] += response.usage.total_tokens
            
            return embeddings
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    def _use_fallback(self, texts: List[str]) -> np.ndarray:
        """Use fallback SentenceTransformer model."""
        if self.fallback_model is None:
            raise ValueError("No fallback model available")
        
        self.stats['fallback_calls'] += 1
        
        embeddings = self.fallback_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        return embeddings
    
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Compatible với SentenceTransformer interface.
        
        Args:
            texts: Single text or list of texts
            convert_to_numpy: Return numpy array (always True for OpenAI)
            show_progress_bar: Show progress (ignored for OpenAI)
            batch_size: Override default batch size
        
        Returns:
            np.ndarray of shape (n_texts, embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.batch_size
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            
            if self.use_cache and cache_key in self.cache:
                self.stats['cache_hits'] += 1
                embeddings.append((i, self.cache[cache_key]))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            try:
                if self.client:
                    # Use OpenAI API
                    for start in range(0, len(texts_to_embed), batch_size):
                        end = min(start + batch_size, len(texts_to_embed))
                        batch_texts = texts_to_embed[start:end]
                        
                        batch_embeddings = self._call_openai_api(batch_texts)
                        
                        for j, emb in enumerate(batch_embeddings):
                            idx = text_indices[start + j]
                            text = texts_to_embed[start + j]
                            
                            # Cache the embedding
                            if self.use_cache:
                                cache_key = self._get_cache_key(text)
                                self.cache[cache_key] = emb
                            
                            embeddings.append((idx, emb))
                else:
                    # Use fallback
                    fallback_embeddings = self._use_fallback(texts_to_embed)
                    for j, emb in enumerate(fallback_embeddings):
                        idx = text_indices[j]
                        embeddings.append((idx, emb))
                        
            except Exception as e:
                print(f"Error getting embeddings: {e}")
                
                # Try fallback
                if self.fallback_to_sentence_transformers and self.fallback_model:
                    print("Falling back to SentenceTransformers...")
                    fallback_embeddings = self._use_fallback(texts_to_embed)
                    for j, emb in enumerate(fallback_embeddings):
                        idx = text_indices[j]
                        embeddings.append((idx, emb))
                else:
                    raise
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings], dtype=np.float32)
        
        return result
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension. Compatible với SentenceTransformer."""
        return self._output_dimensions
    
    def save_cache(self):
        """Manually save cache to disk."""
        self._save_cache()
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear in-memory cache."""
        self.cache = {}
    
    def __del__(self):
        """Save cache on destruction."""
        if self.use_cache and self.cache:
            self._save_cache()


class HybridEmbedder:
    """
    Hybrid embedder cho phép chuyển đổi giữa OpenAI và SentenceTransformers.
    
    Usage:
        embedder = HybridEmbedder(use_openai=True)
        # hoặc
        embedder = HybridEmbedder(use_openai=False)
    """
    
    def __init__(
        self,
        use_openai: bool = True,
        openai_model: str = 'text-embedding-3-small',
        sentence_model: str = 'all-MiniLM-L6-v2',
        openai_dimensions: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize hybrid embedder.
        
        Args:
            use_openai: True để dùng OpenAI, False để dùng SentenceTransformers
            openai_model: OpenAI model name
            sentence_model: SentenceTransformer model name
            openai_dimensions: Output dimensions cho OpenAI (chỉ text-embedding-3-*)
        """
        self.use_openai = use_openai
        
        if use_openai:
            self.embedder = OpenAIEmbedder(
                model=openai_model,
                dimensions=openai_dimensions,
                fallback_model=sentence_model,
                **kwargs
            )
            self._dimensions = self.embedder.get_sentence_embedding_dimension()
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers not installed")
            
            self.embedder = SentenceTransformer(sentence_model)
            self._dimensions = self.embedder.get_sentence_embedding_dimension()
        
        print(f"HybridEmbedder initialized: {'OpenAI' if use_openai else 'SentenceTransformers'}")
        print(f"  Model: {openai_model if use_openai else sentence_model}")
        print(f"  Dimensions: {self._dimensions}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.embedder.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size
        )
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimensions
    
    def to(self, device: str) -> 'HybridEmbedder':
        """Move to device (only for SentenceTransformers)."""
        if not self.use_openai and hasattr(self.embedder, 'to'):
            self.embedder.to(device)
        return self


# Utility function để dễ dàng tạo embedder
def get_embedder(
    use_openai: bool = True,
    openai_model: str = 'text-embedding-3-small',
    sentence_model: str = 'all-MiniLM-L6-v2',
    **kwargs
) -> Union[OpenAIEmbedder, 'SentenceTransformer']:
    """
    Get embedder instance.
    
    Args:
        use_openai: True cho OpenAI, False cho SentenceTransformers
        openai_model: OpenAI model
        sentence_model: SentenceTransformer model
    
    Returns:
        Embedder instance
    """
    return HybridEmbedder(
        use_openai=use_openai,
        openai_model=openai_model,
        sentence_model=sentence_model,
        **kwargs
    )


if __name__ == '__main__':
    # Test
    print("Testing OpenAI Embedder...")
    
    embedder = OpenAIEmbedder(model='text-embedding-3-small')
    
    test_texts = [
        "Software development company specializing in web applications",
        "Mobile app development services for startups",
        "Enterprise software solutions"
    ]
    
    embeddings = embedder.encode(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Stats: {embedder.get_stats()}")
    
    # Save cache
    embedder.save_cache()
