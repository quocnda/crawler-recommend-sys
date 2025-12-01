"""
Base Recommender Module
=======================

Định nghĩa abstract base class cho tất cả recommenders.

Tất cả recommenders phải implement interface này để đảm bảo consistency.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Literal, Union
import pandas as pd
import numpy as np

# Import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.types import (
        UserID,
        TripletID,
        Score,
        RecommendationResult,
        UserRecommendations,
    )
    from models.config import RecommenderConfig, EmbeddingConfig
else:
    try:
        from models.types import (
            UserID,
            TripletID,
            Score,
            RecommendationResult,
            UserRecommendations,
        )
        from models.config import RecommenderConfig, EmbeddingConfig
    except ImportError:
        UserID = str
        TripletID = str
        Score = float
        RecommendationResult = None
        UserRecommendations = None
        # Define placeholder classes for type hints
        @dataclass
        class RecommenderConfig:
            pass
        
        @dataclass
        class EmbeddingConfig:
            pass


# =============================================================================
# Base Recommender Interface
# =============================================================================

class BaseRecommender(ABC):
    """
    Abstract base class for all recommenders.
    
    All recommendation algorithms must inherit from this class and implement
    the required methods.
    
    Attributes:
        name: Human-readable name of the recommender
        is_fitted: Whether the recommender has been fitted
        config: Optional configuration object
    
    Required methods to implement:
        - fit(): Train the recommender on data
        - recommend_triplets(): Generate recommendations for a user
    
    Example:
        >>> class MyRecommender(BaseRecommender):
        ...     def fit(self, df_train, **kwargs):
        ...         # Training logic
        ...         self._is_fitted = True
        ...         return self
        ...     
        ...     def recommend_triplets(self, user_id, top_k=10, **kwargs):
        ...         # Recommendation logic
        ...         return pd.DataFrame([...])
    """
    
    def __init__(
        self,
        name: str = "BaseRecommender",
        config: Optional[RecommenderConfig] = None
    ):
        """
        Initialize base recommender.
        
        Args:
            name: Recommender name
            config: Optional configuration
        """
        self.name = name
        self.config = config
        self._is_fitted = False
    
    @property
    def is_fitted(self) -> bool:
        """Check if recommender has been fitted."""
        return self._is_fitted
    
    def _check_fitted(self):
        """Raise error if recommender not fitted."""
        if not self._is_fitted:
            raise ValueError(
                f"{self.name} has not been fitted. "
                "Call fit() before making recommendations."
            )
    
    @abstractmethod
    def fit(
        self,
        df_train: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> 'BaseRecommender':
        """
        Fit the recommender on training data.
        
        Args:
            df_train: Training DataFrame with triplet column
            df_test: Optional test DataFrame for candidate generation
            **kwargs: Additional fitting parameters
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If required columns are missing
        """
        pass
    
    @abstractmethod
    def recommend_triplets(
        self,
        user_id: UserID,
        top_k: int = 10,
        mode: Literal['train', 'val', 'test'] = 'test',
        exclude_seen: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate triplet recommendations for a user.
        
        Args:
            user_id: User identifier
            top_k: Number of recommendations to return
            mode: Which data to use for candidates ('train', 'val', 'test')
            exclude_seen: Whether to exclude triplets the user has seen
            **kwargs: Additional parameters
        
        Returns:
            DataFrame with columns:
                - triplet: Recommended triplet ID
                - score: Confidence score [0, 1]
                - Additional columns as needed (industry, services, etc.)
        
        Raises:
            ValueError: If recommender not fitted
        """
        pass
    
    def recommend_for_users(
        self,
        user_ids: List[UserID],
        top_k: int = 10,
        mode: Literal['train', 'val', 'test'] = 'test',
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user identifiers
            top_k: Number of recommendations per user
            mode: Which data to use for candidates
            **kwargs: Additional parameters
        
        Returns:
            DataFrame with columns [linkedin_company_outsource, triplet, score]
        """
        self._check_fitted()
        
        all_results = []
        
        for user_id in user_ids:
            try:
                recs = self.recommend_triplets(
                    user_id=user_id,
                    top_k=top_k,
                    mode=mode,
                    **kwargs
                )
                
                if not recs.empty:
                    recs['linkedin_company_outsource'] = user_id
                    all_results.append(recs)
            except Exception as e:
                # Log error but continue with other users
                print(f"Error generating recommendations for user {user_id}: {e}")
                continue
        
        if not all_results:
            return pd.DataFrame(columns=['linkedin_company_outsource', 'triplet', 'score'])
        
        return pd.concat(all_results, ignore_index=True)
    
    def get_info(self) -> Dict:
        """
        Get information about the recommender.
        
        Returns:
            Dictionary with recommender information
        """
        return {
            'name': self.name,
            'is_fitted': self._is_fitted,
            'config': self.config.__dict__ if self.config else None
        }


# =============================================================================
# Content-Based Recommender Base
# =============================================================================

class ContentBasedRecommender(BaseRecommender):
    """
    Base class for content-based recommenders.
    
    Content-based recommenders use item features to make recommendations.
    They build user profiles from historical interactions and match
    against candidate items.
    
    Subclasses should implement:
        - _build_features(): Build feature representations for items
        - build_user_profile(): Build user profile from history
    """
    
    def __init__(
        self,
        name: str = "ContentBasedRecommender",
        config: Optional[RecommenderConfig] = None
    ):
        super().__init__(name=name, config=config)
        
        # Feature matrices
        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        
        # Data references
        self.df_train: Optional[pd.DataFrame] = None
        self.df_val: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
    
    @abstractmethod
    def _build_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Build feature matrix for data.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit any transformers (True for training data)
        
        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def build_user_profile(self, user_id: UserID) -> Optional[np.ndarray]:
        """
        Build user profile vector from historical interactions.
        
        Args:
            user_id: User identifier
        
        Returns:
            User profile vector or None if no history
        """
        pass


# =============================================================================
# Collaborative Filtering Recommender Base
# =============================================================================

class CollaborativeRecommender(BaseRecommender):
    """
    Base class for collaborative filtering recommenders.
    
    Collaborative recommenders use user-user or item-item similarity
    to make recommendations based on interaction patterns.
    
    Subclasses should implement:
        - _build_user_features(): Build user feature representations
        - find_similar_users(): Find similar users to a target user
    """
    
    def __init__(
        self,
        name: str = "CollaborativeRecommender",
        config: Optional[RecommenderConfig] = None,
        min_similarity: float = 0.1,
        top_k_similar_users: int = 20
    ):
        super().__init__(name=name, config=config)
        
        self.min_similarity = min_similarity
        self.top_k_similar_users = top_k_similar_users
        
        # User data
        self.user_features: Dict[UserID, np.ndarray] = {}
        self.user_triplets: Dict[UserID, Set[TripletID]] = {}
        self.triplet_popularity: Dict[TripletID, int] = {}
    
    @abstractmethod
    def find_similar_users(
        self,
        user_id: UserID,
        k: Optional[int] = None
    ) -> List[tuple]:
        """
        Find k most similar users to target user.
        
        Args:
            user_id: Target user identifier
            k: Number of similar users (default: top_k_similar_users)
        
        Returns:
            List of (user_id, similarity_score) tuples, sorted by similarity
        """
        pass
    
    def _recommend_by_popularity(
        self,
        top_k: int,
        exclude: Optional[Set[TripletID]] = None
    ) -> pd.DataFrame:
        """
        Fallback: recommend most popular triplets.
        
        Args:
            top_k: Number of recommendations
            exclude: Set of triplets to exclude
        
        Returns:
            DataFrame with [triplet, score] columns
        """
        exclude = exclude or set()
        
        sorted_triplets = sorted(
            [(t, c) for t, c in self.triplet_popularity.items() if t not in exclude],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        if not sorted_triplets:
            return pd.DataFrame(columns=['triplet', 'score'])
        
        max_count = sorted_triplets[0][1] if sorted_triplets else 1
        
        return pd.DataFrame([
            {'triplet': triplet, 'score': count / max_count}
            for triplet, count in sorted_triplets
        ])


# =============================================================================
# Ensemble Recommender Base
# =============================================================================

class EnsembleRecommender(BaseRecommender):
    """
    Base class for ensemble recommenders.
    
    Ensemble recommenders combine multiple base recommenders
    to produce final recommendations.
    
    Subclasses should implement:
        - _fit_base_models(): Fit all base models
        - _get_base_predictions(): Get predictions from all base models
        - _combine_predictions(): Combine predictions into final scores
    """
    
    def __init__(
        self,
        name: str = "EnsembleRecommender",
        config: Optional[RecommenderConfig] = None,
        model_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__(name=name, config=config)
        
        # Default model weights
        self.model_weights = model_weights or {
            'content': 0.3,
            'enhanced_content': 0.3,
            'user_collab': 0.2,
            'enhanced_collab': 0.2
        }
        
        # Base models
        self.base_models: Dict[str, BaseRecommender] = {}
    
    @abstractmethod
    def _fit_base_models(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame
    ):
        """
        Fit all base recommendation models.
        
        Args:
            df_train: Training data
            df_test: Test/candidate data
        """
        pass
    
    @abstractmethod
    def _get_base_predictions(
        self,
        user_id: UserID,
        top_k: int,
        mode: str
    ) -> Dict[str, Dict[TripletID, Score]]:
        """
        Get predictions from all base models.
        
        Args:
            user_id: User identifier
            top_k: Number of predictions per model
            mode: Recommendation mode ('val' or 'test')
        
        Returns:
            Dictionary mapping model name to {triplet: score} dict
        """
        pass
    
    @abstractmethod
    def _combine_predictions(
        self,
        base_predictions: Dict[str, Dict[TripletID, Score]],
        user_id: UserID
    ) -> Dict[TripletID, Score]:
        """
        Combine predictions from base models into final scores.
        
        Args:
            base_predictions: Predictions from each base model
            user_id: User identifier (for context features)
        
        Returns:
            Dictionary mapping triplet to final combined score
        """
        pass


# =============================================================================
# Utility Mixins
# =============================================================================

class EmbeddingMixin:
    """
    Mixin for recommenders that use embeddings.
    
    Provides common embedding functionality including:
    - Embedding model initialization
    - Text encoding
    - Embedding caching
    """
    
    def _init_embedding_model(
        self,
        embedding_config: Optional[EmbeddingConfig] = None,
        use_openai: bool = True,
        openai_model: str = 'text-embedding-3-small',
        sentence_model: str = 'all-MiniLM-L6-v2'
    ):
        """
        Initialize embedding model.
        
        Args:
            embedding_config: Optional embedding configuration
            use_openai: Whether to use OpenAI embeddings
            openai_model: OpenAI model name
            sentence_model: SentenceTransformer model name
        """
        self.embedding_model = None
        self.embedding_dim = 0
        
        # Try OpenAI first
        if use_openai:
            try:
                from solution.openai_embedder import HybridEmbedder
                self.embedding_model = HybridEmbedder(
                    use_openai=True,
                    openai_model=openai_model,
                    sentence_model=sentence_model
                )
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                return
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embedder: {e}")
        
        # Fallback to SentenceTransformers
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(sentence_model)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            print(f"Warning: Could not initialize SentenceTransformer: {e}")
            self.embedding_dim = 384  # Default
    
    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        
        Returns:
            Embedding matrix of shape (n_texts, embedding_dim)
        """
        if self.embedding_model is None:
            return np.zeros((len(texts), self.embedding_dim))
        
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=batch_size
        )


class CachingMixin:
    """
    Mixin for recommenders that cache computations.
    
    Provides caching functionality for expensive operations.
    """
    
    def _init_cache(self):
        """Initialize cache storage."""
        self._cache: Dict[str, any] = {}
    
    def _cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments."""
        key_parts = [prefix] + [str(arg) for arg in args]
        return "_".join(key_parts)
    
    def _get_cached(self, key: str) -> Optional[any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def _set_cached(self, key: str, value: any):
        """Set value in cache."""
        self._cache[key] = value
    
    def clear_cache(self):
        """Clear all cached values."""
        self._cache = {}
