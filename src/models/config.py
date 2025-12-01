"""
Configuration Classes
=====================

Định nghĩa các configuration classes cho recommendation system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os

from .types import EmbeddingType, RecommenderMode


# =============================================================================
# Embedding Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding models.
    
    Attributes:
        embedding_type: Type of embedding (OpenAI or SentenceTransformers)
        openai_model: OpenAI model name
        sentence_model: SentenceTransformer model name
        embedding_dim: Output embedding dimension
        use_cache: Whether to cache embeddings
        cache_dir: Directory for embedding cache
        batch_size: Batch size for encoding
    """
    embedding_type: EmbeddingType = EmbeddingType.OPENAI
    openai_model: str = "text-embedding-3-small"
    sentence_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    use_cache: bool = True
    cache_dir: Optional[Path] = None
    batch_size: int = 32
    
    def __post_init__(self):
        """Set default cache directory."""
        if self.cache_dir is None:
            self.cache_dir = Path(__file__).parent.parent.parent / "data" / "embedding_cache"
    
    @property
    def use_openai(self) -> bool:
        """Check if using OpenAI embeddings."""
        return self.embedding_type == EmbeddingType.OPENAI
    
    def get_model_name(self) -> str:
        """Get the active model name."""
        if self.use_openai:
            return self.openai_model
        return self.sentence_model


# =============================================================================
# Recommender Configuration
# =============================================================================

@dataclass
class TripletConfig:
    """
    Configuration for triplet creation.
    
    Attributes:
        max_services: Maximum number of services to keep in triplet
        service_separator: Separator for services in triplet
        triplet_separator: Separator between triplet components
        top_k_services: Number of top services to track globally
    """
    max_services: int = 3
    service_separator: str = ","
    triplet_separator: str = "|||"
    top_k_services: int = 70


@dataclass
class ContentRecommenderConfig:
    """
    Configuration for content-based recommender.
    
    Attributes:
        embedding_weights: Weights for different feature types
        use_industry_hierarchy: Whether to use industry clustering
        fusion_method: How to combine embeddings ('concat' or 'weighted_sum')
    """
    embedding_weights: Dict[str, float] = field(default_factory=lambda: {
        'triplet_structure': 0.3,
        'background_text': 0.3,
        'services_text': 0.2,
        'location': 0.1,
        'numerical': 0.1
    })
    use_industry_hierarchy: bool = True
    fusion_method: str = "concat"


@dataclass
class CollaborativeRecommenderConfig:
    """
    Configuration for collaborative filtering recommender.
    
    Attributes:
        min_similarity: Minimum similarity threshold
        top_k_similar_users: Number of similar users to consider
        profile_weight: Weight for profile-based similarity
        history_weight: Weight for history-based similarity
    """
    min_similarity: float = 0.1
    top_k_similar_users: int = 30
    profile_weight: float = 0.4
    history_weight: float = 0.6


@dataclass
class EnsembleRecommenderConfig:
    """
    Configuration for ensemble recommender.
    
    Attributes:
        n_estimators: Number of trees in gradient boosting
        learning_rate: Learning rate for gradient boosting
        max_depth: Maximum depth of trees
        validation_split: Fraction of data for validation
        model_weights: Default weights for each model (if meta-learner not used)
    """
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    validation_split: float = 0.2
    random_state: int = 42
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        'content': 0.25,
        'enhanced_content': 0.35,
        'user_collab': 0.15,
        'enhanced_collab': 0.25
    })


@dataclass
class RecommenderConfig:
    """
    Complete recommender configuration.
    
    Combines all sub-configurations for the recommendation system.
    """
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    triplet: TripletConfig = field(default_factory=TripletConfig)
    content: ContentRecommenderConfig = field(default_factory=ContentRecommenderConfig)
    collaborative: CollaborativeRecommenderConfig = field(default_factory=CollaborativeRecommenderConfig)
    ensemble: EnsembleRecommenderConfig = field(default_factory=EnsembleRecommenderConfig)
    
    @classmethod
    def default(cls) -> "RecommenderConfig":
        """Create default configuration."""
        return cls()
    
    @classmethod
    def with_openai(cls) -> "RecommenderConfig":
        """Create configuration using OpenAI embeddings."""
        config = cls()
        config.embedding.embedding_type = EmbeddingType.OPENAI
        return config
    
    @classmethod
    def with_sentence_transformers(cls) -> "RecommenderConfig":
        """Create configuration using SentenceTransformers."""
        config = cls()
        config.embedding.embedding_type = EmbeddingType.SENTENCE_TRANSFORMERS
        return config


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation.
    
    Attributes:
        top_k: Top-k value for evaluation
        use_partial_match: Whether to use partial matching
        partial_match_threshold: Threshold for partial match
        similarity_weights: Weights for triplet similarity calculation
    """
    top_k: int = 10
    use_partial_match: bool = True
    partial_match_threshold: float = 0.5
    similarity_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # industry, size, services


@dataclass
class ExperimentConfig:
    """
    Configuration for running experiments.
    
    Attributes:
        name: Experiment name
        recommender: Recommender configuration
        evaluation: Evaluation configuration
        experiments_to_run: List of experiment names to run
    """
    name: str = "triplet_recommendation"
    recommender: RecommenderConfig = field(default_factory=RecommenderConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiments_to_run: List[str] = field(default_factory=lambda: [
        "triplet_content",
        "enhanced_content",
        "user_collaborative",
        "enhanced_collaborative",
        "triplet_ensemble",
        "hybrid_ensemble"
    ])
    save_results: bool = True
    verbose: bool = True


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class DataConfig:
    """
    Configuration for data paths.
    
    Attributes:
        base_dir: Base directory for data
        train_file: Training data filename
        test_file: Test data filename
        benchmark_dir: Directory for benchmark results
    """
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    train_file: str = "sample_0_100_update.csv"
    test_file: str = "sample_0_100_update_test.csv"
    benchmark_dir: str = "benchmark"
    
    @property
    def train_path(self) -> Path:
        """Get full path to training data."""
        return self.base_dir / self.train_file
    
    @property
    def test_path(self) -> Path:
        """Get full path to test data."""
        return self.base_dir / self.test_file
    
    @property
    def benchmark_path(self) -> Path:
        """Get full path to benchmark directory."""
        return self.base_dir / self.benchmark_dir


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    Combines data, experiment, and output configurations.
    """
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_paths(
        cls,
        train_path: str,
        test_path: str,
        output_dir: Optional[str] = None
    ) -> "PipelineConfig":
        """
        Create pipeline config from file paths.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            output_dir: Optional output directory
        
        Returns:
            PipelineConfig instance
        """
        train_path = Path(train_path)
        test_path = Path(test_path)
        
        data_config = DataConfig(
            base_dir=train_path.parent,
            train_file=train_path.name,
            test_file=test_path.name
        )
        
        if output_dir:
            data_config.benchmark_dir = output_dir
        
        return cls(data=data_config)
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check data files exist
        if not self.data.train_path.exists():
            raise ValueError(f"Training data not found: {self.data.train_path}")
        
        if not self.data.test_path.exists():
            raise ValueError(f"Test data not found: {self.data.test_path}")
        
        # Create benchmark directory if needed
        self.data.benchmark_path.mkdir(parents=True, exist_ok=True)
        
        # Check embedding cache directory
        if self.experiment.recommender.embedding.use_cache:
            cache_dir = self.experiment.recommender.embedding.cache_dir
            if cache_dir:
                cache_dir.mkdir(parents=True, exist_ok=True)
        
        return True
