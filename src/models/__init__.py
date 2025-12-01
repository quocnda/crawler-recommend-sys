"""
Models Package
==============

Định nghĩa các data models và types cho recommendation system.
"""

from .types import (
    # Data types
    UserID,
    TripletID,
    Score,
    
    # Data classes
    TripletComponents,
    UserProfile,
    RecommendationResult,
    EvaluationMetrics,
    ExperimentResult,
    
    # Enums
    SizeBucket,
    EmbeddingType,
    RecommenderMode,
)

from .config import (
    EmbeddingConfig,
    RecommenderConfig,
    ExperimentConfig,
    PipelineConfig,
)

__all__ = [
    # Types
    "UserID",
    "TripletID", 
    "Score",
    
    # Data classes
    "TripletComponents",
    "UserProfile",
    "RecommendationResult",
    "EvaluationMetrics",
    "ExperimentResult",
    
    # Enums
    "SizeBucket",
    "EmbeddingType",
    "RecommenderMode",
    
    # Config
    "EmbeddingConfig",
    "RecommenderConfig",
    "ExperimentConfig",
    "PipelineConfig",
]
