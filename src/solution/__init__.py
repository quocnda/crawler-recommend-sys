"""
Solution Package - Recommendation Algorithms
=============================================

This package contains all recommendation algorithm implementations.

Base Class:
    - BaseRecommender: Abstract base class defining the interface

Content-Based Recommenders:
    - TripletContentRecommender: Basic content-based with triplet features
    - EnhancedTripletContentRecommender: Advanced with multi-modal embeddings

Collaborative Filtering:
    - UserBasedCollaborativeRecommender: User-user collaborative filtering
    - EnhancedUserCollaborativeRecommender: With profile similarity

Ensemble Methods:
    - TripletEnsembleRecommender: Gradient Boosting meta-learner

Utilities:
    - OpenAIEmbedder: OpenAI embedding wrapper
"""

from .base_recommender import BaseRecommender
from .triplet_recommender import TripletContentRecommender
from .enhanced_triplet_content import EnhancedTripletContentRecommender
from .user_collaborative import UserBasedCollaborativeRecommender
from .enhanced_user_collaborative import EnhancedUserCollaborativeRecommender
from .triplet_ensemble import TripletEnsembleRecommender
from .openai_embedder import OpenAIEmbedder

__all__ = [
    # Base
    "BaseRecommender",
    
    # Content-based
    "TripletContentRecommender",
    "EnhancedTripletContentRecommender",
    
    # Collaborative
    "UserBasedCollaborativeRecommender",
    "EnhancedUserCollaborativeRecommender",
    
    # Ensemble
    "TripletEnsembleRecommender",
    
    # Utilities
    "OpenAIEmbedder",
]
