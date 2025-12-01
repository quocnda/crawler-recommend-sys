"""
Type Definitions
================

Định nghĩa các types, data classes, và enums cho recommendation system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd
import numpy as np


# =============================================================================
# Type Aliases
# =============================================================================

UserID = str  # LinkedIn company outsource identifier
TripletID = str  # Triplet identifier: "industry|||size|||services"
Score = float  # Recommendation/similarity score [0, 1]


# =============================================================================
# Enums
# =============================================================================

class SizeBucket(str, Enum):
    """Client size bucket classification."""
    MICRO = "micro"  # 0-10 employees
    SMALL = "small"  # 11-50 employees
    MEDIUM = "medium"  # 51-200 employees
    LARGE = "large"  # 201-1000 employees
    ENTERPRISE = "enterprise"  # 1000+ employees
    UNKNOWN = "unknown"
    
    @classmethod
    def from_employee_count(cls, count: Optional[float]) -> "SizeBucket":
        """Convert employee count to size bucket."""
        if count is None or pd.isna(count):
            return cls.UNKNOWN
        if count <= 10:
            return cls.MICRO
        if count <= 50:
            return cls.SMALL
        if count <= 200:
            return cls.MEDIUM
        if count <= 1000:
            return cls.LARGE
        return cls.ENTERPRISE
    
    @classmethod
    def get_bucket_range(cls, bucket: "SizeBucket") -> Tuple[int, int]:
        """Get the employee count range for a bucket."""
        ranges = {
            cls.MICRO: (0, 10),
            cls.SMALL: (11, 50),
            cls.MEDIUM: (51, 200),
            cls.LARGE: (201, 1000),
            cls.ENTERPRISE: (1001, float('inf')),
            cls.UNKNOWN: (0, 0),
        }
        return ranges.get(bucket, (0, 0))


class EmbeddingType(str, Enum):
    """Type of embedding model to use."""
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    

class RecommenderMode(str, Enum):
    """Mode for recommendation generation."""
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TripletComponents:
    """
    Parsed components of a triplet.
    
    A triplet represents a unique combination of:
    - industry: The industry sector
    - size: Client size bucket
    - services: Comma-separated services
    
    Example: "Healthcare|||medium|||Web Development,Mobile Development"
    """
    industry: str
    size: SizeBucket
    services: str
    
    def __post_init__(self):
        """Validate and normalize components."""
        if not self.industry or self.industry.lower() == 'nan':
            self.industry = 'unknown'
        
        if isinstance(self.size, str):
            try:
                self.size = SizeBucket(self.size)
            except ValueError:
                self.size = SizeBucket.UNKNOWN
        
        if not self.services or self.services.lower() == 'nan':
            self.services = 'unknown'
    
    def to_triplet_string(self, separator: str = "|||") -> TripletID:
        """Convert to triplet string format."""
        size_value = self.size.value if isinstance(self.size, SizeBucket) else str(self.size)
        return f"{self.industry}{separator}{size_value}{separator}{self.services}"
    
    def get_services_list(self) -> List[str]:
        """Get services as a list."""
        if self.services == 'unknown':
            return []
        return [s.strip() for s in self.services.split(',') if s.strip()]
    
    @classmethod
    def from_triplet_string(
        cls, 
        triplet_str: TripletID, 
        separator: str = "|||"
    ) -> "TripletComponents":
        """Parse triplet string to components."""
        parts = triplet_str.split(separator)
        
        if len(parts) == 3:
            industry, size, services = parts
            try:
                size_bucket = SizeBucket(size)
            except ValueError:
                size_bucket = SizeBucket.UNKNOWN
            return cls(industry=industry, size=size_bucket, services=services)
        
        return cls(industry='unknown', size=SizeBucket.UNKNOWN, services='unknown')


@dataclass
class UserProfile:
    """
    User profile containing company information and interaction history.
    
    Attributes:
        user_id: Unique identifier (linkedin_company_outsource)
        description: Company description
        services_offered: Services the company offers
        interaction_history: Set of triplets the user has interacted with
        feature_vector: Computed feature vector for similarity
    """
    user_id: UserID
    description: Optional[str] = None
    services_offered: Optional[str] = None
    interaction_history: Set[TripletID] = field(default_factory=set)
    feature_vector: Optional[np.ndarray] = None
    
    def has_history(self) -> bool:
        """Check if user has interaction history."""
        return len(self.interaction_history) > 0
    
    def get_history_count(self) -> int:
        """Get number of historical interactions."""
        return len(self.interaction_history)


@dataclass
class RecommendationResult:
    """
    Single recommendation result.
    
    Attributes:
        triplet: Recommended triplet ID
        score: Confidence score [0, 1]
        source: Source model/method that generated this recommendation
        metadata: Additional metadata (industry, size, services parsed)
    """
    triplet: TripletID
    score: Score
    source: str = "unknown"
    metadata: Optional[Dict] = None
    
    def __lt__(self, other: "RecommendationResult") -> bool:
        """Compare by score for sorting."""
        return self.score < other.score
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'triplet': self.triplet,
            'score': self.score,
            'source': self.source,
            'metadata': self.metadata or {}
        }


@dataclass
class UserRecommendations:
    """
    Recommendations for a single user.
    
    Attributes:
        user_id: User identifier
        recommendations: List of recommendation results, sorted by score descending
    """
    user_id: UserID
    recommendations: List[RecommendationResult] = field(default_factory=list)
    
    def add(self, rec: RecommendationResult):
        """Add a recommendation."""
        self.recommendations.append(rec)
    
    def sort(self, descending: bool = True):
        """Sort recommendations by score."""
        self.recommendations.sort(reverse=descending)
    
    def top_k(self, k: int) -> List[RecommendationResult]:
        """Get top-k recommendations."""
        self.sort()
        return self.recommendations[:k]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        if not self.recommendations:
            return pd.DataFrame(columns=['linkedin_company_outsource', 'triplet', 'score'])
        
        return pd.DataFrame([
            {
                'linkedin_company_outsource': self.user_id,
                'triplet': rec.triplet,
                'score': rec.score
            }
            for rec in self.recommendations
        ])


@dataclass
class EvaluationMetrics:
    """
    Evaluation metrics for a recommendation experiment.
    
    Attributes:
        k: Top-k value used for evaluation
        precision: Precision@k
        recall: Recall@k
        f1: F1@k
        map_score: Mean Average Precision@k
        ndcg: Normalized Discounted Cumulative Gain@k
        hit_rate: Hit Rate@k
        match_type: "exact" or "partial"
    """
    k: int
    precision: float
    recall: float
    f1: float
    map_score: float
    ndcg: float
    hit_rate: float
    match_type: str = "exact"
    users_evaluated: int = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'users_evaluated': self.users_evaluated,
            'match_type': self.match_type,
            f'Precision@{self.k}': self.precision,
            f'Recall@{self.k}': self.recall,
            f'F1@{self.k}': self.f1,
            f'MAP@{self.k}': self.map_score,
            f'nDCG@{self.k}': self.ndcg,
            f'HitRate@{self.k}': self.hit_rate,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"EvaluationMetrics(k={self.k}, match={self.match_type}, "
            f"P={self.precision:.4f}, R={self.recall:.4f}, "
            f"F1={self.f1:.4f}, MAP={self.map_score:.4f}, "
            f"nDCG={self.ndcg:.4f}, HR={self.hit_rate:.4f})"
        )


@dataclass
class ExperimentResult:
    """
    Complete result of a recommendation experiment.
    
    Attributes:
        name: Experiment/model name
        exact_metrics: Metrics with exact matching
        partial_metrics: Metrics with partial matching
        recommendations_df: DataFrame of all recommendations
        per_user_metrics: Per-user evaluation metrics
    """
    name: str
    exact_metrics: Optional[EvaluationMetrics] = None
    partial_metrics: Optional[EvaluationMetrics] = None
    recommendations_df: Optional[pd.DataFrame] = None
    per_user_exact: Optional[pd.DataFrame] = None
    per_user_partial: Optional[pd.DataFrame] = None
    
    def get_summary(self) -> Dict:
        """Get summary of results."""
        summary = {'name': self.name}
        
        if self.exact_metrics:
            summary['exact'] = self.exact_metrics.to_dict()
        
        if self.partial_metrics:
            summary['partial'] = self.partial_metrics.to_dict()
        
        if self.recommendations_df is not None:
            summary['n_recommendations'] = len(self.recommendations_df)
            summary['n_users'] = self.recommendations_df['linkedin_company_outsource'].nunique()
        
        return summary


@dataclass 
class GroundTruth:
    """
    Ground truth data for evaluation.
    
    Attributes:
        user_triplets: Mapping from user_id to list of relevant triplets
    """
    user_triplets: Dict[UserID, List[TripletID]] = field(default_factory=dict)
    
    def add(self, user_id: UserID, triplet: TripletID):
        """Add a ground truth triplet for a user."""
        if user_id not in self.user_triplets:
            self.user_triplets[user_id] = []
        self.user_triplets[user_id].append(triplet)
    
    def get(self, user_id: UserID) -> List[TripletID]:
        """Get ground truth triplets for a user."""
        return self.user_triplets.get(user_id, [])
    
    def get_users(self) -> List[UserID]:
        """Get all users with ground truth."""
        return list(self.user_triplets.keys())
    
    def __len__(self) -> int:
        """Get number of users."""
        return len(self.user_triplets)
    
    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame,
        user_col: str = "linkedin_company_outsource",
        triplet_col: str = "triplet"
    ) -> "GroundTruth":
        """Build ground truth from DataFrame."""
        gt = cls()
        
        for _, row in df.iterrows():
            user_id = row.get(user_col)
            triplet = row.get(triplet_col)
            
            if pd.notna(user_id) and pd.notna(triplet):
                gt.add(user_id, triplet)
        
        return gt
