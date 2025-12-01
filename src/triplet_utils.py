"""
Triplet Utilities Module
========================

Utilities for creating, managing, and comparing triplets.

A triplet is a unique combination of:
- Industry: The industry sector of the client
- Client Size: Size bucket (micro, small, medium, large, enterprise)
- Services: Comma-separated list of services provided

Example triplet: "Healthcare|||medium|||Web Development,Mobile Development"

Input:
    - DataFrame with columns: industry, client_min, client_max, services

Output:
    - DataFrame with added 'triplet' column
    - TripletManager for triplet operations

Usage:
    >>> from triplet_utils import TripletManager, add_triplet_column
    >>> manager = TripletManager()
    >>> manager.fit(df_train, services_column='services')
    >>> df = add_triplet_column(df, manager)
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

# Import types from models module
try:
    from models.types import (
        TripletID, 
        TripletComponents, 
        SizeBucket,
        Score,
    )
except ImportError:
    # Fallback for standalone usage
    TripletID = str
    Score = float
    
    class SizeBucket:
        MICRO = "micro"
        SMALL = "small"
        MEDIUM = "medium"
        LARGE = "large"
        ENTERPRISE = "enterprise"
        UNKNOWN = "unknown"


# =============================================================================
# Constants
# =============================================================================

# Size bucket definitions: (min_employees, max_employees)
SIZE_BUCKETS: Dict[str, Tuple[int, int]] = {
    'micro': (0, 10),
    'small': (11, 50),
    'medium': (51, 200),
    'large': (201, 1000),
    'enterprise': (1001, float('inf'))
}

# Ordered list of size buckets for similarity calculation
SIZE_ORDER = ['micro', 'small', 'medium', 'large', 'enterprise']

# Default triplet separator
DEFAULT_TRIPLET_SEPARATOR = "|||"
DEFAULT_SERVICE_SEPARATOR = ","


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TripletSimilarityWeights:
    """
    Weights for triplet similarity calculation.
    
    Attributes:
        industry_weight: Weight for industry matching [0, 1]
        size_weight: Weight for size matching [0, 1]
        services_weight: Weight for services overlap [0, 1]
    """
    industry_weight: float = 0.5
    size_weight: float = 0.3
    services_weight: float = 0.2
    
    def __post_init__(self):
        """Validate weights sum to 1."""
        total = self.industry_weight + self.size_weight + self.services_weight
        if abs(total - 1.0) > 1e-6:
            # Normalize weights
            self.industry_weight /= total
            self.size_weight /= total
            self.services_weight /= total
    
    def as_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple for compatibility."""
        return (self.industry_weight, self.size_weight, self.services_weight)


# =============================================================================
# TripletManager Class
# =============================================================================

class TripletManager:
    """
    Manager for triplet creation, parsing, and similarity calculation.
    
    Handles:
    - Triplet creation from DataFrame rows
    - Triplet parsing back to components
    - Similarity calculation between triplets
    - Service vocabulary management
    
    Attributes:
        max_services: Maximum number of services in a triplet
        service_separator: Separator for services string
        triplet_separator: Separator between triplet components
        top_services: List of most common services from training data
        service_alias_map: Mapping for service name normalization
    
    Example:
        >>> manager = TripletManager(max_services=3)
        >>> manager.fit(df_train)
        >>> triplet = manager.create_triplet(row)
        >>> similarity = manager.calculate_triplet_similarity(triplet1, triplet2)
    """
    
    def __init__(
        self,
        max_services: int = 3,
        service_separator: str = DEFAULT_SERVICE_SEPARATOR,
        triplet_separator: str = DEFAULT_TRIPLET_SEPARATOR,
        top_k_services: int = 70
    ):
        """
        Initialize TripletManager.
        
        Args:
            max_services: Maximum number of services to include in triplet
            service_separator: Separator for services (default: ",")
            triplet_separator: Separator for triplet components (default: "|||")
            top_k_services: Number of top services to track
        """
        self.max_services = max_services
        self.service_separator = service_separator
        self.triplet_separator = triplet_separator
        self.top_k_services = top_k_services
        
        # Populated during fit()
        self.top_services: List[str] = []
        self.service_alias_map: Dict[str, str] = {}
        self.service_counts: Dict[str, int] = {}
        
        # Statistics
        self._fitted = False
        self._stats: Dict[str, any] = {}
    
    @property
    def is_fitted(self) -> bool:
        """Check if manager has been fitted."""
        return self._fitted
    
    # =========================================================================
    # Fitting Methods
    # =========================================================================
    
    def fit(
        self, 
        df: pd.DataFrame, 
        services_column: str = 'services'
    ) -> 'TripletManager':
        """
        Fit the triplet manager on training data.
        
        Learns:
        - Top services (most frequent)
        - Service counts for statistics
        
        IMPORTANT: Only fit on TRAINING data to avoid data leakage.
        
        Args:
            df: Training DataFrame
            services_column: Name of column containing services
        
        Returns:
            Self for chaining
        
        Example:
            >>> manager = TripletManager()
            >>> manager.fit(df_train, services_column='services')
            >>> print(f"Learned {len(manager.top_services)} services")
        """
        # Collect all services
        all_services: List[str] = []
        
        for services_str in df[services_column].dropna():
            services = self._parse_services_string(str(services_str))
            all_services.extend(services)
        
        if not all_services:
            self.top_services = []
            self._fitted = True
            return self
        
        # Count services
        self.service_counts = dict(Counter(all_services))
        
        # Get top-k most common services
        sorted_services = sorted(
            self.service_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.top_services = [svc for svc, _ in sorted_services[:self.top_k_services]]
        
        # Store statistics
        self._stats = {
            'total_services_seen': len(all_services),
            'unique_services': len(self.service_counts),
            'top_services_count': len(self.top_services),
        }
        
        self._fitted = True
        return self
    
    def _parse_services_string(self, services_str: str) -> List[str]:
        """
        Parse services string to list of individual services.
        
        Args:
            services_str: Comma-separated services
        
        Returns:
            List of cleaned service names
        """
        if pd.isna(services_str) or not str(services_str).strip():
            return []
        
        services = [s.strip() for s in str(services_str).split(self.service_separator)]
        return [s for s in services if s and s.lower() != 'nan']
    
    # =========================================================================
    # Size Normalization
    # =========================================================================
    
    def normalize_client_size(self, row: pd.Series) -> str:
        """
        Normalize client size to bucket name.
        
        Uses client_min/client_max if available, falls back to parsing client_size.
        
        Args:
            row: DataFrame row with size information
        
        Returns:
            Size bucket name (micro, small, medium, large, enterprise, unknown)
        
        Example:
            >>> size = manager.normalize_client_size(row)
            >>> print(size)  # "medium"
        """
        # Try client_min/max first
        if 'client_min' in row and 'client_max' in row:
            client_min = row['client_min']
            client_max = row['client_max']
            
            if pd.notna(client_min) and pd.notna(client_max):
                mid = (client_min + client_max) / 2
                return self._midpoint_to_bucket(mid)
        
        # Fallback to parsing client_size string
        if 'client_size' in row and pd.notna(row['client_size']):
            return self._parse_size_string(str(row['client_size']))
        
        return 'unknown'
    
    def _midpoint_to_bucket(self, mid: float) -> str:
        """Convert employee midpoint to size bucket."""
        for bucket_name, (low, high) in SIZE_BUCKETS.items():
            if low <= mid <= high:
                return bucket_name
        return 'unknown'
    
    def _parse_size_string(self, size_str: str) -> str:
        """Parse size string to bucket name."""
        nums = re.findall(r'\d+', size_str.replace(',', ''))
        
        if not nums:
            return 'unknown'
        
        if len(nums) >= 2:
            mid = (int(nums[0]) + int(nums[1])) / 2
        else:
            mid = int(nums[0])
        
        return self._midpoint_to_bucket(mid)
    
    # =========================================================================
    # Service Normalization
    # =========================================================================
    
    def normalize_services(
        self, 
        services_str: str, 
        use_top_k: bool = True
    ) -> str:
        """
        Normalize services string.
        
        Args:
            services_str: Raw services string
            use_top_k: If True, only keep top-k services from fit()
        
        Returns:
            Normalized services string (sorted alphabetically)
        """
        if pd.isna(services_str) or not str(services_str).strip():
            return 'unknown'
        
        services = self._parse_services_string(str(services_str))
        
        if not services:
            return 'unknown'
        
        # Filter to top services if requested
        if use_top_k and self.top_services:
            services = [s for s in services if s in self.top_services]
            if not services:
                return 'unknown'
        
        # Limit to max_services
        services = services[:self.max_services]
        
        return self.service_separator.join(services)
    
    # =========================================================================
    # Triplet Creation and Parsing
    # =========================================================================
    
    def create_triplet(self, row: pd.Series) -> TripletID:
        """
        Create triplet ID from DataFrame row.
        
        Args:
            row: DataFrame row with industry, size, and services info
        
        Returns:
            Triplet string: "industry|||size|||services"
        
        Example:
            >>> triplet = manager.create_triplet(row)
            >>> print(triplet)
            "Healthcare|||medium|||Web Development,Mobile Development"
        """
        # Extract industry
        industry = str(row.get('industry', 'unknown')).strip()
        if not industry or industry.lower() == 'nan':
            industry = 'unknown'
        
        # Normalize size
        client_size = self.normalize_client_size(row)
        
        # Normalize services
        services = self.normalize_services(
            row.get('services', ''),
            use_top_k=True
        )
        
        return f"{industry}{self.triplet_separator}{client_size}{self.triplet_separator}{services}"
    
    def parse_triplet(self, triplet_str: TripletID) -> Tuple[str, str, str]:
        """
        Parse triplet string back to components.
        
        Args:
            triplet_str: Triplet string to parse
        
        Returns:
            Tuple of (industry, client_size, services)
        
        Example:
            >>> industry, size, services = manager.parse_triplet(triplet)
            >>> print(industry)  # "Healthcare"
        """
        parts = triplet_str.split(self.triplet_separator)
        
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        
        return 'unknown', 'unknown', 'unknown'
    
    # =========================================================================
    # Similarity Calculation
    # =========================================================================
    
    def calculate_triplet_similarity(
        self,
        triplet1: TripletID,
        triplet2: TripletID,
        weights: Optional[Union[Tuple[float, float, float], TripletSimilarityWeights]] = None
    ) -> Score:
        """
        Calculate similarity between two triplets.
        
        Similarity is a weighted combination of:
        - Industry match (exact: 1.0, else: 0.0)
        - Size match (exact: 1.0, adjacent: 0.5, 2-apart: 0.2)
        - Services overlap (Jaccard similarity)
        
        Args:
            triplet1: First triplet
            triplet2: Second triplet
            weights: Optional tuple (industry_weight, size_weight, services_weight)
                    or TripletSimilarityWeights object
        
        Returns:
            Similarity score in [0, 1]
        
        Example:
            >>> sim = manager.calculate_triplet_similarity(triplet1, triplet2)
            >>> print(f"Similarity: {sim:.2f}")
        """
        if weights is None:
            weights = TripletSimilarityWeights()
        elif isinstance(weights, tuple):
            weights = TripletSimilarityWeights(*weights)
        
        # Parse triplets
        ind1, size1, svc1 = self.parse_triplet(triplet1)
        ind2, size2, svc2 = self.parse_triplet(triplet2)
        
        # Calculate component similarities
        industry_match = self._calculate_industry_similarity(ind1, ind2)
        size_match = self._calculate_size_similarity(size1, size2)
        services_match = self._calculate_services_similarity(svc1, svc2)
        
        # Weighted combination
        similarity = (
            weights.industry_weight * industry_match +
            weights.size_weight * size_match +
            weights.services_weight * services_match
        )
        
        return similarity
    
    def _calculate_industry_similarity(self, ind1: str, ind2: str) -> float:
        """Calculate industry similarity (exact match only)."""
        return 1.0 if ind1.lower() == ind2.lower() else 0.0
    
    def _calculate_size_similarity(self, size1: str, size2: str) -> float:
        """Calculate size similarity with adjacency bonus."""
        if size1 == size2:
            return 1.0
        
        try:
            idx1 = SIZE_ORDER.index(size1)
            idx2 = SIZE_ORDER.index(size2)
            distance = abs(idx1 - idx2)
            
            if distance == 1:
                return 0.5  # Adjacent sizes
            elif distance == 2:
                return 0.2  # Two steps away
        except ValueError:
            pass
        
        return 0.0
    
    def _calculate_services_similarity(self, svc1: str, svc2: str) -> float:
        """Calculate services similarity using Jaccard similarity."""
        services1 = set(self._parse_services_string(svc1)) if svc1 != 'unknown' else set()
        services2 = set(self._parse_services_string(svc2)) if svc2 != 'unknown' else set()
        
        if not services1 and not services2:
            return 0.0
        
        intersection = len(services1 & services2)
        union = len(services1 | services2)
        
        return intersection / union if union > 0 else 0.0
    
    # =========================================================================
    # Match Checking
    # =========================================================================
    
    def is_exact_match(self, triplet1: TripletID, triplet2: TripletID) -> bool:
        """
        Check if two triplets are exactly the same.
        
        Args:
            triplet1: First triplet
            triplet2: Second triplet
        
        Returns:
            True if triplets are identical
        """
        return triplet1 == triplet2
    
    def is_partial_match(
        self,
        triplet1: TripletID,
        triplet2: TripletID,
        threshold: float = 0.5
    ) -> bool:
        """
        Check if two triplets are partially matching.
        
        Args:
            triplet1: First triplet
            triplet2: Second triplet
            threshold: Minimum similarity for partial match
        
        Returns:
            True if similarity >= threshold
        """
        similarity = self.calculate_triplet_similarity(triplet1, triplet2)
        return similarity >= threshold
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get triplet manager statistics."""
        return {
            'fitted': self._fitted,
            'max_services': self.max_services,
            'top_services_count': len(self.top_services),
            **self._stats
        }


# =============================================================================
# TripletFeatureExtractor Class
# =============================================================================

class TripletFeatureExtractor:
    """
    Extract feature vectors from triplets for ML models.
    
    Creates one-hot/multi-hot encoded feature vectors from triplet strings.
    
    Attributes:
        triplet_manager: TripletManager instance
        industry_encoder: Industry to index mapping
        size_encoder: Size bucket to index mapping
        service_encoder: Service to index mapping
    
    Example:
        >>> extractor = TripletFeatureExtractor(manager)
        >>> extractor.fit(training_triplets)
        >>> features = extractor.transform(triplet)
    """
    
    def __init__(self, triplet_manager: TripletManager):
        """
        Initialize feature extractor.
        
        Args:
            triplet_manager: Fitted TripletManager instance
        """
        self.triplet_manager = triplet_manager
        
        self.industry_encoder: Dict[str, int] = {}
        self.size_encoder: Dict[str, int] = {}
        self.service_encoder: Dict[str, int] = {}
        
        self._feature_dim: Optional[int] = None
    
    def fit(self, triplets: List[TripletID]) -> 'TripletFeatureExtractor':
        """
        Fit encoders on training triplets.
        
        Args:
            triplets: List of triplet strings
        
        Returns:
            Self for chaining
        """
        industries: Set[str] = set()
        sizes: Set[str] = set()
        services: Set[str] = set()
        
        for triplet in triplets:
            ind, size, svc = self.triplet_manager.parse_triplet(triplet)
            
            industries.add(ind)
            sizes.add(size)
            
            for s in svc.split(self.triplet_manager.service_separator):
                if s.strip() and s != 'unknown':
                    services.add(s.strip())
        
        # Create encodings
        self.industry_encoder = {ind: i for i, ind in enumerate(sorted(industries))}
        self.size_encoder = {size: i for i, size in enumerate(sorted(sizes))}
        self.service_encoder = {svc: i for i, svc in enumerate(sorted(services))}
        
        self._feature_dim = (
            len(self.industry_encoder) +
            len(self.size_encoder) +
            len(self.service_encoder)
        )
        
        return self
    
    @property
    def feature_dimension(self) -> int:
        """Get total feature dimension."""
        return self._feature_dim or 0
    
    def transform(self, triplet: TripletID) -> np.ndarray:
        """
        Transform triplet to feature vector.
        
        Args:
            triplet: Triplet string
        
        Returns:
            One-hot/multi-hot encoded feature vector
        """
        ind, size, svc = self.triplet_manager.parse_triplet(triplet)
        
        # One-hot encode industry
        ind_vec = np.zeros(len(self.industry_encoder))
        if ind in self.industry_encoder:
            ind_vec[self.industry_encoder[ind]] = 1.0
        
        # One-hot encode size
        size_vec = np.zeros(len(self.size_encoder))
        if size in self.size_encoder:
            size_vec[self.size_encoder[size]] = 1.0
        
        # Multi-hot encode services
        svc_vec = np.zeros(len(self.service_encoder))
        for s in svc.split(self.triplet_manager.service_separator):
            s = s.strip()
            if s in self.service_encoder:
                svc_vec[self.service_encoder[s]] = 1.0
        
        return np.concatenate([ind_vec, size_vec, svc_vec])
    
    def transform_batch(self, triplets: List[TripletID]) -> np.ndarray:
        """
        Transform multiple triplets to feature matrix.
        
        Args:
            triplets: List of triplet strings
        
        Returns:
            Feature matrix of shape (n_triplets, feature_dim)
        """
        return np.array([self.transform(t) for t in triplets])


# =============================================================================
# Convenience Functions
# =============================================================================

def add_triplet_column(
    df: pd.DataFrame,
    triplet_manager: TripletManager,
    column_name: str = 'triplet'
) -> pd.DataFrame:
    """
    Add triplet column to DataFrame.
    
    Args:
        df: Input DataFrame with industry, size, and services columns
        triplet_manager: Fitted TripletManager
        column_name: Name for the new triplet column
    
    Returns:
        DataFrame with added triplet column
    
    Example:
        >>> df = add_triplet_column(df, manager, column_name='triplet')
        >>> print(df['triplet'].head())
    """
    df = df.copy()
    df[column_name] = df.apply(triplet_manager.create_triplet, axis=1)
    return df


def create_triplet_manager(
    df_train: pd.DataFrame,
    max_services: int = 3,
    services_column: str = 'services'
) -> TripletManager:
    """
    Create and fit a TripletManager on training data.
    
    Convenience function combining TripletManager creation and fitting.
    
    Args:
        df_train: Training DataFrame
        max_services: Maximum services per triplet
        services_column: Name of services column
    
    Returns:
        Fitted TripletManager
    
    Example:
        >>> manager = create_triplet_manager(df_train)
        >>> df_train = add_triplet_column(df_train, manager)
    """
    manager = TripletManager(max_services=max_services)
    manager.fit(df_train, services_column=services_column)
    return manager
