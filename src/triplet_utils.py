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
        service_separator: Separator for services string
        triplet_separator: Separator between triplet components
    
    Example:
        >>> manager = TripletManager()
        >>> manager.fit(df_train)
        >>> triplet = manager.create_triplet(row)
        >>> similarity = manager.calculate_triplet_similarity(triplet1, triplet2)
    """
    
    def __init__(
        self,
        service_separator: str = DEFAULT_SERVICE_SEPARATOR,
        triplet_separator: str = DEFAULT_TRIPLET_SEPARATOR,
    ):
        """
        Initialize TripletManager.
        
        Args:
            service_separator: Separator for services (default: ",")
            triplet_separator: Separator for triplet components (default: "|||")
        """
        self.service_separator = service_separator
        self.triplet_separator = triplet_separator
        
        # Populated during fit()
        self.top_services: List[str] = []
        self.service_counts: Dict[str, int] = {}
        
        # Statistics
        self._fitted = False
        self._stats: Dict[str, any] = {}
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
        client_min = row['client_min']
        client_max = row['client_max']
        
        if pd.notna(client_min) and pd.notna(client_max):
            mid = (client_min + client_max) / 2
            return self._midpoint_to_bucket(mid)

        return 'unknown'
    
    def _midpoint_to_bucket(self, mid: float) -> str:
        """Convert employee midpoint to size bucket."""
        for bucket_name, (low, high) in SIZE_BUCKETS.items():
            if low <= mid <= high:
                return bucket_name
        return 'unknown'
    
    # =========================================================================
    # Service Normalization
    # =========================================================================
    
    def normalize_services(
        self, 
        services_str: str, 
    ) -> str:
        """
        Normalize services string.
        
        Args:
            services_str: Raw services string        
        Returns:
            Normalized services string (sorted alphabetically)
        """
        if pd.isna(services_str) or not str(services_str).strip():
            return 'unknown'
        
        services = self._parse_services_string(str(services_str))
        
        if not services:
            return 'unknown'

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

    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get triplet manager statistics."""
        return {
            'fitted': self._fitted,
            'top_services_count': len(self.top_services),
            **self._stats
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def add_triplet_column(
    df: pd.DataFrame,
    triplet_manager: TripletManager,
    column_name: str = 'triplet'
) -> pd.DataFrame:
    df = df.copy()
    df[column_name] = df.apply(triplet_manager.create_triplet, axis=1)
    return df


def create_triplet_manager(
    df_train: pd.DataFrame,
    services_column: str = 'services'
) -> TripletManager:
    """
    Create and fit a TripletManager on training data.
    
    Convenience function combining TripletManager creation and fitting.
    
    Args:
        df_train: Training DataFrame
        services_column: Name of services column
    
    Returns:
        Fitted TripletManager
    
    Example:
        >>> manager = create_triplet_manager(df_train)
        >>> df_train = add_triplet_column(df_train, manager)
    """
    manager = TripletManager()
    manager.fit(df_train, services_column=services_column)
    return manager
