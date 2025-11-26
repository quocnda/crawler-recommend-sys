"""
Triplet Utilities
=================

Utilities for creating and managing triplets (industry, client_size, services)
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from collections import Counter


class TripletManager:
    """Manages triplet creation and normalization."""
    
    # Client size buckets
    SIZE_BUCKETS = {
        'micro': (0, 10),
        'small': (11, 50),
        'medium': (51, 200),
        'large': (201, 1000),
        'enterprise': (1001, float('inf'))
    }
    
    def __init__(
        self,
        max_services: int = 3,
        service_separator: str = ',',
        triplet_separator: str = '|||'
    ):
        self.max_services = max_services
        self.service_separator = service_separator
        self.triplet_separator = triplet_separator
        
        # Will be populated during fit
        self.top_services: List[str] = []
        self.service_alias_map: Dict[str, str] = {}
        
    def normalize_client_size(self, row: pd.Series) -> str:
        """
        Normalize client size to bucket name.
        Uses client_size, client_min, client_max fields.
        """
        # Try to use client_min and client_max if available
        if 'client_min' in row and 'client_max' in row:
            client_min = row['client_min']
            client_max = row['client_max']
            
            if pd.notna(client_min) and pd.notna(client_max):
                # Use midpoint for bucketing
                mid = (client_min + client_max) / 2
                
                for bucket_name, (low, high) in self.SIZE_BUCKETS.items():
                    if low <= mid <= high:
                        return bucket_name
        
        # Fallback: parse from client_size string
        if 'client_size' in row and pd.notna(row['client_size']):
            client_size_str = str(row['client_size'])
            
            # Extract numbers
            nums = re.findall(r'\d+', client_size_str.replace(',', ''))
            
            if nums:
                # If range like "51-200", use midpoint
                if len(nums) >= 2:
                    mid = (int(nums[0]) + int(nums[1])) / 2
                else:
                    mid = int(nums[0])
                
                for bucket_name, (low, high) in self.SIZE_BUCKETS.items():
                    if low <= mid <= high:
                        return bucket_name
        
        return 'unknown'
    
    def normalize_services(self, services_str: str, use_top_k: bool = True) -> str:
        """
        Normalize services string to top-k canonical services.
        
        Args:
            services_str: Comma-separated services
            use_top_k: If True, only keep top-k most common services from fit()
        
        Returns:
            Normalized services string (sorted alphabetically)
        """
        if pd.isna(services_str) or not str(services_str).strip():
            return 'unknown'
        
        # Split and clean
        services = [s.strip() for s in str(services_str).split(self.service_separator)]
        services = [s for s in services if s and s.lower() != 'nan']
        
        return self.service_separator.join(services)
    
    def fit(self, df: pd.DataFrame, services_column: str = 'services') -> 'TripletManager':
        """
        Fit the triplet manager to learn top services and aliases.
        
        IMPORTANT: Only call this on TRAINING data to avoid data leakage.
        """
        # Count all services
        all_services = []
        
        for services_str in df[services_column].dropna():
            services = [s.strip() for s in str(services_str).split(self.service_separator)]
            services = [s for s in services if s and s.lower() != 'nan']
            all_services.extend(services)
        
        if not all_services:
            self.top_services = []
            return self
        
        # Get top services
        service_counts = Counter(all_services)
        
        # Keep top 50 most common services
        top_k_global = 70
        self.top_services = [svc for svc, _ in service_counts.most_common(top_k_global)]
        
        return self
    
    def create_triplet(self, row: pd.Series) -> str:
        """
        Create triplet ID from row.
        
        Returns:
            Triplet string: "industry|||client_size|||services"
        """
        industry = str(row.get('industry', 'unknown')).strip()
        if not industry or industry.lower() == 'nan':
            industry = 'unknown'
        
        client_size = self.normalize_client_size(row)
        
        services = self.normalize_services(
            row.get('services', ''),
            use_top_k=True
        )
        # print('Services normalized:', services)
        
        return f"{industry}{self.triplet_separator}{client_size}{self.triplet_separator}{services}"
    
    def parse_triplet(self, triplet_str: str) -> Tuple[str, str, str]:
        """
        Parse triplet string back to components.
        
        Returns:
            (industry, client_size, services)
        """
        parts = triplet_str.split(self.triplet_separator)
        
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        else:
            return 'unknown', 'unknown', 'unknown'
    
    def calculate_triplet_similarity(
        self,
        triplet1: str,
        triplet2: str,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    ) -> float:
        """
        Calculate similarity between two triplets.
        
        Args:
            triplet1, triplet2: Triplet strings
            weights: (industry_weight, size_weight, services_weight)
        
        Returns:
            Similarity score [0, 1]
        """
        ind1, size1, svc1 = self.parse_triplet(triplet1)
        ind2, size2, svc2 = self.parse_triplet(triplet2)
        
        # Industry match (exact)
        industry_match = 1.0 if ind1 == ind2 else 0.0
        
        # Client size match (with adjacency bonus)
        size_match = 0.0
        if size1 == size2:
            size_match = 1.0
        else:
            # Adjacent sizes get partial credit
            size_order = ['micro', 'small', 'medium', 'large', 'enterprise']
            try:
                idx1 = size_order.index(size1)
                idx2 = size_order.index(size2)
                distance = abs(idx1 - idx2)
                
                if distance == 1:
                    size_match = 0.5  # Adjacent sizes
                elif distance == 2:
                    size_match = 0.2  # Two steps away
            except ValueError:
                size_match = 0.0
        
        # Services overlap (Jaccard similarity)
        services1 = set(svc1.split(self.service_separator)) if svc1 != 'unknown' else set()
        services2 = set(svc2.split(self.service_separator)) if svc2 != 'unknown' else set()
        
        if services1 or services2:
            intersection = len(services1 & services2)
            union = len(services1 | services2)
            services_match = intersection / union if union > 0 else 0.0
        else:
            services_match = 0.0
        
        # Weighted combination
        w_ind, w_size, w_svc = weights
        similarity = (
            w_ind * industry_match +
            w_size * size_match +
            w_svc * services_match
        )
        
        return similarity
    
    def is_exact_match(self, triplet1: str, triplet2: str) -> bool:
        """Check if two triplets are exactly the same."""
        return triplet1 == triplet2
    
    def is_partial_match(
        self,
        triplet1: str,
        triplet2: str,
        threshold: float = 0.5
    ) -> bool:
        """
        Check if two triplets are partially matching.
        
        Args:
            threshold: Minimum similarity for partial match
        """
        similarity = self.calculate_triplet_similarity(triplet1, triplet2)
        return similarity >= threshold


class TripletFeatureExtractor:
    """Extract features from triplets for ML models."""
    
    def __init__(self, triplet_manager: TripletManager):
        self.triplet_manager = triplet_manager
        self.industry_encoder = {}
        self.size_encoder = {}
        self.service_encoder = {}
    
    def fit(self, triplets: List[str]) -> 'TripletFeatureExtractor':
        """Fit encoders on training triplets."""
        industries = set()
        sizes = set()
        services = set()
        
        for triplet in triplets:
            ind, size, svc = self.triplet_manager.parse_triplet(triplet)
            industries.add(ind)
            sizes.add(size)
            
            for s in svc.split(self.triplet_manager.service_separator):
                if s != 'unknown':
                    services.add(s)
        
        # Create mappings
        self.industry_encoder = {ind: i for i, ind in enumerate(sorted(industries))}
        self.size_encoder = {size: i for i, size in enumerate(sorted(sizes))}
        self.service_encoder = {svc: i for i, svc in enumerate(sorted(services))}
        
        return self
    
    def transform(self, triplet: str) -> np.ndarray:
        """
        Transform triplet to feature vector.
        
        Returns:
            One-hot encoded vector
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
            if s in self.service_encoder:
                svc_vec[self.service_encoder[s]] = 1.0
        
        return np.concatenate([ind_vec, size_vec, svc_vec])


def add_triplet_column(
    df: pd.DataFrame,
    triplet_manager: TripletManager,
    column_name: str = 'triplet'
) -> pd.DataFrame:
    """
    Add triplet column to dataframe.
    
    Args:
        df: Input dataframe
        triplet_manager: Fitted TripletManager
        column_name: Name for the new column
    
    Returns:
        DataFrame with added triplet column
    """
    df = df.copy()
    df[column_name] = df.apply(triplet_manager.create_triplet, axis=1)
    return df
