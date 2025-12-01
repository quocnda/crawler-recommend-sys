"""
Benchmark Evaluation Module
===========================

Module for evaluating recommendation system performance.

Metrics computed:
- Precision@k: Fraction of recommended items that are relevant
- Recall@k: Fraction of relevant items that were recommended
- F1@k: Harmonic mean of Precision and Recall
- MAP@k: Mean Average Precision at k
- nDCG@k: Normalized Discounted Cumulative Gain at k
- HitRate@k: Fraction of users with at least one hit

Input:
    - Recommendations DataFrame: columns [user_col, item_col, score]
    - Ground Truth DataFrame: columns [user_col, item_col]
    - Optional: Similarity function for partial matching

Output:
    - Summary metrics DataFrame
    - Per-user metrics DataFrame

Usage:
    >>> from benchmark_data import BenchmarkEvaluator
    >>> evaluator = BenchmarkEvaluator(recommendations_df, ground_truth_df)
    >>> summary, per_user = evaluator.evaluate(k=10)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from math import log2
from typing import Callable, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np

# Try to import from models module
try:
    from models.types import (
        UserID,
        TripletID,
        Score,
        EvaluationMetrics,
        GroundTruth,
    )
except ImportError:
    UserID = str
    TripletID = str
    Score = float
    EvaluationMetrics = None
    GroundTruth = None


# =============================================================================
# Constants
# =============================================================================

# Default column names
USER_COL = "linkedin_company_outsource"
ITEM_COL = "industry"
TRIPLET_COL = "triplet"
SCORE_COL = "score"


# =============================================================================
# Type Definitions
# =============================================================================

# Similarity function type: (item1, item2) -> similarity score
SimilarityFunction = Callable[[str, str], float]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PerUserMetrics:
    """
    Metrics for a single user.
    
    Attributes:
        user_id: User identifier
        num_predictions: Number of predictions made
        num_ground_truth: Number of ground truth items
        hits: Number of hits (matching predictions)
        precision: Precision@k
        recall: Recall@k
        f1: F1@k
        map_score: Average Precision@k
        ndcg: nDCG@k
        hit_rate: 1 if any hit, else 0
    """
    user_id: UserID
    num_predictions: int = 0
    num_ground_truth: int = 0
    hits: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map_score: float = 0.0
    ndcg: float = 0.0
    hit_rate: float = 0.0
    
    def to_dict(self, k: int) -> Dict:
        """Convert to dictionary with k-suffixed keys."""
        return {
            USER_COL: self.user_id,
            "num_pred": self.num_predictions,
            "num_true": self.num_ground_truth,
            "hits": self.hits,
            f"Precision@{k}": self.precision,
            f"Recall@{k}": self.recall,
            f"F1@{k}": self.f1,
            f"MAP@{k}": self.map_score,
            f"nDCG@{k}": self.ndcg,
            f"HitRate@{k}": self.hit_rate,
        }


@dataclass
class SummaryMetrics:
    """
    Aggregated metrics across all users.
    
    Attributes:
        users_evaluated: Number of users evaluated
        match_type: "exact" or "partial"
        k: Top-k value
        precision: Mean Precision@k
        recall: Mean Recall@k
        f1: Mean F1@k
        map_score: Mean MAP@k
        ndcg: Mean nDCG@k
        hit_rate: Mean HitRate@k
        median_recall: Median Recall@k
        p90_recall: 90th percentile Recall@k
    """
    users_evaluated: int
    match_type: str
    k: int
    precision: float
    recall: float
    f1: float
    map_score: float
    ndcg: float
    hit_rate: float
    median_recall: float = 0.0
    p90_recall: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "users_evaluated": self.users_evaluated,
            "match_type": self.match_type,
            f"Precision@{self.k}": self.precision,
            f"Recall@{self.k}": self.recall,
            f"F1@{self.k}": self.f1,
            f"MAP@{self.k}": self.map_score,
            f"nDCG@{self.k}": self.ndcg,
            f"HitRate@{self.k}": self.hit_rate,
            "median_recall": self.median_recall,
            "p90_recall": self.p90_recall,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Evaluation Results (k={self.k}, {self.match_type}):\n"
            f"  Users evaluated: {self.users_evaluated}\n"
            f"  Precision@{self.k}: {self.precision:.4f}\n"
            f"  Recall@{self.k}: {self.recall:.4f}\n"
            f"  F1@{self.k}: {self.f1:.4f}\n"
            f"  MAP@{self.k}: {self.map_score:.4f}\n"
            f"  nDCG@{self.k}: {self.ndcg:.4f}\n"
            f"  HitRate@{self.k}: {self.hit_rate:.4f}"
        )


# =============================================================================
# Metric Calculation Functions
# =============================================================================

def precision_at_k(hits: List[float], k: int) -> float:
    """
    Calculate Precision@k.
    
    Precision@k = (number of relevant items in top-k) / k
    
    Args:
        hits: List of hit indicators (1 for hit, 0 for miss, or similarity scores)
        k: Number of top items
    
    Returns:
        Precision score
    """
    return sum(hits[:k]) / k


def recall_at_k(hits: List[float], num_relevant: int, k: int) -> float:
    """
    Calculate Recall@k.
    
    Recall@k = (number of relevant items in top-k) / (total relevant items)
    
    Args:
        hits: List of hit indicators
        num_relevant: Total number of relevant items
        k: Number of top items
    
    Returns:
        Recall score
    """
    if num_relevant == 0:
        return 0.0
    return sum(hits[:k]) / num_relevant


def average_precision_at_k(hits: List[float], k: int, num_relevant: int) -> float:
    """
    Calculate Average Precision@k.
    
    AP@k = (1/min(k, num_relevant)) * sum(P@i * rel_i for i in 1..k)
    
    Args:
        hits: List of hit indicators
        k: Number of top items
        num_relevant: Total number of relevant items
    
    Returns:
        Average Precision score
    """
    if num_relevant == 0:
        return 0.0
    
    ap = 0.0
    hit_count = 0
    
    for i in range(min(k, len(hits))):
        if hits[i] > 0:  # Hit (can be 1 or similarity score)
            hit_count += 1
            ap += hit_count / (i + 1)
    
    return ap / min(num_relevant, k)


def ndcg_at_k(hits: List[float], k: int, num_relevant: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@k.
    
    DCG@k = sum(rel_i / log2(i + 2) for i in 0..k-1)
    IDCG@k = sum(1 / log2(i + 2) for i in 0..min(k, num_relevant)-1)
    nDCG@k = DCG@k / IDCG@k
    
    Args:
        hits: List of hit indicators
        k: Number of top items
        num_relevant: Total number of relevant items
    
    Returns:
        nDCG score
    """
    # Calculate DCG
    dcg = 0.0
    for i in range(min(k, len(hits))):
        if hits[i] > 0:
            dcg += 1.0 / log2(i + 2)  # i=0 -> log2(2) = 1
    
    # Calculate IDCG (ideal DCG)
    idcg = sum(1.0 / log2(i + 2) for i in range(min(num_relevant, k)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


# =============================================================================
# BenchmarkEvaluator Class
# =============================================================================

class BenchmarkEvaluator:
    """
    Evaluator for recommendation system benchmarks.
    
    Computes standard recommendation metrics comparing predictions
    against ground truth.
    
    Supports:
    - Exact matching: item must match exactly
    - Partial matching: uses similarity function with threshold
    
    Attributes:
        predictions: DataFrame with columns [user_col, item_col, score]
        ground_truth: DataFrame with columns [user_col, item_col]
        similarity_fn: Optional function for partial matching
        user_col: Name of user column
        item_col: Name of item/triplet column
    
    Example:
        >>> evaluator = BenchmarkEvaluator(
        ...     predictions_df, 
        ...     ground_truth_df,
        ...     similarity_fn=triplet_similarity
        ... )
        >>> summary, per_user = evaluator.evaluate(k=10, use_partial_match=True)
        >>> print(summary)
    """
    
    def __init__(
        self,
        predictions: pd.DataFrame,
        ground_truth: pd.DataFrame,
        similarity_fn: Optional[SimilarityFunction] = None,
        user_col: str = USER_COL,
        item_col: Optional[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            predictions: DataFrame with recommendations
            ground_truth: DataFrame with ground truth
            similarity_fn: Optional similarity function for partial matching
            user_col: Name of user column
            item_col: Name of item column (auto-detected if None)
        """
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.similarity_fn = similarity_fn
        self.user_col = user_col
        
        # Auto-detect item column
        if item_col is None:
            if TRIPLET_COL in predictions.columns:
                self.item_col = TRIPLET_COL
            else:
                self.item_col = ITEM_COL
        else:
            self.item_col = item_col
    
    def _get_unique_ordered(self, items: List) -> List:
        """Get unique items while preserving order."""
        seen = set()
        result = []
        for item in items:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    
    def _build_prediction_lists(self) -> Dict[UserID, List[str]]:
        """Build ordered prediction lists per user."""
        return (
            self.predictions
            .groupby(self.user_col)[self.item_col]
            .apply(list)
            .apply(self._get_unique_ordered)
            .to_dict()
        )
    
    def _build_ground_truth_sets(self) -> Dict[UserID, Set[str]]:
        """Build ground truth sets per user."""
        return (
            self.ground_truth
            .groupby(self.user_col)[self.item_col]
            .apply(set)
            .to_dict()
        )
    
    def _compute_exact_hits(
        self,
        predictions: List[str],
        ground_truth: Set[str]
    ) -> List[float]:
        """Compute hits using exact matching."""
        return [1.0 if pred in ground_truth else 0.0 for pred in predictions]
    
    def _compute_partial_hits(
        self,
        predictions: List[str],
        ground_truth: Set[str],
        threshold: float
    ) -> List[float]:
        """
        Compute hits using partial matching.
        
        For each prediction, find the best matching ground truth item.
        If similarity >= threshold, count as a weighted hit.
        
        Args:
            predictions: List of predicted items
            ground_truth: Set of ground truth items
            threshold: Minimum similarity for a match
        
        Returns:
            List of hit scores (similarity scores or 0)
        """
        if self.similarity_fn is None:
            # Fall back to exact matching
            return self._compute_exact_hits(predictions, ground_truth)
        
        hits = []
        for pred in predictions:
            max_similarity = 0.0
            
            for gt_item in ground_truth:
                similarity = self.similarity_fn(pred, gt_item)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= threshold:
                hits.append(max_similarity)
            else:
                hits.append(0.0)
        
        return hits
    
    def _evaluate_user(
        self,
        user_id: UserID,
        predictions: List[str],
        ground_truth: Set[str],
        k: int,
        use_partial_match: bool,
        partial_match_threshold: float
    ) -> PerUserMetrics:
        """Evaluate metrics for a single user."""
        # Get top-k predictions
        predictions_k = predictions[:k]
        
        # Compute hits
        if use_partial_match:
            hits = self._compute_partial_hits(
                predictions_k, 
                ground_truth, 
                partial_match_threshold
            )
        else:
            hits = self._compute_exact_hits(predictions_k, ground_truth)
        
        num_relevant = len(ground_truth)
        
        # Calculate metrics
        prec = precision_at_k(hits, k)
        rec = recall_at_k(hits, num_relevant, k)
        
        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        
        ap = average_precision_at_k(hits, k, num_relevant)
        ndcg = ndcg_at_k(hits, k, num_relevant)
        hr = 1.0 if sum(hits) > 0 else 0.0
        
        return PerUserMetrics(
            user_id=user_id,
            num_predictions=len(predictions_k),
            num_ground_truth=num_relevant,
            hits=int(sum(1 for h in hits if h > 0)),
            precision=prec,
            recall=rec,
            f1=f1,
            map_score=ap,
            ndcg=ndcg,
            hit_rate=hr
        )
    
    def evaluate(
        self,
        k: int = 10,
        use_partial_match: bool = False,
        partial_match_threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate recommendations.
        
        Args:
            k: Top-k value for evaluation
            use_partial_match: Whether to use partial matching
            partial_match_threshold: Minimum similarity for partial match
        
        Returns:
            Tuple of (summary_df, per_user_df)
        
        Example:
            >>> summary, per_user = evaluator.evaluate(k=10)
            >>> print(summary[['Precision@10', 'Recall@10', 'MAP@10']])
        """
        # Build prediction and ground truth dictionaries
        pred_lists = self._build_prediction_lists()
        gt_sets = self._build_ground_truth_sets()
        
        # Get all users
        all_users = sorted(set(pred_lists.keys()) | set(gt_sets.keys()))
        
        # Evaluate each user
        per_user_metrics: List[PerUserMetrics] = []
        
        for user_id in all_users:
            predictions = pred_lists.get(user_id, [])
            ground_truth = gt_sets.get(user_id, set())
            
            metrics = self._evaluate_user(
                user_id=user_id,
                predictions=predictions,
                ground_truth=ground_truth,
                k=k,
                use_partial_match=use_partial_match,
                partial_match_threshold=partial_match_threshold
            )
            per_user_metrics.append(metrics)
        
        # Build per-user DataFrame
        per_user_df = pd.DataFrame([m.to_dict(k) for m in per_user_metrics])
        
        # Calculate summary statistics
        if per_user_metrics:
            summary = SummaryMetrics(
                users_evaluated=len(per_user_metrics),
                match_type="partial" if use_partial_match else "exact",
                k=k,
                precision=per_user_df[f"Precision@{k}"].mean(),
                recall=per_user_df[f"Recall@{k}"].mean(),
                f1=per_user_df[f"F1@{k}"].mean(),
                map_score=per_user_df[f"MAP@{k}"].mean(),
                ndcg=per_user_df[f"nDCG@{k}"].mean(),
                hit_rate=per_user_df[f"HitRate@{k}"].mean(),
                median_recall=per_user_df[f"Recall@{k}"].median(),
                p90_recall=per_user_df[f"Recall@{k}"].quantile(0.9)
            )
        else:
            summary = SummaryMetrics(
                users_evaluated=0,
                match_type="partial" if use_partial_match else "exact",
                k=k,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                map_score=0.0,
                ndcg=0.0,
                hit_rate=0.0
            )
        
        summary_df = summary.to_dataframe()
        
        return summary_df, per_user_df
    
    def evaluate_topk(
        self,
        k: int = 10,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        use_partial_match: bool = False,
        partial_match_threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate at top-k (legacy interface for backward compatibility).
        
        Args:
            k: Top-k value
            user_col: User column name (unused, kept for compatibility)
            item_col: Item column name (unused, kept for compatibility)
            use_partial_match: Whether to use partial matching
            partial_match_threshold: Minimum similarity threshold
        
        Returns:
            Tuple of (summary_df, per_user_df)
        """
        return self.evaluate(
            k=k,
            use_partial_match=use_partial_match,
            partial_match_threshold=partial_match_threshold
        )


# =============================================================================
# Legacy Class (for backward compatibility)
# =============================================================================

class BenchmarkOutput(BenchmarkEvaluator):
    """
    Legacy class name for backward compatibility.
    
    Use BenchmarkEvaluator for new code.
    """
    
    def __init__(
        self,
        data_output: pd.DataFrame,
        data_ground_truth: pd.DataFrame,
        similarity_fn: Optional[SimilarityFunction] = None
    ):
        """
        Initialize with legacy parameter names.
        
        Args:
            data_output: Recommendations DataFrame
            data_ground_truth: Ground truth DataFrame
            similarity_fn: Optional similarity function
        """
        super().__init__(
            predictions=data_output,
            ground_truth=data_ground_truth,
            similarity_fn=similarity_fn
        )
        
        # Store legacy references
        self.data_output = data_output
        self.data_ground_truth = data_ground_truth
    
    # Legacy method aliases
    def _unique_preserve(self, seq):
        """Legacy method - use _get_unique_ordered instead."""
        return self._get_unique_ordered(seq)
    
    def _precision_at_k(self, hits: list, k: int) -> float:
        """Legacy method."""
        return precision_at_k(hits, k)
    
    def _recall_at_k(self, hits: list, num_true: int, k: int) -> float:
        """Legacy method."""
        return recall_at_k(hits, num_true, k)
    
    def _ap_at_k(self, hits, k, num_true):
        """Legacy method."""
        return average_precision_at_k(hits, k, num_true)
    
    def _ndcg_at_k(self, hits, k, num_true):
        """Legacy method."""
        return ndcg_at_k(hits, k, num_true)
    
    def _compute_partial_hits(
        self,
        predictions: list,
        ground_truth: set,
        threshold: float
    ) -> list:
        """Legacy method with original signature."""
        return super()._compute_partial_hits(predictions, ground_truth, threshold)