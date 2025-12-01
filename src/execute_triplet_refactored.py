"""
Triplet-based Recommendation Pipeline
=====================================

Main execution module for running triplet-based recommendation experiments.

This module provides:
1. Data preparation pipeline (load, preprocess, create triplets)
2. Individual experiment runners for each recommendation algorithm
3. Complete experiment pipeline with benchmarking
4. Result comparison and reporting

Available Experiments:
1. Triplet Content-Based Recommendation
2. Enhanced Triplet Content-Based (with advanced embeddings)
3. User-Based Collaborative Filtering
4. Enhanced User Collaborative (with profile similarity)
5. Triplet Ensemble (Gradient Boosting meta-learner)
6. Hybrid Ensemble (weighted combination)

Input:
    - Training data CSV: columns include industry, services, client size, etc.
    - Test data CSV: same format as training data
    
Output:
    - Recommendation results (CSV per experiment)
    - Evaluation metrics (exact and partial matching)
    - Per-user metrics for detailed analysis

Usage:
    >>> from execute_triplet import run_pipeline, PipelineConfig
    >>> config = PipelineConfig.default()
    >>> results = run_pipeline(config)
    
    Or from command line:
    $ python execute_triplet.py

Author: Quoc Nguyen
Version: 2.0.0
"""

from __future__ import annotations
import os
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
import traceback

import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Local imports
from preprocessing_data import preprocess_data
from triplet_utils import TripletManager, add_triplet_column
from benchmark_data import BenchmarkEvaluator

# Solution imports
from solution.triplet_recommender import TripletContentRecommender
from solution.user_collaborative import UserBasedCollaborativeRecommender
from solution.enhanced_user_collaborative import EnhancedUserCollaborativeRecommender
from solution.enhanced_triplet_content import EnhancedTripletContentRecommender
from solution.triplet_ensemble import TripletEnsembleRecommender

# Try to import models
try:
    from models.types import (
        UserID,
        TripletID,
        GroundTruth,
        ExperimentResult,
        EvaluationMetrics,
    )
    from models.config import (
        PipelineConfig as ModelPipelineConfig,
        EmbeddingConfig,
        EvaluationConfig,
    )
except ImportError:
    UserID = str
    TripletID = str
    GroundTruth = None
    ExperimentResult = None


# =============================================================================
# Constants and Configuration
# =============================================================================

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"

# Default files
DEFAULT_TRAIN_FILE = "sample_0_100_update.csv"
DEFAULT_TEST_FILE = "sample_0_100_update_test.csv"


class ExperimentType(str, Enum):
    """Available experiment types."""
    TRIPLET_CONTENT = "triplet_content"
    ENHANCED_CONTENT = "enhanced_content"
    USER_COLLABORATIVE = "user_collaborative"
    ENHANCED_COLLABORATIVE = "enhanced_collaborative"
    TRIPLET_ENSEMBLE = "triplet_ensemble"
    HYBRID_ENSEMBLE = "hybrid_ensemble"


@dataclass
class EmbeddingSettings:
    """Settings for embedding models."""
    use_openai: bool = True
    openai_model: str = "text-embedding-3-small"
    sentence_model: str = "all-MiniLM-L6-v2"


@dataclass
class ExperimentSettings:
    """Settings for running experiments."""
    top_k: int = 10
    use_partial_match: bool = True
    partial_match_threshold: float = 0.5
    save_results: bool = True
    verbose: bool = True


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    Attributes:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
        output_dir: Directory for benchmark results
        embedding: Embedding model settings
        experiment: Experiment settings
        experiments_to_run: List of experiments to execute
    """
    train_path: Path = field(default_factory=lambda: DATA_DIR / DEFAULT_TRAIN_FILE)
    test_path: Path = field(default_factory=lambda: DATA_DIR / DEFAULT_TEST_FILE)
    output_dir: Path = field(default_factory=lambda: BENCHMARK_DIR)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    experiments_to_run: List[ExperimentType] = field(default_factory=lambda: [
        ExperimentType.TRIPLET_CONTENT,
        ExperimentType.ENHANCED_CONTENT,
    ])
    
    @classmethod
    def default(cls) -> "PipelineConfig":
        """Create default configuration."""
        return cls()
    
    @classmethod
    def full(cls) -> "PipelineConfig":
        """Create configuration with all experiments enabled."""
        config = cls()
        config.experiments_to_run = list(ExperimentType)
        return config
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.train_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_path}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return True


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ExperimentResult:
    """
    Result of a single experiment.
    
    Attributes:
        name: Experiment name
        exact_summary: Summary metrics with exact matching
        partial_summary: Summary metrics with partial matching
        recommendations: DataFrame of all recommendations
        per_user_exact: Per-user metrics with exact matching
        per_user_partial: Per-user metrics with partial matching
    """
    name: str
    exact_summary: Optional[pd.DataFrame] = None
    partial_summary: Optional[pd.DataFrame] = None
    recommendations: Optional[pd.DataFrame] = None
    per_user_exact: Optional[pd.DataFrame] = None
    per_user_partial: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if experiment succeeded."""
        return self.error is None and self.recommendations is not None


@dataclass
class PipelineResult:
    """
    Result of the complete pipeline.
    
    Attributes:
        experiments: Dictionary mapping experiment name to results
        df_train: Preprocessed training data
        df_test: Preprocessed test data
        triplet_manager: Fitted TripletManager
        ground_truth: Ground truth mapping
    """
    experiments: Dict[str, ExperimentResult] = field(default_factory=dict)
    df_train: Optional[pd.DataFrame] = None
    df_test: Optional[pd.DataFrame] = None
    triplet_manager: Optional[TripletManager] = None
    ground_truth: Optional[Dict[UserID, List[TripletID]]] = None


# =============================================================================
# Data Preparation Functions
# =============================================================================

def prepare_data(
    train_path: Path,
    test_path: Path,
    max_services: int = 3,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, TripletManager]:
    """
    Load and prepare data with triplets.
    
    Steps:
    1. Load CSV files
    2. Preprocess data (clean columns, parse client size)
    3. Fit TripletManager on training data
    4. Add triplet column to both datasets
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        max_services: Maximum services per triplet
        verbose: Print progress information
    
    Returns:
        Tuple of (df_train, df_test, triplet_manager)
    
    Example:
        >>> df_train, df_test, manager = prepare_data(train_path, test_path)
        >>> print(f"Train triplets: {df_train['triplet'].nunique()}")
    """
    if verbose:
        print("=" * 80)
        print("STEP 1: Loading and preprocessing data")
        print("=" * 80)
    
    # Load and preprocess
    df_train = preprocess_data(train_path)
    df_test = preprocess_data(test_path)
    
    if verbose:
        print(f"Train set: {len(df_train)} rows, "
              f"{df_train['linkedin_company_outsource'].nunique()} users")
        print(f"Test set: {len(df_test)} rows, "
              f"{df_test['linkedin_company_outsource'].nunique()} users")
    
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: Creating triplets")
        print("=" * 80)
    
    # Initialize and fit TripletManager on TRAINING data only
    triplet_manager = TripletManager(max_services=max_services)
    triplet_manager.fit(df_train, services_column='services')
    
    # Add triplet column to both datasets
    df_train = add_triplet_column(df_train, triplet_manager, column_name='triplet')
    df_test = add_triplet_column(df_test, triplet_manager, column_name='triplet')
    
    if verbose:
        print(f"\nTrain: {df_train['triplet'].nunique()} unique triplets")
        print(f"Test: {df_test['triplet'].nunique()} unique triplets")
    
    return df_train, df_test, triplet_manager


def build_ground_truth(df_test: pd.DataFrame) -> Dict[UserID, List[TripletID]]:
    """
    Build ground truth dictionary mapping user -> list of triplets.
    
    Args:
        df_test: Test DataFrame with user and triplet columns
    
    Returns:
        Dictionary mapping user_id to list of relevant triplets
    
    Example:
        >>> ground_truth = build_ground_truth(df_test)
        >>> print(f"Users with ground truth: {len(ground_truth)}")
    """
    ground_truth: Dict[UserID, List[TripletID]] = {}
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        triplet = row.get("triplet")
        
        if pd.isna(user_id) or pd.isna(triplet):
            continue
        
        if user_id not in ground_truth:
            ground_truth[user_id] = []
        
        ground_truth[user_id].append(triplet)
    
    return ground_truth


# =============================================================================
# Experiment Runner Functions
# =============================================================================

def _create_similarity_function(triplet_manager: TripletManager) -> Callable[[str, str], float]:
    """Create similarity function for partial matching."""
    def similarity_fn(triplet1: str, triplet2: str) -> float:
        return triplet_manager.calculate_triplet_similarity(triplet1, triplet2)
    return similarity_fn


def _evaluate_recommendations(
    recommendations: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    top_k: int,
    use_partial_match: bool,
    partial_match_threshold: float,
    verbose: bool
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate recommendations with both exact and partial matching.
    
    Returns:
        Tuple of (exact_summary, per_user_exact, partial_summary, per_user_partial)
    """
    # Exact match evaluation
    if verbose:
        print("\n--- Evaluation: EXACT MATCH ---")
    
    benchmark_exact = BenchmarkEvaluator(recommendations, df_test)
    summary_exact, per_user_exact = benchmark_exact.evaluate(
        k=top_k,
        use_partial_match=False
    )
    
    if verbose:
        print(summary_exact)
    
    # Partial match evaluation
    if verbose:
        print("\n--- Evaluation: PARTIAL MATCH ---")
    
    similarity_fn = _create_similarity_function(triplet_manager)
    benchmark_partial = BenchmarkEvaluator(
        recommendations, 
        df_test, 
        similarity_fn=similarity_fn
    )
    summary_partial, per_user_partial = benchmark_partial.evaluate(
        k=top_k,
        use_partial_match=use_partial_match,
        partial_match_threshold=partial_match_threshold
    )
    
    if verbose:
        print(summary_partial)
    
    return summary_exact, per_user_exact, summary_partial, per_user_partial


def _generate_recommendations(
    recommender: Any,
    df_test: pd.DataFrame,
    top_k: int,
    mode: str = 'test',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate recommendations for all test users.
    
    Args:
        recommender: Fitted recommender object
        df_test: Test DataFrame
        top_k: Number of recommendations per user
        mode: Recommendation mode
        verbose: Print progress
    
    Returns:
        DataFrame with columns [linkedin_company_outsource, triplet, score]
    """
    if verbose:
        print("\nGenerating recommendations...")
    
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        
        if pd.isna(user_id) or user_id in seen_users:
            continue
        
        seen_users.add(user_id)
        
        try:
            recs = recommender.recommend_triplets(user_id, top_k=top_k, mode=mode)
            
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'triplet': rec['triplet'],
                    'score': rec['score']
                })
        except Exception as e:
            if verbose:
                print(f"Error for user {str(user_id)[:20]}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"Generated {len(results_df)} recommendations for {len(seen_users)} users")
    
    return results_df


def run_triplet_content_experiment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    config: PipelineConfig
) -> ExperimentResult:
    """
    Run Triplet Content-Based Recommendation experiment.
    
    Uses item features (industry, services, background) to build
    user profiles and match against candidates.
    
    Args:
        df_train: Training data
        df_test: Test data
        triplet_manager: Fitted TripletManager
        config: Pipeline configuration
    
    Returns:
        ExperimentResult with metrics and recommendations
    """
    name = "Triplet Content-Based"
    verbose = config.experiment.verbose
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {name}")
        print(f"Using {'OpenAI' if config.embedding.use_openai else 'SentenceTransformers'} embeddings")
        print("=" * 80)
    
    try:
        # Build recommender
        recommender = TripletContentRecommender(
            df_history=df_train,
            df_test=df_test,
            triplet_manager=triplet_manager,
            use_openai=config.embedding.use_openai,
            openai_model=config.embedding.openai_model
        )
        
        # Generate recommendations
        recommendations = _generate_recommendations(
            recommender, df_test, config.experiment.top_k, verbose=verbose
        )
        
        # Evaluate
        exact_summary, per_user_exact, partial_summary, per_user_partial = _evaluate_recommendations(
            recommendations, df_test, triplet_manager,
            config.experiment.top_k,
            config.experiment.use_partial_match,
            config.experiment.partial_match_threshold,
            verbose
        )
        
        # Save results
        if config.experiment.save_results:
            recommendations.to_csv(
                config.output_dir / 'recommend_triplet.csv', 
                index=False
            )
            exact_summary.to_csv(
                config.output_dir / "triplet_content_exact.csv", 
                index=False
            )
            partial_summary.to_csv(
                config.output_dir / "triplet_content_partial.csv", 
                index=False
            )
            per_user_exact.to_csv(
                config.output_dir / "triplet_content_per_user_exact.csv", 
                index=False
            )
            per_user_partial.to_csv(
                config.output_dir / "triplet_content_per_user_partial.csv", 
                index=False
            )
        
        return ExperimentResult(
            name=name,
            exact_summary=exact_summary,
            partial_summary=partial_summary,
            recommendations=recommendations,
            per_user_exact=per_user_exact,
            per_user_partial=per_user_partial
        )
        
    except Exception as e:
        traceback.print_exc()
        return ExperimentResult(name=name, error=str(e))


def run_enhanced_content_experiment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    config: PipelineConfig
) -> ExperimentResult:
    """
    Run Enhanced Triplet Content-Based Recommendation experiment.
    
    Uses advanced embeddings (SentenceTransformers or OpenAI) with
    multi-modal features and industry hierarchy clustering.
    
    Args:
        df_train: Training data
        df_test: Test data
        triplet_manager: Fitted TripletManager
        config: Pipeline configuration
    
    Returns:
        ExperimentResult with metrics and recommendations
    """
    name = "Enhanced Triplet Content-Based"
    verbose = config.experiment.verbose
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {name}")
        print(f"Using {'OpenAI' if config.embedding.use_openai else 'SentenceTransformers'} embeddings")
        print("=" * 80)
    
    try:
        # Build recommender
        recommender = EnhancedTripletContentRecommender(
            df_history=df_train,
            df_test=df_test,
            triplet_manager=triplet_manager,
            embedding_config={
                'sentence_model_name': config.embedding.sentence_model,
                'embedding_dim': 384,
                'use_industry_hierarchy': True,
                'fusion_method': 'concat'
            },
            use_openai=config.embedding.use_openai,
            openai_model=config.embedding.openai_model
        )
        
        # Generate recommendations
        recommendations = _generate_recommendations(
            recommender, df_test, config.experiment.top_k, verbose=verbose
        )
        
        # Evaluate
        exact_summary, per_user_exact, partial_summary, per_user_partial = _evaluate_recommendations(
            recommendations, df_test, triplet_manager,
            config.experiment.top_k,
            config.experiment.use_partial_match,
            config.experiment.partial_match_threshold,
            verbose
        )
        
        # Save results
        if config.experiment.save_results:
            recommendations.to_csv(
                config.output_dir / 'recommend_triplet_enhanced_content.csv', 
                index=False
            )
            exact_summary.to_csv(
                config.output_dir / "enhanced_triplet_content_exact.csv", 
                index=False
            )
            partial_summary.to_csv(
                config.output_dir / "enhanced_triplet_content_partial.csv", 
                index=False
            )
            per_user_exact.to_csv(
                config.output_dir / "enhanced_triplet_content_per_user_exact.csv", 
                index=False
            )
            per_user_partial.to_csv(
                config.output_dir / "enhanced_triplet_content_per_user_partial.csv", 
                index=False
            )
        
        return ExperimentResult(
            name=name,
            exact_summary=exact_summary,
            partial_summary=partial_summary,
            recommendations=recommendations,
            per_user_exact=per_user_exact,
            per_user_partial=per_user_partial
        )
        
    except Exception as e:
        traceback.print_exc()
        return ExperimentResult(name=name, error=str(e))


def run_user_collaborative_experiment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    config: PipelineConfig
) -> ExperimentResult:
    """
    Run User-Based Collaborative Filtering experiment.
    
    Recommends based on similar users' interactions.
    
    Args:
        df_train: Training data
        df_test: Test data
        triplet_manager: Fitted TripletManager
        config: Pipeline configuration
    
    Returns:
        ExperimentResult with metrics and recommendations
    """
    name = "User-Based Collaborative Filtering"
    verbose = config.experiment.verbose
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {name}")
        print(f"Using {'OpenAI' if config.embedding.use_openai else 'SentenceTransformers'} embeddings")
        print("=" * 80)
    
    try:
        # Build and fit recommender
        recommender = UserBasedCollaborativeRecommender(
            min_similarity=0.1,
            top_k_similar_users=20,
            use_openai=config.embedding.use_openai,
            openai_model=config.embedding.openai_model
        )
        recommender.fit(df_history=df_train, df_user_info=None)
        
        # Generate recommendations
        if verbose:
            print("\nGenerating recommendations...")
        
        results = []
        seen_users = set()
        
        for _, row in df_test.iterrows():
            user_id = row.get("linkedin_company_outsource")
            
            if pd.isna(user_id) or user_id in seen_users:
                continue
            
            seen_users.add(user_id)
            
            try:
                recs = recommender.recommend_triplets(
                    user_id, 
                    top_k=config.experiment.top_k, 
                    exclude_seen=True
                )
                
                for _, rec in recs.iterrows():
                    results.append({
                        'linkedin_company_outsource': user_id,
                        'triplet': rec['triplet'],
                        'score': rec['score']
                    })
            except Exception as e:
                if verbose:
                    print(f"Error for user {user_id}: {e}")
                continue
        
        recommendations = pd.DataFrame(results)
        
        if verbose:
            print(f"Generated {len(recommendations)} recommendations for {len(seen_users)} users")
        
        # Evaluate
        exact_summary, per_user_exact, partial_summary, per_user_partial = _evaluate_recommendations(
            recommendations, df_test, triplet_manager,
            config.experiment.top_k,
            config.experiment.use_partial_match,
            config.experiment.partial_match_threshold,
            verbose
        )
        
        # Save results
        if config.experiment.save_results:
            recommendations.to_csv(
                config.output_dir / 'recommend_triplet_collab.csv', 
                index=False
            )
            exact_summary.to_csv(
                config.output_dir / "user_collab_exact.csv", 
                index=False
            )
            partial_summary.to_csv(
                config.output_dir / "user_collab_partial.csv", 
                index=False
            )
            per_user_exact.to_csv(
                config.output_dir / "user_collab_per_user_exact.csv", 
                index=False
            )
            per_user_partial.to_csv(
                config.output_dir / "user_collab_per_user_partial.csv", 
                index=False
            )
        
        return ExperimentResult(
            name=name,
            exact_summary=exact_summary,
            partial_summary=partial_summary,
            recommendations=recommendations,
            per_user_exact=per_user_exact,
            per_user_partial=per_user_partial
        )
        
    except Exception as e:
        traceback.print_exc()
        return ExperimentResult(name=name, error=str(e))


def run_enhanced_collaborative_experiment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    config: PipelineConfig
) -> ExperimentResult:
    """
    Run Enhanced User-Based Collaborative Filtering experiment.
    
    Uses both user profile features and interaction history for similarity.
    
    Args:
        df_train: Training data
        df_test: Test data
        triplet_manager: Fitted TripletManager
        config: Pipeline configuration
    
    Returns:
        ExperimentResult with metrics and recommendations
    """
    name = "Enhanced User-Based Collaborative"
    verbose = config.experiment.verbose
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {name}")
        print("(Profile + History Similarity)")
        print(f"Using {'OpenAI' if config.embedding.use_openai else 'SentenceTransformers'} embeddings")
        print("=" * 80)
    
    try:
        # Build and fit recommender
        recommender = EnhancedUserCollaborativeRecommender(
            min_similarity=0.1,
            top_k_similar_users=30,
            profile_weight=0.4,
            history_weight=0.6,
            use_openai=config.embedding.use_openai,
            openai_model=config.embedding.openai_model
        )
        recommender.fit(df_history=df_train, df_user_info=None)
        
        # Generate recommendations
        if verbose:
            print("\nGenerating recommendations...")
        
        results = []
        seen_users = set()
        
        for _, row in df_test.iterrows():
            user_id = row.get("linkedin_company_outsource")
            
            if pd.isna(user_id) or user_id in seen_users:
                continue
            
            seen_users.add(user_id)
            
            try:
                recs = recommender.recommend_triplets(
                    user_id, 
                    top_k=config.experiment.top_k, 
                    exclude_seen=True
                )
                
                for _, rec in recs.iterrows():
                    results.append({
                        'linkedin_company_outsource': user_id,
                        'triplet': rec['triplet'],
                        'score': rec['score']
                    })
            except Exception as e:
                if verbose:
                    print(f"Error for user {user_id}: {e}")
                continue
        
        recommendations = pd.DataFrame(results)
        
        if verbose:
            print(f"Generated {len(recommendations)} recommendations for {len(seen_users)} users")
        
        # Evaluate
        exact_summary, per_user_exact, partial_summary, per_user_partial = _evaluate_recommendations(
            recommendations, df_test, triplet_manager,
            config.experiment.top_k,
            config.experiment.use_partial_match,
            config.experiment.partial_match_threshold,
            verbose
        )
        
        # Save results
        if config.experiment.save_results:
            recommendations.to_csv(
                config.output_dir / 'recommend_triplet_enhanced_collab.csv', 
                index=False
            )
            exact_summary.to_csv(
                config.output_dir / "enhanced_collab_exact.csv", 
                index=False
            )
            partial_summary.to_csv(
                config.output_dir / "enhanced_collab_partial.csv", 
                index=False
            )
            per_user_exact.to_csv(
                config.output_dir / "enhanced_collab_per_user_exact.csv", 
                index=False
            )
            per_user_partial.to_csv(
                config.output_dir / "enhanced_collab_per_user_partial.csv", 
                index=False
            )
        
        return ExperimentResult(
            name=name,
            exact_summary=exact_summary,
            partial_summary=partial_summary,
            recommendations=recommendations,
            per_user_exact=per_user_exact,
            per_user_partial=per_user_partial
        )
        
    except Exception as e:
        traceback.print_exc()
        return ExperimentResult(name=name, error=str(e))


def run_triplet_ensemble_experiment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    ground_truth: Dict[UserID, List[TripletID]],
    config: PipelineConfig
) -> ExperimentResult:
    """
    Run Triplet Ensemble experiment with Gradient Boosting meta-learner.
    
    Combines multiple base models using learned weights.
    
    Args:
        df_train: Training data
        df_test: Test data
        triplet_manager: Fitted TripletManager
        ground_truth: Ground truth mapping
        config: Pipeline configuration
    
    Returns:
        ExperimentResult with metrics and recommendations
    """
    name = "Triplet Ensemble (Gradient Boosting)"
    verbose = config.experiment.verbose
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {name}")
        print(f"Using {'OpenAI' if config.embedding.use_openai else 'SentenceTransformers'} embeddings")
        print("=" * 80)
    
    try:
        # Build ensemble
        ensemble = TripletEnsembleRecommender(
            triplet_manager=triplet_manager,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_openai=config.embedding.use_openai,
            openai_model=config.embedding.openai_model
        )
        
        # Fit ensemble
        ensemble.fit(df_train, df_test, ground_truth)
        
        # Generate recommendations
        recommendations = _generate_recommendations(
            ensemble, df_test, config.experiment.top_k, verbose=verbose
        )
        
        # Evaluate
        exact_summary, per_user_exact, partial_summary, per_user_partial = _evaluate_recommendations(
            recommendations, df_test, triplet_manager,
            config.experiment.top_k,
            config.experiment.use_partial_match,
            config.experiment.partial_match_threshold,
            verbose
        )
        
        # Save results
        if config.experiment.save_results:
            recommendations.to_csv(
                config.output_dir / 'recommend_triplet_ensemble.csv', 
                index=False
            )
            exact_summary.to_csv(
                config.output_dir / "triplet_ensemble_exact.csv", 
                index=False
            )
            partial_summary.to_csv(
                config.output_dir / "triplet_ensemble_partial.csv", 
                index=False
            )
            per_user_exact.to_csv(
                config.output_dir / "triplet_ensemble_per_user_exact.csv", 
                index=False
            )
            per_user_partial.to_csv(
                config.output_dir / "triplet_ensemble_per_user_partial.csv", 
                index=False
            )
        
        return ExperimentResult(
            name=name,
            exact_summary=exact_summary,
            partial_summary=partial_summary,
            recommendations=recommendations,
            per_user_exact=per_user_exact,
            per_user_partial=per_user_partial
        )
        
    except Exception as e:
        traceback.print_exc()
        return ExperimentResult(name=name, error=str(e))


def run_hybrid_ensemble_experiment(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    content_results: pd.DataFrame,
    collab_results: pd.DataFrame,
    config: PipelineConfig,
    weights: Tuple[float, float] = (0.7, 0.3)
) -> ExperimentResult:
    """
    Run Hybrid Ensemble experiment (weighted combination).
    
    Simple weighted average of content-based and collaborative scores.
    
    Args:
        df_train: Training data
        df_test: Test data
        triplet_manager: Fitted TripletManager
        content_results: Results from content-based experiment
        collab_results: Results from collaborative experiment
        config: Pipeline configuration
        weights: (content_weight, collab_weight)
    
    Returns:
        ExperimentResult with metrics and recommendations
    """
    name = "Hybrid Ensemble"
    verbose = config.experiment.verbose
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {name}")
        print(f"Weights: Content={weights[0]}, Collaborative={weights[1]}")
        print("=" * 80)
    
    try:
        # Merge results
        content_df = content_results.copy()
        collab_df = collab_results.copy()
        
        content_df = content_df.rename(columns={'score': 'score_content'})
        collab_df = collab_df.rename(columns={'score': 'score_collab'})
        
        # Merge on user and triplet
        merged = pd.merge(
            content_df,
            collab_df,
            on=['linkedin_company_outsource', 'triplet'],
            how='outer'
        )
        
        # Fill missing scores with 0
        merged['score_content'] = merged['score_content'].fillna(0)
        merged['score_collab'] = merged['score_collab'].fillna(0)
        
        # Compute ensemble score
        merged['score'] = (
            weights[0] * merged['score_content'] + 
            weights[1] * merged['score_collab']
        )
        
        # Keep only needed columns
        recommendations = merged[['linkedin_company_outsource', 'triplet', 'score']].copy()
        
        # Re-rank per user
        recommendations = (
            recommendations
            .sort_values(['linkedin_company_outsource', 'score'], ascending=[True, False])
            .groupby('linkedin_company_outsource')
            .head(config.experiment.top_k)
            .reset_index(drop=True)
        )
        
        if verbose:
            print(f"Generated {len(recommendations)} ensemble recommendations")
        
        # Evaluate
        exact_summary, per_user_exact, partial_summary, per_user_partial = _evaluate_recommendations(
            recommendations, df_test, triplet_manager,
            config.experiment.top_k,
            config.experiment.use_partial_match,
            config.experiment.partial_match_threshold,
            verbose
        )
        
        # Save results
        if config.experiment.save_results:
            recommendations.to_csv(
                config.output_dir / 'recommend_triplet_hybrid.csv', 
                index=False
            )
            exact_summary.to_csv(
                config.output_dir / "hybrid_ensemble_exact.csv", 
                index=False
            )
            partial_summary.to_csv(
                config.output_dir / "hybrid_ensemble_partial.csv", 
                index=False
            )
            per_user_exact.to_csv(
                config.output_dir / "hybrid_ensemble_per_user_exact.csv", 
                index=False
            )
            per_user_partial.to_csv(
                config.output_dir / "hybrid_ensemble_per_user_partial.csv", 
                index=False
            )
        
        return ExperimentResult(
            name=name,
            exact_summary=exact_summary,
            partial_summary=partial_summary,
            recommendations=recommendations,
            per_user_exact=per_user_exact,
            per_user_partial=per_user_partial
        )
        
    except Exception as e:
        traceback.print_exc()
        return ExperimentResult(name=name, error=str(e))


# =============================================================================
# Main Pipeline Function
# =============================================================================

def run_pipeline(config: Optional[PipelineConfig] = None) -> PipelineResult:
    """
    Run the complete recommendation pipeline.
    
    Steps:
    1. Prepare data (load, preprocess, create triplets)
    2. Build ground truth
    3. Run selected experiments
    4. Generate comparison report
    
    Args:
        config: Pipeline configuration (uses defaults if None)
    
    Returns:
        PipelineResult with all experiment results
    
    Example:
        >>> config = PipelineConfig.default()
        >>> config.experiment.top_k = 20
        >>> results = run_pipeline(config)
        >>> print(results.experiments['triplet_content'].exact_summary)
    """
    if config is None:
        config = PipelineConfig.default()
    
    # Validate configuration
    config.validate()
    
    result = PipelineResult()
    
    print("=" * 80)
    print("TRIPLET-BASED RECOMMENDATION SYSTEM")
    print("=" * 80)
    
    # Step 1: Prepare data
    df_train, df_test, triplet_manager = prepare_data(
        config.train_path,
        config.test_path,
        verbose=config.experiment.verbose
    )
    
    result.df_train = df_train
    result.df_test = df_test
    result.triplet_manager = triplet_manager
    
    # Save test data with triplets
    if config.experiment.save_results:
        df_test.to_csv(config.output_dir / 'recommend_test_triplet.csv')
    
    # Step 2: Build ground truth
    ground_truth = build_ground_truth(df_test)
    result.ground_truth = ground_truth
    
    if config.experiment.verbose:
        print(f"\nGround truth: {len(ground_truth)} users")
    
    # Step 3: Run experiments
    experiment_runners = {
        ExperimentType.TRIPLET_CONTENT: lambda: run_triplet_content_experiment(
            df_train, df_test, triplet_manager, config
        ),
        ExperimentType.ENHANCED_CONTENT: lambda: run_enhanced_content_experiment(
            df_train, df_test, triplet_manager, config
        ),
        ExperimentType.USER_COLLABORATIVE: lambda: run_user_collaborative_experiment(
            df_train, df_test, triplet_manager, config
        ),
        ExperimentType.ENHANCED_COLLABORATIVE: lambda: run_enhanced_collaborative_experiment(
            df_train, df_test, triplet_manager, config
        ),
        ExperimentType.TRIPLET_ENSEMBLE: lambda: run_triplet_ensemble_experiment(
            df_train, df_test, triplet_manager, ground_truth, config
        ),
    }
    
    # Run each experiment
    content_result = None
    collab_result = None
    
    for exp_type in config.experiments_to_run:
        if exp_type == ExperimentType.HYBRID_ENSEMBLE:
            # Hybrid needs content and collab results
            if content_result and collab_result:
                exp_result = run_hybrid_ensemble_experiment(
                    df_train, df_test, triplet_manager,
                    content_result.recommendations,
                    collab_result.recommendations,
                    config
                )
            else:
                if config.experiment.verbose:
                    print(f"\nSkipping {exp_type.value}: requires content and collab results")
                continue
        elif exp_type in experiment_runners:
            exp_result = experiment_runners[exp_type]()
        else:
            continue
        
        result.experiments[exp_type.value] = exp_result
        
        # Track for hybrid ensemble
        if exp_type == ExperimentType.TRIPLET_CONTENT:
            content_result = exp_result
        elif exp_type == ExperimentType.USER_COLLABORATIVE:
            collab_result = exp_result
    
    # Step 4: Print comparison
    _print_comparison(result, config)
    
    return result


def _print_comparison(result: PipelineResult, config: PipelineConfig):
    """Print final comparison of all experiments."""
    if not config.experiment.verbose:
        return
    
    top_k = config.experiment.top_k
    
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    print("\n{:<30} {:>10} {:>10} {:>10} {:>10}".format(
        "Method", "Precision", "Recall", "MAP", "HitRate"
    ))
    print("-" * 75)
    
    for match_type in ["EXACT MATCH", "PARTIAL MATCH"]:
        print(f"\n{match_type}:")
        
        for name, exp in result.experiments.items():
            if not exp.success:
                continue
            
            if match_type == "EXACT MATCH":
                summary = exp.exact_summary
            else:
                summary = exp.partial_summary
            
            if summary is not None and not summary.empty:
                row = summary.iloc[0]
                print("{:<30} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    exp.name[:30],
                    row.get(f"Precision@{top_k}", 0),
                    row.get(f"Recall@{top_k}", 0),
                    row.get(f"MAP@{top_k}", 0),
                    row.get(f"HitRate@{top_k}", 0)
                ))
    
    print("\n" + "=" * 80)
    print(f"All results saved to: {config.output_dir}")
    print("=" * 80)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for command-line execution."""
    # Create default configuration
    config = PipelineConfig.default()
    
    # You can customize configuration here:
    # config.experiment.top_k = 1700
    # config.experiments_to_run = [ExperimentType.TRIPLET_CONTENT, ExperimentType.ENHANCED_CONTENT]
    
    # Run pipeline
    results = run_pipeline(config)
    
    return results


if __name__ == "__main__":
    main()
