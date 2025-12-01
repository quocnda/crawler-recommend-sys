"""
Triplet-based Recommendation Pipeline
======================================

Execute and evaluate triplet-based recommendations.

Experiments:
1. Triplet Content-Based Recommendation
2. User-Based Collaborative Filtering
3. Enhanced User Collaborative (with profile similarity)
4. Triplet Ensemble (Gradient Boosting meta-learner)
5. Hybrid Ensemble (weighted combination)

NEW: Support for OpenAI embeddings (use_openai=True)
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from preprocessing_data import full_pipeline_preprocess_data
from benchmark_data import BenchmarkOutput

# Base directories (relative to this file's location)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BENCHMARK_DIR = DATA_DIR / "benchmark"
from triplet_utils import TripletManager, add_triplet_column
from solution.triplet_recommender import TripletContentRecommender
from solution.user_collaborative import UserBasedCollaborativeRecommender
from solution.enhanced_user_collaborative import EnhancedUserCollaborativeRecommender
from solution.enhanced_triplet_content import EnhancedTripletContentRecommender
from solution.triplet_ensemble import TripletEnsembleRecommender
import warnings
warnings.filterwarnings('ignore')

# Global config for embedding type
USE_OPENAI_EMBEDDINGS = True  # Set to False to use SentenceTransformers
OPENAI_MODEL = 'text-embedding-3-small'  # or 'text-embedding-3-large'


def prepare_triplet_data(
    data_path: str,
    data_test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, TripletManager]:
    """
    Load and prepare data with triplets.
    
    Returns:
        (df_train, df_test, triplet_manager)
    """
    print("=" * 80)
    print("STEP 1: Loading and preprocessing data")
    print("=" * 80)
    
    # Load and preprocess
    df_train = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    # df_train = df_train[:2]
    print(f"Train set: {len(df_train)} rows, {df_train['linkedin_company_outsource'].nunique()} users")
    print(f"Test set: {len(df_test)} rows, {df_test['linkedin_company_outsource'].nunique()} users")
    
    print("\n" + "=" * 80)
    print("STEP 2: Creating triplets")
    print("=" * 80)
    
    # Initialize and fit TripletManager on TRAINING data only (avoid data leakage)
    triplet_manager = TripletManager(max_services=3)
    triplet_manager.fit(df_train, services_column='services')
    
    # Add triplet column to both datasets
    df_train = add_triplet_column(df_train, triplet_manager, column_name='triplet')
    df_test = add_triplet_column(df_test, triplet_manager, column_name='triplet')

    
    # Statistics
    print(f"\nTrain: {df_train['triplet'].nunique()} unique triplets")
    print(f"Test: {df_test['triplet'].nunique()} unique triplets")
    
    return df_train, df_test, triplet_manager


def build_ground_truth_triplets(df_test: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build ground truth dictionary mapping user -> list of triplets.
    """
    ground_truth = {}
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        triplet = row.get("triplet")
        
        if pd.isna(user_id) or pd.isna(triplet):
            continue
        
        if user_id not in ground_truth:
            ground_truth[user_id] = []
        
        ground_truth[user_id].append(triplet)
    
    return ground_truth


def experiment_triplet_content_based(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    ground_truth: Dict[str, List[str]],
    top_k: int = 10
):
    """
    Experiment 1: Triplet-based Content Recommendation
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Triplet Content-Based Recommendation")
    print(f"Using {'OpenAI' if USE_OPENAI_EMBEDDINGS else 'SentenceTransformers'} embeddings")
    print("=" * 80)
    
    # Build recommender
    recommender = TripletContentRecommender(
        df_history=df_train,
        df_test=df_test,
        triplet_manager=triplet_manager,
        use_openai=USE_OPENAI_EMBEDDINGS,
        openai_model=OPENAI_MODEL
    )
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        
        if pd.isna(user_id) or user_id in seen_users:
            continue
        
        seen_users.add(user_id)
        
        try:
            recs = recommender.recommend_triplets(user_id, top_k=top_k, mode='test')
            
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'triplet': rec['triplet'],
                    'score': rec['score']
                })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error for user {user_id[:10]}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(BENCHMARK_DIR / 'recommend_triplet.csv', index=False)
    print(f"Generated {len(results_df)} recommendations for {len(seen_users)} users")
    
    # Evaluate with exact match
    print("\n--- Evaluation: EXACT MATCH ---")
    benchmark_exact = BenchmarkOutput(results_df, df_test)
    summary_exact, per_user_exact = benchmark_exact.evaluate_topk(
        k=top_k,
        use_partial_match=False
    )
    print(summary_exact)
    
    # Evaluate with partial match
    print("\n--- Evaluation: PARTIAL MATCH ---")
    
    def similarity_fn(triplet1: str, triplet2: str) -> float:
        return triplet_manager.calculate_triplet_similarity(triplet1, triplet2)
    
    benchmark_partial = BenchmarkOutput(results_df, df_test, similarity_fn=similarity_fn)
    summary_partial, per_user_partial = benchmark_partial.evaluate_topk(
        k=top_k,
        use_partial_match=True,
        partial_match_threshold=0.5
    )
    print(summary_partial)
    
    # Save results
    summary_exact.to_csv(BENCHMARK_DIR / "triplet_content_exact.csv", index=False)
    summary_partial.to_csv(BENCHMARK_DIR / "triplet_content_partial.csv", index=False)
    per_user_exact.to_csv(BENCHMARK_DIR / "triplet_content_per_user_exact.csv", index=False)
    per_user_partial.to_csv(BENCHMARK_DIR / "triplet_content_per_user_partial.csv", index=False)
    
    return {
        'exact': summary_exact,
        'partial': summary_partial,
        'results': results_df
    }


def experiment_enhanced_triplet_content(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    ground_truth: Dict[str, List[str]],
    top_k: int = 10
):
    """
    Experiment 1b: Enhanced Triplet Content-Based Recommendation
    
    Uses SentenceTransformers + multi-modal embeddings (text + categorical + numerical)
    with industry hierarchy clustering.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1b: Enhanced Triplet Content-Based (SentenceTransformers)")
    print(f"Using {'OpenAI' if USE_OPENAI_EMBEDDINGS else 'SentenceTransformers'} embeddings")
    print("=" * 80)
    
    # Build recommender
    recommender = EnhancedTripletContentRecommender(
        df_history=df_train,
        df_test=df_test,
        triplet_manager=triplet_manager,
        embedding_config={
            'sentence_model_name': 'all-MiniLM-L6-v2',
            'embedding_dim': 384,
            'use_industry_hierarchy': True,
            'fusion_method': 'concat'
        },
        use_openai=USE_OPENAI_EMBEDDINGS,
        openai_model=OPENAI_MODEL
    )
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        
        if pd.isna(user_id) or user_id in seen_users:
            continue
        
        seen_users.add(user_id)
        
        try:
            recs = recommender.recommend_triplets(user_id, top_k=top_k, mode='test')
            
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'triplet': rec['triplet'],
                    'score': rec['score']
                })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error for user {user_id[:10]}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(BENCHMARK_DIR / 'recommend_triplet_enhanced_content.csv', index=False)
    print(f"Generated {len(results_df)} recommendations for {len(seen_users)} users")
    
    # Evaluate with exact match
    print("\n--- Evaluation: EXACT MATCH ---")
    benchmark_exact = BenchmarkOutput(results_df, df_test)
    summary_exact, per_user_exact = benchmark_exact.evaluate_topk(
        k=top_k,
        use_partial_match=False
    )
    print(summary_exact)
    
    # Evaluate with partial match
    print("\n--- Evaluation: PARTIAL MATCH ---")
    
    def similarity_fn(triplet1: str, triplet2: str) -> float:
        return triplet_manager.calculate_triplet_similarity(triplet1, triplet2)
    
    benchmark_partial = BenchmarkOutput(results_df, df_test, similarity_fn=similarity_fn)
    summary_partial, per_user_partial = benchmark_partial.evaluate_topk(
        k=top_k,
        use_partial_match=True,
        partial_match_threshold=0.5
    )
    print(summary_partial)
    
    # Save results
    summary_exact.to_csv(BENCHMARK_DIR / "enhanced_triplet_content_exact.csv", index=False)
    summary_partial.to_csv(BENCHMARK_DIR / "enhanced_triplet_content_partial.csv", index=False)
    per_user_exact.to_csv(BENCHMARK_DIR / "enhanced_triplet_content_per_user_exact.csv", index=False)
    per_user_partial.to_csv(BENCHMARK_DIR / "enhanced_triplet_content_per_user_partial.csv", index=False)
    
    return {
        'exact': summary_exact,
        'partial': summary_partial,
        'results': results_df
    }


def experiment_user_collaborative(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    ground_truth: Dict[str, List[str]],
    top_k: int = 10
):
    """
    Experiment 2: User-Based Collaborative Filtering
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: User-Based Collaborative Filtering")
    print(f"Using {'OpenAI' if USE_OPENAI_EMBEDDINGS else 'SentenceTransformers'} embeddings")
    print("=" * 80)
    
    # Build recommender
    recommender = UserBasedCollaborativeRecommender(
        min_similarity=0.1,
        top_k_similar_users=20,
        use_openai=USE_OPENAI_EMBEDDINGS,
        openai_model=OPENAI_MODEL
    )
    
    # Fit on training data
    recommender.fit(df_history=df_train, df_user_info=None)
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        
        if pd.isna(user_id) or user_id in seen_users:
            continue
        
        seen_users.add(user_id)
        
        try:
            recs = recommender.recommend_triplets(user_id, top_k=top_k, exclude_seen=True)
            
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'triplet': rec['triplet'],
                    'score': rec['score']
                })
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(BENCHMARK_DIR / 'recommend_triplet_collab.csv', index=False)
    print(f"Generated {len(results_df)} recommendations for {len(seen_users)} users")
    
    # Evaluate with exact match
    print("\n--- Evaluation: EXACT MATCH ---")
    benchmark_exact = BenchmarkOutput(results_df, df_test)
    summary_exact, per_user_exact = benchmark_exact.evaluate_topk(
        k=top_k,
        use_partial_match=False
    )
    print(summary_exact)
    
    # Evaluate with partial match
    print("\n--- Evaluation: PARTIAL MATCH ---")
    
    def similarity_fn(triplet1: str, triplet2: str) -> float:
        return triplet_manager.calculate_triplet_similarity(triplet1, triplet2)
    
    benchmark_partial = BenchmarkOutput(results_df, df_test, similarity_fn=similarity_fn)
    summary_partial, per_user_partial = benchmark_partial.evaluate_topk(
        k=top_k,
        use_partial_match=True,
        partial_match_threshold=0.5
    )
    print(summary_partial)
    
    # Save results
    summary_exact.to_csv(BENCHMARK_DIR / "user_collab_exact.csv", index=False)
    summary_partial.to_csv(BENCHMARK_DIR / "user_collab_partial.csv", index=False)
    per_user_exact.to_csv(BENCHMARK_DIR / "user_collab_per_user_exact.csv", index=False)
    per_user_partial.to_csv(BENCHMARK_DIR / "user_collab_per_user_partial.csv", index=False)
    
    return {
        'exact': summary_exact,
        'partial': summary_partial,
        'results': results_df
    }


def experiment_hybrid_ensemble(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    content_results: pd.DataFrame,
    collab_results: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
    top_k: int = 10,
    weights: Tuple[float, float] = (0.7, 0.3)
):
    """
    Experiment 3: Hybrid Ensemble (Content + Collaborative)
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Hybrid Ensemble")
    print(f"Weights: Content={weights[0]}, Collaborative={weights[1]}")
    print("=" * 80)
    
    # Merge results
    content_results = content_results.copy()
    collab_results = collab_results.copy()
    
    content_results = content_results.rename(columns={'score': 'score_content'})
    collab_results = collab_results.rename(columns={'score': 'score_collab'})
    
    # Merge on user and triplet
    merged = pd.merge(
        content_results,
        collab_results,
        on=['linkedin_company_outsource', 'triplet'],
        how='outer'
    )
    
    # Fill missing scores with 0
    merged['score_content'] = merged['score_content'].fillna(0)
    merged['score_collab'] = merged['score_collab'].fillna(0)
    
    # Ensemble score
    merged['score'] = (
        weights[0] * merged['score_content'] + 
        weights[1] * merged['score_collab']
    )
    
    # Keep only needed columns
    results_df = merged[['linkedin_company_outsource', 'triplet', 'score']].copy()
    
    # Re-rank per user
    results_df = (
        results_df
        .sort_values(['linkedin_company_outsource', 'score'], ascending=[True, False])
        .groupby('linkedin_company_outsource')
        .head(top_k)
        .reset_index(drop=True)
    )
    results_df.to_csv(BENCHMARK_DIR / 'recommend_triplet_hybrid.csv', index=False)
    print(f"Generated {len(results_df)} ensemble recommendations")
    
    # Evaluate with exact match
    print("\n--- Evaluation: EXACT MATCH ---")
    benchmark_exact = BenchmarkOutput(results_df, df_test)
    summary_exact, per_user_exact = benchmark_exact.evaluate_topk(
        k=top_k,
        use_partial_match=False
    )
    print(summary_exact)
    
    # Evaluate with partial match
    print("\n--- Evaluation: PARTIAL MATCH ---")
    
    def similarity_fn(triplet1: str, triplet2: str) -> float:
        return triplet_manager.calculate_triplet_similarity(triplet1, triplet2)
    
    benchmark_partial = BenchmarkOutput(results_df, df_test, similarity_fn=similarity_fn)
    summary_partial, per_user_partial = benchmark_partial.evaluate_topk(
        k=top_k,
        use_partial_match=True,
        partial_match_threshold=0.5
    )
    print(summary_partial)
    
    # Save results
    summary_exact.to_csv(BENCHMARK_DIR / "hybrid_ensemble_exact.csv", index=False)
    summary_partial.to_csv(BENCHMARK_DIR / "hybrid_ensemble_partial.csv", index=False)
    per_user_exact.to_csv(BENCHMARK_DIR / "hybrid_ensemble_per_user_exact.csv", index=False)
    per_user_partial.to_csv(BENCHMARK_DIR / "hybrid_ensemble_per_user_partial.csv", index=False)
    
    return {
        'exact': summary_exact,
        'partial': summary_partial,
        'results': results_df
    }


def experiment_enhanced_user_collaborative(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    ground_truth: Dict[str, List[str]],
    top_k: int = 10
):
    """
    Experiment 3: Enhanced User-Based Collaborative Filtering
    
    Uses BOTH user profile features (description, services_company_outsource)
    AND interaction history for user-user similarity.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Enhanced User-Based Collaborative Filtering")
    print("(Profile + History Similarity)")
    print(f"Using {'OpenAI' if USE_OPENAI_EMBEDDINGS else 'SentenceTransformers'} embeddings")
    print("=" * 80)
    
    # Build recommender
    recommender = EnhancedUserCollaborativeRecommender(
        min_similarity=0.1,
        top_k_similar_users=30,
        profile_weight=0.4,  # 40% weight on profile similarity
        history_weight=0.6,   # 60% weight on history similarity
        use_openai=USE_OPENAI_EMBEDDINGS,
        openai_model=OPENAI_MODEL
    )
    
    # Fit on training data
    recommender.fit(df_history=df_train, df_user_info=None)
    
    # Generate recommendations
    print("\nGenerating recommendations...")
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        
        if pd.isna(user_id) or user_id in seen_users:
            continue
        
        seen_users.add(user_id)
        
        try:
            recs = recommender.recommend_triplets(user_id, top_k=top_k, exclude_seen=True)
            
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'triplet': rec['triplet'],
                    'score': rec['score']
                })
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(BENCHMARK_DIR / 'recommend_triplet_enhanced_collab.csv', index=False)
    print(f"Generated {len(results_df)} recommendations for {len(seen_users)} users")
    
    # Evaluate with exact match
    print("\n--- Evaluation: EXACT MATCH ---")
    benchmark_exact = BenchmarkOutput(results_df, df_test)
    summary_exact, per_user_exact = benchmark_exact.evaluate_topk(
        k=top_k,
        use_partial_match=False
    )
    print(summary_exact)
    
    # Evaluate with partial match
    print("\n--- Evaluation: PARTIAL MATCH ---")
    
    def similarity_fn(triplet1: str, triplet2: str) -> float:
        return triplet_manager.calculate_triplet_similarity(triplet1, triplet2)
    
    benchmark_partial = BenchmarkOutput(results_df, df_test, similarity_fn=similarity_fn)
    summary_partial, per_user_partial = benchmark_partial.evaluate_topk(
        k=top_k,
        use_partial_match=True,
        partial_match_threshold=0.5
    )
    print(summary_partial)
    
    # Save results
    summary_exact.to_csv(BENCHMARK_DIR / "enhanced_collab_exact.csv", index=False)
    summary_partial.to_csv(BENCHMARK_DIR / "enhanced_collab_partial.csv", index=False)
    per_user_exact.to_csv(BENCHMARK_DIR / "enhanced_collab_per_user_exact.csv", index=False)
    per_user_partial.to_csv(BENCHMARK_DIR / "enhanced_collab_per_user_partial.csv", index=False)
    
    return {
        'exact': summary_exact,
        'partial': summary_partial,
        'results': results_df
    }


def experiment_triplet_ensemble(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    triplet_manager: TripletManager,
    ground_truth: Dict[str, List[str]],
    top_k: int = 10
):
    """
    Experiment 4: Triplet Ensemble with Gradient Boosting Meta-Learner
    
    Combines:
    - Triplet Content-Based
    - User Collaborative
    - Enhanced User Collaborative
    
    Using Gradient Boosting to learn optimal combination.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Triplet Ensemble (Gradient Boosting Meta-Learner)")
    print(f"Using {'OpenAI' if USE_OPENAI_EMBEDDINGS else 'SentenceTransformers'} embeddings")
    print("=" * 80)
    
    # Build ensemble
    ensemble = TripletEnsembleRecommender(
        triplet_manager=triplet_manager,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_openai=USE_OPENAI_EMBEDDINGS,
        openai_model=OPENAI_MODEL
    )
    
    # Fit ensemble
    ensemble.fit(df_train, df_test, ground_truth)
    
    # Generate recommendations
    print("\nGenerating ensemble recommendations...")
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        
        if pd.isna(user_id) or user_id in seen_users:
            continue
        
        seen_users.add(user_id)
        
        try:
            recs = ensemble.recommend_triplets(user_id, top_k=top_k, mode='test')
            
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'triplet': rec['triplet'],
                    'score': rec['score']
                })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error for user {user_id}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(BENCHMARK_DIR / 'recommend_triplet_ensemble.csv', index=False)
    print(f"Generated {len(results_df)} ensemble recommendations for {len(seen_users)} users")
    
    # Evaluate with exact match
    print("\n--- Evaluation: EXACT MATCH ---")
    benchmark_exact = BenchmarkOutput(results_df, df_test)
    summary_exact, per_user_exact = benchmark_exact.evaluate_topk(
        k=top_k,
        use_partial_match=False
    )
    print(summary_exact)
    
    # Evaluate with partial match
    print("\n--- Evaluation: PARTIAL MATCH ---")
    
    def similarity_fn(triplet1: str, triplet2: str) -> float:
        return triplet_manager.calculate_triplet_similarity(triplet1, triplet2)
    
    benchmark_partial = BenchmarkOutput(results_df, df_test, similarity_fn=similarity_fn)
    summary_partial, per_user_partial = benchmark_partial.evaluate_topk(
        k=top_k,
        use_partial_match=True,
        partial_match_threshold=0.5
    )
    print(summary_partial)
    
    # Save results
    summary_exact.to_csv(BENCHMARK_DIR / "triplet_ensemble_exact.csv", index=False)
    summary_partial.to_csv(BENCHMARK_DIR / "triplet_ensemble_partial.csv", index=False)
    per_user_exact.to_csv(BENCHMARK_DIR / "triplet_ensemble_per_user_exact.csv", index=False)
    per_user_partial.to_csv(BENCHMARK_DIR / "triplet_ensemble_per_user_partial.csv", index=False)
    
    return {
        'exact': summary_exact,
        'partial': summary_partial,
        'results': results_df
    }


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("TRIPLET-BASED RECOMMENDATION SYSTEM")
    print("=" * 80)
    
    # Paths
    data_path = DATA_DIR / "sample_0_100_update.csv"
    data_test_path = DATA_DIR / "sample_0_100_update_test.csv"
    
    # Prepare data
    df_train, df_test, triplet_manager = prepare_triplet_data(data_path, data_test_path)
    df_test.to_csv(BENCHMARK_DIR / 'recommend_test_triplet.csv')
    
    # Build ground truth
    ground_truth = build_ground_truth_triplets(df_test)
    print(f"\nGround truth: {len(ground_truth)} users")
    
    # Run experiments
    top_k = 1700
    
    # Experiment 1: Content-Based
    content_exp = experiment_triplet_content_based(
        df_train, df_test, triplet_manager, ground_truth, top_k
    )
    
    # Experiment 1b: Enhanced Content-Based (SentenceTransformers)
    enhanced_content_exp = experiment_enhanced_triplet_content(
        df_train, df_test, triplet_manager, ground_truth, top_k
    )

    # # Experiment 2: Basic User Collaborative
    collab_exp = experiment_user_collaborative(
        df_train, df_test, triplet_manager, ground_truth, top_k
    )
    
    # # Experiment 3: Enhanced User Collaborative (with profile similarity)
    enhanced_collab_exp = experiment_enhanced_user_collaborative(
        df_train, df_test, triplet_manager, ground_truth, top_k
    )
    
    # Experiment 4: Triplet Ensemble (Gradient Boosting)
    ensemble_exp = experiment_triplet_ensemble(
        df_train, df_test, triplet_manager, ground_truth, top_k
    )
    
    # Experiment 5: Simple Hybrid Ensemble (weighted average)
    hybrid_exp = experiment_hybrid_ensemble(
        df_train, df_test, triplet_manager,
        content_exp['results'],
        collab_exp['results'],
        ground_truth, top_k,
        weights=(0.7, 0.3)  # 70% content, 30% collaborative
    )
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    summary_data = []
    
    experiments = {
        'Content-Based': content_exp,
        'Enhanced Content': enhanced_content_exp,
        'User Collaborative': collab_exp,
        'Enhanced Collaborative': enhanced_collab_exp,
        'Triplet Ensemble (GB)': ensemble_exp,
        'Hybrid Ensemble': hybrid_exp
    }
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Method", "Precision", "Recall", "MAP", "HitRate"
    ))
    print("-" * 75)
    
    print("\nEXACT MATCH:")
    for name, exp in experiments.items():
        if exp and 'exact' in exp and not exp['exact'].empty:
            summary = exp['exact'].iloc[0]
            p_col = f"Precision@{top_k}"
            r_col = f"Recall@{top_k}"
            m_col = f"MAP@{top_k}"
            h_col = f"HitRate@{top_k}"
            
            print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                name,
                summary.get(p_col, 0),
                summary.get(r_col, 0),
                summary.get(m_col, 0),
                summary.get(h_col, 0)
            ))
    
    print("\nPARTIAL MATCH:")
    for name, exp in experiments.items():
        if exp and 'partial' in exp and not exp['partial'].empty:
            summary = exp['partial'].iloc[0]
            p_col = f"Precision@{top_k}"
            r_col = f"Recall@{top_k}"
            m_col = f"MAP@{top_k}"
            h_col = f"HitRate@{top_k}"
            
            print("{:<25} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
                name,
                summary.get(p_col, 0),
                summary.get(r_col, 0),
                summary.get(m_col, 0),
                summary.get(h_col, 0)
            ))
    
    print("\n" + "=" * 80)
    print(f"All results saved to: {BENCHMARK_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
