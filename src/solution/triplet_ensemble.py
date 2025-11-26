"""
Triplet-based Advanced Ensemble
================================

Ensemble methods for triplet recommendation combining:
1. Triplet Content-Based (using item features)
2. Enhanced Triplet Content-Based (using Sentence Transformers or OpenAI)
3. User-Based Collaborative (using interaction history similarity)
4. Enhanced User Collaborative (using user profile + history similarity)

Uses Gradient Boosting as meta-learner to combine predictions.

NEW: Support for OpenAI embeddings (use_openai=True)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from triplet_utils import TripletManager

# Global config for embedding type
USE_OPENAI_EMBEDDINGS = True  # Set to False to use SentenceTransformers
OPENAI_MODEL = 'text-embedding-3-small'  # or 'text-embedding-3-large'


class TripletEnsembleRecommender:
    """
    Advanced Ensemble for Triplet Recommendation.
    
    Combines multiple base models using Gradient Boosting as meta-learner:
    1. TripletContentRecommender - Basic content-based
    2. EnhancedTripletContentRecommender - Advanced content with SentenceTransformers
    3. UserBasedCollaborativeRecommender - Basic user CF
    4. EnhancedUserCollaborativeRecommender - User CF with profile similarity
    """
    
    def __init__(
        self,
        triplet_manager: TripletManager,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
        self.triplet_manager = triplet_manager
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.use_openai = use_openai
        self.openai_model = openai_model
        
        # Meta-learner
        self.meta_learner = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            subsample=0.8,
            min_samples_leaf=10
        )
        
        self.scaler = StandardScaler()
        
        # Base models
        self.content_model = None
        self.enhanced_content_model = None  # NEW: Enhanced content-based
        self.user_collab_model = None
        self.enhanced_collab_model = None
        
        # Training data
        self.df_train = None
        self.df_test = None
        
        # Feature statistics
        self.feature_stats = {}
        
        self.is_fitted = False
    
    def fit(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        ground_truth: Dict[str, List[str]],
        validation_split: float = 0.2
    ):
        """
        Fit ensemble model.
        
        Args:
            df_train: Training data with triplet column
            df_test: Test/candidate data with triplet column
            ground_truth: {user_id: [relevant_triplets]}
            validation_split: Fraction for validation
        """
        print("=" * 80)
        print("FITTING TRIPLET ENSEMBLE")
        print("=" * 80)
        
        self.df_train = df_train.copy()
        self.df_test = df_test.copy()
        
        # Split train/val users for meta-learner training
        train_users = list(ground_truth.keys())
        np.random.seed(self.random_state)
        np.random.shuffle(train_users)
        
        n_val = int(len(train_users) * validation_split)
        val_users = set(train_users[:n_val])
        fit_users = set(train_users[n_val:])
        
        print(f"Users for base model fitting: {len(fit_users)}")
        print(f"Users for meta-learner training: {len(val_users)}")
        
        # 1. Fit base models
        self._fit_base_models(df_train, df_test)
        
        # 2. Prepare meta-learner training data
        X_meta, y_meta = self._prepare_meta_training_data(
            val_users, ground_truth
        )
        
        if len(X_meta) == 0:
            print("Warning: No meta-training data available. Using default weights.")
            self.is_fitted = False
            return self
        
        # 3. Scale features
        X_meta_scaled = self.scaler.fit_transform(X_meta)
        
        # 4. Fit meta-learner
        print("\nFitting meta-learner...")
        self.meta_learner.fit(X_meta_scaled, y_meta)
        
        # Print feature importances
        feature_names = [
            'content_score', 'enhanced_content_score', 'user_collab_score', 'enhanced_collab_score',
            'score_variance', 'max_min_spread', 'user_history_count',
            'triplet_popularity', 'industry_match', 'size_match', 'service_overlap',
            'n_models_recommending'
        ]
        
        importances = self.meta_learner.feature_importances_
        print("\nFeature Importances:")
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.4f}")
        
        self.is_fitted = True
        print("\nEnsemble fitting completed!")
        
        return self
    
    def _fit_base_models(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        """Fit all base recommendation models."""
        print("\nFitting base models...")
        print(f"Using {'OpenAI' if self.use_openai else 'SentenceTransformers'} embeddings")
        
        # 1. Basic Content-based
        print("  - Fitting Triplet Content-Based...")
        from solution.triplet_recommender import TripletContentRecommender
        self.content_model = TripletContentRecommender(
            df_history=df_train,
            df_test=df_test,
            triplet_manager=self.triplet_manager,
            use_openai=self.use_openai,
            openai_model=self.openai_model
        )
        
        # 2. Enhanced Content-based with SentenceTransformers/OpenAI
        print("  - Fitting Enhanced Triplet Content-Based...")
        try:
            from solution.enhanced_triplet_content import EnhancedTripletContentRecommender
            self.enhanced_content_model = EnhancedTripletContentRecommender(
                df_history=df_train,
                df_test=df_test,
                triplet_manager=self.triplet_manager,
                embedding_config={
                    'sentence_model_name': 'all-MiniLM-L6-v2',
                    'embedding_dim': 384,
                    'use_industry_hierarchy': True,
                    'fusion_method': 'concat'
                },
                use_openai=self.use_openai,
                openai_model=self.openai_model
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ! Enhanced Content-Based failed: {e}")
            self.enhanced_content_model = None
        
        # 3. User-based Collaborative
        print("  - Fitting User-Based Collaborative...")
        from solution.user_collaborative import UserBasedCollaborativeRecommender
        self.user_collab_model = UserBasedCollaborativeRecommender(
            min_similarity=0.1,
            top_k_similar_users=20,
            use_openai=self.use_openai,
            openai_model=self.openai_model
        )
        self.user_collab_model.fit(df_history=df_train, df_user_info=None)
        
        # 4. Enhanced User Collaborative (profile-based)
        print("  - Fitting Enhanced User Collaborative...")
        from solution.enhanced_user_collaborative import EnhancedUserCollaborativeRecommender
        self.enhanced_collab_model = EnhancedUserCollaborativeRecommender(
            min_similarity=0.1,
            top_k_similar_users=30,
            profile_weight=0.4,
            history_weight=0.6,
            use_openai=self.use_openai,
            openai_model=self.openai_model
        )
        self.enhanced_collab_model.fit(df_history=df_train, df_user_info=None)
        
        # Calculate feature statistics for normalization
        self._calculate_feature_stats(df_train)
        
        n_models = sum([
            self.content_model is not None,
            self.enhanced_content_model is not None,
            self.user_collab_model is not None,
            self.enhanced_collab_model is not None
        ])
        print(f"Base models fitted! ({n_models} models)")
    
    def _calculate_feature_stats(self, df_train: pd.DataFrame):
        """Calculate statistics for feature normalization."""
        # User history counts
        user_counts = df_train.groupby('linkedin_company_outsource').size()
        self.feature_stats['user_count_mean'] = user_counts.mean()
        self.feature_stats['user_count_std'] = user_counts.std() + 1e-6
        
        # Triplet popularity
        if 'triplet' in df_train.columns:
            triplet_counts = df_train.groupby('triplet').size()
            self.feature_stats['triplet_pop_mean'] = triplet_counts.mean()
            self.feature_stats['triplet_pop_std'] = triplet_counts.std() + 1e-6
    
    def _prepare_meta_training_data(
        self,
        val_users: set,
        ground_truth: Dict[str, List[str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for meta-learner.
        
        Creates feature vectors for (user, triplet) pairs with binary relevance labels.
        """
        print("\nPreparing meta-learner training data...")
        
        X_meta = []
        y_meta = []
        
        for user_id in val_users:
            if user_id not in ground_truth:
                continue
            
            gt_triplets = set(ground_truth[user_id])
            
            # Get base predictions
            base_preds = self._get_base_predictions(user_id, top_k=50, mode='val')
            
            if not base_preds:
                continue
            
            # Get all candidate triplets
            all_triplets = set()
            for model_preds in base_preds.values():
                all_triplets.update(model_preds.keys())
            
            # Also add some ground truth triplets
            all_triplets.update(gt_triplets)
            
            # Get user context features
            user_history = self.df_train[self.df_train['linkedin_company_outsource'] == user_id]
            user_history_count = len(user_history)
            
            for triplet in all_triplets:
                # Create feature vector
                features = self._create_meta_features(
                    user_id, triplet, base_preds, user_history_count
                )
                
                # Create label (1 if relevant, 0 otherwise)
                label = 1.0 if triplet in gt_triplets else 0.0
                
                X_meta.append(features)
                y_meta.append(label)
        
        print(f"Created {len(X_meta)} training samples")
        print(f"Positive samples: {sum(y_meta)}, Negative: {len(y_meta) - sum(y_meta)}")
        
        return np.array(X_meta), np.array(y_meta)
    
    def _get_base_predictions(
        self,
        user_id: str,
        top_k: int = 50,
        mode: Literal['val', 'test'] = 'test'
    ) -> Dict[str, Dict[str, float]]:
        """Get predictions from all base models."""
        predictions = {}
        
        # 1. Basic Content-based
        try:
            content_recs = self.content_model.recommend_triplets(user_id, top_k=top_k, mode=mode)
            predictions['content'] = dict(zip(content_recs['triplet'], content_recs['score']))
        except Exception as e:
            predictions['content'] = {}
        
        # 2. Enhanced Content-based (SentenceTransformers)
        if self.enhanced_content_model is not None:
            try:
                enhanced_recs = self.enhanced_content_model.recommend_triplets(user_id, top_k=top_k, mode=mode)
                predictions['enhanced_content'] = dict(zip(enhanced_recs['triplet'], enhanced_recs['score']))
            except Exception as e:
                predictions['enhanced_content'] = {}
        
        # 3. User Collaborative
        try:
            collab_recs = self.user_collab_model.recommend_triplets(user_id, top_k=top_k)
            predictions['user_collab'] = dict(zip(collab_recs['triplet'], collab_recs['score']))
        except Exception as e:
            predictions['user_collab'] = {}
        
        # 4. Enhanced User Collaborative
        try:
            enhanced_collab_recs = self.enhanced_collab_model.recommend_triplets(user_id, top_k=top_k)
            predictions['enhanced_collab'] = dict(zip(enhanced_collab_recs['triplet'], enhanced_collab_recs['score']))
        except Exception as e:
            predictions['enhanced_collab'] = {}
        
        return predictions
    
    def _create_meta_features(
        self,
        user_id: str,
        triplet: str,
        base_preds: Dict[str, Dict[str, float]],
        user_history_count: int
    ) -> np.ndarray:
        """
        Create meta-features for a (user, triplet) pair.
        
        Features (12 features):
        1. content_score: Score from basic content-based model
        2. enhanced_content_score: Score from enhanced content-based model (SentenceTransformers)
        3. user_collab_score: Score from user collaborative model
        4. enhanced_collab_score: Score from enhanced collaborative model
        5. score_variance: Variance across model scores (uncertainty)
        6. max_min_spread: Max - Min score (agreement)
        7. user_history_count: Number of user's past interactions (normalized)
        8. triplet_popularity: How popular is this triplet (normalized)
        9. industry_match: Whether industry appears in user's history
        10. size_match: Whether size bucket appears in user's history
        11. service_overlap: Overlap of services with user's history
        12. n_models_recommending: Number of models that recommend this triplet
        """
        features = []
        
        # 1-4. Base model scores
        content_score = base_preds.get('content', {}).get(triplet, 0.0)
        enhanced_content_score = base_preds.get('enhanced_content', {}).get(triplet, 0.0)
        collab_score = base_preds.get('user_collab', {}).get(triplet, 0.0)
        enhanced_collab_score = base_preds.get('enhanced_collab', {}).get(triplet, 0.0)
        
        features.extend([content_score, enhanced_content_score, collab_score, enhanced_collab_score])
        
        # 5-6. Cross-model statistics
        scores = [content_score, enhanced_content_score, collab_score, enhanced_collab_score]
        non_zero_scores = [s for s in scores if s > 0]
        
        score_variance = np.var(scores)
        max_min_spread = max(scores) - min(scores)
        
        features.extend([score_variance, max_min_spread])
        
        # 7. User history count (normalized)
        user_count_norm = (user_history_count - self.feature_stats.get('user_count_mean', 0)) / \
                         self.feature_stats.get('user_count_std', 1)
        features.append(user_count_norm)
        
        # 8. Triplet popularity
        triplet_pop = self.user_collab_model.triplet_popularity.get(triplet, 0) if \
                     self.user_collab_model else 0
        triplet_pop_norm = (triplet_pop - self.feature_stats.get('triplet_pop_mean', 0)) / \
                          self.feature_stats.get('triplet_pop_std', 1)
        features.append(triplet_pop_norm)
        
        # 9-11. Triplet component matching with user history
        industry_match, size_match, service_overlap = self._calculate_triplet_user_match(
            user_id, triplet
        )
        features.extend([industry_match, size_match, service_overlap])
        
        # 12. Number of models recommending this triplet
        n_models_recommending = len(non_zero_scores)
        features.append(n_models_recommending / 4.0)  # Normalized by total models
        
        return np.array(features)
    
    def _calculate_triplet_user_match(
        self,
        user_id: str,
        triplet: str
    ) -> Tuple[float, float, float]:
        """
        Calculate how well a triplet matches user's historical preferences.
        
        Returns:
            (industry_match, size_match, service_overlap)
        """
        # Parse triplet
        industry, size, services = self.triplet_manager.parse_triplet(triplet)
        
        # Get user history
        user_history = self.df_train[self.df_train['linkedin_company_outsource'] == user_id]
        
        if user_history.empty:
            return 0.0, 0.0, 0.0
        
        # Industry match (fraction of history with same industry)
        industry_counts = user_history['industry'].value_counts()
        total = len(user_history)
        industry_match = industry_counts.get(industry, 0) / total
        
        # Size match (based on client size bucket)
        size_match = 0.0
        size_counts = defaultdict(int)
        
        for _, row in user_history.iterrows():
            if 'client_min' in row and 'client_max' in row:
                if pd.notna(row['client_min']) and pd.notna(row['client_max']):
                    mid = (row['client_min'] + row['client_max']) / 2
                    
                    if mid <= 10:
                        bucket = 'micro'
                    elif mid <= 50:
                        bucket = 'small'
                    elif mid <= 200:
                        bucket = 'medium'
                    elif mid <= 1000:
                        bucket = 'large'
                    else:
                        bucket = 'enterprise'
                    
                    size_counts[bucket] += 1
        
        if size_counts:
            size_match = size_counts.get(size, 0) / sum(size_counts.values())
        
        # Service overlap (Jaccard similarity)
        triplet_services = set(s.strip() for s in services.split(',') if s.strip() and s != 'unknown')
        
        user_services = set()
        for svc_str in user_history['services'].dropna():
            for s in str(svc_str).split(','):
                if s.strip():
                    user_services.add(s.strip())
        
        if triplet_services and user_services:
            intersection = len(triplet_services & user_services)
            union = len(triplet_services | user_services)
            service_overlap = intersection / union if union > 0 else 0.0
        else:
            service_overlap = 0.0
        
        return industry_match, size_match, service_overlap
    
    def recommend_triplets(
        self,
        user_id: str,
        top_k: int = 10,
        mode: Literal['val', 'test'] = 'test'
    ) -> pd.DataFrame:
        """
        Generate triplet recommendations using ensemble.
        """
        # Get base predictions
        base_preds = self._get_base_predictions(user_id, top_k=top_k * 5, mode=mode)
        
        if not any(base_preds.values()):
            # Fallback to popularity
            return self._recommend_by_popularity(top_k)
        
        # Get all candidate triplets
        all_triplets = set()
        for model_preds in base_preds.values():
            all_triplets.update(model_preds.keys())
        
        if not all_triplets:
            return self._recommend_by_popularity(top_k)
        
        # Get user context
        user_history = self.df_train[self.df_train['linkedin_company_outsource'] == user_id]
        user_history_count = len(user_history)
        
        # Score each triplet
        triplet_scores = {}
        
        for triplet in all_triplets:
            if self.is_fitted:
                # Use meta-learner
                features = self._create_meta_features(
                    user_id, triplet, base_preds, user_history_count
                )
                features_scaled = self.scaler.transform([features])
                score = self.meta_learner.predict(features_scaled)[0]
            else:
                # Fallback to weighted average (include enhanced_content)
                content_score = base_preds.get('content', {}).get(triplet, 0.0)
                enhanced_content_score = base_preds.get('enhanced_content', {}).get(triplet, 0.0)
                collab_score = base_preds.get('user_collab', {}).get(triplet, 0.0)
                enhanced_collab_score = base_preds.get('enhanced_collab', {}).get(triplet, 0.0)
                
                # Default weights: emphasize enhanced models
                score = (0.25 * content_score + 
                        0.35 * enhanced_content_score +  # Higher weight for enhanced content
                        0.15 * collab_score + 
                        0.25 * enhanced_collab_score)
            
            triplet_scores[triplet] = score
        
        # Sort by score
        sorted_triplets = sorted(
            triplet_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = pd.DataFrame([
            {'triplet': triplet, 'score': score}
            for triplet, score in sorted_triplets
        ])
        
        return results
    
    def _recommend_by_popularity(self, top_k: int) -> pd.DataFrame:
        """Fallback: recommend most popular triplets."""
        if self.user_collab_model:
            sorted_triplets = sorted(
                self.user_collab_model.triplet_popularity.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            if sorted_triplets:
                max_count = sorted_triplets[0][1]
                results = pd.DataFrame([
                    {'triplet': triplet, 'score': count / max_count}
                    for triplet, count in sorted_triplets
                ])
                return results
        
        return pd.DataFrame(columns=['triplet', 'score'])


def run_triplet_ensemble_experiment(
    data_path: str,
    data_test_path: str,
    top_k: int = 10
):
    """
    Run complete triplet ensemble experiment.
    """
    from preprocessing_data import full_pipeline_preprocess_data
    from benchmark_data import BenchmarkOutput
    from triplet_utils import TripletManager, add_triplet_column
    
    print("=" * 80)
    print("TRIPLET ENSEMBLE EXPERIMENT")
    print("=" * 80)
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df_train = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    
    print(f"Train set: {len(df_train)} rows, {df_train['linkedin_company_outsource'].nunique()} users")
    print(f"Test set: {len(df_test)} rows, {df_test['linkedin_company_outsource'].nunique()} users")
    
    # 2. Create triplets
    print("\n2. Creating triplets...")
    triplet_manager = TripletManager(max_services=3)
    triplet_manager.fit(df_train, services_column='services')
    
    df_train = add_triplet_column(df_train, triplet_manager, column_name='triplet')
    df_test = add_triplet_column(df_test, triplet_manager, column_name='triplet')
    
    print(f"Train: {df_train['triplet'].nunique()} unique triplets")
    print(f"Test: {df_test['triplet'].nunique()} unique triplets")
    
    # 3. Build ground truth
    print("\n3. Building ground truth...")
    ground_truth = {}
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        triplet = row.get("triplet")
        
        if pd.isna(user_id) or pd.isna(triplet):
            continue
        
        if user_id not in ground_truth:
            ground_truth[user_id] = []
        ground_truth[user_id].append(triplet)
    
    print(f"Ground truth: {len(ground_truth)} users")
    
    # 4. Fit ensemble
    print("\n4. Fitting ensemble...")
    ensemble = TripletEnsembleRecommender(
        triplet_manager=triplet_manager,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
    ensemble.fit(df_train, df_test, ground_truth)
    
    # 5. Generate recommendations
    print("\n5. Generating recommendations...")
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
            print(f"Error for user {user_id}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    print(f"Generated {len(results_df)} recommendations for {len(seen_users)} users")
    
    # 6. Evaluate
    print("\n6. Evaluating...")
    
    # Exact match
    print("\n--- Evaluation: EXACT MATCH ---")
    benchmark_exact = BenchmarkOutput(results_df, df_test)
    summary_exact, per_user_exact = benchmark_exact.evaluate_topk(
        k=top_k,
        use_partial_match=False
    )
    print(summary_exact)
    
    # Partial match
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
    
    # 7. Save results
    out_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
    summary_exact.to_csv(out_dir + "triplet_ensemble_exact.csv", index=False)
    summary_partial.to_csv(out_dir + "triplet_ensemble_partial.csv", index=False)
    results_df.to_csv(out_dir + "triplet_ensemble_recommendations.csv", index=False)
    
    print(f"\nResults saved to {out_dir}")
    
    return {
        'exact': summary_exact,
        'partial': summary_partial,
        'results': results_df
    }


if __name__ == "__main__":
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_0_100_update.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_0_100_update_test.csv"
    
    run_triplet_ensemble_experiment(data_path, data_test_path, top_k=10)
