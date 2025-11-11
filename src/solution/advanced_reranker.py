"""
Advanced Reranking Strategy for Recommendation System
===================================================

This module implements sophisticated reranking techniques to improve recommendation quality:
1. Learning-to-Rank (LTR) with LightGBM
2. Feature engineering for reranking
3. Business rule integration
4. Diversity optimization
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedReranker:
    """
    Advanced reranking system that combines multiple signals to optimize recommendation quality.
    
    Features used for reranking:
    1. Base recommendation scores (content-based, collaborative, fusion)
    2. User-item interaction features
    3. Popularity and freshness signals
    4. Diversity metrics
    5. Business constraints
    """
    
    def __init__(
        self,
        diversity_weight: float = 0.15,
        popularity_weight: float = 0.1,
        freshness_weight: float = 0.05,
        business_boost: float = 0.1,
        min_diversity_threshold: float = 0.3,
        lgb_params: Optional[Dict] = None
    ):
        self.diversity_weight = diversity_weight
        self.popularity_weight = popularity_weight  
        self.freshness_weight = freshness_weight
        self.business_boost = business_boost
        self.min_diversity_threshold = min_diversity_threshold
        
        # LightGBM parameters for learning-to-rank
        self.lgb_params = lgb_params or {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10],
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.model = None
        self.feature_scaler = StandardScaler()
        self.industry_encoder = LabelEncoder()
        self.location_encoder = LabelEncoder()
        self.is_fitted = False
        
    def extract_features(
        self,
        candidates: pd.DataFrame,
        user_history: pd.DataFrame,
        user_id: str,
        base_scores: Dict[str, float],
        industry_stats: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Extract comprehensive features for reranking candidates.
        
        Args:
            candidates: DataFrame with candidate items and their metadata
            user_history: Historical interactions of the user  
            user_id: User identifier
            base_scores: Dict mapping industry -> base recommendation score
            industry_stats: Global industry statistics (popularity, etc.)
            
        Returns:
            DataFrame with extracted features for each candidate
        """
        features = []
        
        # Get user profile from history
        user_industries = set(user_history['industry'].values) if not user_history.empty else set()
        user_locations = set(user_history['location'].values) if not user_history.empty else set()
        user_services = ' '.join(user_history['services'].fillna('').values) if not user_history.empty else ''
        
        # Calculate user interaction stats
        user_avg_client_size = user_history['client_size_mid'].mean() if not user_history.empty else 0
        user_avg_budget = user_history['project_budget_mid'].mean() if not user_history.empty else 0
        user_interaction_count = len(user_history)
        
        for _, candidate in candidates.iterrows():
            industry = candidate['industry']
            location = candidate.get('location', '')
            services = candidate.get('services', '')
            
            # 1. Base recommendation scores
            base_score = base_scores.get(industry, 0.0)
            
            # 2. User-item affinity features
            industry_familiarity = 1.0 if industry in user_industries else 0.0
            location_familiarity = 1.0 if location in user_locations else 0.0
            
            # Services similarity (simple word overlap)
            if user_services and services:
                user_services_set = set(user_services.lower().split())
                candidate_services_set = set(services.lower().split())
                services_overlap = len(user_services_set & candidate_services_set) / max(len(user_services_set), 1)
            else:
                services_overlap = 0.0
                
            # 3. Popularity and global stats
            if industry_stats:
                industry_popularity = industry_stats.get(industry, {}).get('popularity', 0.0)
                industry_avg_rating = industry_stats.get(industry, {}).get('avg_rating', 0.0)
                industry_project_count = industry_stats.get(industry, {}).get('project_count', 0)
            else:
                industry_popularity = 0.0
                industry_avg_rating = 0.0
                industry_project_count = 0
                
            # 4. Client size and budget compatibility
            candidate_client_size = candidate.get('client_size_mid', 0)
            candidate_budget = candidate.get('project_budget_mid', 0)
            
            client_size_match = 1.0 if abs(candidate_client_size - user_avg_client_size) < user_avg_client_size * 0.5 else 0.0
            budget_match = 1.0 if abs(candidate_budget - user_avg_budget) < user_avg_budget * 0.5 else 0.0
            
            # 5. Freshness (assume recent projects are better)
            # This would need actual timestamp data - using a placeholder
            freshness_score = 1.0  # Placeholder
            
            # 6. User experience level (more interactions = more experienced)
            user_experience = min(user_interaction_count / 10.0, 1.0)
            
            feature_row = {
                'user_id': user_id,
                'industry': industry,
                'base_score': base_score,
                'industry_familiarity': industry_familiarity,
                'location_familiarity': location_familiarity,
                'services_overlap': services_overlap,
                'industry_popularity': industry_popularity,
                'industry_avg_rating': industry_avg_rating,
                'industry_project_count': industry_project_count,
                'client_size_match': client_size_match,
                'budget_match': budget_match,
                'freshness_score': freshness_score,
                'user_experience': user_experience,
                'user_interaction_count': user_interaction_count
            }
            
            features.append(feature_row)
            
        return pd.DataFrame(features)
    
    def calculate_diversity_score(self, recommended_industries: List[str]) -> float:
        """
        Calculate diversity score of recommendation list.
        Higher score = more diverse recommendations.
        """
        if len(recommended_industries) <= 1:
            return 0.0
            
        # Simple diversity: ratio of unique industries
        unique_ratio = len(set(recommended_industries)) / len(recommended_industries)
        return unique_ratio
    
    def apply_business_rules(
        self, 
        candidates: pd.DataFrame, 
        user_history: pd.DataFrame,
        boost_factors: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Apply business rules and constraints to boost/penalize certain recommendations.
        """
        candidates = candidates.copy()
        
        if boost_factors is None:
            # Default business boosts
            boost_factors = {
                'IT': 1.2,  # IT services are high-value
                'Consulting': 1.1,  # Consulting has good margins
                'eCommerce': 0.9,   # eCommerce might be competitive
            }
            
        # Apply industry-specific boosts
        for industry, boost in boost_factors.items():
            mask = candidates['industry'] == industry
            candidates.loc[mask, 'base_score'] *= boost
            
        # Penalize over-representation in user history
        if not user_history.empty:
            user_top_industries = user_history['industry'].value_counts().head(3).index
            for industry in user_top_industries:
                mask = candidates['industry'] == industry
                candidates.loc[mask, 'base_score'] *= 0.8  # Small penalty for over-exposure
                
        return candidates
    
    def fit(
        self,
        training_data: pd.DataFrame,
        user_histories: Dict[str, pd.DataFrame],
        ground_truth: Dict[str, List[str]]
    ):
        """
        Fit the learning-to-rank model using training data.
        
        Args:
            training_data: DataFrame with user_id, industry, base_score, etc.
            user_histories: Dict mapping user_id -> historical interactions DataFrame
            ground_truth: Dict mapping user_id -> list of relevant industries
        """
        print("Fitting advanced reranking model...")
        
        # Prepare training dataset
        X_list = []
        y_list = []
        group_list = []
        
        for user_id, relevant_items in ground_truth.items():
            if user_id not in user_histories:
                continue
                
            user_data = training_data[training_data['user_id'] == user_id].copy()
            if user_data.empty:
                continue
                
            # Create relevance labels (1 if in ground truth, 0 otherwise)
            user_data['relevance'] = user_data['industry'].apply(
                lambda x: 1 if x in relevant_items else 0
            )
            
            if user_data['relevance'].sum() == 0:  # Skip if no positive examples
                continue
                
            # Extract features for this user's candidates
            user_history = user_histories[user_id]
            base_scores = dict(zip(user_data['industry'], user_data['base_score']))
            
            # Calculate industry stats for this training batch
            industry_stats = self._calculate_industry_stats(training_data)
            
            features_df = self.extract_features(
                user_data, user_history, user_id, base_scores, industry_stats
            )
            
            # Prepare features for LightGBM
            feature_cols = [col for col in features_df.columns 
                          if col not in ['user_id', 'industry']]
            
            X_user = features_df[feature_cols].values
            y_user = user_data['relevance'].values
            
            X_list.append(X_user)
            y_list.append(y_user)
            group_list.append(len(X_user))
            
        if not X_list:
            raise ValueError("No valid training data found")
            
        # Combine all training data
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        groups = np.array(group_list)
        
        # Encode categorical features
        categorical_features = []
        if 'industry' in training_data.columns:
            self.industry_encoder.fit(training_data['industry'])
            categorical_features.append('industry_encoded')
        
        # Scale numerical features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X_train_scaled, 
            label=y_train, 
            group=groups,
            categorical_feature=categorical_features
        )
        
        # Train model
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.is_fitted = True
        print("Reranking model training completed!")
        
    def _calculate_industry_stats(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate global statistics for each industry."""
        stats = {}
        
        industry_groups = data.groupby('industry')
        total_projects = len(data)
        
        for industry, group in industry_groups:
            stats[industry] = {
                'popularity': len(group) / total_projects,
                'avg_rating': group.get('rating', pd.Series([0])).mean(),
                'project_count': len(group)
            }
            
        return stats
    
    def rerank(
        self,
        candidates: pd.DataFrame,
        user_history: pd.DataFrame, 
        user_id: str,
        base_scores: Dict[str, float],
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Rerank candidates using the trained model and business rules.
        
        Args:
            candidates: DataFrame with candidate items
            user_history: User's historical interactions
            user_id: User identifier  
            base_scores: Base recommendation scores
            top_k: Number of items to return
            
        Returns:
            Reranked candidates DataFrame
        """
        if candidates.empty:
            return candidates
            
        # Apply business rules first
        candidates = self.apply_business_rules(candidates, user_history)
        
        # If model is fitted, use ML reranking
        if self.is_fitted and self.model is not None:
            # Calculate industry stats (in production, this would be pre-computed)
            industry_stats = {}  # Simplified for demo
            
            # Extract features
            features_df = self.extract_features(
                candidates, user_history, user_id, base_scores, industry_stats
            )
            
            # Prepare features for prediction
            feature_cols = [col for col in features_df.columns 
                          if col not in ['user_id', 'industry']]
            
            X = features_df[feature_cols].values
            X_scaled = self.feature_scaler.transform(X)
            
            # Predict relevance scores
            ml_scores = self.model.predict(X_scaled)
            candidates = candidates.copy()
            candidates['ml_score'] = ml_scores
            
            # Combine base score with ML score
            candidates['final_score'] = (
                0.7 * candidates['base_score'] + 0.3 * candidates['ml_score']
            )
        else:
            # Fallback to rule-based reranking
            candidates = candidates.copy()
            candidates['final_score'] = candidates['base_score']
            
        # Apply diversity optimization
        candidates_sorted = candidates.sort_values('final_score', ascending=False)
        
        # Ensure minimum diversity
        final_candidates = self._ensure_diversity(candidates_sorted, top_k)
        
        return final_candidates.head(top_k)
    
    def _ensure_diversity(self, candidates: pd.DataFrame, top_k: int) -> pd.DataFrame:
        """
        Ensure recommendation list meets minimum diversity requirements.
        Uses a greedy approach to balance relevance and diversity.
        """
        if len(candidates) <= top_k:
            return candidates
            
        selected = []
        remaining = candidates.copy()
        selected_industries = set()
        
        # Always include the top candidate
        if not remaining.empty:
            top_candidate = remaining.iloc[0]
            selected.append(top_candidate)
            selected_industries.add(top_candidate['industry'])
            remaining = remaining.iloc[1:]
        
        # Greedily add remaining candidates balancing score and diversity
        while len(selected) < top_k and not remaining.empty:
            best_idx = 0
            best_score = -1
            
            for idx, candidate in remaining.iterrows():
                # Base relevance score
                relevance = candidate['final_score']
                
                # Diversity bonus (higher if industry not seen)
                diversity_bonus = 0
                if candidate['industry'] not in selected_industries:
                    diversity_bonus = self.diversity_weight
                
                # Combined score
                combined_score = relevance + diversity_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            # Add best candidate
            selected_candidate = remaining.loc[best_idx]
            selected.append(selected_candidate)
            selected_industries.add(selected_candidate['industry'])
            remaining = remaining.drop(best_idx)
        
        return pd.DataFrame(selected)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the trained model."""
        if not self.is_fitted or self.model is None:
            return None
            
        importance = self.model.feature_importance(importance_type='gain')
        # Note: feature names would need to be tracked during training
        # This is a simplified version
        return dict(enumerate(importance))


def integrate_advanced_reranking(
    content_app,
    collab_app, 
    df_test: pd.DataFrame,
    df_history: pd.DataFrame,
    top_k: int = 10,
    enable_learning_to_rank: bool = False
) -> pd.DataFrame:
    """
    Integration function to apply advanced reranking to existing recommendation pipeline.
    
    Args:
        content_app: Fitted content-based recommender
        collab_app: Fitted collaborative recommender
        df_test: Test dataset with users to get recommendations for
        df_history: Historical interaction data
        top_k: Number of recommendations per user
        enable_learning_to_rank: Whether to train and use LTR model
        
    Returns:
        DataFrame with reranked recommendations
    """
    reranker = AdvancedReranker()
    results = []
    
    # If LTR is enabled, we would need to fit the model first
    # This is simplified for demo - in practice you'd need proper train/test split
    
    seen_users = set()
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        if pd.isna(user_id) or user_id in seen_users:
            continue
        seen_users.add(user_id)
        
        # Get base recommendations from both models
        try:
            cb_recs = content_app.recommend_items(user_id, top_k=top_k*2)  # Get more for reranking
            cf_recs = collab_app.recommend_items(user_id, top_k=top_k*2)
            
            # Combine and create base scores
            all_industries = set(cb_recs['industry'].tolist() + cf_recs['industry'].tolist())
            base_scores = {}
            
            # Simple fusion of content-based and collaborative scores
            for industry in all_industries:
                cb_score = cb_recs[cb_recs['industry'] == industry]['score'].iloc[0] if industry in cb_recs['industry'].values else 0
                cf_score = cf_recs[cf_recs['industry'] == industry]['score'].iloc[0] if industry in cf_recs['industry'].values else 0
                base_scores[industry] = 0.6 * cb_score + 0.4 * cf_score
            
            # Create candidates DataFrame
            candidates_list = []
            for industry in all_industries:
                # Get metadata from content-based results (more complete)
                if industry in cb_recs['industry'].values:
                    metadata = cb_recs[cb_recs['industry'] == industry].iloc[0]
                else:
                    metadata = cf_recs[cf_recs['industry'] == industry].iloc[0]
                
                candidates_list.append({
                    'industry': industry,
                    'location': metadata.get('location', ''),
                    'services': metadata.get('example_services', ''),
                    'project_description': metadata.get('example_project', ''),
                    'base_score': base_scores[industry]
                })
            
            candidates_df = pd.DataFrame(candidates_list)
            
            # Get user history
            user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
            
            # Apply reranking
            reranked = reranker.rerank(
                candidates_df, user_history, user_id, base_scores, top_k
            )
            
            # Format results
            for _, rec in reranked.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'industry': rec['industry'],
                    'score': rec.get('final_score', rec.get('base_score', 0))
                })
                
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue
    
    return pd.DataFrame(results)