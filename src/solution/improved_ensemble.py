"""
Improved Ensemble with Advanced Features
======================================

This module combines multiple improvement strategies:
1. Enhanced Feature Engineering
2. Multi-Stage Candidate Generation
3. Diversity-Aware Ranking
4. Advanced Fusion Strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from solution.feature_engineering import AdvancedFeatureEngineer
    from solution.advanced_ensemble import AdvancedEnsembleRecommender
    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ADVANCED_COMPONENTS_AVAILABLE = False
    print("Advanced components not available. Using simplified approach.")


class MultiStageRecommender:
    """
    Multi-stage recommendation pipeline for higher recall.
    """
    
    def __init__(
        self,
        stage1_multiplier: int = 8,  # Increased from 3x to 8x
        stage2_multiplier: int = 4,  # Rerank top 4x candidates
        diversity_weight: float = 0.3,
        use_advanced_features: bool = True
    ):
        self.stage1_multiplier = stage1_multiplier
        self.stage2_multiplier = stage2_multiplier  
        self.diversity_weight = diversity_weight
        self.use_advanced_features = use_advanced_features
        
        # Initialize components
        self.base_models = {}
        self.feature_engineer = None
        self.ensemble_recommender = None
        
        if ADVANCED_COMPONENTS_AVAILABLE and use_advanced_features:
            self.feature_engineer = AdvancedFeatureEngineer()
            self.ensemble_recommender = AdvancedEnsembleRecommender()
    
    def fit(self, df_history: pd.DataFrame, df_test: pd.DataFrame, ground_truth: Dict[str, List[str]]):
        """Fit all components of the multi-stage pipeline."""
        print("Fitting Multi-Stage Recommender...")
        
        # Fit feature engineer
        if self.feature_engineer:
            print("- Fitting feature engineer...")
            self.feature_engineer.fit(df_history)
        
        # Fit base models
        self._fit_base_models(df_history, df_test)
        
        # Fit ensemble if available
        if self.ensemble_recommender:
            print("- Fitting ensemble recommender...")
            self.ensemble_recommender.fit_ensemble(df_history, df_test, ground_truth)
    
    def _fit_base_models(self, df_history: pd.DataFrame, df_test: pd.DataFrame):
        """Fit base recommendation models."""
        print("- Fitting base models...")
        
        # Content-based with OpenAI
        try:
            from solution.content_base_for_item import ContentBaseBasicApproach
            self.base_models['content_openai'] = ContentBaseBasicApproach(df_history, df_test)
            print("  ✓ Content-Based OpenAI fitted")
        except Exception as e:
            print(f"  ✗ Content-Based OpenAI failed: {e}")
        
        # Enhanced embeddings
        try:
            from solution.enhanced_embeddings import EnhancedContentBasedRecommender
            self.base_models['enhanced_embeddings'] = EnhancedContentBasedRecommender(df_history, df_test)
            print("  ✓ Enhanced Embeddings fitted")
        except Exception as e:
            print(f"  ✗ Enhanced Embeddings failed: {e}")
        
        # Collaborative filtering
        try:
            from solution.collborative_for_item import CollaborativeIndustryRecommender
            collab = CollaborativeIndustryRecommender(
                n_components=128,
                min_user_interactions=1,
                min_item_interactions=1,
                use_tfidf_weighting=True,
                random_state=42
            ).fit(df_history=df_history, df_candidates=df_test)
            self.base_models['collaborative'] = collab
            print("  ✓ Collaborative Filtering fitted")
        except Exception as e:
            print(f"  ✗ Collaborative Filtering failed: {e}")
        
        # Graph-based if available
        try:
            from solution.graph_recommendations import GraphBasedRecommender
            graph_model = GraphBasedRecommender()
            graph_model.fit(df_history)
            self.base_models['graph_based'] = graph_model
            print("  ✓ Graph-Based fitted")
        except Exception as e:
            print(f"  ✗ Graph-Based failed: {e}")
    
    def recommend_items(
        self,
        user_id: str,
        df_history: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """Generate recommendations using multi-stage pipeline."""
        
        # Stage 1: Cast wide net with high fanout
        stage1_candidates = self._stage1_candidate_generation(
            user_id, df_history, top_k * self.stage1_multiplier
        )
        
        if stage1_candidates.empty:
            return pd.DataFrame(columns=['industry', 'score'])
        
        # Stage 2: Enhanced ranking with behavioral patterns
        stage2_candidates = self._stage2_enhanced_ranking(
            user_id, df_history, stage1_candidates, top_k * self.stage2_multiplier
        )
        
        # Stage 3: Diversity-aware final selection
        final_recommendations = self._stage3_diversity_selection(
            user_id, df_history, stage2_candidates, top_k
        )
        
        return final_recommendations
    
    def _stage1_candidate_generation(
        self,
        user_id: str,
        df_history: pd.DataFrame,
        top_k: int
    ) -> pd.DataFrame:
        """Stage 1: Generate candidates with high coverage."""
        print(f"Stage 1: Generating {top_k} candidates for user {user_id[:8]}...")
        
        all_candidates = []
        model_weights = self._get_stage1_weights()
        for model_name, model in self.base_models.items():
            try:
                # Get recommendations from base model
                if model_name == 'collaborative':
                    recs = model.recommend_items(user_id, top_k=top_k//2)
                else:
                    recs = model.recommend_items(user_id, top_k=top_k//2)
                
                # Add model info and weight
                recs['model'] = model_name
                recs['base_weight'] = model_weights.get(model_name, 1.0)
                recs['weighted_score'] = recs['score'] * recs['base_weight']
                
                all_candidates.append(recs)
                
            except Exception as e:
                print(f"Error in {model_name}: {e}")
                continue
        
        if not all_candidates:
            return pd.DataFrame(columns=['industry', 'score'])
        
        # Combine and aggregate candidates
        combined = pd.concat(all_candidates, ignore_index=True)
        
        # Aggregate scores by industry (sum weighted scores)
        aggregated = combined.groupby('industry').agg({
            'weighted_score': 'sum',
            'score': 'mean',
            'model': lambda x: ','.join(x.unique())
        }).reset_index()
        
        # Sort and return top candidates
        aggregated = aggregated.sort_values('weighted_score', ascending=False).head(top_k)
        aggregated['score'] = aggregated['weighted_score']  # Use aggregated score
        
        return aggregated[['industry', 'score']].reset_index(drop=True)
    
    def _stage2_enhanced_ranking(
        self,
        user_id: str,
        df_history: pd.DataFrame,
        candidates: pd.DataFrame,
        top_k: int
    ) -> pd.DataFrame:
        """Stage 2: Enhanced ranking with user behavioral patterns."""
        print(f"Stage 2: Enhanced ranking to {top_k} candidates...")
        
        if candidates.empty:
            return candidates
        
        user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
        
        # Extract user preferences if feature engineering is available
        user_preferences = {}
        if self.feature_engineer:
            try:
                user_preferences = self.feature_engineer.extract_user_features(user_id, user_history)
            except Exception as e:
                print(f"Feature extraction error: {e}")
        
        enhanced_scores = []
        
        for _, row in candidates.iterrows():
            industry = row['industry']
            base_score = row['score']
            
            # Calculate enhancement factors
            enhancement_score = self._calculate_enhancement_score(
                industry, user_history, user_preferences
            )
            
            # Enhanced final score
            final_score = base_score * (1.0 + enhancement_score)
            enhanced_scores.append(final_score)
        
        candidates['enhanced_score'] = enhanced_scores
        
        # Re-rank by enhanced score
        candidates = candidates.sort_values('enhanced_score', ascending=False).head(top_k)
        candidates['score'] = candidates['enhanced_score']
        
        return candidates[['industry', 'score']].reset_index(drop=True)
    
    def _stage3_diversity_selection(
        self,
        user_id: str,
        df_history: pd.DataFrame,
        candidates: pd.DataFrame,
        top_k: int
    ) -> pd.DataFrame:
        """Stage 3: Diversity-aware final selection."""
        print(f"Stage 3: Diversity-aware selection to {top_k} items...")
        
        if len(candidates) <= top_k:
            return candidates
        
        # Greedy diversity selection
        selected = []
        selected_categories = set()
        remaining = candidates.copy()
        
        # Industry categories for diversity
        category_mapping = self._get_industry_categories()
        
        while len(selected) < top_k and not remaining.empty:
            best_idx = 0
            best_score = -float('inf')
            
            for idx, row in remaining.iterrows():
                industry = row['industry']
                base_score = row['score']
                
                # Diversity bonus
                category = category_mapping.get(industry, 'other')
                diversity_bonus = 0.0 if category in selected_categories else self.diversity_weight
                
                # Combined score
                total_score = base_score + diversity_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx
            
            # Select best item
            best_row = remaining.loc[best_idx]
            selected.append(best_row)
            
            # Update selected categories
            category = category_mapping.get(best_row['industry'], 'other')
            selected_categories.add(category)
            
            # Remove from remaining
            remaining = remaining.drop(best_idx)
        
        result = pd.DataFrame(selected)[['industry', 'score']].reset_index(drop=True)
        return result
    
    def _get_stage1_weights(self) -> Dict[str, float]:
        """Get weights for stage 1 model combination."""
        return {
            'content_openai': 1.2,      # Strong for content understanding
            'enhanced_embeddings': 1.0,  # Balanced approach
            'collaborative': 1.4,       # Strong for user behavior
            'graph_based': 0.8          # Supplementary
        }
    
    def _calculate_enhancement_score(
        self,
        industry: str,
        user_history: pd.DataFrame,
        user_preferences: Dict
    ) -> float:
        """Calculate enhancement score based on user behavioral patterns."""
        enhancement = 0.0
        
        if user_history.empty:
            return enhancement
        
        # Historical industry affinity
        industry_counts = user_history['industry'].value_counts()
        if industry in industry_counts:
            # Boost score for previously interacted industries
            frequency = industry_counts[industry]
            total_interactions = len(user_history)
            affinity_boost = min(frequency / total_interactions, 0.5)  # Cap at 50%
            enhancement += affinity_boost
        
        # Industry category preferences
        if user_preferences:
            tech_pref = user_preferences.get('tech_preference', 0.0)
            service_pref = user_preferences.get('service_preference', 0.0)
            
            tech_industries = {'Software', 'IT Services', 'Information technology'}
            service_industries = {'Consulting', 'Business services'}
            
            if industry in tech_industries:
                enhancement += tech_pref * 0.3
            elif industry in service_industries:
                enhancement += service_pref * 0.3
        
        # Complexity alignment
        if user_preferences and 'avg_service_complexity' in user_preferences:
            user_complexity = user_preferences['avg_service_complexity']
            
            # Estimate industry complexity (simplified)
            high_complexity = {'Software', 'IT Services', 'Information technology'}
            medium_complexity = {'Consulting', 'Financial services'}
            
            if industry in high_complexity:
                item_complexity = 0.8
            elif industry in medium_complexity:
                item_complexity = 0.6
            else:
                item_complexity = 0.4
            
            # Boost if complexity matches
            complexity_diff = abs(user_complexity - item_complexity)
            if complexity_diff < 0.2:
                enhancement += 0.15
        
        return enhancement
    
    def _get_industry_categories(self) -> Dict[str, str]:
        """Map industries to broader categories for diversity."""
        return {
            'Software': 'tech',
            'IT Services': 'tech',
            'Information technology': 'tech',
            'Telecommunications': 'tech',
            'Consulting': 'services',
            'Business services': 'services',
            'Professional services': 'services',
            'Financial services': 'finance',
            'Banking': 'finance',
            'Insurance': 'finance',
            'Healthcare': 'healthcare',
            'Biotechnology': 'healthcare',
            'Manufacturing': 'industrial',
            'Construction': 'industrial',
            'Energy & natural resources': 'industrial',
            'Retail': 'consumer',
            'Consumer goods': 'consumer',
            'Education': 'education',
            'Government': 'public'
        }


def main_improved_ensemble_experiment(df_history: pd.DataFrame, df_test: pd.DataFrame, ground_truth: Dict[str, List[str]], top_k: int = 10) -> pd.DataFrame:
    """
    Main function to run improved ensemble experiment.
    """
    print("="*60)
    print("IMPROVED ENSEMBLE WITH ADVANCED FEATURES")
    print("="*60)
    
    # Load data
    import sys
    import os
    sys.path.append('/home/ubuntu/crawl/crawler-recommend-sys/src')
    
    # from preprocessing_data import load_and_preprocess_data
    # df_history, df_test = load_and_preprocess_data()
    
    # from benchmark_data import create_ground_truth
    # ground_truth = create_ground_truth(df_history)
    
    print(f"Loaded {len(df_history)} history records, {len(df_test)} test candidates")
    print(f"Ground truth for {len(ground_truth)} users")
    
    # Initialize improved recommender
    recommender = MultiStageRecommender(
        stage1_multiplier=8,  # Cast wider net
        stage2_multiplier=4,  # More candidates for reranking
        diversity_weight=0.25,
        use_advanced_features=True
    )
    
    # Fit recommender
    recommender.fit(df_history, df_test, ground_truth)
    
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
            recs = recommender.recommend_items(user_id, df_history, top_k=top_k)
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'industry': rec['industry'], 
                    'score': rec['score']
                })
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue
    
    result_df = pd.DataFrame(results)
    print(f"\nGenerated {len(result_df)} recommendations for {len(seen_users)} users")
    
    return result_df
