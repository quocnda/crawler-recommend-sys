"""
Multi-Stage Recommendation Pipeline for Higher Recall
===================================================

Approach:
1. Stage 1: Cast wide net with multiple candidate generators (high recall)
2. Stage 2: Apply sophisticated ranking and filtering (maintain precision)
3. Stage 3: Diversity optimization and business rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from solution.content_base_for_item import ContentBaseBasicApproach
from solution.collborative_for_item import CollaborativeIndustryRecommender
from solution.advanced_reranker import AdvancedReranker
import warnings
warnings.filterwarnings('ignore')


class MultiStageRecommendationPipeline:
    """
    Multi-stage pipeline optimized for high recall while maintaining precision.
    """
    
    def __init__(
        self,
        stage1_candidates_multiplier: float = 5.0,  # Cast wider net
        popularity_boost: float = 0.1,
        geographic_boost: float = 0.05,
        temporal_boost: float = 0.05
    ):
        self.stage1_multiplier = stage1_candidates_multiplier
        self.popularity_boost = popularity_boost
        self.geographic_boost = geographic_boost
        self.temporal_boost = temporal_boost
        
        self.content_model = None
        self.collab_model = None
        self.reranker = AdvancedReranker()
        
    def fit(self, df_history: pd.DataFrame, df_test: pd.DataFrame):
        """Fit all component models."""
        print("Fitting multi-stage pipeline...")
        
        # Content-based model
        print("- Fitting content-based model...")
        self.content_model = ContentBaseBasicApproach(df_history, df_test)
        
        # Collaborative model
        print("- Fitting collaborative model...")
        self.collab_model = CollaborativeIndustryRecommender(
            n_components=150,  # Increased from 128
            min_user_interactions=1,
            min_item_interactions=1,
            use_tfidf_weighting=True,
            random_state=42
        ).fit(df_history=df_history, df_candidates=df_test)
        
    def stage1_candidate_generation(
        self, 
        user_id: str, 
        target_candidates: int
    ) -> Dict[str, float]:
        """
        Stage 1: Generate diverse candidate set with high recall.
        Uses multiple strategies to ensure we don't miss relevant items.
        """
        all_candidates = {}
        
        # Strategy 1: Content-based (wider net)
        try:
            cb_recs = self.content_model.recommend_items(user_id, top_k=target_candidates)
            for _, row in cb_recs.iterrows():
                industry = row['industry']
                score = row['score']
                all_candidates[industry] = max(all_candidates.get(industry, 0), score * 0.4)
        except Exception as e:
            print(f"Content-based failed for {user_id}: {e}")
        
        # Strategy 2: Collaborative filtering (wider net)  
        try:
            cf_recs = self.collab_model.recommend_items(user_id, top_k=target_candidates)
            for _, row in cf_recs.iterrows():
                industry = row['industry']
                score = row['score']
                all_candidates[industry] = max(all_candidates.get(industry, 0), score * 0.4)
        except Exception as e:
            print(f"Collaborative failed for {user_id}: {e}")
            
        # Strategy 3: Popularity-based fallback
        popularity_candidates = self._get_popular_industries(target_candidates // 3)
        for industry, pop_score in popularity_candidates.items():
            boosted_score = pop_score * self.popularity_boost
            all_candidates[industry] = max(all_candidates.get(industry, 0), boosted_score)
        
        # Strategy 4: Geographic similarity
        geo_candidates = self._get_geographic_similar_industries(user_id, target_candidates // 4)
        for industry, geo_score in geo_candidates.items():
            boosted_score = geo_score * self.geographic_boost  
            all_candidates[industry] = max(all_candidates.get(industry, 0), boosted_score)
            
        return all_candidates
    
    def stage2_ranking_and_filtering(
        self,
        candidates: Dict[str, float],
        user_id: str,
        user_history: pd.DataFrame,
        target_size: int
    ) -> List[Tuple[str, float]]:
        """
        Stage 2: Apply sophisticated ranking while maintaining high recall.
        """
        if not candidates:
            return []
            
        # Enhanced scoring with multiple signals
        enhanced_candidates = {}
        
        for industry, base_score in candidates.items():
            enhanced_score = base_score
            
            # User history affinity boost
            if not user_history.empty:
                history_industries = set(user_history['industry'].values)
                
                # Direct history match (strong signal)
                if industry in history_industries:
                    enhanced_score *= 1.3
                
                # Industry category similarity
                category_boost = self._calculate_category_similarity(
                    industry, history_industries
                )
                enhanced_score *= (1 + category_boost * 0.2)
            
            # Diversity penalty for over-represented categories
            diversity_penalty = self._calculate_diversity_penalty(
                industry, list(candidates.keys())
            )
            enhanced_score *= (1 - diversity_penalty * 0.1)
            
            enhanced_candidates[industry] = enhanced_score
        
        # Sort and return top candidates
        sorted_candidates = sorted(
            enhanced_candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_candidates[:target_size]
    
    def stage3_diversity_optimization(
        self,
        ranked_candidates: List[Tuple[str, float]],
        target_size: int,
        diversity_threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Stage 3: Optimize for diversity while maintaining relevance.
        """
        if len(ranked_candidates) <= target_size:
            return ranked_candidates
        
        # Greedy diversity optimization
        selected = []
        remaining = ranked_candidates.copy()
        selected_categories = set()
        
        # Always include top candidate
        if remaining:
            top_candidate = remaining.pop(0)
            selected.append(top_candidate)
            selected_categories.add(self._get_industry_category(top_candidate[0]))
        
        # Greedily select remaining candidates
        while len(selected) < target_size and remaining:
            best_idx = 0
            best_score = -1
            
            for idx, (industry, score) in enumerate(remaining):
                # Base relevance score
                relevance = score
                
                # Diversity bonus
                category = self._get_industry_category(industry)
                diversity_bonus = 0
                if category not in selected_categories:
                    diversity_bonus = diversity_threshold * score
                
                combined_score = relevance + diversity_bonus
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            # Add best candidate
            selected_candidate = remaining.pop(best_idx)
            selected.append(selected_candidate)
            selected_categories.add(self._get_industry_category(selected_candidate[0]))
        
        return selected
    
    def recommend_items(
        self,
        user_id: str,
        user_history: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Main recommendation pipeline.
        """
        # Stage 1: Generate broad candidate set
        stage1_target = int(top_k * self.stage1_multiplier)
        candidates = self.stage1_candidate_generation(user_id, stage1_target)
        
        if not candidates:
            return pd.DataFrame(columns=['industry', 'score'])
        
        # Stage 2: Sophisticated ranking
        stage2_target = min(top_k * 3, len(candidates))  # 3x final size for stage 3
        ranked_candidates = self.stage2_ranking_and_filtering(
            candidates, user_id, user_history, stage2_target
        )
        
        # Stage 3: Diversity optimization
        final_candidates = self.stage3_diversity_optimization(
            ranked_candidates, top_k
        )
        
        # Format results
        results = []
        for industry, score in final_candidates:
            results.append({
                'industry': industry,
                'score': float(score)
            })
        
        return pd.DataFrame(results)
    
    def _get_popular_industries(self, top_k: int) -> Dict[str, float]:
        """Get most popular industries as fallback candidates."""
        # This would be computed from historical data
        # Simplified version:
        popular_industries = {
            'Software': 0.8,
            'IT Services': 0.7,
            'Consulting': 0.6,
            'Financial services': 0.5,
            'Healthcare': 0.4,
            'Manufacturing': 0.3,
            'Education': 0.25,
            'Retail': 0.2
        }
        
        return dict(list(popular_industries.items())[:top_k])
    
    def _get_geographic_similar_industries(self, user_id: str, top_k: int) -> Dict[str, float]:
        """Get industries popular in user's geographic area."""
        # Simplified - would use actual geographic data
        geo_industries = {
            'IT Services': 0.3,
            'Software': 0.25,
            'Business services': 0.2,
            'Consulting': 0.15
        }
        
        return dict(list(geo_industries.items())[:top_k])
    
    def _calculate_category_similarity(self, industry: str, history_industries: Set[str]) -> float:
        """Calculate similarity between industry and user's historical industries."""
        # Define industry categories
        tech_categories = {'Software', 'IT Services', 'Information technology', 'Telecommunications'}
        service_categories = {'Consulting', 'Business services', 'Professional services'}
        finance_categories = {'Financial services', 'Banking', 'Insurance'}
        
        industry_category = None
        if industry in tech_categories:
            industry_category = 'tech'
        elif industry in service_categories:
            industry_category = 'service'
        elif industry in finance_categories:
            industry_category = 'finance'
        
        if industry_category:
            category_map = {
                'tech': tech_categories,
                'service': service_categories, 
                'finance': finance_categories
            }
            
            overlap = len(history_industries & category_map[industry_category])
            if overlap > 0:
                return min(overlap * 0.2, 0.5)  # Cap at 50% boost
                
        return 0.0
    
    def _calculate_diversity_penalty(self, industry: str, all_candidates: List[str]) -> float:
        """Calculate penalty for over-represented categories."""
        category = self._get_industry_category(industry)
        
        # Count how many candidates are in same category
        same_category_count = sum(
            1 for candidate in all_candidates 
            if self._get_industry_category(candidate) == category
        )
        
        # Penalty increases with over-representation
        penalty = max(0, (same_category_count - 3) * 0.1)
        return min(penalty, 0.3)  # Cap penalty at 30%
    
    def _get_industry_category(self, industry: str) -> str:
        """Get broad category for an industry."""
        tech_categories = {'Software', 'IT Services', 'Information technology', 'Telecommunications'}
        service_categories = {'Consulting', 'Business services', 'Professional services'}
        finance_categories = {'Financial services', 'Banking', 'Insurance'}
        health_categories = {'Healthcare', 'Medical', 'Pharmaceuticals'}
        
        if industry in tech_categories:
            return 'tech'
        elif industry in service_categories:
            return 'service'
        elif industry in finance_categories:
            return 'finance'
        elif industry in health_categories:
            return 'health'
        else:
            return 'other'


def integrate_multi_stage_pipeline(
    df_history: pd.DataFrame,
    df_test: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Integration function for multi-stage pipeline.
    """
    pipeline = MultiStageRecommendationPipeline(
        stage1_candidates_multiplier=6.0,  # Cast very wide net
        popularity_boost=0.15,
        geographic_boost=0.1
    )
    
    # Fit pipeline
    pipeline.fit(df_history, df_test)
    
    # Generate recommendations
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        if pd.isna(user_id) or user_id in seen_users:
            continue
        seen_users.add(user_id)
        
        try:
            user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
            recs = pipeline.recommend_items(user_id, user_history, top_k=top_k)
            
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user_id,
                    'industry': rec['industry'],
                    'score': rec['score']
                })
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")
            continue
    
    return pd.DataFrame(results)