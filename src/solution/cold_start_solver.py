"""
Cold Start Solutions for Recommendation System
=============================================

This module implements various strategies to handle cold start problems:
1. Meta-learning for quick user adaptation
2. Knowledge-based recommendations using business rules
3. Demographic-based filtering
4. Content-based bootstrapping
5. Transfer learning from similar users/items
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class ColdStartSolver:
    """
    Comprehensive cold start solution combining multiple strategies.
    """
    
    def __init__(
        self,
        min_interactions_for_cf: int = 3,
        similarity_threshold: float = 0.3,
        demographic_weight: float = 0.3,
        content_weight: float = 0.4,
        popularity_weight: float = 0.3,
        use_industry_knowledge: bool = True
    ):
        self.min_interactions_for_cf = min_interactions_for_cf
        self.similarity_threshold = similarity_threshold
        self.demographic_weight = demographic_weight
        self.content_weight = content_weight
        self.popularity_weight = popularity_weight
        self.use_industry_knowledge = use_industry_knowledge
        
        # Learned components
        self.user_clusters = None
        self.industry_popularity = {}
        self.industry_compatibility_matrix = {}
        self.demographic_profiles = {}
        self.nn_model = None
        self.scaler = StandardScaler()
        
    def _extract_user_features(self, user_data: pd.DataFrame) -> np.ndarray:
        """
        Extract demographic and contextual features from user data.
        """
        if user_data.empty:
            return np.zeros(10)  # Default feature vector
            
        features = []
        
        # Industry diversity
        unique_industries = len(user_data['industry'].unique())
        total_interactions = len(user_data)
        industry_diversity = unique_industries / max(total_interactions, 1)
        features.append(industry_diversity)
        
        # Location diversity 
        unique_locations = len(user_data['location'].dropna().unique())
        location_diversity = unique_locations / max(total_interactions, 1)
        features.append(location_diversity)
        
        # Average client size
        if 'client_size_mid' in user_data.columns:
            avg_client_size = user_data['client_size_mid'].mean()
        else:
            avg_client_size = 0
        features.append(avg_client_size)
        
        # Average project budget
        if 'project_budget_mid' in user_data.columns:
            avg_budget = user_data['project_budget_mid'].mean()
        else:
            avg_budget = 0
        features.append(avg_budget)
        
        # Project complexity (based on services)
        services_text = ' '.join(user_data['services'].fillna(''))
        complexity_keywords = ['AI', 'Machine Learning', 'Blockchain', 'Custom', 'Enterprise']
        complexity_score = sum(1 for keyword in complexity_keywords if keyword.lower() in services_text.lower())
        features.append(complexity_score)
        
        # Engagement level (interactions per time period - placeholder)
        engagement_level = min(total_interactions / 12.0, 1.0)  # Assume 12 months max
        features.append(engagement_level)
        
        # Most frequent industry (encoded)
        if not user_data['industry'].empty:
            top_industry = user_data['industry'].value_counts().index[0]
            # Simple hash-based encoding
            industry_code = hash(top_industry) % 100 / 100.0
        else:
            industry_code = 0
        features.append(industry_code)
        
        # Geographic focus
        if not user_data['location'].dropna().empty:
            top_location = user_data['location'].value_counts().index[0]
            location_code = hash(top_location) % 100 / 100.0
        else:
            location_code = 0
        features.append(location_code)
        
        # Recency (placeholder - would use actual timestamps)
        recency_score = 1.0  # Assume recent
        features.append(recency_score)
        
        # Seasonality (placeholder)
        seasonality_score = 0.5
        features.append(seasonality_score)
        
        return np.array(features)
    
    def _build_industry_knowledge_base(self, df: pd.DataFrame) -> Dict:
        """
        Build industry compatibility and knowledge base.
        """
        knowledge_base = {
            'compatibility': {},
            'typical_services': {},
            'typical_budgets': {},
            'growth_trends': {}
        }
        
        # Industry compatibility based on co-occurrence
        user_industry_matrix = df.groupby(['linkedin_company_outsource', 'industry']).size().unstack(fill_value=0)
        
        for ind1 in user_industry_matrix.columns:
            compatible_industries = []
            for ind2 in user_industry_matrix.columns:
                if ind1 != ind2:
                    # Calculate how often users who use ind1 also use ind2
                    users_ind1 = set(user_industry_matrix[user_industry_matrix[ind1] > 0].index)
                    users_ind2 = set(user_industry_matrix[user_industry_matrix[ind2] > 0].index)
                    
                    if users_ind1:
                        overlap = len(users_ind1 & users_ind2) / len(users_ind1)
                        if overlap > 0.1:  # 10% threshold
                            compatible_industries.append((ind2, overlap))
            
            knowledge_base['compatibility'][ind1] = sorted(compatible_industries, key=lambda x: x[1], reverse=True)
        
        # Typical services per industry
        for industry in df['industry'].unique():
            industry_data = df[df['industry'] == industry]
            services_text = ' '.join(industry_data['services'].fillna(''))
            # Simple keyword extraction (in practice, use more sophisticated NLP)
            common_services = []
            service_keywords = ['Development', 'Consulting', 'Design', 'Analytics', 'Support']
            for keyword in service_keywords:
                if keyword.lower() in services_text.lower():
                    count = services_text.lower().count(keyword.lower())
                    common_services.append((keyword, count))
            
            knowledge_base['typical_services'][industry] = sorted(common_services, key=lambda x: x[1], reverse=True)
        
        # Typical budgets
        for industry in df['industry'].unique():
            industry_data = df[df['industry'] == industry]
            if 'project_budget_mid' in industry_data.columns:
                avg_budget = industry_data['project_budget_mid'].mean()
                knowledge_base['typical_budgets'][industry] = avg_budget
            else:
                knowledge_base['typical_budgets'][industry] = 0
        
        return knowledge_base
    
    def fit(self, df_history: pd.DataFrame):
        """
        Fit cold start models on historical data.
        """
        print("Fitting cold start solver...")
        
        # Build industry knowledge base
        if self.use_industry_knowledge:
            self.industry_knowledge = self._build_industry_knowledge_base(df_history)
        
        # Calculate industry popularity
        industry_counts = df_history['industry'].value_counts()
        total_interactions = len(df_history)
        self.industry_popularity = {
            industry: count / total_interactions 
            for industry, count in industry_counts.items()
        }
        
        # Build user demographic profiles
        user_features_list = []
        user_ids = []
        
        for user_id in df_history['linkedin_company_outsource'].unique():
            user_data = df_history[df_history['linkedin_company_outsource'] == user_id]
            if len(user_data) >= self.min_interactions_for_cf:  # Only users with sufficient data
                features = self._extract_user_features(user_data)
                user_features_list.append(features)
                user_ids.append(user_id)
        
        if user_features_list:
            user_features_matrix = np.vstack(user_features_list)
            
            # Fit scaler
            user_features_scaled = self.scaler.fit_transform(user_features_matrix)
            
            # Build nearest neighbors model for finding similar users
            self.nn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
            self.nn_model.fit(user_features_scaled)
            
            # Cluster users for demographic-based recommendations
            n_clusters = min(10, len(user_ids) // 5 + 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(user_features_scaled)
            
            # Build cluster profiles
            for cluster_id in range(n_clusters):
                cluster_users = [user_ids[i] for i, c in enumerate(clusters) if c == cluster_id]
                cluster_interactions = df_history[
                    df_history['linkedin_company_outsource'].isin(cluster_users)
                ]
                
                # Most popular industries in this cluster
                cluster_industry_counts = cluster_interactions['industry'].value_counts()
                total_cluster_interactions = len(cluster_interactions)
                
                self.demographic_profiles[cluster_id] = {
                    'industry_preferences': {
                        industry: count / total_cluster_interactions
                        for industry, count in cluster_industry_counts.head(10).items()
                    },
                    'avg_budget': cluster_interactions.get('project_budget_mid', pd.Series([0])).mean(),
                    'common_services': cluster_interactions['services'].fillna('').str.cat(sep=' ')
                }
            
            self.user_clusters = kmeans
            self._user_features = user_features_scaled
            self._user_ids = user_ids
            
        print("Cold start solver fitting completed!")
    
    def recommend_for_cold_user(
        self,
        user_context: Dict,
        available_industries: List[str],
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Generate recommendations for a completely new user.
        
        Args:
            user_context: Dict with available info about user (company_size, location, etc.)
            available_industries: List of industries to choose from
            top_k: Number of recommendations
        """
        recommendations = []
        
        # Strategy 1: Knowledge-based recommendations
        knowledge_scores = self._get_knowledge_based_scores(user_context, available_industries)
        
        # Strategy 2: Popularity-based recommendations
        popularity_scores = {
            industry: self.industry_popularity.get(industry, 0.001)
            for industry in available_industries
        }
        
        # Strategy 3: Demographic-based recommendations (if we have enough context)
        demographic_scores = self._get_demographic_scores(user_context, available_industries)
        
        # Combine scores
        for industry in available_industries:
            combined_score = (
                self.content_weight * knowledge_scores.get(industry, 0) +
                self.popularity_weight * popularity_scores.get(industry, 0) +
                self.demographic_weight * demographic_scores.get(industry, 0)
            )
            
            recommendations.append({
                'industry': industry,
                'score': combined_score,
                'knowledge_score': knowledge_scores.get(industry, 0),
                'popularity_score': popularity_scores.get(industry, 0),
                'demographic_score': demographic_scores.get(industry, 0)
            })
        
        # Sort and return top K
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('score', ascending=False).head(top_k)
        
        return recommendations_df
    
    def _get_knowledge_based_scores(self, user_context: Dict, industries: List[str]) -> Dict[str, float]:
        """
        Score industries based on business rules and domain knowledge.
        """
        scores = {}
        
        # Default business rules
        business_rules = {
            'IT': {'min_size': 10, 'min_budget': 50000, 'growth_multiplier': 1.3},
            'Consulting': {'min_size': 1, 'min_budget': 10000, 'growth_multiplier': 1.2},
            'eCommerce': {'min_size': 5, 'min_budget': 25000, 'growth_multiplier': 1.1},
            'Healthcare': {'min_size': 20, 'min_budget': 100000, 'growth_multiplier': 1.4},
            'Finance': {'min_size': 50, 'min_budget': 200000, 'growth_multiplier': 1.5}
        }
        
        user_size = user_context.get('company_size', 10)
        user_budget = user_context.get('budget', 50000)
        user_location = user_context.get('location', '')
        
        for industry in industries:
            score = 0.5  # Base score
            
            # Apply business rules if available
            if industry in business_rules:
                rules = business_rules[industry]
                
                # Size compatibility
                if user_size >= rules['min_size']:
                    score += 0.2
                
                # Budget compatibility  
                if user_budget >= rules['min_budget']:
                    score += 0.2
                
                # Growth potential
                score *= rules['growth_multiplier']
            
            # Location-based adjustments
            if user_location:
                if 'US' in user_location or 'United States' in user_location:
                    if industry in ['IT', 'Healthcare', 'Finance']:
                        score *= 1.1  # These industries are strong in US
                elif 'Europe' in user_location:
                    if industry in ['Consulting', 'Manufacturing']:
                        score *= 1.1
            
            scores[industry] = min(score, 1.0)  # Cap at 1.0
        
        return scores
    
    def _get_demographic_scores(self, user_context: Dict, industries: List[str]) -> Dict[str, float]:
        """
        Score industries based on similar user demographics.
        """
        if not hasattr(self, 'user_clusters') or self.user_clusters is None:
            return {industry: 0.5 for industry in industries}
        
        # Create feature vector for new user
        mock_df = pd.DataFrame([{
            'industry': 'Unknown',
            'location': user_context.get('location', ''),
            'client_size_mid': user_context.get('company_size', 10),
            'project_budget_mid': user_context.get('budget', 50000),
            'services': user_context.get('services', '')
        }])
        
        user_features = self._extract_user_features(mock_df)
        user_features_scaled = self.scaler.transform([user_features])
        
        # Find most likely cluster
        cluster_id = self.user_clusters.predict(user_features_scaled)[0]
        
        # Get preferences for this cluster
        if cluster_id in self.demographic_profiles:
            cluster_prefs = self.demographic_profiles[cluster_id]['industry_preferences']
            scores = {
                industry: cluster_prefs.get(industry, 0.1)
                for industry in industries
            }
        else:
            scores = {industry: 0.5 for industry in industries}
        
        return scores
    
    def recommend_for_warm_user(
        self,
        user_id: str,
        user_history: pd.DataFrame,
        available_industries: List[str],
        df_full_history: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Generate recommendations for users with limited interaction history.
        """
        n_interactions = len(user_history)
        
        if n_interactions == 0:
            # Completely cold - use cold user strategy
            user_context = {'company_size': 10, 'budget': 50000}  # Default context
            return self.recommend_for_cold_user(user_context, available_industries, top_k)
        
        # Find similar users based on limited history
        similar_users = self._find_similar_users(user_history, df_full_history)
        
        # Strategy 1: Collaborative filtering with similar users
        cf_scores = self._collaborative_with_similar_users(similar_users, df_full_history, available_industries)
        
        # Strategy 2: Content-based on limited history
        content_scores = self._content_based_limited_history(user_history, available_industries)
        
        # Strategy 3: Popularity with user bias
        popularity_scores = self._popularity_with_user_bias(user_history, available_industries)
        
        # Combine strategies with adaptive weights based on interaction count
        cf_weight = min(n_interactions / self.min_interactions_for_cf, 0.5)
        content_weight = 0.3
        popularity_weight = max(0.2, 0.7 - cf_weight - content_weight)
        
        recommendations = []
        for industry in available_industries:
            combined_score = (
                cf_weight * cf_scores.get(industry, 0) +
                content_weight * content_scores.get(industry, 0) +
                popularity_weight * popularity_scores.get(industry, 0)
            )
            
            recommendations.append({
                'industry': industry,
                'score': combined_score,
                'cf_score': cf_scores.get(industry, 0),
                'content_score': content_scores.get(industry, 0),
                'popularity_score': popularity_scores.get(industry, 0)
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        return recommendations_df.sort_values('score', ascending=False).head(top_k)
    
    def _find_similar_users(
        self, 
        user_history: pd.DataFrame,
        df_full_history: pd.DataFrame
    ) -> List[str]:
        """
        Find users with similar interaction patterns.
        """
        user_industries = set(user_history['industry'])
        similar_users = []
        
        for other_user in df_full_history['linkedin_company_outsource'].unique():
            other_history = df_full_history[df_full_history['linkedin_company_outsource'] == other_user]
            other_industries = set(other_history['industry'])
            
            # Calculate Jaccard similarity
            intersection = len(user_industries & other_industries)
            union = len(user_industries | other_industries)
            
            if union > 0:
                similarity = intersection / union
                if similarity > self.similarity_threshold:
                    similar_users.append(other_user)
        
        return similar_users[:20]  # Limit to top 20 similar users
    
    def _collaborative_with_similar_users(
        self,
        similar_users: List[str],
        df_full_history: pd.DataFrame,
        available_industries: List[str]
    ) -> Dict[str, float]:
        """
        Collaborative filtering using similar users.
        """
        if not similar_users:
            return {industry: 0.0 for industry in available_industries}
        
        # Get industries preferred by similar users
        similar_interactions = df_full_history[
            df_full_history['linkedin_company_outsource'].isin(similar_users)
        ]
        
        industry_counts = similar_interactions['industry'].value_counts()
        total_interactions = len(similar_interactions)
        
        scores = {}
        for industry in available_industries:
            score = industry_counts.get(industry, 0) / max(total_interactions, 1)
            scores[industry] = score
            
        return scores
    
    def _content_based_limited_history(
        self,
        user_history: pd.DataFrame,
        available_industries: List[str]
    ) -> Dict[str, float]:
        """
        Content-based recommendations from limited user history.
        """
        if user_history.empty:
            return {industry: 0.0 for industry in available_industries}
        
        # Extract user preferences
        user_industries = set(user_history['industry'])
        user_services = ' '.join(user_history['services'].fillna(''))
        
        scores = {}
        for industry in available_industries:
            score = 0.0
            
            # Direct match
            if industry in user_industries:
                score += 0.5
            
            # Industry compatibility based on knowledge base
            if hasattr(self, 'industry_knowledge') and industry in self.industry_knowledge['compatibility']:
                for user_ind in user_industries:
                    compatible = self.industry_knowledge['compatibility'][industry]
                    for comp_ind, comp_score in compatible:
                        if comp_ind == user_ind:
                            score += 0.3 * comp_score
                            break
            
            # Service compatibility (simple keyword matching)
            if hasattr(self, 'industry_knowledge') and industry in self.industry_knowledge['typical_services']:
                typical_services = self.industry_knowledge['typical_services'][industry]
                for service, _ in typical_services:
                    if service.lower() in user_services.lower():
                        score += 0.1
            
            scores[industry] = min(score, 1.0)
        
        return scores
    
    def _popularity_with_user_bias(
        self,
        user_history: pd.DataFrame,
        available_industries: List[str]
    ) -> Dict[str, float]:
        """
        Popularity-based recommendations with user preference bias.
        """
        base_popularity = {
            industry: self.industry_popularity.get(industry, 0.001)
            for industry in available_industries
        }
        
        if user_history.empty:
            return base_popularity
        
        # Boost industries similar to user's history
        user_industries = set(user_history['industry'])
        
        for industry in available_industries:
            if industry in user_industries:
                base_popularity[industry] *= 1.5  # Boost familiar industries
            elif hasattr(self, 'industry_knowledge'):
                # Check compatibility with user industries
                compatibility_boost = 0
                for user_ind in user_industries:
                    if (industry in self.industry_knowledge.get('compatibility', {}) and
                        any(comp[0] == user_ind for comp in self.industry_knowledge['compatibility'][industry])):
                        compatibility_boost += 0.2
                
                base_popularity[industry] *= (1 + compatibility_boost)
        
        return base_popularity