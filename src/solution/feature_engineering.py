"""
Advanced Feature Engineering for Higher Recall
==============================================

Focus on extracting more nuanced signals that can improve recall
by better understanding user preferences and item characteristics.
"""

import pandas as pd
from typing import Dict, List, Tuple, Set
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineer:
    """
    Extract sophisticated features to improve recommendation recall.
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        self.industry_embeddings = {}
        self.location_embeddings = {}
        self.service_patterns = {}
        
    def fit(self, df_history: pd.DataFrame):
        """Fit feature extractors on historical data."""
        print("Fitting advanced feature engineer...")
        
        # Extract service patterns
        self._extract_service_patterns(df_history)
        
        # Create industry co-occurrence embeddings
        self._create_industry_embeddings(df_history)
        
        # Geographic patterns
        self._create_location_embeddings(df_history)
        
    def extract_user_features(
        self, 
        user_id: str, 
        user_history: pd.DataFrame
    ) -> Dict[str, float]:
        """Extract comprehensive user features."""
        if user_history.empty:
            return self._get_default_user_features()
        
        features = {}
        
        # 1. Behavioral patterns
        features.update(self._extract_behavioral_patterns(user_history))
        
        # 2. Industry preferences
        # features.update(self._extract_industry_preferences(user_history))
        
        # 3. Geographic preferences  
        features.update(self._extract_geographic_preferences(user_history))
        
        # 4. Service complexity preferences
        features.update(self._extract_service_complexity(user_history))
        
        # 5. Temporal patterns
        # features.update(self._extract_temporal_patterns(user_history))
        
        # 6. Budget and scale preferences
        # features.update(self._extract_scale_preferences(user_history))
        
        return features
    
    def extract_item_features(
        self, 
        industry: str, 
        context: Dict = None
    ) -> Dict[str, float]:
        """Extract item-specific features."""
        features = {}
        
        # Industry characteristics
        features.update(self._get_industry_characteristics(industry))
        
        # Market position
        features.update(self._get_market_position(industry))
        
        # Service complexity
        features.update(self._get_service_complexity(industry))
        
        return features
    
    def calculate_user_item_compatibility(
        self,
        user_features: Dict[str, float],
        item_features: Dict[str, float]
    ) -> float:
        """Calculate compatibility score between user and item."""
        compatibility = 0.0
        
        # Industry preference alignment
        for pref_key in user_features:
            if pref_key.startswith('industry_pref_'):
                industry = pref_key.replace('industry_pref_', '')
                if f'is_{industry.lower().replace(" ", "_")}' in item_features:
                    compatibility += user_features[pref_key] * item_features[f'is_{industry.lower().replace(" ", "_")}'] * 0.3
        
        # Service complexity alignment
        user_complexity = user_features.get('avg_service_complexity', 0.5)
        item_complexity = item_features.get('service_complexity', 0.5)
        complexity_diff = abs(user_complexity - item_complexity)
        complexity_score = max(0, 1 - complexity_diff)
        compatibility += complexity_score * 0.2
        
        # Scale preferences
        user_scale = user_features.get('avg_project_scale', 0.5)
        item_scale = item_features.get('typical_project_scale', 0.5)
        scale_diff = abs(user_scale - item_scale)
        scale_score = max(0, 1 - scale_diff)
        compatibility += scale_score * 0.15
        
        return min(compatibility, 1.0)
    
    def _extract_behavioral_patterns(self, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract user behavioral patterns."""
        features = {}
        
        # Interaction frequency
        features['interaction_count'] = len(user_history)
        
        # Industry diversity
        unique_industries = user_history['industry'].nunique()
        total_interactions = len(user_history)
        features['industry_diversity'] = unique_industries / max(total_interactions, 1)
        
        # Location diversity
        unique_locations = user_history['location'].nunique()
        features['location_diversity'] = unique_locations / max(total_interactions, 1)
        
        # # Consistency patterns
        # industry_counts = user_history['industry'].value_counts()
        # if len(industry_counts) > 0:
        #     features['top_industry_ratio'] = industry_counts.iloc[0] / total_interactions
        #     features['industry_concentration'] = (industry_counts ** 2).sum() / (total_interactions ** 2)
        
        return features
    
    def _extract_industry_preferences(self, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract industry-specific preferences."""
        features = {}
        
        industry_counts = user_history['industry'].value_counts()
        total_interactions = len(user_history)
        
        # # Top industries preferences
        # for industry, count in industry_counts.head(5).items():
        #     safe_industry = industry.lower().replace(' ', '_').replace('&', 'and')
        #     features[f'industry_pref_{safe_industry}'] = count / total_interactions
        
        # Industry category preferences
        tech_industries = {'Software', 'IT Services', 'Information technology'}
        service_industries = {'Consulting', 'Business services'}
        finance_industries = {'Financial services', 'Banking'}
        
        tech_count = sum(industry_counts.get(ind, 0) for ind in tech_industries)
        service_count = sum(industry_counts.get(ind, 0) for ind in service_industries)  
        finance_count = sum(industry_counts.get(ind, 0) for ind in finance_industries)
        
        features['tech_preference'] = tech_count / max(total_interactions, 1)
        features['service_preference'] = service_count / max(total_interactions, 1)
        features['finance_preference'] = finance_count / max(total_interactions, 1)
        
        return features
    
    def _extract_geographic_preferences(self, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract geographic preferences."""
        features = {}
        
        location_counts = user_history['location'].value_counts()
        total_interactions = len(user_history)
        
        if len(location_counts) > 0:
            # Domestic vs international preference
            # Simplified: assume locations with 'United States' are domestic
            # domestic_locations = [loc for loc in location_counts.index if 'United States' in str(loc)]
            # domestic_count = sum(location_counts.get(loc, 0) for loc in domestic_locations)
            
            # features['domestic_preference'] = domestic_count / max(total_interactions, 1)
            # features['international_preference'] = 1 - features['domestic_preference']
            
            # Top location preference
            features['location_concentration'] = location_counts.iloc[0] / total_interactions
        
        return features
    
    def _extract_service_complexity(self, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract service complexity preferences."""
        features = {}
        
        # Define complexity keywords
        high_complexity_keywords = {
            'AI', 'Machine Learning', 'Blockchain', 'Enterprise', 'Custom',
            'Advanced', 'Complex', 'Architecture', 'Integration', 'Migration'
        }
        
        medium_complexity_keywords = {
            'Development', 'Implementation', 'Consulting', 'Analysis',
            'Optimization', 'Support', 'Maintenance'
        }
        
        complexity_scores = []
        for _, row in user_history.iterrows():
            services_text = str(row.get('services', '')).upper()
            
            high_matches = sum(1 for keyword in high_complexity_keywords if keyword in services_text)
            medium_matches = sum(1 for keyword in medium_complexity_keywords if keyword in services_text)
            
            # Calculate complexity score (0-1)
            if high_matches > 0:
                score = 0.8 + min(high_matches * 0.05, 0.2)
            elif medium_matches > 0:
                score = 0.4 + min(medium_matches * 0.1, 0.3)
            else:
                score = 0.2
                
            complexity_scores.append(score)
        
        if complexity_scores:
            features['avg_service_complexity'] = sum(complexity_scores) / len(complexity_scores)
            features['max_service_complexity'] = max(complexity_scores)
            features['complexity_variance'] = pd.Series(complexity_scores).var()
        
        return features
    
    def _extract_temporal_patterns(self, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract temporal interaction patterns."""
        features = {}
        
        # For now, simplified without actual timestamp data
        # In real implementation, would analyze:
        # - Seasonal patterns
        # - Recency effects
        # - Growth trends
        
        # Placeholder features
        features['recency_bias'] = 1.0
        features['seasonal_factor'] = 1.0
        features['growth_trend'] = 0.0
        
        return features
    
    def _extract_scale_preferences(self, user_history: pd.DataFrame) -> Dict[str, float]:
        """Extract project scale and budget preferences."""
        features = {}
        
        if 'client_size_mid' in user_history.columns:
            client_sizes = user_history['client_size_mid'].dropna()
            if len(client_sizes) > 0:
                features['avg_client_size'] = client_sizes.mean()
                features['client_size_std'] = client_sizes.std()
                features['prefers_large_clients'] = (client_sizes > client_sizes.median()).mean()
        
        if 'project_budget_mid' in user_history.columns:
            budgets = user_history['project_budget_mid'].dropna()
            if len(budgets) > 0:
                features['avg_project_budget'] = budgets.mean()
                features['budget_std'] = budgets.std()
                features['prefers_high_budget'] = (budgets > budgets.median()).mean()
        
        # Project scale proxy
        features['avg_project_scale'] = features.get('avg_client_size', 0) * 0.6 + features.get('avg_project_budget', 0) * 0.4
        
        return features
    
    def _get_industry_characteristics(self, industry: str) -> Dict[str, float]:
        """Get characteristics of specific industry."""
        features = {}
        
        # Tech industries
        tech_industries = {'Software', 'IT Services', 'Information technology', 'Telecommunications'}
        features['is_tech'] = 1.0 if industry in tech_industries else 0.0
        
        # Service industries
        service_industries = {'Consulting', 'Business services', 'Professional services'}
        features['is_service'] = 1.0 if industry in service_industries else 0.0
        
        # Regulated industries
        regulated_industries = {'Healthcare', 'Financial services', 'Banking', 'Insurance', 'Government'}
        features['is_regulated'] = 1.0 if industry in regulated_industries else 0.0
        
        # Manufacturing/Physical
        physical_industries = {'Manufacturing', 'Construction', 'Agriculture', 'Energy & natural resources'}
        features['is_physical'] = 1.0 if industry in physical_industries else 0.0
        
        return features
    
    def _get_market_position(self, industry: str) -> Dict[str, float]:
        """Get market position characteristics."""
        # Simplified market data - in practice would use real market data
        market_data = {
            'Software': {'growth_rate': 0.8, 'competition': 0.9, 'demand': 0.9},
            'IT Services': {'growth_rate': 0.7, 'competition': 0.8, 'demand': 0.8},
            'Consulting': {'growth_rate': 0.6, 'competition': 0.7, 'demand': 0.7},
            'Healthcare': {'growth_rate': 0.5, 'competition': 0.4, 'demand': 0.9},
            'Financial services': {'growth_rate': 0.4, 'competition': 0.6, 'demand': 0.6}
        }
        
        return market_data.get(industry, {'growth_rate': 0.5, 'competition': 0.5, 'demand': 0.5})
    
    def _get_service_complexity(self, industry: str) -> Dict[str, float]:
        """Get typical service complexity for industry."""
        complexity_mapping = {
            'Software': 0.8,
            'IT Services': 0.7,
            'Consulting': 0.6,
            'Financial services': 0.7,
            'Healthcare': 0.6,
            'Manufacturing': 0.5,
            'Retail': 0.4,
            'Education': 0.4
        }
        
        return {
            'service_complexity': complexity_mapping.get(industry, 0.5),
            'typical_project_scale': complexity_mapping.get(industry, 0.5)
        }
    
    def _get_default_user_features(self) -> Dict[str, float]:
        """Default features for cold start users."""
        return {
            'interaction_count': 0,
            'interaction_frequency': 0,
            'industry_diversity': 0,
            'avg_service_complexity': 0.5,
            'tech_preference': 0.3,
            'service_preference': 0.2,
            'domestic_preference': 0.7,
            'avg_project_scale': 0.5
        }
    
    def _extract_service_patterns(self, df_history: pd.DataFrame):
        """Extract common service patterns from historical data."""
        # Simplified implementation
        self.service_patterns = {}
        
    def _create_industry_embeddings(self, df_history: pd.DataFrame):
        """Create industry co-occurrence embeddings."""
        # Simplified implementation  
        self.industry_embeddings = {}
        
    def _create_location_embeddings(self, df_history: pd.DataFrame):
        """Create location-based embeddings."""
        # Simplified implementation
        self.location_embeddings = {}