"""
Enhanced User-Based Collaborative Filtering
============================================

Recommends triplets based on BOTH:
1. User interaction history similarity (what items users interacted with)
2. User profile similarity (company description, services offered)

Key insight: Two outsource companies with similar services/description
are likely to work with similar types of clients (industry, size, services).

Supports both OpenAI and SentenceTransformers embeddings.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Literal
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from triplet_utils import TripletManager

# Import embedder với fallback
try:
    from solution.openai_embedder import HybridEmbedder, get_embedder
    OPENAI_EMBEDDER_AVAILABLE = True
except ImportError:
    OPENAI_EMBEDDER_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EnhancedUserFeatureBuilder:
    """
    Build comprehensive user features combining:
    1. Company profile features (description, services offered)
    2. Interaction history features (industries, sizes, locations worked with)
    """
    
    def __init__(
        self,
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        profile_weight: float = 0.4,  # Weight for profile-based similarity
        history_weight: float = 0.6,   # Weight for history-based similarity
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
        self.sentence_model_name = sentence_model_name
        self.profile_weight = profile_weight
        self.history_weight = history_weight
        self.use_openai = use_openai
        
        # Initialize embedder (OpenAI or SentenceTransformers)
        self.sentence_model = None
        
        if use_openai and OPENAI_EMBEDDER_AVAILABLE:
            try:
                self.sentence_model = HybridEmbedder(
                    use_openai=True,
                    openai_model=openai_model,
                    sentence_model=sentence_model_name
                )
                print(f"EnhancedUserFeatureBuilder using OpenAI: {openai_model}")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embedder: {e}")
                self.sentence_model = None
        
        if self.sentence_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(sentence_model_name)
                print(f"EnhancedUserFeatureBuilder using SentenceTransformers: {sentence_model_name}")
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.sentence_model = None
        
        self.tfidf_services = TfidfVectorizer(max_features=100, stop_words='english')
        self.scaler = StandardScaler()
        
        # Learned statistics from training data
        self.global_industry_freq: Dict[str, float] = {}
        self.global_service_freq: Dict[str, float] = {}
        self.global_location_freq: Dict[str, float] = {}
        
        # User profile embeddings cache
        self.user_profile_embeddings: Dict[str, np.ndarray] = {}
        self.user_history_embeddings: Dict[str, np.ndarray] = {}
        self.user_combined_embeddings: Dict[str, np.ndarray] = {}
        
    def fit(
        self,
        df_history: pd.DataFrame,
        df_user_profiles: Optional[pd.DataFrame] = None
    ) -> 'EnhancedUserFeatureBuilder':
        """
        Fit on training data to learn global statistics.
        
        Args:
            df_history: Historical interactions
            df_user_profiles: Optional user profile data with columns:
                - linkedin_company_outsource
                - description_company_outsource
                - services_company_outsource
        """
        print("Fitting Enhanced User Feature Builder...")
        
        # 1. Calculate global frequencies from history
        self._fit_global_frequencies(df_history)
        
        # 2. Fit TF-IDF on user services (from profiles or history)
        self._fit_services_vectorizer(df_history, df_user_profiles)
        
        print("Enhanced User Feature Builder fitted!")
        return self
    
    def _fit_global_frequencies(self, df_history: pd.DataFrame):
        """Calculate global frequencies for industries, services, locations."""
        total = len(df_history)
        
        # Industry frequencies
        industry_counts = df_history['industry'].value_counts()
        self.global_industry_freq = {
            ind: count / total 
            for ind, count in industry_counts.items()
        }
        
        # Location frequencies
        location_counts = df_history['location'].value_counts()
        self.global_location_freq = {
            loc: count / total 
            for loc, count in location_counts.items()
        }
        
        # Service frequencies
        all_services = []
        for services_str in df_history['services'].dropna():
            services = [s.strip() for s in str(services_str).split(',')]
            all_services.extend([s for s in services if s])
        
        from collections import Counter
        service_counts = Counter(all_services)
        total_services = sum(service_counts.values())
        self.global_service_freq = {
            svc: count / total_services
            for svc, count in service_counts.items()
        }
    
    def _fit_services_vectorizer(
        self,
        df_history: pd.DataFrame,
        df_user_profiles: Optional[pd.DataFrame]
    ):
        """Fit TF-IDF vectorizer on services text."""
        services_texts = []
        
        # Collect services from user profiles if available
        if df_user_profiles is not None and 'services_company_outsource' in df_user_profiles.columns:
            for svc in df_user_profiles['services_company_outsource'].dropna():
                services_texts.append(str(svc))
        
        # Also collect from history
        for svc in df_history['services'].dropna():
            services_texts.append(str(svc))
        
        if services_texts:
            self.tfidf_services.fit(services_texts)
    
    def build_user_profile_features(
        self,
        user_id: str,
        user_info: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Build features from user's company profile.
        
        Features:
        - Embedding of description_company_outsource
        - Embedding of services_company_outsource
        
        Returns:
            Profile feature vector
        """
        features = []
        
        # Default embedding dimension
        embed_dim = self.sentence_model.get_sentence_embedding_dimension() if self.sentence_model else 1536
        
        # 1. Embed company description
        if user_info is not None and 'description_company_outsource' in user_info:
            desc = user_info.get('description_company_outsource')
            if pd.notna(desc) and str(desc).strip():
                desc_text = str(desc)[:1000]  # Truncate
                if self.sentence_model:
                    desc_emb = self.sentence_model.encode(
                        [desc_text], 
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )[0]
                else:
                    desc_emb = np.zeros(embed_dim)
            else:
                desc_emb = np.zeros(embed_dim)
        else:
            desc_emb = np.zeros(embed_dim)
        
        features.append(desc_emb)
        
        # 2. Embed company services
        if user_info is not None and 'services_company_outsource' in user_info:
            svc = user_info.get('services_company_outsource')
            if pd.notna(svc) and str(svc).strip():
                svc_text = str(svc)
                if self.sentence_model:
                    svc_emb = self.sentence_model.encode(
                        [svc_text],
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )[0]
                else:
                    svc_emb = np.zeros(embed_dim)
            else:
                svc_emb = np.zeros(embed_dim)
        else:
            svc_emb = np.zeros(embed_dim)
        
        features.append(svc_emb)
        
        # Concatenate
        profile_vector = np.concatenate(features)
        
        # L2 normalize
        norm = np.linalg.norm(profile_vector)
        if norm > 0:
            profile_vector = profile_vector / norm
        
        return profile_vector
    
    def build_user_history_features(
        self,
        user_id: str,
        df_history: pd.DataFrame
    ) -> np.ndarray:
        """
        Build features from user's interaction history.
        
        Features:
        - Industry preference vector (TF-IDF weighted)
        - Service usage vector
        - Size preference distribution
        - Location preference distribution
        - Numerical stats (interaction count, avg client size)
        """
        user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
        
        if user_history.empty:
            # Return zero vector for cold start users
            n_industries = len(self.global_industry_freq)
            n_services = len(self.global_service_freq)
            n_locations = min(20, len(self.global_location_freq))
            return np.zeros(n_industries + n_services + 5 + n_locations + 5)
        
        features = []
        
        # 1. Industry preference vector (TF-IDF style)
        industry_vector = self._build_industry_vector(user_history)
        features.append(industry_vector)
        
        # 2. Service usage vector
        service_vector = self._build_service_vector(user_history)
        features.append(service_vector)
        
        # 3. Size preference distribution
        size_vector = self._build_size_vector(user_history)
        features.append(size_vector)
        
        # 4. Location preference vector
        location_vector = self._build_location_vector(user_history)
        features.append(location_vector)
        
        # 5. Numerical statistics
        num_stats = self._build_numerical_stats(user_history)
        features.append(num_stats)
        
        # Concatenate
        history_vector = np.concatenate(features)
        
        # L2 normalize
        norm = np.linalg.norm(history_vector)
        if norm > 0:
            history_vector = history_vector / norm
        
        return history_vector
    
    def _build_industry_vector(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build TF-IDF style industry preference vector."""
        industry_counts = user_history['industry'].value_counts()
        total = len(user_history)
        
        vector = np.zeros(len(self.global_industry_freq))
        industry_to_idx = {ind: i for i, ind in enumerate(self.global_industry_freq.keys())}
        
        for industry, count in industry_counts.items():
            if industry in industry_to_idx:
                tf = count / total
                idf = np.log(1.0 / (self.global_industry_freq.get(industry, 1e-6) + 1e-6))
                vector[industry_to_idx[industry]] = tf * idf
        
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _build_service_vector(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build service usage vector."""
        service_counts = defaultdict(int)
        
        for services_str in user_history['services'].dropna():
            services = [s.strip() for s in str(services_str).split(',')]
            for svc in services:
                if svc:
                    service_counts[svc] += 1
        
        vector = np.zeros(len(self.global_service_freq))
        service_to_idx = {svc: i for i, svc in enumerate(self.global_service_freq.keys())}
        
        total = sum(service_counts.values()) or 1
        
        for svc, count in service_counts.items():
            if svc in service_to_idx:
                vector[service_to_idx[svc]] = count / total
        
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _build_size_vector(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build client size preference distribution."""
        size_buckets = ['micro', 'small', 'medium', 'large', 'enterprise']
        size_counts = np.zeros(5)
        
        for _, row in user_history.iterrows():
            if 'client_min' in row and 'client_max' in row:
                client_min = row['client_min']
                client_max = row['client_max']
                
                if pd.notna(client_min) and pd.notna(client_max):
                    mid = (client_min + client_max) / 2
                    
                    if mid <= 10:
                        size_counts[0] += 1
                    elif mid <= 50:
                        size_counts[1] += 1
                    elif mid <= 200:
                        size_counts[2] += 1
                    elif mid <= 1000:
                        size_counts[3] += 1
                    else:
                        size_counts[4] += 1
        
        # Normalize to probability distribution
        total = size_counts.sum()
        if total > 0:
            size_counts = size_counts / total
        
        return size_counts
    
    def _build_location_vector(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build location preference vector (top 20 locations)."""
        # Get top 20 global locations
        top_locations = list(self.global_location_freq.keys())[:20]
        
        location_counts = user_history['location'].value_counts()
        total = len(user_history)
        
        vector = np.zeros(len(top_locations))
        
        for i, loc in enumerate(top_locations):
            if loc in location_counts:
                vector[i] = location_counts[loc] / total
        
        return vector
    
    def _build_numerical_stats(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build numerical statistics."""
        stats = []
        
        # Interaction count (normalized)
        n_interactions = len(user_history)
        stats.append(min(n_interactions / 50.0, 1.0))
        
        # Industry diversity
        n_industries = user_history['industry'].nunique()
        stats.append(n_industries / max(n_interactions, 1))
        
        # Average client size (log normalized)
        if 'client_min' in user_history.columns and 'client_max' in user_history.columns:
            client_mids = []
            for _, row in user_history.iterrows():
                if pd.notna(row.get('client_min')) and pd.notna(row.get('client_max')):
                    client_mids.append((row['client_min'] + row['client_max']) / 2)
            
            if client_mids:
                avg_size = np.mean(client_mids)
                stats.append(np.log1p(avg_size) / 10.0)
            else:
                stats.append(0.0)
        else:
            stats.append(0.0)
        
        # Service diversity
        all_services = set()
        for svc_str in user_history['services'].dropna():
            services = [s.strip() for s in str(svc_str).split(',')]
            all_services.update([s for s in services if s])
        stats.append(min(len(all_services) / 20.0, 1.0))
        
        # Location concentration
        n_locations = user_history['location'].nunique()
        stats.append(1.0 - (n_locations / max(n_interactions, 1)))
        
        return np.array(stats)
    
    def build_combined_features(
        self,
        user_id: str,
        df_history: pd.DataFrame,
        user_info: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Build combined features from both profile and history.
        
        Returns:
            Combined weighted feature vector
        """
        # Build profile features
        profile_features = self.build_user_profile_features(user_id, user_info)
        
        # Build history features
        history_features = self.build_user_history_features(user_id, df_history)
        
        # Weighted combination (same dimension for comparison)
        # Use separate vectors for similarity calculation
        combined = np.concatenate([
            profile_features * self.profile_weight,
            history_features * self.history_weight
        ])
        
        # L2 normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def compute_user_similarity(
        self,
        user1_id: str,
        user2_id: str,
        df_history: pd.DataFrame,
        user_info_map: Dict[str, pd.Series] = None
    ) -> float:
        """
        Compute similarity between two users combining profile and history.
        """
        # Get or compute user1 features
        if user1_id not in self.user_combined_embeddings:
            user1_info = user_info_map.get(user1_id) if user_info_map else None
            self.user_combined_embeddings[user1_id] = self.build_combined_features(
                user1_id, df_history, user1_info
            )
        
        # Get or compute user2 features
        if user2_id not in self.user_combined_embeddings:
            user2_info = user_info_map.get(user2_id) if user_info_map else None
            self.user_combined_embeddings[user2_id] = self.build_combined_features(
                user2_id, df_history, user2_info
            )
        
        user1_vec = self.user_combined_embeddings[user1_id]
        user2_vec = self.user_combined_embeddings[user2_id]
        
        # Cosine similarity (vectors are already L2 normalized)
        similarity = np.dot(user1_vec, user2_vec)
        
        return float(similarity)


class EnhancedUserCollaborativeRecommender:
    """
    Enhanced User-Based Collaborative Filtering for Triplet Recommendation.
    
    Combines:
    1. Profile-based similarity (company description, services offered)
    2. History-based similarity (interaction patterns)
    
    Two users with similar profiles AND/OR similar histories
    will have similar triplet recommendations.
    """
    
    def __init__(
        self,
        min_similarity: float = 0.1,
        top_k_similar_users: int = 30,
        profile_weight: float = 0.4,
        history_weight: float = 0.6,
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
        self.min_similarity = min_similarity
        self.top_k_similar_users = top_k_similar_users
        self.profile_weight = profile_weight
        self.history_weight = history_weight
        
        # Initialize feature builder
        self.feature_builder = EnhancedUserFeatureBuilder(
            sentence_model_name=sentence_model_name,
            profile_weight=profile_weight,
            history_weight=history_weight,
            use_openai=use_openai,
            openai_model=openai_model
        )
        
        # Fitted data
        self.user_features: Dict[str, np.ndarray] = {}
        self.user_triplets: Dict[str, Set[str]] = defaultdict(set)
        self.triplet_popularity: Dict[str, int] = defaultdict(int)
        self.triplet_scores: Dict[str, Dict[str, float]] = {}  # {user: {triplet: score}}
        
        # User info cache
        self.user_info_map: Dict[str, pd.Series] = {}
        self.df_history: Optional[pd.DataFrame] = None
    
    def fit(
        self,
        df_history: pd.DataFrame,
        df_user_info: Optional[pd.DataFrame] = None
    ) -> 'EnhancedUserCollaborativeRecommender':
        """
        Fit the enhanced collaborative recommender.
        
        Args:
            df_history: Training data with triplet column
            df_user_info: Optional user profile data
        """
        print("Fitting Enhanced User-Based Collaborative Recommender...")
        
        self.df_history = df_history.copy()
        
        # Build user info map
        if df_user_info is not None:
            for _, row in df_user_info.iterrows():
                user_id = row.get('linkedin_company_outsource')
                if pd.notna(user_id):
                    self.user_info_map[user_id] = row
        else:
            # Extract user info from history if not provided separately
            for _, row in df_history.iterrows():
                user_id = row.get('linkedin_company_outsource')
                if pd.notna(user_id) and user_id not in self.user_info_map:
                    self.user_info_map[user_id] = row
        
        # Fit feature builder
        self.feature_builder.fit(df_history, df_user_info)
        
        # Build features for each user
        unique_users = df_history['linkedin_company_outsource'].unique()
        
        print(f"Building features for {len(unique_users)} users...")
        
        for i, user_id in enumerate(unique_users):
            if i % 50 == 0:
                print(f"  Processing user {i+1}/{len(unique_users)}")
            
            # Get user info
            user_info = self.user_info_map.get(user_id)
            
            # Build combined features
            combined_features = self.feature_builder.build_combined_features(
                user_id, df_history, user_info
            )
            self.user_features[user_id] = combined_features
            
            # Store user's triplets
            user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
            if 'triplet' in user_history.columns:
                self.user_triplets[user_id] = set(user_history['triplet'].dropna().unique())
                
                # Update popularity
                for triplet in self.user_triplets[user_id]:
                    self.triplet_popularity[triplet] += 1
        
        print(f"Fitted on {len(self.user_features)} users")
        return self
    
    def find_similar_users(
        self,
        target_user_id: str,
        k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar users to target user.
        
        Uses combined profile + history similarity.
        """
        if k is None:
            k = self.top_k_similar_users
        
        # Build features for target user if not seen
        if target_user_id not in self.user_features:
            user_info = self.user_info_map.get(target_user_id)
            self.user_features[target_user_id] = self.feature_builder.build_combined_features(
                target_user_id, self.df_history, user_info
            )
        
        target_vector = self.user_features[target_user_id]
        
        # Calculate similarities with all other users
        similarities = []
        
        for user_id, user_vector in self.user_features.items():
            if user_id == target_user_id:
                continue
            
            # Cosine similarity (vectors are L2 normalized)
            sim = np.dot(target_vector, user_vector)
            
            if sim >= self.min_similarity:
                similarities.append((user_id, float(sim)))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def recommend_triplets(
        self,
        user_id: str,
        top_k: int = 10,
        exclude_seen: bool = True
    ) -> pd.DataFrame:
        """
        Recommend triplets based on similar users' preferences.
        """
        # Find similar users
        similar_users = self.find_similar_users(user_id)
        
        if not similar_users:
            # Fallback to popularity
            return self._recommend_by_popularity(top_k)
        
        # Aggregate triplets from similar users
        triplet_scores = defaultdict(float)
        triplet_sources = defaultdict(list)  # Track which users contributed
        
        seen_triplets = self.user_triplets.get(user_id, set())
        
        for similar_user, similarity in similar_users:
            similar_user_triplets = self.user_triplets.get(similar_user, set())
            
            for triplet in similar_user_triplets:
                # Skip if user has already seen this triplet
                if exclude_seen and triplet in seen_triplets:
                    continue
                
                # Weight by similarity
                triplet_scores[triplet] += similarity
                triplet_sources[triplet].append((similar_user, similarity))
        
        if not triplet_scores:
            return self._recommend_by_popularity(top_k, exclude=seen_triplets if exclude_seen else set())
        
        # Normalize scores by number of sources (avoid bias towards popular items)
        for triplet in triplet_scores:
            n_sources = len(triplet_sources[triplet])
            # Use harmonic mean-like normalization
            triplet_scores[triplet] = triplet_scores[triplet] / np.sqrt(n_sources)
        
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
    
    def _recommend_by_popularity(
        self,
        top_k: int,
        exclude: Set[str] = None
    ) -> pd.DataFrame:
        """Fallback: recommend most popular triplets."""
        exclude = exclude or set()
        
        # Filter out excluded triplets
        filtered_triplets = [
            (triplet, count) 
            for triplet, count in self.triplet_popularity.items()
            if triplet not in exclude
        ]
        
        # Sort by popularity
        sorted_triplets = sorted(
            filtered_triplets,
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        if not sorted_triplets:
            return pd.DataFrame(columns=['triplet', 'score'])
        
        # Normalize scores
        max_count = sorted_triplets[0][1] if sorted_triplets else 1
        
        results = pd.DataFrame([
            {'triplet': triplet, 'score': count / max_count}
            for triplet, count in sorted_triplets
        ])
        
        return results
    
    def get_user_profile_similarity(
        self,
        user1_id: str,
        user2_id: str
    ) -> float:
        """Get profile-based similarity between two users."""
        return self.feature_builder.compute_user_similarity(
            user1_id, user2_id, self.df_history, self.user_info_map
        )
