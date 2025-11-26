"""
User-Based Collaborative Filtering
===================================

Recommends items based on similarity between users (outsource companies).
If User A and User B are similar (same services, similar past projects),
then items that A interacted with are recommended to B.

Supports both OpenAI and SentenceTransformers embeddings.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

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


class UserFeatureBuilder:
    """
    Build comprehensive feature vectors for users (outsource companies).
    """
    
    def __init__(
        self,
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        device: str = None,
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
        self.sentence_model_name = sentence_model_name
        self.device = device or 'cpu'
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
                print(f"UserFeatureBuilder using OpenAI: {openai_model}")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embedder: {e}")
                self.sentence_model = None
        
        if self.sentence_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(sentence_model_name)
                self.sentence_model.to(self.device)
                print(f"UserFeatureBuilder using SentenceTransformers: {sentence_model_name}")
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.sentence_model = None
        
        # Learned statistics
        self.global_industry_freq: Dict[str, float] = {}
        self.global_service_freq: Dict[str, float] = {}
        self.scaler = StandardScaler()
        
    def fit(self, df_history: pd.DataFrame) -> 'UserFeatureBuilder':
        """
        Fit on training history to learn global statistics.
        
        Args:
            df_history: Historical interactions (training data only)
        """
        # Calculate global industry frequencies
        industry_counts = df_history['industry'].value_counts()
        total = len(df_history)
        self.global_industry_freq = {
            ind: count / total 
            for ind, count in industry_counts.items()
        }
        
        # Calculate global service frequencies
        all_services = []
        for services_str in df_history['services'].dropna():
            services = [s.strip() for s in str(services_str).split(',')]
            all_services.extend(services)
        
        from collections import Counter
        service_counts = Counter(all_services)
        total_services = sum(service_counts.values())
        
        self.global_service_freq = {
            svc: count / total_services
            for svc, count in service_counts.items()
        }
        
        # Fit scaler on numerical features
        numerical_features = []
        
        for user_id in df_history['linkedin_company_outsource'].unique():
            user_hist = df_history[df_history['linkedin_company_outsource'] == user_id]
            
            # Basic stats
            n_interactions = len(user_hist)
            
            # Client size stats
            if 'client_size_mid' in user_hist.columns:
                avg_client_size = user_hist['client_size_mid'].mean()
            else:
                avg_client_size = 0.0
            
            numerical_features.append([n_interactions, avg_client_size])
        
        if numerical_features:
            self.scaler.fit(numerical_features)
        
        return self
    
    def build_user_features(
        self,
        user_id: str,
        df_history: pd.DataFrame,
        user_info: Optional[pd.Series] = None
    ) -> Dict[str, np.ndarray]:
        """
        Build comprehensive features for a user.
        
        Args:
            user_id: User identifier (linkedin_company_outsource)
            df_history: Historical interaction data
            user_info: Optional row with user's company info
                      (website_outsource_url, description_company_outsource, 
                       services_company_outsource)
        
        Returns:
            Dictionary with different feature types
        """
        features = {}
        
        # 1. Historical interaction features
        user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
        
        if not user_history.empty:
            # Industry preference vector (TF-IDF style)
            industry_vector = self._build_industry_vector(user_history)
            features['industry_preferences'] = industry_vector
            
            # Service usage vector
            service_vector = self._build_service_vector(user_history)
            features['service_usage'] = service_vector
            
            # Client size statistics
            size_stats = self._build_size_statistics(user_history)
            features['size_statistics'] = size_stats
            
            # Location preferences
            location_vector = self._build_location_vector(user_history)
            features['location_preferences'] = location_vector
            
            # Numerical statistics
            numerical_stats = self._build_numerical_stats(user_history)
            features['numerical_stats'] = numerical_stats
        else:
            # Cold start user - use defaults
            features['industry_preferences'] = np.zeros(len(self.global_industry_freq))
            features['service_usage'] = np.zeros(len(self.global_service_freq))
            features['size_statistics'] = np.zeros(5)  # 5 size buckets
            features['location_preferences'] = np.zeros(10)  # Top 10 locations
            features['numerical_stats'] = np.zeros(2)
        
        # 2. Company profile features (if available)
        if user_info is not None:
            # Get embedding dimension
            embed_dim = self.sentence_model.get_sentence_embedding_dimension() if self.sentence_model else 1536
            
            # Embed company description
            if 'description_company_outsource' in user_info and pd.notna(user_info['description_company_outsource']):
                desc_text = str(user_info['description_company_outsource'])
                if self.sentence_model:
                    desc_emb = self.sentence_model.encode([desc_text], convert_to_numpy=True)[0]
                    features['company_description'] = desc_emb
                else:
                    features['company_description'] = np.zeros(embed_dim)
            else:
                features['company_description'] = np.zeros(embed_dim if self.sentence_model else 1536)
            
            # Embed company services
            if 'services_company_outsource' in user_info and pd.notna(user_info['services_company_outsource']):
                svc_text = str(user_info['services_company_outsource'])
                if self.sentence_model:
                    svc_emb = self.sentence_model.encode([svc_text], convert_to_numpy=True)[0]
                    features['company_services'] = svc_emb
                else:
                    features['company_services'] = np.zeros(embed_dim if self.sentence_model else 1536)
            else:
                features['company_services'] = np.zeros(embed_dim if self.sentence_model else 1536)
        else:
            embed_dim = self.sentence_model.get_sentence_embedding_dimension() if self.sentence_model else 1536
            features['company_description'] = np.zeros(embed_dim)
            features['company_services'] = np.zeros(embed_dim)
        
        return features
    
    def _build_industry_vector(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build TF-IDF style industry preference vector."""
        # Count industries for this user
        industry_counts = user_history['industry'].value_counts()
        
        # Calculate TF-IDF
        vector = np.zeros(len(self.global_industry_freq))
        industry_to_idx = {ind: i for i, ind in enumerate(self.global_industry_freq.keys())}
        
        total_user_interactions = len(user_history)
        
        for industry, count in industry_counts.items():
            if industry in industry_to_idx:
                tf = count / total_user_interactions
                idf = np.log(1.0 / (self.global_industry_freq[industry] + 1e-6))
                vector[industry_to_idx[industry]] = tf * idf
        
        # L2 normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _build_service_vector(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build service usage vector from historical projects."""
        service_counts = defaultdict(int)
        
        for services_str in user_history['services'].dropna():
            services = [s.strip() for s in str(services_str).split(',')]
            for svc in services:
                if svc:
                    service_counts[svc] += 1
        
        # Create vector
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
    
    def _build_size_statistics(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build client size preference distribution."""
        size_buckets = ['micro', 'small', 'medium', 'large', 'enterprise']
        size_counts = np.zeros(len(size_buckets))
        
        # We need to infer size bucket from client_min/max
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
        
        # Normalize to distribution
        total = size_counts.sum()
        if total > 0:
            size_counts = size_counts / total
        
        return size_counts
    
    def _build_location_vector(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build location preference vector (top 10 locations)."""
        location_counts = user_history['location'].value_counts()
        
        # Top 10 locations
        top_locations = location_counts.head(10)
        vector = np.zeros(10)
        
        total = len(user_history)
        for i, (loc, count) in enumerate(top_locations.items()):
            vector[i] = count / total
        
        return vector
    
    def _build_numerical_stats(self, user_history: pd.DataFrame) -> np.ndarray:
        """Build numerical statistics."""
        n_interactions = len(user_history)
        
        # Average client size
        if 'client_size_mid' in user_history.columns:
            avg_client_size = user_history['client_size_mid'].mean()
            if pd.isna(avg_client_size):
                avg_client_size = 0.0
        else:
            avg_client_size = 0.0
        
        stats = np.array([n_interactions, avg_client_size])
        
        # Scale
        try:
            stats = self.scaler.transform([stats])[0]
        except Exception:
            pass
        
        return stats
    
    def get_combined_feature_vector(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine all feature vectors into one."""
        vectors = [
            features.get('industry_preferences', np.array([])),
            features.get('service_usage', np.array([])),
            features.get('size_statistics', np.array([])),
            features.get('location_preferences', np.array([])),
            features.get('numerical_stats', np.array([])),
            features.get('company_description', np.array([])),
            features.get('company_services', np.array([]))
        ]
        
        # Filter out empty arrays
        vectors = [v for v in vectors if len(v) > 0]
        
        if not vectors:
            return np.array([])
        
        combined = np.concatenate(vectors)
        
        # L2 normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined


class UserBasedCollaborativeRecommender:
    """
    User-based collaborative filtering for triplet recommendation.
    """
    
    def __init__(
        self,
        min_similarity: float = 0.1,
        top_k_similar_users: int = 20,
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
        self.min_similarity = min_similarity
        self.top_k_similar_users = top_k_similar_users
        
        self.feature_builder = UserFeatureBuilder(
            sentence_model_name=sentence_model_name,
            use_openai=use_openai,
            openai_model=openai_model
        )
        
        # Fitted data
        self.user_features: Dict[str, np.ndarray] = {}
        self.user_triplets: Dict[str, Set[str]] = defaultdict(set)
        self.triplet_popularity: Dict[str, int] = defaultdict(int)
        
    def fit(
        self,
        df_history: pd.DataFrame,
        df_user_info: Optional[pd.DataFrame] = None
    ) -> 'UserBasedCollaborativeRecommender':
        """
        Fit the collaborative recommender on training data.
        
        Args:
            df_history: Training interaction history with triplet column
            df_user_info: Optional user company information
        """
        print("Fitting User-Based Collaborative Recommender...")
        
        # Fit feature builder
        self.feature_builder.fit(df_history)
        
        # Build features for each user
        unique_users = df_history['linkedin_company_outsource'].unique()
        
        for user_id in unique_users:
            # Get user info if available
            user_info = None
            if df_user_info is not None:
                user_rows = df_user_info[df_user_info['linkedin_company_outsource'] == user_id]
                if not user_rows.empty:
                    user_info = user_rows.iloc[0]
            
            # Build features
            features = self.feature_builder.build_user_features(
                user_id, df_history, user_info
            )
            
            # Combine into single vector
            combined = self.feature_builder.get_combined_feature_vector(features)
            self.user_features[user_id] = combined
            
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
        
        Returns:
            List of (user_id, similarity_score) tuples, sorted by similarity
        """
        if k is None:
            k = self.top_k_similar_users
        
        if target_user_id not in self.user_features:
            return []
        
        target_vector = self.user_features[target_user_id]
        
        # Calculate similarities
        similarities = []
        
        for user_id, user_vector in self.user_features.items():
            if user_id == target_user_id:
                continue
            
            # Cosine similarity
            sim = np.dot(target_vector, user_vector)
            
            if sim >= self.min_similarity:
                similarities.append((user_id, float(sim)))
        
        # Sort by similarity
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
        
        Returns:
            DataFrame with columns: [triplet, score]
        """
        # Find similar users
        similar_users = self.find_similar_users(user_id)
        
        if not similar_users:
            # Fallback to popularity
            return self._recommend_by_popularity(top_k)
        
        # Aggregate triplets from similar users
        triplet_scores = defaultdict(float)
        seen_triplets = self.user_triplets.get(user_id, set())
        
        for similar_user, similarity in similar_users:
            similar_user_triplets = self.user_triplets.get(similar_user, set())
            
            for triplet in similar_user_triplets:
                # Skip if user has already seen this triplet
                if exclude_seen and triplet in seen_triplets:
                    continue
                
                # Weight by similarity
                triplet_scores[triplet] += similarity
        
        if not triplet_scores:
            return self._recommend_by_popularity(top_k)
        
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
        sorted_triplets = sorted(
            self.triplet_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Normalize scores
        max_count = sorted_triplets[0][1] if sorted_triplets else 1
        
        results = pd.DataFrame([
            {'triplet': triplet, 'score': count / max_count}
            for triplet, count in sorted_triplets
        ])
        
        return results
