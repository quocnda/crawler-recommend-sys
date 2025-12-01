"""
Triplet-based Content Recommender
==================================

Content-based recommendation using triplet embeddings.
Supports both OpenAI and SentenceTransformers embeddings.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal, Union
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
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


class TripletContentRecommender:
    """
    Content-based recommender using triplet embeddings.
    """
    
    def __init__(
        self,
        df_history: pd.DataFrame,
        df_test: pd.DataFrame,
        triplet_manager: TripletManager,
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        embedding_weights: Dict[str, float] = None,
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
        """
        Args:
            df_history: Training data
            df_test: Test/candidate data
            triplet_manager: Fitted TripletManager
            sentence_model_name: Model for text embeddings (fallback)
            embedding_weights: Weights for different embedding components
            use_openai: Whether to use OpenAI embeddings (default: True)
            openai_model: OpenAI embedding model name
        """
        # Split train/val
        data_train, data_val = train_test_split(df_history, test_size=0.2, random_state=42)
        
        self.data_train = data_train.copy()
        self.data_val = data_val.copy()
        self.df_test = df_test.copy()
        self.triplet_manager = triplet_manager
        self.use_openai = use_openai
        
        # Default embedding weights
        self.embedding_weights = embedding_weights or {
            'triplet_structure': 0.3,    # Industry, size, services structure
            'background_text': 0.3,      # Background description
            'services_text': 0.2,        # Services detailed text
            'location': 0.1,             # Geographic features
            'numerical': 0.1             # Client size, budget numerical
        }
        
        # Initialize embedder (OpenAI or SentenceTransformers)
        self.sentence_model = None
        
        if use_openai and OPENAI_EMBEDDER_AVAILABLE:
            try:
                self.sentence_model = HybridEmbedder(
                    use_openai=True,
                    openai_model=openai_model,
                    sentence_model=sentence_model_name
                )
                print(f"Using OpenAI embeddings: {openai_model}")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embedder: {e}")
                self.sentence_model = None
        
        if self.sentence_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(sentence_model_name)
                print(f"Using SentenceTransformers: {sentence_model_name}")
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                self.sentence_model = None
        
        self.scaler = StandardScaler()
        
        # Vocabularies (will be fitted on training data)
        self.industry_vocab = None
        self.size_vocab = None
        self.service_vocab = None
        
        # Build features
        print("Building triplet-based features...")
        self._build_features()
    
    def _build_features(self):
        """Build feature matrices for train, val, and test sets."""
        # Build vocabularies from TRAINING data only
        print("Building vocabularies from training data...")
        self._build_vocabularies(self.data_train)
        
        # Build features for each set using same vocabulary
        print("Transforming training data...")
        self.X_train = self._transform_to_features(self.data_train, fit=True)
        print(f'X_train shape: {self.X_train.shape}')
        
        print("Transforming validation data...")
        self.X_val = self._transform_to_features(self.data_val, fit=False)
        print(f'X_val shape: {self.X_val.shape}')
        
        print("Transforming test data...")
        self.X_test = self._transform_to_features(self.df_test, fit=False)
        print(f'X_test shape: {self.X_test.shape}')
    
    def _build_vocabularies(self, df: pd.DataFrame):
        """
        Build vocabularies from training data only to ensure consistent dimensions.
        """
        if 'triplet' not in df.columns:
            raise ValueError("DataFrame must have 'triplet' column")
        
        # Parse all triplets
        parsed = df['triplet'].apply(self.triplet_manager.parse_triplet)
        
        industries = [p[0] for p in parsed]
        sizes = [p[1] for p in parsed]
        services_list = [p[2] for p in parsed]
        
        # Build industry vocabulary
        self.industry_vocab = sorted(set(industries))
        print(f"  Industry vocabulary: {len(self.industry_vocab)} unique industries")
        
        # Build size vocabulary
        self.size_vocab = ['micro', 'small', 'medium', 'large', 'enterprise', 'unknown']
        print(f"  Size vocabulary: {len(self.size_vocab)} size buckets")
        
        # Build service vocabulary
        all_services = set()
        for svc_str in services_list:
            if svc_str != 'unknown':
                services = [s.strip() for s in svc_str.split(',') if s.strip()]
                all_services.update(services)
        
        self.service_vocab = sorted(all_services)
        print(f"  Service vocabulary: {len(self.service_vocab)} unique services")
            
    def _embed_triplet_structure(self, df: pd.DataFrame) -> np.ndarray:
        """
        Embed the triplet structure (industry, size, services) using one-hot encoding.
        Uses pre-built vocabularies to ensure consistent dimensions.
        """
        if 'triplet' not in df.columns:
            raise ValueError("DataFrame must have 'triplet' column")
        
        if self.industry_vocab is None or self.size_vocab is None or self.service_vocab is None:
            raise ValueError("Vocabularies not built. Call _build_vocabularies first.")
        
        # Parse triplets
        parsed = df['triplet'].apply(self.triplet_manager.parse_triplet)
        
        industries = [p[0] for p in parsed]
        sizes = [p[1] for p in parsed]
        services_list = [p[2] for p in parsed]
        
        # One-hot encode industry using vocabulary
        industry_map = {ind: i for i, ind in enumerate(self.industry_vocab)}
        industry_matrix = np.zeros((len(df), len(self.industry_vocab)))
        
        for i, ind in enumerate(industries):
            if ind in industry_map:
                industry_matrix[i, industry_map[ind]] = 1.0
            # If industry not in vocab (new industry in test), leave as zeros
        
        # One-hot encode size using vocabulary
        size_map = {s: i for i, s in enumerate(self.size_vocab)}
        size_matrix = np.zeros((len(df), len(self.size_vocab)))
        
        for i, size in enumerate(sizes):
            if size in size_map:
                size_matrix[i, size_map[size]] = 1.0
        
        # Multi-hot encode services using vocabulary
        service_map = {svc: i for i, svc in enumerate(self.service_vocab)}
        service_matrix = np.zeros((len(df), len(self.service_vocab)))
        
        for i, svc_str in enumerate(services_list):
            if svc_str != 'unknown':
                services = [s.strip() for s in svc_str.split(',') if s.strip()]
                for svc in services:
                    if svc in service_map:
                        service_matrix[i, service_map[svc]] = 1.0
                    # If service not in vocab, ignore (OOV handling)
        
        # Concatenate
        triplet_features = np.hstack([industry_matrix, size_matrix, service_matrix])
        
        return triplet_features
    
    def _embed_text_fields(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Embed text fields using OpenAI or SentenceTransformers."""
        embeddings = {}
        
        # Get embedding dimension
        if self.sentence_model is None:
            embed_dim = 1536 if self.use_openai else 384
            return {
                'background': np.zeros((len(df), embed_dim)),
                'services': np.zeros((len(df), embed_dim))
            }
        
        embed_dim = self.sentence_model.get_sentence_embedding_dimension()
        
        # Background text
        background_texts = []
        for _, row in df.iterrows():
            bg = str(row.get('background', '')).strip()
            if not bg or bg.lower() == 'nan':
                bg = "No background information"
            background_texts.append(bg[:1000])  # Truncate long texts
        
        embeddings['background'] = self.sentence_model.encode(
            background_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        # Services text
        services_texts = []
        for _, row in df.iterrows():
            svc = str(row.get('services', '')).strip()
            if not svc or svc.lower() == 'nan':
                svc = "No services specified"
            services_texts.append(svc)
        
        embeddings['services'] = self.sentence_model.encode(
            services_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        return embeddings
    
    def _embed_location(self, df: pd.DataFrame) -> np.ndarray:
        """Simple one-hot encoding for top locations."""
        locations = df['location'].fillna('Unknown')
        
        # Get top 20 locations from training data
        if hasattr(self, 'top_locations'):
            top_locs = self.top_locations
        else:
            top_locs = self.data_train['location'].value_counts().head(20).index.tolist()
            self.top_locations = top_locs
        
        location_matrix = np.zeros((len(df), len(top_locs)))
        
        for i, loc in enumerate(locations):
            if loc in top_locs:
                idx = top_locs.index(loc)
                location_matrix[i, idx] = 1.0
        
        return location_matrix
    
    def _embed_numerical(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Embed numerical features."""
        numerical_features = []
        
        for _, row in df.iterrows():
            features = []
            
            # Client size
            if 'client_min' in row and 'client_max' in row:
                client_min = row['client_min'] if pd.notna(row['client_min']) else 0
                client_max = row['client_max'] if pd.notna(row['client_max']) else 0
                client_mid = (client_min + client_max) / 2
            else:
                client_mid = 0
            
            features.append(client_mid)
                        
            numerical_features.append(features)
        
        numerical_features = np.array(numerical_features)
        
        # Scale
        if fit:
            numerical_features = self.scaler.fit_transform(numerical_features)
        else:
            numerical_features = self.scaler.transform(numerical_features)
        
        return numerical_features
    
    def _transform_to_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Transform dataframe to feature matrix.
        
        Args:
            df: Input dataframe with triplet column
            fit: Whether to fit transformers
        """
        feature_blocks = []
        weights = []
        
        # 1. Triplet structure features
        triplet_features = self._embed_triplet_structure(df)
        feature_blocks.append(triplet_features)
        weights.append(self.embedding_weights['triplet_structure'])
        
        # 2. Text embeddings
        text_embeddings = self._embed_text_fields(df)
        
        # Background
        feature_blocks.append(text_embeddings['background'])
        weights.append(self.embedding_weights['background_text'])
        
        # Services
        feature_blocks.append(text_embeddings['services'])
        weights.append(self.embedding_weights['services_text'])
        
        # 3. Location
        location_features = self._embed_location(df)
        feature_blocks.append(location_features)
        weights.append(self.embedding_weights['location'])
        
        # 4. Numerical
        numerical_features = self._embed_numerical(df, fit=fit)
        feature_blocks.append(numerical_features)
        weights.append(self.embedding_weights['numerical'])
        
        # Combine with weights
        weighted_blocks = []
        for block, weight in zip(feature_blocks, weights):
            weighted_blocks.append(block * weight)
        
        combined = np.hstack(weighted_blocks)
        
        # L2 normalize rows
        norms = np.linalg.norm(combined, axis=1, keepdims=True) + 1e-12
        combined = combined / norms
        
        return combined.astype(np.float32)
    
    def build_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """
        Build user profile from historical interactions.
        
        Returns:
            User profile vector (mean pooling of history)
        """
        user_history = self.data_train[self.data_train['linkedin_company_outsource'] == user_id]
        
        if user_history.empty:
            return None
        
        # Get feature vectors for user's history
        user_features = self._transform_to_features(user_history, fit=False)
        
        # Mean pooling
        user_profile = np.mean(user_features, axis=0, keepdims=True)
        
        return user_profile
    
    def recommend_triplets(
        self,
        user_id: str,
        top_k: int = 10,
        mode: Literal['val', 'test'] = 'test'
    ) -> pd.DataFrame:
        """
        Recommend triplets for a user.
        
        Args:
            user_id: User identifier
            top_k: Number of recommendations
            mode: 'val' for validation set, 'test' for test set
        
        Returns:
            DataFrame with columns: [triplet, score, industry, client_size, services]
        """
        # Build user profile
        user_profile = self.build_user_profile(user_id)
        
        if user_profile is None:
            # Cold start - return empty or most popular
            return pd.DataFrame(columns=['triplet', 'score', 'industry', 'client_size', 'services'])
        
        # Select candidate set
        if mode == 'val':
            X_candidates = self.X_val
            df_candidates = self.data_val
        else:
            X_candidates = self.X_test
            df_candidates = self.df_test
        
        # Compute similarities
        similarities = cosine_similarity(X_candidates, user_profile).ravel()
        
        # Add scores to dataframe
        df_scored = df_candidates.copy()
        df_scored['score'] = similarities
        
        # Aggregate by triplet (max score)
        if 'triplet' in df_scored.columns:
            triplet_results = (
                df_scored.groupby('triplet')
                .agg({
                    'score': 'max',
                    'industry': 'first',
                    'location': 'first',
                    'services': 'first'
                })
                .reset_index()
                .sort_values('score', ascending=False)
                .head(top_k)
            )
            
            # Parse triplet components
            triplet_results['triplet_industry'] = triplet_results['triplet'].apply(
                lambda x: self.triplet_manager.parse_triplet(x)[0]
            )
            triplet_results['triplet_client_size'] = triplet_results['triplet'].apply(
                lambda x: self.triplet_manager.parse_triplet(x)[1]
            )
            triplet_results['triplet_services'] = triplet_results['triplet'].apply(
                lambda x: self.triplet_manager.parse_triplet(x)[2]
            )
            
            return triplet_results
        else:
            return pd.DataFrame(columns=['triplet', 'score', 'industry', 'client_size', 'services'])
