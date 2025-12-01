"""
Enhanced Triplet Content-Based Recommender
==========================================

Advanced content-based recommendation for triplets using:
1. OpenAI or Sentence Transformers for semantic text understanding
2. Multi-modal embeddings (text + categorical + numerical)
3. Industry hierarchy clustering
4. Cross-modal fusion

Adapted from EnhancedContentBasedRecommender to work with triplets.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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


class EnhancedTripletEmbedder:
    """
    Advanced embedding system for triplet-based recommendations.
    
    Combines:
    1. OpenAI or Sentence Transformers for semantic text understanding
    2. Triplet component embeddings (industry, size, services)
    3. Categorical embeddings for structured data
    4. Hierarchical industry embeddings
    5. Cross-modal fusion
    """
    
    def __init__(
        self,
        triplet_manager: TripletManager,
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        embedding_dim: int = 384,
        use_industry_hierarchy: bool = True,
        fusion_method: str = 'concat',  # 'concat', 'weighted_sum'
        device: str = None,
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
        self.triplet_manager = triplet_manager
        self.sentence_model_name = sentence_model_name
        self.embedding_dim = embedding_dim
        self.use_industry_hierarchy = use_industry_hierarchy
        self.fusion_method = fusion_method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
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
                print(f"EnhancedTripletEmbedder using OpenAI: {openai_model}")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embedder: {e}")
                self.sentence_model = None
        
        if self.sentence_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print(f"Loading Sentence Transformer: {sentence_model_name}")
                self.sentence_model = SentenceTransformer(sentence_model_name)
                self.sentence_model.to(self.device)
                print(f"EnhancedTripletEmbedder using SentenceTransformers: {sentence_model_name}")
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                raise ValueError("No embedding model available")
        
        # Learnable components
        self.industry_embeddings: Dict[str, np.ndarray] = {}
        self.industry_hierarchy: Dict[str, Dict] = {}
        self.location_embeddings: Dict[str, np.ndarray] = {}
        self.size_embeddings: Dict[str, np.ndarray] = {}
        self.service_embeddings: Dict[str, np.ndarray] = {}
        
        self.scaler = StandardScaler()
        self.pca = None
        self.industry_clusters = None
        
        # Embedding dimensions
        self.text_dim = self.sentence_model.get_sentence_embedding_dimension()
        
    def _create_industry_hierarchy(self, industries: List[str]) -> Dict[str, Dict]:
        """
        Create hierarchical clustering of industries for better embeddings.
        """
        if not self.use_industry_hierarchy or len(industries) < 3:
            return {}
        
        print(f"Creating industry hierarchy for {len(industries)} industries...")
        
        # Encode industry names
        industry_texts = [ind.lower().replace('&', 'and') for ind in industries]
        industry_embeddings = self.sentence_model.encode(
            industry_texts, 
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Cluster industries into groups
        n_clusters = min(10, max(2, len(industries) // 3))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(industry_embeddings)
        
        hierarchy = {}
        for i, industry in enumerate(industries):
            hierarchy[industry] = {
                'cluster': int(clusters[i]),
                'embedding': industry_embeddings[i],
                'cluster_center': kmeans.cluster_centers_[clusters[i]]
            }
            self.industry_embeddings[industry] = industry_embeddings[i]
        
        self.industry_clusters = kmeans
        return hierarchy
    
    def _create_size_embeddings(self):
        """Create embeddings for client size buckets."""
        size_buckets = ['micro', 'small', 'medium', 'large', 'enterprise', 'unknown']
        
        # Create semantic embeddings for size descriptions
        size_descriptions = [
            "micro company with 1-10 employees, very small team",
            "small company with 11-50 employees, small business",
            "medium company with 51-200 employees, medium enterprise",
            "large company with 201-1000 employees, large corporation",
            "enterprise company with over 1000 employees, major corporation",
            "unknown company size"
        ]
        
        size_embs = self.sentence_model.encode(
            size_descriptions,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        for bucket, emb in zip(size_buckets, size_embs):
            self.size_embeddings[bucket] = emb
    
    def _create_service_embeddings(self, all_services: List[str]):
        """Create embeddings for services."""
        if not all_services:
            return
        
        print(f"Creating service embeddings for {len(all_services)} unique services...")
        
        # Batch encode services
        service_embs = self.sentence_model.encode(
            all_services,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        for service, emb in zip(all_services, service_embs):
            self.service_embeddings[service] = emb
    
    def _encode_text_fields(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode text fields using Sentence Transformers.
        Combines services + background with smart concatenation.
        """
        combined_texts = []
        
        for _, row in df.iterrows():
            services = str(row.get('services', '')).strip()
            background = str(row.get('background', '')).strip()
            industry = str(row.get('industry', '')).strip()
            
            text_parts = []
            
            if industry and industry.lower() != 'nan':
                text_parts.append(f"Industry: {industry}")
            
            if services and services.lower() != 'nan':
                text_parts.append(f"Services: {services}")
            
            if background and background.lower() != 'nan':
                bg = background[:800] + "..." if len(background) > 800 else background
                text_parts.append(f"Background: {bg}")
            
            combined_text = " | ".join(text_parts) if text_parts else "No description available"
            combined_texts.append(combined_text)
        
        # Encode with Sentence Transformers
        embeddings = self.sentence_model.encode(
            combined_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32
        )
        
        return embeddings
    
    def _encode_triplet_components(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode triplet components (industry, size, services) using pre-computed embeddings.
        """
        if 'triplet' not in df.columns:
            return np.zeros((len(df), self.text_dim * 3))
        
        triplet_embeddings = []
        
        for _, row in df.iterrows():
            triplet = row.get('triplet', '')
            
            if pd.isna(triplet) or not triplet:
                triplet_embeddings.append(np.zeros(self.text_dim * 3))
                continue
            
            # Parse triplet
            industry, size, services_str = self.triplet_manager.parse_triplet(triplet)
            
            # Industry embedding
            if industry in self.industry_embeddings:
                ind_emb = self.industry_embeddings[industry]
            else:
                # Fallback: encode on the fly
                ind_emb = self.sentence_model.encode(
                    [industry],
                    convert_to_numpy=True,
                    show_progress_bar=False
                )[0]
            
            # Size embedding
            if size in self.size_embeddings:
                size_emb = self.size_embeddings[size]
            else:
                size_emb = self.size_embeddings.get('unknown', np.zeros(self.text_dim))
            
            # Services embedding (average of individual service embeddings)
            services = [s.strip() for s in services_str.split(',') if s.strip() and s != 'unknown']
            
            if services:
                service_embs = []
                for svc in services:
                    if svc in self.service_embeddings:
                        service_embs.append(self.service_embeddings[svc])
                    else:
                        # Encode on the fly
                        svc_emb = self.sentence_model.encode(
                            [svc],
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )[0]
                        service_embs.append(svc_emb)
                
                svc_emb = np.mean(service_embs, axis=0)
            else:
                svc_emb = np.zeros(self.text_dim)
            
            # Concatenate triplet components
            triplet_emb = np.concatenate([ind_emb, size_emb, svc_emb])
            triplet_embeddings.append(triplet_emb)
        
        return np.array(triplet_embeddings)
    
    def _encode_categorical_fields(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create embeddings for categorical fields (location).
        """
        categorical_features = []
        
        # Location embeddings
        if 'location' in df.columns:
            location_vecs = []
            for loc in df['location'].fillna('Unknown'):
                if loc in self.location_embeddings:
                    location_vecs.append(self.location_embeddings[loc])
                else:
                    # Create hash-based embedding for unseen locations
                    np.random.seed(hash(str(loc)) % 2**31)
                    location_vecs.append(np.random.normal(0, 0.1, 32))
            
            categorical_features.append(np.array(location_vecs))
        
        if categorical_features:
            return np.hstack(categorical_features)
        else:
            return np.zeros((len(df), 32))
    
    def _encode_numerical_fields(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Encode and normalize numerical fields.
        """
        numerical_cols = []
        df_temp = df.copy()
        
        if 'client_min' in df.columns and 'client_max' in df.columns:
            df_temp['client_size_mid'] = df_temp[['client_min', 'client_max']].mean(axis=1, skipna=True)
            numerical_cols.append('client_size_mid')
        
        if numerical_cols:
            numerical_data = df_temp[numerical_cols].fillna(0).values
            
            if fit:
                return self.scaler.fit_transform(numerical_data)
            else:
                return self.scaler.transform(numerical_data)
        else:
            return np.zeros((len(df), 1))
    
    def _fuse_embeddings(
        self,
        text_emb: np.ndarray,
        triplet_emb: np.ndarray,
        cat_emb: np.ndarray,
        num_emb: np.ndarray,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Fuse different types of embeddings.
        """
        weights = weights or {
            'text': 0.3,
            'triplet': 0.4,
            'categorical': 0.2,
            'numerical': 0.1
        }
        
        if self.fusion_method == 'concat':
            # Weighted concatenation
            combined = np.hstack([
                text_emb * weights['text'],
                triplet_emb * weights['triplet'],
                cat_emb * weights['categorical'],
                num_emb * weights['numerical']
            ])
            
            # Reduce dimensionality if needed
            if combined.shape[1] > self.embedding_dim * 2:
                if self.pca is None:
                    n_components = min(self.embedding_dim, combined.shape[1] - 1, combined.shape[0] - 1)
                    self.pca = PCA(n_components=n_components, random_state=42)
                    combined = self.pca.fit_transform(combined)
                else:
                    combined = self.pca.transform(combined)
            
            return combined
        
        elif self.fusion_method == 'weighted_sum':
            # Resize all to text_dim and weighted sum
            target_dim = self.text_dim
            
            # Resize triplet (3 * text_dim -> text_dim)
            triplet_resized = triplet_emb[:, :target_dim] if triplet_emb.shape[1] >= target_dim else \
                np.pad(triplet_emb, ((0, 0), (0, target_dim - triplet_emb.shape[1])))
            
            # Resize categorical
            cat_resized = np.pad(cat_emb, ((0, 0), (0, max(0, target_dim - cat_emb.shape[1]))))[:, :target_dim]
            
            # Resize numerical
            num_resized = np.pad(num_emb, ((0, 0), (0, max(0, target_dim - num_emb.shape[1]))))[:, :target_dim]
            
            combined = (
                weights['text'] * text_emb +
                weights['triplet'] * triplet_resized +
                weights['categorical'] * cat_resized +
                weights['numerical'] * num_resized
            )
            
            return combined
        
        else:
            # Default to concat
            return np.hstack([text_emb, triplet_emb, cat_emb, num_emb])
    
    def fit(self, df: pd.DataFrame) -> 'EnhancedTripletEmbedder':
        """
        Fit the embedder on training data.
        """
        print("Fitting Enhanced Triplet Embedder...")
        
        # 1. Create industry hierarchy
        if 'industry' in df.columns:
            unique_industries = df['industry'].dropna().unique().tolist()
            self.industry_hierarchy = self._create_industry_hierarchy(unique_industries)
        
        # 2. Create size embeddings
        self._create_size_embeddings()
        
        # 3. Create service embeddings
        all_services = set()
        for svc_str in df['services'].dropna():
            services = [s.strip() for s in str(svc_str).split(',')]
            all_services.update([s for s in services if s])
        
        self._create_service_embeddings(list(all_services))
        
        # 4. Create location embeddings
        if 'location' in df.columns:
            unique_locations = df['location'].dropna().unique().tolist()
            for loc in unique_locations[:100]:
                np.random.seed(hash(str(loc)) % 2**31)
                self.location_embeddings[loc] = np.random.normal(0, 0.1, 32)
        
        # 5. Fit numerical scaler
        self._encode_numerical_fields(df, fit=True)
        
        print("Enhanced Triplet Embedder fitted!")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data to enhanced triplet embeddings.
        """
        # Encode different modalities
        text_emb = self._encode_text_fields(df)
        triplet_emb = self._encode_triplet_components(df)
        cat_emb = self._encode_categorical_fields(df)
        num_emb = self._encode_numerical_fields(df, fit=False)
        
        # Fuse embeddings
        final_embeddings = self._fuse_embeddings(text_emb, triplet_emb, cat_emb, num_emb)
        
        # L2 normalize
        norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True) + 1e-12
        final_embeddings = final_embeddings / norms
        
        return final_embeddings.astype(np.float32)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class EnhancedTripletContentRecommender:
    """
    Enhanced content-based recommender for triplets using advanced embeddings.
    """
    
    def __init__(
        self,
        df_history: pd.DataFrame,
        df_test: pd.DataFrame,
        triplet_manager: TripletManager,
        embedding_config: Optional[Dict] = None,
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
        # Split train/val
        data_train, data_val = train_test_split(df_history, test_size=0.2, random_state=42)
        
        self.data_train = data_train.copy()
        self.data_val = data_val.copy()
        self.df_test = df_test.copy()
        self.triplet_manager = triplet_manager
        
        # Default embedding config
        self.embedding_config = embedding_config or {
            'sentence_model_name': 'all-MiniLM-L6-v2',
            'embedding_dim': 384,
            'use_industry_hierarchy': True,
            'fusion_method': 'concat'
        }
        
        # Add OpenAI config
        self.embedding_config['use_openai'] = use_openai
        self.embedding_config['openai_model'] = openai_model
        
        print("Building enhanced triplet feature representations...")
        
        # Initialize and fit embedder
        self.embedder = EnhancedTripletEmbedder(
            triplet_manager=triplet_manager,
            **self.embedding_config
        )
        self.embedder.fit(data_train)
        
        # Transform datasets
        print("Transforming training data...")
        self.X_train = self.embedder.transform(data_train)
        
        print("Transforming validation data...")
        self.X_val = self.embedder.transform(data_val)
        
        print("Transforming test data...")
        self.X_test = self.embedder.transform(df_test)
        
        print(f"Enhanced triplet embeddings shape: {self.X_test.shape}")
    
    def build_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """
        Build enhanced user profile from historical interactions.
        """
        user_history = self.data_train[self.data_train['linkedin_company_outsource'] == user_id]
        
        if user_history.empty:
            return None
        
        # Transform user history to embeddings
        user_embeddings = self.embedder.transform(user_history)
        
        # Aggregate user profile (mean pooling)
        user_profile = np.mean(user_embeddings, axis=0, keepdims=True)
        
        return user_profile
    
    def recommend_triplets(
        self,
        user_id: str,
        top_k: int = 10,
        mode: Literal['val', 'test'] = 'test'
    ) -> pd.DataFrame:
        """
        Generate triplet recommendations using enhanced embeddings.
        """
        user_profile = self.build_user_profile(user_id)
        
        if user_profile is None:
            return pd.DataFrame(columns=['triplet', 'score'])
        
        # Select candidate set
        if mode == 'val':
            X_candidates = self.X_val
            df_candidates = self.data_val
        else:
            X_candidates = self.X_test
            df_candidates = self.df_test
        
        # Compute similarities
        similarities = np.dot(X_candidates, user_profile.T).ravel()
        
        # Add scores to dataframe
        df_scored = df_candidates.copy()
        df_scored['score'] = similarities
        
        # Aggregate by triplet (taking max score)
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
            
            return triplet_results
        else:
            return pd.DataFrame(columns=['triplet', 'score'])
