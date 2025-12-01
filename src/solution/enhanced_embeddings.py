"""
Deep Learning Embedding Enhancement
==================================

This module implements advanced embedding strategies to improve content representation:
1. OpenAI Embeddings or Sentence Transformers for better semantic understanding
2. Multi-modal embeddings (text + categorical features)
3. Domain-specific embedding fine-tuning
4. Hierarchical embeddings for industry categories
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Optional, Union
import torch
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
from sklearn.model_selection import train_test_split
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


class EnhancedEmbedder:
    """
    Advanced embedding system that combines multiple embedding techniques:
    1. OpenAI Embeddings or Sentence Transformers for semantic text understanding
    2. Categorical embeddings for structured data
    3. Hierarchical industry embeddings
    4. Cross-modal fusion
    """
    
    def __init__(
        self,
        sentence_model_name: str = 'all-MiniLM-L6-v2',
        embedding_dim: int = 384,
        use_industry_hierarchy: bool = True,
        fusion_method: str = 'concat',  # 'concat', 'weighted_sum', 'attention'
        device: str = None,
        use_openai: bool = True,  # NEW: Use OpenAI embeddings
        openai_model: str = 'text-embedding-3-small'  # NEW: OpenAI model
    ):
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
                print(f"EnhancedEmbedder using OpenAI: {openai_model}")
            except Exception as e:
                print(f"Warning: Could not initialize OpenAI embedder: {e}")
                self.sentence_model = None
        
        if self.sentence_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(sentence_model_name)
                self.sentence_model.to(self.device)
                print(f"EnhancedEmbedder using SentenceTransformers: {sentence_model_name}")
            except Exception as e:
                print(f"Warning: Could not load sentence transformer: {e}")
                raise ValueError("No embedding model available")
        
        # Learnable components
        self.industry_embeddings = {}
        self.location_embeddings = {}
        self.scaler = StandardScaler()
        self.pca = None
        self.industry_clusters = None
        
        # Embedding dimensions
        self.text_dim = self.sentence_model.get_sentence_embedding_dimension()
        self.cat_dim = 0  # Will be set during fit
        self.final_dim = embedding_dim
        
    def _create_industry_hierarchy(self, industries: List[str]) -> Dict[str, Dict]:
        """
        Create hierarchical clustering of industries for better embeddings.
        """
        if not self.use_industry_hierarchy:
            return {}
            
        # Simple text-based clustering of industries
        industry_texts = [ind.lower().replace('&', 'and') for ind in industries]
        industry_embeddings = self.sentence_model.encode(industry_texts, convert_to_numpy=True)
        
        # Cluster industries into groups
        n_clusters = min(10, len(industries) // 3 + 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(industry_embeddings)
        
        hierarchy = {}
        for i, industry in enumerate(industries):
            hierarchy[industry] = {
                'cluster': int(clusters[i]),
                'embedding': industry_embeddings[i],
                'cluster_center': kmeans.cluster_centers_[clusters[i]]
            }
            
        self.industry_clusters = kmeans
        return hierarchy
    
    def _encode_text_fields(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode text fields using Sentence Transformers.
        Combines services + project_description with smart concatenation.
        """
        combined_texts = []
        
        for _, row in df.iterrows():
            # Combine multiple text fields intelligently
            services = str(row.get('services', '')).strip()
            # description = str(row.get('project_description', '')).strip()
            background = str(row.get('background', '')).strip()
            
            # Create rich text representation
            text_parts = []
            
            if services and services.lower() != 'nan':
                text_parts.append(f"Services: {services}")
                
            # if description and description.lower() != 'nan':
            #     # Truncate very long descriptions
            #     desc = description[:500] + "..." if len(description) > 500 else description
            #     text_parts.append(f"Project: {desc}")
                
            if background and background.lower() != 'nan':
                bg = background[:1000] + "..." if len(background) > 1000 else background
                text_parts.append(f"Background: {bg}")
                
            combined_text = " | ".join(text_parts) if text_parts else "No description available"
            combined_texts.append(combined_text)
        
        # Encode with Sentence Transformers
        embeddings = self.sentence_model.encode(
            combined_texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        return embeddings
    
    def _encode_categorical_fields(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create learned embeddings for categorical fields.
        """
        categorical_features = []
        
        # Industry embeddings (using hierarchy if available)
        if 'industry' in df.columns:
            industries = df['industry'].fillna('Unknown').values
            if self.industry_embeddings:
                industry_vecs = np.array([
                    self.industry_embeddings.get(ind, {}).get('embedding', np.zeros(384)) if ind in self.industry_embeddings else np.zeros(384)
                    for ind in industries
                ])
            else:
                # Fallback: simple one-hot style
                unique_industries = list(set(industries))
                industry_vecs = np.array([
                    [float(ind == target) for target in unique_industries[:32]]  # Limit dimensions
                    for ind in industries
                ])
                # Pad if needed
                if industry_vecs.shape[1] < 32:
                    padding = np.zeros((industry_vecs.shape[0], 32 - industry_vecs.shape[1]))
                    industry_vecs = np.hstack([industry_vecs, padding])
            categorical_features.append(industry_vecs)
        
        # Location embeddings
        if 'location' in df.columns:
            locations = df['location'].fillna('Unknown').values
            if self.location_embeddings:
                location_vecs = np.array([
                    self.location_embeddings.get(loc, np.zeros(16))
                    for loc in locations
                ])
            else:
                # Simple geographic embeddings (placeholder)
                location_vecs = np.random.normal(0, 0.1, (len(locations), 16))
                    
            categorical_features.append(location_vecs)
        
        if categorical_features:
            return np.hstack(categorical_features)
        else:
            return np.zeros((len(df), 1))
    
    def _encode_numerical_fields(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode and normalize numerical fields.
        """
        numerical_cols = []
        
        # Add derived numerical features
        if 'client_min' in df.columns and 'client_max' in df.columns:
            df = df.copy()
            df['client_size_mid'] = df[['client_min', 'client_max']].mean(axis=1, skipna=True)
            numerical_cols.append('client_size_mid')
            
        if numerical_cols:
            numerical_data = df[numerical_cols].fillna(0).values
            return self.scaler.transform(numerical_data)
        else:
            return np.zeros((len(df), 1))
    
    def _fuse_embeddings(
        self, 
        text_emb: np.ndarray,
        cat_emb: np.ndarray, 
        num_emb: np.ndarray
    ) -> np.ndarray:
        """
        Fuse different types of embeddings using the specified method.
        """
        if self.fusion_method == 'concat':
            # Simple concatenation
            combined = np.hstack([text_emb, cat_emb, num_emb])
            
            # Reduce dimensionality if needed
            if combined.shape[1] > self.final_dim:
                if self.pca is None:
                    self.pca = PCA(n_components=self.final_dim, random_state=42)
                    combined = self.pca.fit_transform(combined)
                else:
                    combined = self.pca.transform(combined)
            
            return combined
            
        elif self.fusion_method == 'weighted_sum':
            # Weighted combination (requires same dimensions)
            # Resize to common dimension first
            target_dim = self.text_dim
            
            if cat_emb.shape[1] != target_dim:
                cat_emb = cat_emb[:, :target_dim] if cat_emb.shape[1] > target_dim else np.pad(
                    cat_emb, ((0, 0), (0, target_dim - cat_emb.shape[1]))
                )
                
            if num_emb.shape[1] != target_dim:
                num_emb = np.pad(num_emb, ((0, 0), (0, target_dim - num_emb.shape[1])))[:, :target_dim]
            
            # Weighted sum
            weights = [0.7, 0.2, 0.1]  # Text, categorical, numerical
            combined = (weights[0] * text_emb + 
                       weights[1] * cat_emb + 
                       weights[2] * num_emb)
            
            return combined
            
        else:  # attention (simplified)
            # Simple attention mechanism
            all_embeddings = [text_emb, cat_emb, num_emb]
            attention_weights = []
            
            for emb in all_embeddings:
                # Simple attention: average magnitude
                weight = np.mean(np.linalg.norm(emb, axis=1))
                attention_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(attention_weights)
            attention_weights = [w/total_weight for w in attention_weights]
            
            # Apply weighted combination
            combined = np.zeros_like(text_emb)
            for i, (emb, weight) in enumerate(zip(all_embeddings, attention_weights)):
                if emb.shape[1] == combined.shape[1]:
                    combined += weight * emb
                    
            return combined
    
    def fit(self, df: pd.DataFrame) -> 'EnhancedEmbedder':
        """
        Fit the embedder on training data.
        """
        print("Fitting Enhanced Embedder...")
        
        # Create industry hierarchy
        if 'industry' in df.columns:
            unique_industries = df['industry'].dropna().unique().tolist()
            self.industry_embeddings = self._create_industry_hierarchy(unique_industries)
            
        # Create location embeddings (simplified)
        if 'location' in df.columns:
            unique_locations = df['location'].dropna().unique().tolist()
            for i, loc in enumerate(unique_locations[:100]):  # Limit to prevent explosion
                # Simple hash-based embedding
                np.random.seed(hash(loc) % 2**31)
                self.location_embeddings[loc] = np.random.normal(0, 0.1, 16)
        
        # Fit numerical scaler
        numerical_cols = []
        df_temp = df.copy()
        
        if 'client_min' in df.columns and 'client_max' in df.columns:
            df_temp['client_size_mid'] = df_temp[['client_min', 'client_max']].mean(axis=1, skipna=True)
            numerical_cols.append('client_size_mid')
            
        # if 'project_min' in df.columns and 'project_max' in df.columns:
        #     df_temp['project_budget_mid'] = df_temp[['project_min', 'project_max']].mean(axis=1, skipna=True)
        #     numerical_cols.append('project_budget_mid')
        
        if numerical_cols:
            self.scaler.fit(df_temp[numerical_cols].fillna(0))
        
        print("Enhanced Embedder fitting completed!")
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data to enhanced embeddings.
        """
        # Encode different modalities
        text_emb = self._encode_text_fields(df)
        cat_emb = self._encode_categorical_fields(df)
        num_emb = self._encode_numerical_fields(df)
        
        # Fuse embeddings
        final_embeddings = self._fuse_embeddings(text_emb, cat_emb, num_emb)
        
        # L2 normalize
        norms = np.linalg.norm(final_embeddings, axis=1, keepdims=True) + 1e-12
        final_embeddings = final_embeddings / norms
        
        return final_embeddings.astype(np.float32)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        return self.fit(df).transform(df)


class EnhancedContentBasedRecommender:
    """
    Enhanced content-based recommender using advanced embeddings.
    """
    
    def __init__(
        self,
        df_history: pd.DataFrame,
        df_test: pd.DataFrame,
        embedding_config: Optional[Dict] = None
    ):
        data_train, data_val = train_test_split(df_history, test_size=0.2, random_state=42)
        self.data_raw = data_train
        self.data_val = data_val
        self.df_test = df_test.copy()
        
        # Enhanced embedder configuration
        self.embedder_config = embedding_config or {
            'sentence_model_name': 'all-MiniLM-L6-v2',
            'embedding_dim': 256,
            'use_industry_hierarchy': True,
            'fusion_method': 'concat'
        }
        
        print("Building enhanced feature representations...")
        
        # Initialize and fit embedder
        self.embedder = EnhancedEmbedder(**self.embedder_config)
        self.embedder.fit(data_train)
        
        # Transform candidate items
        print("Transforming candidate items...")
        self.X_candidates = self.embedder.transform(df_test)
        print('Data val :', data_val.shape)
        print('Data train :', data_train.shape)
        self.X_val_candidates = self.embedder.transform(data_val)
        print(f"Enhanced embeddings shape: {self.X_candidates.shape}")
    
    def build_user_profile(self, user_id: str) -> Optional[np.ndarray]:
        """
        Build enhanced user profile from historical interactions.
        """
        user_history = self.data_raw[self.data_raw['linkedin_company_outsource'] == user_id]
        
        if user_history.empty:
            return None
            
        # Transform user history to embeddings
        user_embeddings = self.embedder.transform(user_history)
        
        # Aggregate user profile (could use more sophisticated methods)
        user_profile = np.mean(user_embeddings, axis=0, keepdims=True)
        
        return user_profile
    
    def recommend_items(self, user_id: str, top_k: int = 10, mode: Literal['val','test'] = 'test') -> pd.DataFrame:
        """
        Generate recommendations using enhanced embeddings.
        """
        user_profile = self.build_user_profile(user_id)
        
        if user_profile is None:
            # Cold start - return most popular items
            return pd.DataFrame(columns=['industry', 'score'])
        
        # Compute similarities
        if mode == 'val' :
            similarities = np.dot(self.X_val_candidates, user_profile.T).ravel()
            results = self.data_val.copy()
            results['score'] = similarities  
        else:
            similarities = np.dot(self.X_candidates, user_profile.T).ravel()  
            results = self.df_test.copy()
            results['score'] = similarities  
        # Create results DataFrame
        
        
        # Aggregate by industry (taking max score)
        industry_results = (
            results.groupby('industry')
            .agg({
                'score': 'max',
                'location': 'first',
                'services': 'first', 
                'project_description': 'first'
            })
            .reset_index()
            .sort_values('score', ascending=False)
            .head(top_k)
        )
        
        return industry_results


def compare_embedding_approaches(
    df_history: pd.DataFrame,
    df_test: pd.DataFrame,
    embedding_configs: List[Dict],
    config_names: List[str]
) -> pd.DataFrame:
    """
    Compare different embedding configurations.
    """
    results = []
    
    for config, name in zip(embedding_configs, config_names):
        print(f"\nTesting configuration: {name}")
        print("-" * 50)
        
        try:
            # Build recommender with this config
            recommender = EnhancedContentBasedRecommender(
                df_history, df_test, embedding_config=config
            )
            
            # Test on a subset of users
            test_users = df_test['linkedin_company_outsource'].unique()[:10]
            
            total_score = 0
            valid_users = 0
            
            for user in test_users:
                recs = recommender.recommend_items(user, top_k=5)
                if not recs.empty:
                    avg_score = recs['score'].mean()
                    total_score += avg_score
                    valid_users += 1
            
            avg_performance = total_score / max(valid_users, 1)
            
            results.append({
                'config_name': name,
                'avg_score': avg_performance,
                'valid_users': valid_users,
                'embedding_dim': config.get('embedding_dim', 0),
                'fusion_method': config.get('fusion_method', 'unknown')
            })
            
        except Exception as e:
            print(f"Error with config {name}: {e}")
            results.append({
                'config_name': name,
                'avg_score': 0,
                'valid_users': 0,
                'embedding_dim': config.get('embedding_dim', 0),
                'fusion_method': config.get('fusion_method', 'unknown')
            })
    
    return pd.DataFrame(results).sort_values('avg_score', ascending=False)


# Example usage and configurations
EMBEDDING_CONFIGS = {
    'basic_sentence_transformer': {
        'sentence_model_name': 'all-MiniLM-L6-v2',
        'embedding_dim': 256,
        'use_industry_hierarchy': False,
        'fusion_method': 'concat',
        'use_openai': False
    },
    'hierarchical_concat': {
        'sentence_model_name': 'all-MiniLM-L6-v2', 
        'embedding_dim': 384,
        'use_industry_hierarchy': True,
        'fusion_method': 'concat',
        'use_openai': False
    },
    'weighted_fusion': {
        'sentence_model_name': 'all-mpnet-base-v2',
        'embedding_dim': 512,
        'use_industry_hierarchy': True,
        'fusion_method': 'weighted_sum',
        'use_openai': False
    },
    'attention_fusion': {
        'sentence_model_name': 'all-mpnet-base-v2',
        'embedding_dim': 384,
        'use_industry_hierarchy': True,
        'fusion_method': 'attention',
        'use_openai': False
    },
    # NEW: OpenAI configurations
    'openai_small': {
        'sentence_model_name': 'all-MiniLM-L6-v2',  # Fallback
        'embedding_dim': 512,
        'use_industry_hierarchy': True,
        'fusion_method': 'concat',
        'use_openai': True,
        'openai_model': 'text-embedding-3-small'
    },
    'openai_large': {
        'sentence_model_name': 'all-MiniLM-L6-v2',  # Fallback
        'embedding_dim': 768,
        'use_industry_hierarchy': True,
        'fusion_method': 'concat',
        'use_openai': True,
        'openai_model': 'text-embedding-3-large'
    }
}