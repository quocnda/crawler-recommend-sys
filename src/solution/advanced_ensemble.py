"""
Advanced Ensemble Methods for Recommendation System
=================================================

This module implements sophisticated ensemble techniques to combine multiple recommendation approaches:
1. Stacking with meta-learner
2. Gradient boosting for recommendation scores
3. Neural ensemble with attention mechanism
4. Dynamic ensemble selection based on user characteristics
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Neural ensemble will be disabled.")


class StackingEnsemble:
    """
    Stacking ensemble that learns how to combine different recommendation approaches.
    """
    
    def __init__(
        self,
        meta_learner: str = 'lightgbm',
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.meta_learner_name = meta_learner
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Initialize meta-learner
        if meta_learner == 'lightgbm':
            self.meta_learner = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state,
                verbosity=-1
            )
        elif meta_learner == 'gradient_boosting':
            self.meta_learner = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=random_state
            )
        elif meta_learner == 'ridge':
            self.meta_learner = Ridge(alpha=1.0, random_state=random_state)
        elif meta_learner == 'elastic':
            self.meta_learner = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state)
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner}")
            
        self.is_fitted = False
        self.scaler = StandardScaler()
        
    def create_meta_features(
        self,
        base_predictions: Dict[str, np.ndarray],
        user_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create meta-features from base model predictions and user features.
        """
        features = []
        
        # Base predictions
        for model_name, preds in base_predictions.items():
            features.append(preds.reshape(-1, 1))
        
        # Cross-model features
        pred_values = list(base_predictions.values())
        if len(pred_values) >= 2:
            # Variance across models (uncertainty)
            pred_matrix = np.column_stack(pred_values)
            variance = np.var(pred_matrix, axis=1, keepdims=True)
            features.append(variance)
            
            # Max-min spread
            max_min_spread = (np.max(pred_matrix, axis=1) - np.min(pred_matrix, axis=1)).reshape(-1, 1)
            features.append(max_min_spread)
            
            # Pairwise correlations (for first two models)
            if len(pred_values) >= 2:
                prod = (pred_values[0] * pred_values[1]).reshape(-1, 1)
                features.append(prod)
        
        # User features if available
        if user_features is not None:
            features.append(user_features)
        
        return np.hstack(features)
    
    def fit(
        self,
        base_predictions_train: Dict[str, Dict[str, np.ndarray]], 
        ground_truth: Dict[str, List[str]],
        user_features: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Fit stacking ensemble using cross-validation.
        
        Args:
            base_predictions_train: {user_id: {model_name: predictions}}
            ground_truth: {user_id: [relevant_industries]}
            user_features: {user_id: feature_vector}
        """
        print("Fitting Stacking Ensemble...")
        
        # Prepare training data
        X_meta = []
        y_meta = []
        
        for user_id, gt_items in ground_truth.items():
            if user_id not in base_predictions_train:
                continue
                
            user_base_preds = base_predictions_train[user_id]
            user_feat = user_features.get(user_id) if user_features else None
            
            # Get all predicted items for this user
            all_items = set()
            for model_preds in user_base_preds.values():
                all_items.update(model_preds.keys())
            
            for item in all_items:
                # Create base prediction features
                base_pred_dict = {
                    model: preds.get(item, 0.0) 
                    for model, preds in user_base_preds.items()
                }
                
                # Create meta-features
                base_pred_array = np.array(list(base_pred_dict.values()))
                meta_feat = self.create_meta_features(
                    {model: np.array([score]) for model, score in base_pred_dict.items()},
                    user_feat.reshape(1, -1) if user_feat is not None else None
                )
                
                # Create target (relevance score)
                relevance = 1.0 if item in gt_items else 0.0
                
                X_meta.append(meta_feat.flatten())
                y_meta.append(relevance)
        
        if not X_meta:
            raise ValueError("No training data available for stacking")
            
        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)
        
        # Scale features
        X_meta_scaled = self.scaler.fit_transform(X_meta)
        
        # Fit meta-learner
        self.meta_learner.fit(X_meta_scaled, y_meta)
        self.is_fitted = True
        
        print("Stacking ensemble training completed!")
    
    def predict(
        self,
        base_predictions: Dict[str, np.ndarray],
        user_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict using fitted stacking ensemble.
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        meta_features = self.create_meta_features(base_predictions, user_features)
        meta_features_scaled = self.scaler.transform(meta_features)
        
        return self.meta_learner.predict(meta_features_scaled)


class NeuralEnsemble(nn.Module):
    """
    Neural network ensemble with attention mechanism.
    """
    
    def __init__(
        self,
        n_models: int,
        user_feature_dim: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.n_models = n_models
        self.user_feature_dim = user_feature_dim
        
        # Input dimension: base predictions + user features + cross-model features
        input_dim = n_models + user_feature_dim + 3  # +3 for variance, spread, product
        
        # Attention mechanism for base models
        self.attention = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_models),
            nn.Softmax(dim=-1)
        )
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, base_preds, user_features=None):
        batch_size = base_preds.size(0)
        
        # Attention weights for base predictions
        attention_weights = self.attention(base_preds)
        weighted_preds = base_preds * attention_weights
        
        # Cross-model features
        variance = torch.var(base_preds, dim=1, keepdim=True)
        max_vals, _ = torch.max(base_preds, dim=1, keepdim=True)
        min_vals, _ = torch.min(base_preds, dim=1, keepdim=True)
        spread = max_vals - min_vals
        
        # Product of first two predictions (if available)
        if base_preds.size(1) >= 2:
            product = (base_preds[:, 0] * base_preds[:, 1]).unsqueeze(1)
        else:
            product = torch.zeros(batch_size, 1, device=base_preds.device)
        
        # Combine all features
        features = [weighted_preds, variance, spread, product]
        
        if user_features is not None:
            features.append(user_features)
        
        combined_features = torch.cat(features, dim=1)
        
        return self.network(combined_features).squeeze()


class AdvancedEnsembleRecommender:
    """
    Advanced ensemble recommender combining multiple sophisticated techniques.
    """
    
    def __init__(
        self,
        ensemble_methods: List[str] = ['stacking', 'neural', 'weighted'],
        neural_config: Optional[Dict] = None
    ):
        self.ensemble_methods = ensemble_methods
        self.neural_config = neural_config or {
            'hidden_dim': 64,
            'dropout': 0.3,
            'epochs': 100,
            'lr': 0.001
        }
        
        # Initialize ensemble components
        self.stacking_ensemble = None
        self.neural_ensemble = None
        self.base_models = {}
        self.user_feature_extractor = None
        
    def extract_user_features(self, user_history: pd.DataFrame, user_id: str) -> np.ndarray:
        """
        Extract comprehensive user features for ensemble.
        """
        if user_history.empty:
            return np.zeros(10)
        
        features = []
        
        # Interaction statistics
        n_interactions = len(user_history)
        features.append(min(n_interactions / 20.0, 1.0))  # Normalized interaction count
        
        # Industry diversity
        unique_industries = len(user_history['industry'].unique())
        industry_diversity = unique_industries / max(n_interactions, 1)
        features.append(industry_diversity)
        
        # Temporal patterns (if timestamp available)
        features.append(0.5)  # Placeholder for recency
        features.append(0.5)  # Placeholder for seasonality
        
        # Business characteristics
        if 'client_size_mid' in user_history.columns:
            avg_client_size = user_history['client_size_mid'].mean()
            features.append(np.log1p(avg_client_size) / 10.0)  # Log-normalized
        else:
            features.append(0.0)
            
        # if 'project_budget_mid' in user_history.columns:
        #     avg_budget = user_history['project_budget_mid'].mean()
        #     features.append(np.log1p(avg_budget) / 15.0)  # Log-normalized
        # else:
        #     features.append(0.0)
        
        # Geographic focus
        location_concentration = 1.0 - (len(user_history['location'].unique()) / max(n_interactions, 1))
        features.append(location_concentration)
        
        # Service complexity
        services_text = ' '.join(user_history['services'].fillna(''))
        complex_keywords = ['AI', 'Machine Learning', 'Blockchain', 'Enterprise', 'Custom']
        complexity = sum(1 for kw in complex_keywords if kw.lower() in services_text.lower()) / len(complex_keywords)
        features.append(complexity)
        
        # Engagement patterns
        features.append(min(n_interactions / 365.0, 1.0))  # Annual engagement rate
        features.append(0.7)  # Placeholder for consistency score
        
        return np.array(features)
    
    def fit_base_models(self, df_history: pd.DataFrame, df_test: pd.DataFrame):
        """
        Fit all base recommendation models.
        """
        print("Fitting base models for ensemble...")
        
        # Content-based with OpenAI
        from solution.content_base_for_item import ContentBaseBasicApproach
        print("- Fitting Content-Based OpenAI...")
        self.base_models['content_openai'] = ContentBaseBasicApproach(df_history, df_test)
        
        # Enhanced embeddings
        try:
            from solution.enhanced_embeddings import EnhancedContentBasedRecommender
            print("- Fitting Enhanced Embeddings...")
            self.base_models['enhanced_embeddings'] = EnhancedContentBasedRecommender(df_history, df_test)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"- Enhanced Embeddings failed: {e}")
        
        # Collaborative filtering
        try:
            from solution.collborative_for_item import CollaborativeIndustryRecommender
            print("- Fitting Collaborative Filtering...")
            collab = CollaborativeIndustryRecommender(
                n_components=128,
                min_user_interactions=1,
                min_item_interactions=1,
                use_tfidf_weighting=True,
                random_state=42
            ).fit(df_history=df_history, df_candidates=df_test)
            self.base_models['collaborative'] = collab
        except Exception as e:
            print(f"- Collaborative Filtering failed: {e}")
        
        print(f"Successfully fitted {len(self.base_models)} base models")
    
    def get_base_predictions(self, user_id: str, top_k: int = 20) -> Dict[str, Dict[str, float]]:
        """
        Get predictions from all base models for a user.
        """
        predictions = {}
        for model_name, model in self.base_models.items():
            try:
                if model_name == 'collaborative':
                    recs = model.recommend_items(user_id, top_k=top_k)
                else:
                    recs = model.recommend_items(user_id, top_k=top_k)
                
                # Convert to dict format
                pred_dict = dict(zip(recs['industry'], recs['score']))
                predictions[model_name] = pred_dict
                
            except Exception as e:
                print(f"Error getting predictions from {model_name}: {e}")
                predictions[model_name] = {}
        
        return predictions
    
    def fit_ensemble(
        self,
        df_history: pd.DataFrame,
        df_test: pd.DataFrame,
        ground_truth: Dict[str, List[str]],
        validation_split: float = 0.2
    ):
        """
        Fit ensemble models using base model predictions.
        """
        print("Fitting ensemble models...")
        
        # First fit base models
        self.fit_base_models(df_history, df_test)
        
        # Prepare ensemble training data
        train_users = list(ground_truth.keys())
        n_val = int(len(train_users) * validation_split)
        
        np.random.seed(42)
        val_users = set(np.random.choice(train_users, n_val, replace=False))
        train_users = [u for u in train_users if u not in val_users]
        
        # Get base predictions for training users
        base_predictions_train = {}
        user_features_train = {}
        
        for user_id in train_users:
            user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
            
            # Get base predictions
            base_preds = self.get_base_predictions(user_id, top_k=30)
            if any(base_preds.values()):  # Only include if we have predictions
                base_predictions_train[user_id] = base_preds
                user_features_train[user_id] = self.extract_user_features(user_history, user_id)
        
        # Fit stacking ensemble
        if 'stacking' in self.ensemble_methods:
            print("START STACKING")
            self.stacking_ensemble = StackingEnsemble(meta_learner='lightgbm')
            try:
                self.stacking_ensemble.fit(
                    base_predictions_train, 
                    {u: gt for u, gt in ground_truth.items() if u in train_users},
                    user_features_train
                )
            except Exception as e:
                print(f"Stacking ensemble fitting failed: {e}")
                self.stacking_ensemble = None
        
        # Fit neural ensemble
        if 'neural' in self.ensemble_methods and TORCH_AVAILABLE:
            print("START NEURAL ENSEMBLE")
            try:
                self._fit_neural_ensemble(
                    base_predictions_train, 
                    {u: gt for u, gt in ground_truth.items() if u in train_users},
                    user_features_train,
                    val_users, ground_truth, df_history
                )
            except Exception as e:
                print(f"Neural ensemble fitting failed: {e}")
                self.neural_ensemble = None
        
        print("Ensemble fitting completed!")
    
    def _fit_neural_ensemble(
        self,
        base_predictions_train: Dict,
        ground_truth_train: Dict,
        user_features_train: Dict,
        val_users: set,
        full_ground_truth: Dict,
        df_history: pd.DataFrame
    ):
        """
        Fit neural ensemble model.
        """
        if not TORCH_AVAILABLE:
            return
            
        # Prepare training data
        X_base = []
        X_user = []
        y = []
        
        n_models = len(self.base_models)
        user_feat_dim = len(next(iter(user_features_train.values())))
        
        for user_id, gt_items in ground_truth_train.items():
            if user_id not in base_predictions_train:
                continue
                
            user_base_preds = base_predictions_train[user_id]
            user_feat = user_features_train[user_id]
            
            # Get all items for this user
            all_items = set()
            for model_preds in user_base_preds.values():
                all_items.update(model_preds.keys())
            
            for item in all_items:
                # Base predictions vector
                base_pred_vector = np.array([
                    user_base_preds.get(model, {}).get(item, 0.0) 
                    for model in self.base_models.keys()
                ])
                
                # Pad if necessary
                if len(base_pred_vector) < n_models:
                    base_pred_vector = np.pad(base_pred_vector, (0, n_models - len(base_pred_vector)))
                
                relevance = 1.0 if item in gt_items else 0.0
                
                X_base.append(base_pred_vector)
                X_user.append(user_feat)
                y.append(relevance)
        
        if not X_base:
            return
            
        X_base = torch.FloatTensor(np.array(X_base))
        X_user = torch.FloatTensor(np.array(X_user))
        y = torch.FloatTensor(np.array(y))
        
        # Initialize and train neural ensemble
        self.neural_ensemble = NeuralEnsemble(n_models, user_feat_dim, **self.neural_config)
        optimizer = optim.Adam(self.neural_ensemble.parameters(), lr=self.neural_config['lr'])
        criterion = nn.BCELoss()
        
        # Training loop
        batch_size = 256
        n_batches = len(X_base) // batch_size + 1
        
        self.neural_ensemble.train()
        for epoch in range(self.neural_config['epochs']):
            total_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_base))
                
                if start_idx >= end_idx:
                    continue
                
                batch_base = X_base[start_idx:end_idx]
                batch_user = X_user[start_idx:end_idx]
                batch_y = y[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = self.neural_ensemble(batch_base, batch_user)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                print(f"Neural ensemble epoch {epoch}, loss: {total_loss/n_batches:.4f}")
    
    def recommend_items(
        self,
        user_id: str,
        df_history: pd.DataFrame,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Generate recommendations using ensemble methods.
        """
        # Get base predictions
        base_predictions = self.get_base_predictions(user_id, top_k=top_k*3)
        
        if not any(base_predictions.values()):
            return pd.DataFrame(columns=['industry', 'score'])
        
        # Get all candidate items
        all_items = set()
        for model_preds in base_predictions.values():
            all_items.update(model_preds.keys())
        
        # Extract user features
        user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
        user_features = self.extract_user_features(user_history, user_id)
        
        # Ensemble predictions
        ensemble_scores = {}
        
        for item in all_items:
            scores = []
            
            # Base model scores
            base_scores = {
                model: preds.get(item, 0.0) 
                for model, preds in base_predictions.items()
            }
            
            # Stacking ensemble
            if self.stacking_ensemble and self.stacking_ensemble.is_fitted:
                try:
                    base_array = {model: np.array([score]) for model, score in base_scores.items()}
                    stacking_score = self.stacking_ensemble.predict(base_array, user_features.reshape(1, -1))
                    scores.append(('stacking', stacking_score[0]))
                except Exception as e:
                    print(f"Stacking prediction error: {e}")
            
            # Neural ensemble
            if self.neural_ensemble and TORCH_AVAILABLE:
                try:
                    with torch.no_grad():
                        base_tensor = torch.FloatTensor([list(base_scores.values())])
                        user_tensor = torch.FloatTensor([user_features])
                        neural_score = self.neural_ensemble(base_tensor, user_tensor).item()
                        scores.append(('neural', neural_score))
                except Exception as e:
                    print(f"Neural prediction error: {e}")
            
            # Weighted ensemble (fallback)
            if 'weighted' in self.ensemble_methods or not scores:
                # Dynamic weighting based on user characteristics
                weights = self._get_dynamic_weights(user_features, base_scores)
                weighted_score = sum(weights[model] * score for model, score in base_scores.items())
                scores.append(('weighted', weighted_score))
            
            # Combine ensemble methods
            if scores:
                # Use the best performing method or average
                final_score = np.mean([score for _, score in scores])
                ensemble_scores[item] = final_score
        
        # Create results DataFrame
        sorted_items = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = pd.DataFrame([
            {'industry': item, 'score': score} 
            for item, score in sorted_items
        ])
        
        return results
    
    def _get_dynamic_weights(self, user_features: np.ndarray, base_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Get dynamic weights for base models based on user characteristics.
        """
        weights = {}
        n_models = len(base_scores)
        
        # Default equal weights
        default_weight = 1.0 / n_models
        
        # Extract relevant user characteristics
        interaction_level = user_features[0]  # Normalized interaction count
        industry_diversity = user_features[1]
        complexity_score = user_features[7] if len(user_features) > 7 else 0.5
        
        for i, model in enumerate(base_scores.keys()):
            weight = default_weight
            
            # Adjust based on user characteristics
            if model == 'content_openai':
                # Content-based works better for users with diverse interests
                weight *= (1.0 + 0.3 * industry_diversity)
                
            elif model == 'collaborative':
                # Collaborative works better for users with more interactions
                weight *= (1.0 + 0.5 * interaction_level)
                
            elif model == 'enhanced_embeddings':
                # Enhanced embeddings work better for complex projects
                weight *= (1.0 + 0.2 * complexity_score)
            
            weights[model] = weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {model: w/total_weight for model, w in weights.items()}
        
        return weights


def integrate_advanced_ensemble(
    df_history: pd.DataFrame,
    df_test: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
    top_k: int = 10
) -> pd.DataFrame:
    """
    Integration function for advanced ensemble recommendation.
    """
    print("Initializing Advanced Ensemble Recommender...")
    
    ensemble = AdvancedEnsembleRecommender(
        ensemble_methods=['stacking', 'neural', 'weighted'],
    )
    
    # Fit ensemble
    ensemble.fit_ensemble(df_history, df_test, ground_truth)
    
    # Generate recommendations for all test users
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        if pd.isna(user_id) or user_id in seen_users:
            continue
        seen_users.add(user_id)
        
        try:
            recs = ensemble.recommend_items(user_id, df_history, top_k=top_k)
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