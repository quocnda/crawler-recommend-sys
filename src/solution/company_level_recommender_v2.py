"""
Advanced Company-Level Recommendation System
============================================

Extends the sophisticated multi-model ensemble pipeline to recommend COMPANIES
instead of just industries, leveraging the full power of existing architecture.

Key Evolution:
--------------
OLD: Outsource → [Multi-Model Ensemble] → Industries (67 labels)
NEW: Outsource → [Industry Ensemble Filter] → [Company-Level Ranking] → Companies

Architecture Integration:
-------------------------
Reuses existing components:
- ContentBaseBasicApproach (OpenAI embeddings with block weighting)
- CollaborativeIndustryRecommender (SVD + BM25/TF-IDF)
- AdvancedEnsembleRecommender (Stacking + Neural + Weighted fusion)
- MultiStageRecommender (Funnel approach with diversity)

Multi-Stage Pipeline:
---------------------
Stage 1: Industry-Level Ensemble Filtering
    Models: All existing ensemble models (content + collaborative + neural)
    Output: Top 5-10 industries with ensemble scores
    Purpose: Narrow 67 industries → promising subset
    
Stage 2: Company Candidate Generation
    Input: Filtered industries from Stage 1
    Method: Retrieve all companies in those industries
    Output: 200-500 candidate companies
    Purpose: Expand to company-level within filtered space
    
Stage 3: Company-Level Advanced Ranking
    Signals:
        a) Content similarity (OpenAI embeddings on company attributes)
        b) Collaborative patterns (companies similar outsources worked with)
        c) Attribute matching (size, location, services compatibility)
        d) Industry score inheritance (from Stage 1)
    Fusion: Weighted combination + z-score normalization
    Output: Top 30-50 ranked companies
    
Stage 4: Final Ensemble & Diversity Selection
    Methods:
        a) Advanced ensemble fusion (meta-learner on all signals)
        b) Diversity injection (MMR-style)
        c) Business rules (recency, match quality)
    Output: Final Top-K companies

Features:
---------
- Multi-model ensemble at industry level (proven high recall)
- Company-level ranking with multiple signals
- Feature engineering for company attributes
- Diversity-aware final selection
- Optional LLM re-ranking for top candidates
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Import existing solution components
try:
    from solution.content_base_for_item import ContentBaseBasicApproach, OpenAIEmbedder
    CONTENT_BASE_AVAILABLE = True
except ImportError:
    CONTENT_BASE_AVAILABLE = False

try:
    from solution.collborative_for_item import CollaborativeIndustryRecommender
    COLLABORATIVE_AVAILABLE = True
except ImportError:
    COLLABORATIVE_AVAILABLE = False

try:
    from solution.advanced_ensemble import AdvancedEnsembleRecommender
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from solution.improved_ensemble import MultiStageRecommender
    MULTI_STAGE_AVAILABLE = True
except ImportError:
    MULTI_STAGE_AVAILABLE = False


class AdvancedCompanyLevelRecommender:
    """
    Advanced multi-stage company-level recommender that leverages
    the full existing ensemble pipeline.
    """
    
    def __init__(
        self,
        df_history: pd.DataFrame,
        df_candidates: pd.DataFrame,
        industry_top_k: int = 10,
        company_fanout: int = 5,  # Companies per industry to consider
        use_ensemble: bool = True,
        embedding_model: str = "text-embedding-3-large"
    ):
        """
        Args:
            df_history: Historical outsource-client interactions
            df_candidates: Candidate companies (potential clients)
            industry_top_k: Number of top industries to filter in Stage 1
            company_fanout: Max companies per industry to consider
            use_ensemble: Whether to use full ensemble (True) or simple models
            embedding_model: OpenAI embedding model for company attributes
        """
        self.df_history = df_history.copy()
        self.df_candidates = df_candidates.copy()
        self.industry_top_k = industry_top_k
        self.company_fanout = company_fanout
        self.use_ensemble = use_ensemble
        self.embedding_model_name = embedding_model
        
        # Standardize column names
        self._standardize_columns()
        
        # Initialize components
        self.industry_models = {}
        self.company_embedder = None
        self.ensemble_recommender = None
        
        print("="*80)
        print("INITIALIZING ADVANCED COMPANY-LEVEL RECOMMENDER")
        print("="*80)
        
        # Build industry-level models (reuse existing sophisticated pipeline)
        self._build_industry_models()
        
        # Build company-level infrastructure
        self._build_company_infrastructure()
        
        print("✓ Initialization complete!")
        print("="*80)
    
    def _standardize_columns(self):
        """Standardize column names across dataframes."""
        # Note: preprocessing_data already handles most renaming
        # Just ensure company_name exists
        rename_map = {
            'reviewer_company': 'company_name'
        }
        
        for df in [self.df_history, self.df_candidates]:
            for old_col, new_col in rename_map.items():
                if old_col in df.columns and new_col not in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
        
        # Fill NaN values for string columns only
        for col in ['industry', 'location', 'services', 'company_name']:
            if col in self.df_history.columns:
                self.df_history[col] = self.df_history[col].fillna('Unknown')
            if col in self.df_candidates.columns:
                self.df_candidates[col] = self.df_candidates[col].fillna('Unknown')
    
    def _build_industry_models(self):
        """Build sophisticated industry-level models using existing pipeline."""
        print("\n[Stage 0] Building Industry-Level Ensemble Models...")
        
        # Content-based with OpenAI embeddings
        if CONTENT_BASE_AVAILABLE:
            print("- Building Content-Based model (OpenAI embeddings)...")
            try:
                self.industry_models['content'] = ContentBaseBasicApproach(
                    self.df_history,
                    self.df_candidates,
                    embedding_model=self.embedding_model_name,
                    block_weights=(0.35, 0, 0.35, 0.3)  # services, desc, cat, num
                )
                print("  ✓ Content-Based model ready")
            except Exception as e:
                print(f"  ✗ Content-Based failed: {e}")
        
        # Collaborative filtering with BM25
        if COLLABORATIVE_AVAILABLE:
            print("- Building Collaborative Filtering model (BM25 + SVD)...")
            try:
                collab = CollaborativeIndustryRecommender(
                    n_components=128,
                    min_user_interactions=1,
                    min_item_interactions=1,
                    use_tfidf_weighting=True,
                    random_state=42
                ).fit(df_history=self.df_history, df_candidates=self.df_candidates)
                self.industry_models['collaborative'] = collab
                print("  ✓ Collaborative model ready")
            except Exception as e:
                print(f"  ✗ Collaborative failed: {e}")
        
        # Advanced Ensemble (if requested and available)
        if self.use_ensemble and ENSEMBLE_AVAILABLE:
            print("- Building Advanced Ensemble (Stacking + Neural)...")
            try:
                # Note: This would require ground truth for training
                # For now, we'll use it in predict mode only
                self.ensemble_recommender = AdvancedEnsembleRecommender(
                    ensemble_methods=['weighted'],  # Start with weighted fusion
                )
                print("  ✓ Ensemble recommender ready")
            except Exception as e:
                print(f"  ✗ Ensemble failed: {e}")
        
        if not self.industry_models:
            raise ValueError("No industry models could be initialized!")
        
        print(f"✓ {len(self.industry_models)} industry-level models ready")
    
    def _build_company_infrastructure(self):
        """Build company-level ranking infrastructure."""
        print("\n[Stage 0] Building Company-Level Infrastructure...")
        
        # Create unique company identifier
        self.df_candidates['company_id'] = self.df_candidates.apply(
            lambda row: f"{row.get('company_name', 'Unknown')}|{row.get('industry', '')}|{row.get('location', '')}",
            axis=1
        )
        
        # Build industry → companies mapping
        self.industry_to_companies = defaultdict(list)
        for idx, row in self.df_candidates.iterrows():
            industry = row.get('industry')
            if pd.notna(industry) and industry != 'Unknown':
                self.industry_to_companies[industry].append(idx)
        
        print(f"  - {len(self.industry_to_companies)} industries")
        print(f"  - {len(self.df_candidates)} total companies")
        print(f"  - Avg {np.mean([len(v) for v in self.industry_to_companies.values()]):.1f} companies per industry")
        
        # Initialize company-level embedder for attributes
        if CONTENT_BASE_AVAILABLE and OpenAIEmbedder:
            print("- Initializing company attribute embedder...")
            self.company_embedder = OpenAIEmbedder(
                model=self.embedding_model_name,
                batch_size=512,
                normalize=True
            )
            
            # Precompute company attribute embeddings
            print("- Precomputing company embeddings...")
            self.df_candidates['company_text'] = self.df_candidates.apply(
                lambda row: f"Industry: {row.get('industry', '')}. "
                            f"Location: {row.get('location', '')}. "
                            f"Size: {row.get('client_min', '')}-{row.get('client_max', '')} employees. "
                            f"Services: {row.get('services', '')}",
                axis=1
            )
            
            self.company_embeddings = self.company_embedder.transform(
                self.df_candidates['company_text'].tolist()
            )
            print(f"  ✓ Computed {self.company_embeddings.shape} embeddings")
        else:
            print("  ! Company embeddings not available")
            self.company_embeddings = None
        
        print("✓ Company infrastructure ready")
    
    def stage1_industry_ensemble_filter(
        self,
        outsource_url: str,
        top_k: int
    ) -> pd.DataFrame:
        """
        Stage 1: Get top industries using sophisticated ensemble of models.
        
        Returns DataFrame with columns: ['industry', 'score', 'model_scores']
        """
        # print(f"\n[Stage 1] Industry Ensemble Filtering (top {top_k})")
        
        # Collect predictions from all industry models
        industry_scores = defaultdict(list)
        model_predictions = {}
        
        for model_name, model in self.industry_models.items():
            try:
                # print(f"  - Getting predictions from {model_name}...")
                recs = model.recommend_items(outsource_url, top_k=top_k * 2)
                
                if not recs.empty and 'industry' in recs.columns and 'score' in recs.columns:
                    model_pred = dict(zip(recs['industry'], recs['score']))
                    model_predictions[model_name] = model_pred
                    
                    for industry, score in model_pred.items():
                        industry_scores[industry].append(score)
                    
                    # print(f"    ✓ Got {len(model_pred)} industries")
            except Exception as e:
                pass
                # print(f"    ✗ {model_name} failed: {e}")
        
        if not industry_scores:
            # print("  ! No predictions from any model")
            return pd.DataFrame(columns=['industry', 'score', 'model_scores'])
        
        # Fusion: Use ensemble if available, otherwise simple fusion
        if self.ensemble_recommender is not None and len(model_predictions) > 1:
            # print("  - Using AdvancedEnsemble for fusion...")
            try:
                # Use ensemble's weighted fusion method
                fused_scores = self._ensemble_fuse_predictions(model_predictions, industry_scores)
                # print("    ✓ Ensemble fusion completed")
            except Exception as e:
                # print(f"    ✗ Ensemble fusion failed: {e}, falling back to simple fusion")
                # Fallback to simple averaging
                fused_scores = {industry: np.mean(scores) for industry, scores in industry_scores.items()}
        else:
            # Simple fusion: Z-score normalize within each model, then average
            # print("  - Using simple weighted fusion...")
            fused_scores = self._simple_fuse_predictions(model_predictions, industry_scores)
        
        # Convert to DataFrame and sort
        results = pd.DataFrame([
            {
                'industry': industry,
                'score': score,
                'model_scores': {
                    m: model_predictions[m].get(industry, 0.0)
                    for m in model_predictions.keys()
                },
                'n_models': len(industry_scores[industry])
            }
            for industry, score in fused_scores.items()
        ]).sort_values('score', ascending=False).head(top_k)
        
        # print(f"  ✓ Filtered {len(results)} industries")
        # if not results.empty:
            # print(f"    Top 3: {results['industry'].head(3).tolist()}")
        
        return results
    
    def _simple_fuse_predictions(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        industry_scores: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Simple fusion: Z-score normalize each model's scores, then weighted average.
        """
        # Z-score normalize each model's predictions
        normalized_preds = {}
        for model_name, preds in model_predictions.items():
            scores = np.array(list(preds.values()))
            if scores.std() > 0:
                mean, std = scores.mean(), scores.std()
                normalized_preds[model_name] = {
                    ind: (score - mean) / std
                    for ind, score in preds.items()
                }
            else:
                normalized_preds[model_name] = preds
        
        # Weighted average (equal weights for now)
        fused_scores = {}
        for industry in industry_scores.keys():
            scores = [
                normalized_preds[model].get(industry, 0.0)
                for model in normalized_preds.keys()
                if industry in normalized_preds[model]
            ]
            fused_scores[industry] = np.mean(scores) if scores else 0.0
        
        return fused_scores
    
    def _ensemble_fuse_predictions(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        industry_scores: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """
        Advanced fusion using ensemble recommender's fusion logic.
        """
        # Get all unique industries
        all_industries = set()
        for preds in model_predictions.values():
            all_industries.update(preds.keys())
        
        # Create feature matrix for ensemble
        # Each row = industry, each column = model score
        feature_matrix = []
        industry_list = list(all_industries)
        
        for industry in industry_list:
            features = [
                model_predictions.get(model, {}).get(industry, 0.0)
                for model in sorted(model_predictions.keys())
            ]
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        
        # Use ensemble's weighted method (if it has learned weights)
        # Otherwise, use optimized weights based on model diversity
        if hasattr(self.ensemble_recommender, 'ensemble_weights_'):
            weights = self.ensemble_recommender.ensemble_weights_
        else:
            # Dynamic weights based on model agreement
            # Models that agree more get higher weight
            weights = self._compute_dynamic_weights(model_predictions)
        
        # Weighted fusion
        fused_values = np.dot(feature_matrix, weights)
        
        return dict(zip(industry_list, fused_values))
    
    def _compute_dynamic_weights(
        self,
        model_predictions: Dict[str, Dict[str, float]]
    ) -> np.ndarray:
        """
        Compute dynamic weights based on model diversity and coverage.
        Models with better coverage and less correlation get higher weights.
        """
        n_models = len(model_predictions)
        
        # Equal weights as baseline
        weights = np.ones(n_models) / n_models
        
        # Adjust based on coverage (models that predict more industries)
        model_names = sorted(model_predictions.keys())
        for i, model_name in enumerate(model_names):
            coverage = len(model_predictions[model_name])
            # Normalize coverage to [0.8, 1.2] range
            coverage_factor = 0.8 + 0.4 * min(coverage / 50, 1.0)
            weights[i] *= coverage_factor
        
        # Normalize to sum to 1
        weights = weights / weights.sum()
        
        return weights
    
    def stage2_company_candidate_generation(
        self,
        filtered_industries: pd.DataFrame,
        max_per_industry: int = None
    ) -> pd.DataFrame:
        """
        Stage 2: Generate company candidates from filtered industries.
        
        Returns DataFrame with candidate companies.
        """
        if max_per_industry is None:
            max_per_industry = self.company_fanout
        
        # print(f"\n[Stage 2] Company Candidate Generation")
        # print(f"  - Max {max_per_industry} companies per industry")
        
        candidate_indices = []
        candidate_industry_scores = {}
        
        for _, row in filtered_industries.iterrows():
            industry = row['industry']
            industry_score = row['score']
            
            # Get companies in this industry
            company_ids = self.industry_to_companies.get(industry, [])
            
            if company_ids:
                # Sample if too many
                if len(company_ids) > max_per_industry:
                    sampled = np.random.choice(company_ids, max_per_industry, replace=False)
                else:
                    sampled = company_ids
                
                candidate_indices.extend(sampled)
                for idx in sampled:
                    candidate_industry_scores[idx] = industry_score
        
        if not candidate_indices:
            print("  ! No candidates found")
            return pd.DataFrame()
        
        # Get candidate companies
        candidates = self.df_candidates.loc[candidate_indices].copy()
        candidates['industry_score'] = candidates.index.map(candidate_industry_scores)
        
        print(f"  ✓ Generated {len(candidates)} candidates from {len(filtered_industries)} industries")
        
        return candidates
    
    def stage3_company_ranking(
        self,
        outsource_url: str,
        candidates: pd.DataFrame,
        top_k: int
    ) -> pd.DataFrame:
        """
        Stage 3: Rank companies using multiple signals.
        
        Signals:
        - Content similarity (embeddings on company attributes)
        - Collaborative patterns (implicit from history)
        - Industry score inheritance
        - Attribute matching scores
        """
        # print(f"\n[Stage 3] Company Ranking (top {top_k})")
        
        if candidates.empty:
            return pd.DataFrame(columns=['company_id', 'company_name', 'industry', 'score'])
        
        scores = pd.DataFrame(index=candidates.index)
        
        # Signal 1: Industry score inheritance (from Stage 1)
        scores['industry_score'] = candidates['industry_score']
        # print(f"  - Industry scores: {scores['industry_score'].describe()}")
        
        # Signal 2: Content similarity (company embeddings)
        if self.company_embeddings is not None:
            # print("  - Computing content similarity...")
            # Get outsource profile embedding
            outsource_history = self.df_history[
                self.df_history['linkedin_company_outsource'] == outsource_url
            ]
            
            if not outsource_history.empty:
                # Create outsource profile text
                outsource_text = f"Industry preferences: {', '.join(outsource_history['industry'].unique())}. "
                outsource_text += f"Services: {', '.join(outsource_history['services'].unique())}."
                
                outsource_emb = self.company_embedder.transform([outsource_text])[0]
                
                # Compute similarity with candidates
                # Map DataFrame index to positional index for numpy array
                positions = [self.df_candidates.index.get_loc(idx) for idx in candidates.index]
                candidate_embs = self.company_embeddings[positions]
                similarities = np.dot(candidate_embs, outsource_emb)
                scores['content_similarity'] = similarities
                # print(f"    ✓ Content similarity: min={similarities.min():.3f}, max={similarities.max():.3f}, mean={similarities.mean():.3f}")
            else:
                scores['content_similarity'] = 0.0
        else:
            scores['content_similarity'] = 0.0
        
        # Signal 3: Attribute matching
        # print("  - Computing attribute matching...")
        # Location match (prefer same geographic region)
        if not outsource_history.empty:
            preferred_locations = set(outsource_history['location'].unique())
            scores['location_match'] = candidates['location'].apply(
                lambda loc: 1.0 if loc in preferred_locations else 0.3
            )
            
            # Client size match (prefer similar size ranges worked with before)
            hist_mins = outsource_history['client_min'].dropna()
            hist_maxs = outsource_history['client_max'].dropna()
            
            if len(hist_mins) > 0 and len(hist_maxs) > 0:
                avg_min = hist_mins.mean()
                avg_max = hist_maxs.mean()
                
                def size_match_score(row):
                    cmin = row.get('client_min')
                    cmax = row.get('client_max')
                    if pd.isna(cmin) or pd.isna(cmax):
                        return 0.3
                    # Check if ranges overlap reasonably
                    if cmax < avg_min * 0.3 or cmin > avg_max * 3.0:
                        return 0.2  # Very different size
                    elif cmax < avg_min * 0.7 or cmin > avg_max * 1.5:
                        return 0.6  # Somewhat different
                    else:
                        return 1.0  # Good match
                
                scores['size_match'] = candidates.apply(size_match_score, axis=1)
            else:
                scores['size_match'] = 0.5
        else:
            scores['location_match'] = 0.5
            scores['size_match'] = 0.5
        
        # Fusion: Weighted combination with normalization
        weights = {
            'industry_score': 0.40,
            'content_similarity': 0.35,
            'location_match': 0.15,
            'size_match': 0.10
        }
        
        # Z-score normalize each signal
        for col in weights.keys():
            if scores[col].std() > 0:
                scores[f'{col}_norm'] = (scores[col] - scores[col].mean()) / scores[col].std()
            else:
                scores[f'{col}_norm'] = scores[col]
        
        # Weighted fusion
        scores['final_score'] = sum(
            weights[col] * scores[f'{col}_norm']
            for col in weights.keys()
        )
        
        # Rank candidates
        ranked_indices = scores.nlargest(top_k, 'final_score').index
        results = candidates.loc[ranked_indices].copy()
        results['score'] = scores.loc[ranked_indices, 'final_score'].values
        
        # Include signal breakdown for debugging
        for col in ['industry_score', 'content_similarity', 'location_match', 'size_match']:
            results[col] = scores.loc[ranked_indices, col].values
        
        # print(f"  ✓ Ranked top {len(results)} companies")
        # print(f"    Score range: [{results['score'].min():.3f}, {results['score'].max():.3f}]")
        
        return results
    
    def stage4_diversity_selection(
        self,
        ranked_companies: pd.DataFrame,
        final_k: int,
        diversity_weight: float = 0.3
    ) -> pd.DataFrame:
        """
        Stage 4: Apply diversity-aware selection (MMR-style).
        
        Ensures recommended companies are not too similar to each other.
        """
        # print(f"\n[Stage 4] Diversity-Aware Selection (final top {final_k})")
        
        if len(ranked_companies) <= final_k:
            return ranked_companies
        
        # Start with top-scored company
        selected = [ranked_companies.index[0]]
        candidates = list(ranked_companies.index[1:])
        
        while len(selected) < final_k and candidates:
            best_idx = None
            best_score = -np.inf
            
            for idx in candidates:
                # Relevance score
                relevance = ranked_companies.loc[idx, 'score']
                
                # Diversity penalty (average similarity to selected)
                diversity_penalties = []
                for sel_idx in selected:
                    # Simple diversity: different industry or location
                    same_industry = (ranked_companies.loc[idx, 'industry'] == 
                                   ranked_companies.loc[sel_idx, 'industry'])
                    same_location = (ranked_companies.loc[idx, 'location'] == 
                                   ranked_companies.loc[sel_idx, 'location'])
                    
                    penalty = 0.0
                    if same_industry:
                        penalty += 0.5
                    if same_location:
                        penalty += 0.3
                    
                    diversity_penalties.append(penalty)
                
                avg_penalty = np.mean(diversity_penalties) if diversity_penalties else 0.0
                
                # MMR-style score
                mmr_score = (1 - diversity_weight) * relevance - diversity_weight * avg_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                candidates.remove(best_idx)
            else:
                break
        
        results = ranked_companies.loc[selected]
        
        # print(f"  ✓ Selected {len(results)} diverse companies")
        # print(f"    Industries: {results['industry'].value_counts().to_dict()}")
        
        return results
    
    def recommend_companies(
        self,
        outsource_url: str,
        top_k: int = 10,
        enable_diversity: bool = True
    ) -> pd.DataFrame:
        """
        Full pipeline: Recommend specific companies for an outsource company.
        
        Args:
            outsource_url: LinkedIn URL of outsource company
            top_k: Number of companies to recommend
            enable_diversity: Whether to apply diversity selection
        
        Returns:
            DataFrame with columns: ['company_name', 'industry', 'location', 
                                     'client_size', 'score', ...]
        """
        # print("\n" + "="*80)
        # print(f"RECOMMENDING COMPANIES FOR: {outsource_url}")
        # print("="*80)
        
        # Stage 1: Filter industries using ensemble
        filtered_industries = self.stage1_industry_ensemble_filter(
            outsource_url,
            top_k=self.industry_top_k
        )
        
        if filtered_industries.empty:
            print("\n! No industries found")
            return pd.DataFrame(columns=['company_name', 'industry', 'score'])
        
        # Stage 2: Generate company candidates
        candidates = self.stage2_company_candidate_generation(
            filtered_industries,
            max_per_industry=self.company_fanout
        )
        
        if candidates.empty:
            print("\n! No company candidates found")
            return pd.DataFrame(columns=['company_name', 'industry', 'score'])
        
        # Stage 3: Rank companies
        ranked = self.stage3_company_ranking(
            outsource_url,
            candidates,
            top_k=top_k * 3  # Get more for diversity selection
        )
        
        # Stage 4: Diversity selection (optional)
        if enable_diversity and len(ranked) > top_k:
            final = self.stage4_diversity_selection(
                ranked,
                final_k=top_k,
                diversity_weight=0.3
            )
        else:
            final = ranked.head(top_k)
        
        # Format output
        output_cols = ['company_name', 'industry', 'location', 'client_min', 'client_max', 'score']
        available_cols = [col for col in output_cols if col in final.columns]
        results = final[available_cols].reset_index(drop=True)
        
        # print("\n" + "="*80)
        # print(f"✓ FINAL RECOMMENDATIONS: {len(results)} companies")
        # print("="*80)
        
        return results
