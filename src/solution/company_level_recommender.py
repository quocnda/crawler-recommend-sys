"""
Multi-Stage Company-Level Recommender System

Architecture:
    Stage 1: Industry-based Candidate Generation (Broad)
    Stage 2: Multi-attribute Filtering (Client Size, Location, Services)
    Stage 3: Semantic Similarity Ranking (Embeddings)
    Stage 4: LLM Re-ranking (Optional, for final refinement)

Flow:
    Outsource Profile → Industry Match (top 10-15) → 
    Attribute Filter (100-200 companies) → 
    Semantic Rank (top 30) → 
    [Optional: LLM] → Final Top 10 Companies
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class CompanyLevelRecommender:
    """
    Multi-stage funnel recommender that recommends specific companies
    instead of just industries.
    """
    
    def __init__(
        self,
        df_history: pd.DataFrame,
        df_candidates: pd.DataFrame,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Args:
            df_history: Historical data with outsource-client interactions
            df_candidates: Candidate companies (potential clients)
            embedding_model_name: Name of sentence transformer model
        """
        self.df_history = df_history.copy()
        self.df_candidates = df_candidates.copy()
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Preprocess data
        self._preprocess_data()
        
        # Build lookup structures
        self._build_lookups()
        
        # Precompute embeddings
        self._precompute_embeddings()
        
    def _preprocess_data(self):
        """Clean and standardize data."""
        # Standardize column names
        for df in [self.df_history, self.df_candidates]:
            if 'Industry' in df.columns and 'industry' not in df.columns:
                df.rename(columns={'Industry': 'industry'}, inplace=True)
            if 'Location' in df.columns and 'location' not in df.columns:
                df.rename(columns={'Location': 'location'}, inplace=True)
            if 'Client size' in df.columns and 'client_size' not in df.columns:
                df.rename(columns={'Client size': 'client_size'}, inplace=True)
            if 'Services' in df.columns and 'services' not in df.columns:
                df.rename(columns={'Services': 'services'}, inplace=True)
            if 'linkedin Company Outsource' in df.columns and 'linkedin_company_outsource' not in df.columns:
                df.rename(columns={'linkedin Company Outsource': 'linkedin_company_outsource'}, inplace=True)
        
        # Fill NaN values
        for col in ['industry', 'location', 'client_size', 'services']:
            if col in self.df_history.columns:
                self.df_history[col] = self.df_history[col].fillna('Unknown')
            if col in self.df_candidates.columns:
                self.df_candidates[col] = self.df_candidates[col].fillna('Unknown')
        
        # Create unique company identifier
        self.df_candidates['company_id'] = self.df_candidates.apply(
            lambda row: f"{row.get('reviewer_company', 'Unknown')}_{row.get('industry', 'Unknown')}_{row.get('location', 'Unknown')}",
            axis=1
        )
        
    def _build_lookups(self):
        """Build lookup structures for fast filtering."""
        # Outsource company → industries they worked with
        self.outsource_to_industries = defaultdict(set)
        for _, row in self.df_history.iterrows():
            outsource = row.get('linkedin_company_outsource')
            industry = row.get('industry')
            if pd.notna(outsource) and pd.notna(industry):
                self.outsource_to_industries[outsource].add(industry)
        
        # Outsource company → client sizes they worked with
        self.outsource_to_sizes = defaultdict(set)
        for _, row in self.df_history.iterrows():
            outsource = row.get('linkedin_company_outsource')
            size = row.get('client_size')
            if pd.notna(outsource) and pd.notna(size):
                self.outsource_to_sizes[outsource].add(size)
        
        # Outsource company → services they provided
        self.outsource_to_services = defaultdict(set)
        for _, row in self.df_history.iterrows():
            outsource = row.get('linkedin_company_outsource')
            services = row.get('services', '')
            if pd.notna(outsource) and pd.notna(services):
                # Split services by common delimiters
                service_list = [s.strip() for s in str(services).split()]
                self.outsource_to_services[outsource].update(service_list)
        
        # Industry → list of companies (store positional indices)
        self.industry_to_companies = defaultdict(list)
        # Reset candidate index to ensure iloc works correctly
        self.df_candidates = self.df_candidates.reset_index(drop=True)
        for idx in range(len(self.df_candidates)):
            industry = self.df_candidates.iloc[idx].get('industry')
            if pd.notna(industry):
                self.industry_to_companies[industry].append(idx)
        
        print(f"✓ Loaded {len(self.outsource_to_industries)} outsource companies")
        print(f"✓ Indexed {len(self.industry_to_companies)} industries")
        print(f"✓ Total candidates: {len(self.df_candidates)}")
        
    def _precompute_embeddings(self):
        """Precompute embeddings for all companies."""
        print("Precomputing candidate company embeddings...")
        
        # Create rich text representation for each candidate company
        self.candidate_texts = []
        for _, row in self.df_candidates.iterrows():
            text_parts = []
            
            # Company name and role
            if pd.notna(row.get('reviewer_company')):
                text_parts.append(f"Company: {row['reviewer_company']}")
            if pd.notna(row.get('reviewer_role')):
                text_parts.append(f"Role: {row['reviewer_role']}")
            
            # Industry and location
            if pd.notna(row.get('industry')):
                text_parts.append(f"Industry: {row['industry']}")
            if pd.notna(row.get('location')):
                text_parts.append(f"Location: {row['location']}")
            
            # Client size and services
            if pd.notna(row.get('client_size')):
                text_parts.append(f"Size: {row['client_size']}")
            if pd.notna(row.get('services')):
                text_parts.append(f"Services: {row['services']}")
            
            # Background info if available
            if pd.notna(row.get('background')):
                text_parts.append(f"Background: {row['background'][:200]}")  # Limit length
            
            self.candidate_texts.append(" | ".join(text_parts))
        
        # Encode all candidates
        self.candidate_embeddings = self.embedding_model.encode(
            self.candidate_texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"Computed embeddings for {len(self.candidate_embeddings)} candidates")
        
    def _build_outsource_profile(self, outsource_url: str) -> str:
        """Build rich text profile for outsource company."""
        # Get all records for this outsource
        outsource_records = self.df_history[
            self.df_history['linkedin_company_outsource'] == outsource_url
        ]
        
        if len(outsource_records) == 0:
            return "Unknown outsource company"
        
        text_parts = []
        
        # Industries worked with
        industries = self.outsource_to_industries.get(outsource_url, set())
        if industries:
            text_parts.append(f"Industries: {', '.join(list(industries)[:5])}")
        
        # Client sizes worked with
        sizes = self.outsource_to_sizes.get(outsource_url, set())
        if sizes:
            text_parts.append(f"Client Sizes: {', '.join(list(sizes)[:5])}")
        
        # Services provided
        services = self.outsource_to_services.get(outsource_url, set())
        if services:
            text_parts.append(f"Services: {', '.join(list(services)[:10])}")
        
        # Sample project descriptions
        sample_projects = outsource_records['background'].dropna().head(3).tolist()
        if sample_projects:
            text_parts.append(f"Sample Projects: {' | '.join([p[:100] for p in sample_projects])}")
        
        return " | ".join(text_parts)
    
    # ========== STAGE 1: Industry-based Candidate Generation ==========
    
    def stage1_industry_filter(
        self,
        outsource_url: str,
        top_k_industries: int = 15
    ) -> List[int]:
        """
        Stage 1: Generate broad candidate set based on industry matching.
        
        Returns:
            List of candidate indices (row numbers in df_candidates)
        """
        # Get industries this outsource has worked with
        worked_industries = self.outsource_to_industries.get(outsource_url, set())
        
        if not worked_industries:
            # Cold start: return all industries with weighted sampling
            all_industries = list(self.industry_to_companies.keys())
            worked_industries = set(all_industries[:top_k_industries])
        
        # Get all companies in these industries
        candidate_indices = []
        for industry in worked_industries:
            candidate_indices.extend(self.industry_to_companies.get(industry, []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in candidate_indices:
            # Validate index is within bounds
            if idx not in seen and idx < len(self.df_candidates):
                seen.add(idx)
                unique_indices.append(idx)
        
        print(f"Stage 1: {len(worked_industries)} industries → {len(unique_indices)} candidates")
        return unique_indices
    
    # ========== STAGE 2: Multi-attribute Filtering ==========
    
    def _size_compatibility_score(self, outsource_sizes: set, candidate_size: str) -> float:
        """Calculate compatibility score based on client size."""
        if not outsource_sizes or pd.isna(candidate_size):
            return 0.5  # Neutral score
        
        if candidate_size in outsource_sizes:
            return 1.0  # Perfect match
        
        # Parse size ranges for proximity matching
        size_order = [
            '1-10 Employees',
            '11-50 Employees', 
            '51-200 Employees',
            '201-500 Employees',
            '501-1,000 Employees',
            '1,001-5,000 Employees',
            '5,001-10,000 Employees',
            '10,001+ Employees'
        ]
        
        try:
            candidate_idx = size_order.index(candidate_size)
            outsource_indices = [size_order.index(s) for s in outsource_sizes if s in size_order]
            
            if outsource_indices:
                min_distance = min(abs(candidate_idx - oi) for oi in outsource_indices)
                # Exponential decay: adjacent = 0.7, 2 away = 0.5, 3+ = 0.3
                return max(0.3, 1.0 - (min_distance * 0.15))
        except (ValueError, IndexError):
            pass
        
        return 0.4  # Some compatibility for unknown
    
    def _location_compatibility_score(self, outsource_url: str, candidate_location: str) -> float:
        """Calculate compatibility score based on location."""
        # Get locations this outsource has worked with
        outsource_locations = set()
        outsource_records = self.df_history[
            self.df_history['linkedin_company_outsource'] == outsource_url
        ]
        for loc in outsource_records['location'].dropna():
            outsource_locations.add(loc)
        
        if not outsource_locations or pd.isna(candidate_location):
            return 0.5  # Neutral
        
        # Exact match
        if candidate_location in outsource_locations:
            return 1.0
        
        # Country/region proximity (basic heuristic)
        candidate_parts = str(candidate_location).split(',')
        for worked_loc in outsource_locations:
            worked_parts = str(worked_loc).split(',')
            # Same country or region
            if any(cp.strip() in wp for cp in candidate_parts for wp in worked_parts):
                return 0.7
        
        return 0.3  # Different region
    
    def _service_compatibility_score(self, outsource_url: str, candidate_services: str) -> float:
        """Calculate compatibility score based on services."""
        outsource_services = self.outsource_to_services.get(outsource_url, set())
        
        if not outsource_services or pd.isna(candidate_services):
            return 0.5
        
        candidate_service_set = set([s.strip() for s in str(candidate_services).split()])
        
        # Jaccard similarity
        intersection = outsource_services & candidate_service_set
        union = outsource_services | candidate_service_set
        
        if not union:
            return 0.5
        
        return len(intersection) / len(union)
    
    def stage2_attribute_filter(
        self,
        outsource_url: str,
        candidate_indices: List[int],
        min_score: float = 0.3,
        top_k: int = 200
    ) -> List[Tuple[int, float]]:
        """
        Stage 2: Filter candidates based on multiple attributes.
        
        Returns:
            List of (candidate_idx, score) tuples, sorted by score
        """
        outsource_sizes = self.outsource_to_sizes.get(outsource_url, set())
        
        scored_candidates = []
        for idx in candidate_indices:
            # Validate index before access
            if idx >= len(self.df_candidates):
                continue
                
            row = self.df_candidates.iloc[idx]
            
            # Calculate individual scores
            size_score = self._size_compatibility_score(outsource_sizes, row.get('client_size'))
            location_score = self._location_compatibility_score(outsource_url, row.get('location'))
            service_score = self._service_compatibility_score(outsource_url, row.get('services'))
            
            # Weighted combination
            # Weight services highest (most important for compatibility)
            final_score = (
                0.4 * service_score +
                0.35 * size_score +
                0.25 * location_score
            )
            
            if final_score >= min_score:
                scored_candidates.append((idx, final_score))
        
        # Sort by score and take top_k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        result = scored_candidates[:top_k]
        
        print(f"Stage 2: {len(candidate_indices)} → {len(result)} candidates (score >= {min_score})")
        return result
    
    # ========== STAGE 3: Semantic Similarity Ranking ==========
    
    def stage3_semantic_ranking(
        self,
        outsource_url: str,
        scored_candidates: List[Tuple[int, float]],
        top_k: int = 30,
        alpha: float = 0.6  # Weight for semantic similarity vs attribute score
    ) -> List[Tuple[int, float]]:
        """
        Stage 3: Re-rank using semantic similarity.
        
        Args:
            alpha: Weight for semantic score (1-alpha for attribute score)
        """
        # Build outsource profile
        outsource_profile = self._build_outsource_profile(outsource_url)
        
        # Encode outsource profile
        outsource_embedding = self.embedding_model.encode([outsource_profile])[0]
        
        # Re-score with semantic similarity
        reranked = []
        for idx, attr_score in scored_candidates:
            candidate_embedding = self.candidate_embeddings[idx]
            
            # Cosine similarity
            semantic_score = float(cosine_similarity(
                [outsource_embedding],
                [candidate_embedding]
            )[0][0])
            
            # Combine scores
            final_score = alpha * semantic_score + (1 - alpha) * attr_score
            
            reranked.append((idx, final_score, semantic_score, attr_score))
        
        # Sort and take top_k
        reranked.sort(key=lambda x: x[1], reverse=True)
        result = [(idx, score) for idx, score, _, _ in reranked[:top_k]]
        
        print(f"Stage 3: {len(scored_candidates)} → {len(result)} candidates (semantic ranking)")
        return result
    
    # ========== Main Recommendation Function ==========
    
    def recommend_companies(
        self,
        outsource_url: str,
        top_k: int = 10,
        stage1_industries: int = 15,
        stage2_candidates: int = 200,
        stage3_candidates: int = 30,
        return_details: bool = True
    ) -> pd.DataFrame:
        """
        Multi-stage recommendation pipeline.
        
        Args:
            outsource_url: LinkedIn URL of outsource company
            top_k: Final number of recommendations
            stage1_industries: Number of industries to consider
            stage2_candidates: Max candidates after attribute filtering
            stage3_candidates: Max candidates after semantic ranking
            return_details: Include detailed information in output
        
        Returns:
            DataFrame with recommended companies and scores
        """
        print(f"\n{'='*60}")
        print(f"Recommending for: {outsource_url}")
        print(f"{'='*60}")
        
        # Stage 1: Industry filtering
        stage1_candidates = self.stage1_industry_filter(outsource_url, stage1_industries)
        
        if not stage1_candidates:
            print("No candidates found in Stage 1")
            return pd.DataFrame()
        
        # Stage 2: Attribute filtering
        stage2_results = self.stage2_attribute_filter(
            outsource_url,
            stage1_candidates,
            top_k=stage2_candidates
        )
        
        if not stage2_results:
            print("No candidates passed Stage 2 filters")
            return pd.DataFrame()
        
        # Stage 3: Semantic ranking
        stage3_results = self.stage3_semantic_ranking(
            outsource_url,
            stage2_results,
            top_k=stage3_candidates
        )
        
        # Take final top_k
        final_results = stage3_results[:top_k]
        
        # Build output DataFrame
        recommendations = []
        for idx, score in final_results:
            row = self.df_candidates.iloc[idx]
            rec = {
                'company_name': row.get('reviewer_company', 'Unknown'),
                'industry': row.get('industry', 'Unknown'),
                'location': row.get('location', 'Unknown'),
                'client_size': row.get('client_size', 'Unknown'),
                'score': score
            }
            
            if return_details:
                rec.update({
                    'services': row.get('services', 'Unknown'),
                    'reviewer_role': row.get('reviewer_role', 'Unknown'),
                    'company_id': row.get('company_id', 'Unknown')
                })
            
            recommendations.append(rec)
        
        result_df = pd.DataFrame(recommendations)
        
        print(f"\n{'='*60}")
        print(f"Final: Top {len(result_df)} companies")
        print(f"{'='*60}\n")
        
        return result_df


def integrate_company_level_recommendations(
    df_history: pd.DataFrame,
    df_test: pd.DataFrame,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Integration function for evaluation pipeline.
    
    Returns:
        DataFrame with columns: [linkedin_company_outsource, company_name, industry, score]
    """
    print("Initializing Company-Level Recommender...")
    recommender = CompanyLevelRecommender(df_history, df_test)
    
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user = row.get('linkedin Company Outsource') or row.get('linkedin_company_outsource')
        if pd.isna(user) or user in seen_users:
            continue
        seen_users.add(user)
        
        try:
            recs = recommender.recommend_companies(user, top_k=top_k, return_details=False)
            
            for _, rec in recs.iterrows():
                results.append({
                    'linkedin_company_outsource': user,
                    'company_name': rec['company_name'],
                    'industry': rec['industry'],
                    'score': rec['score']
                })
        except Exception as e:
            print(f"Error recommending for {user}: {e}")
            continue
    
    return pd.DataFrame(results)
