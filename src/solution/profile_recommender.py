"""
Multi-Stage Client Profile Recommender System
==============================================

This module implements a hierarchical recommendation system that recommends
client profiles instead of specific companies, using advanced ensemble methods
at each stage.

Architecture:
    Stage 1: Industry Recommendation (Advanced Ensemble)
    Stage 2: Client Size Recommendation (Per Industry)
    Stage 3: Location Recommendation (Per Industry + Size)
    Stage 4: Services Recommendation (Per Industry + Size + Location)
    Stage 5: LLM Synthesis → Profile + Example Companies

Author: Advanced Recommendation System
Date: November 2025
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Import lazy to avoid circular dependencies
def get_embedder():
    from solution.content_base_for_item import OpenAIEmbedder
    return OpenAIEmbedder

def get_advanced_ensemble():
    from solution.advanced_ensemble import integrate_advanced_ensemble
    return integrate_advanced_ensemble

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. LLM synthesis will be disabled.")


class AttributeEnsembleRecommender:
    """
    Generic ensemble recommender for any categorical attribute (client_size, location, services).
    Reuses the advanced ensemble architecture.
    """
    
    def __init__(
        self,
        attribute_name: str,
        df_history: pd.DataFrame,
        use_embeddings: bool = True
    ):
        """
        Args:
            attribute_name: Name of attribute to recommend ('client_size', 'location', 'services')
            df_history: Historical interaction data
            use_embeddings: Whether to use OpenAI embeddings
        """
        self.attribute_name = attribute_name
        self.df_history = df_history.copy()
        self.use_embeddings = use_embeddings
        
        # Normalize column names
        self._normalize_columns()
        
        # Build lookup structures
        self._build_lookups()
        
        # Initialize embeddings if needed
        if self.use_embeddings:
            self._init_embeddings()
    
    def _normalize_columns(self):
        """Standardize column names."""
        rename_map = {
            'Industry': 'industry',
            'Location': 'location', 
            'Client size': 'client_size',
            'Services': 'services',
            'linkedin Company Outsource': 'linkedin_company_outsource'
        }
        self.df_history.rename(columns=rename_map, inplace=True)
        
        # Fill NaN
        for col in ['industry', 'location', 'client_size', 'services']:
            if col in self.df_history.columns:
                self.df_history[col] = self.df_history[col].fillna('Unknown')
    
    def _build_lookups(self):
        """Build lookup structures."""
        # Outsource → attribute values
        self.outsource_to_attrs = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.df_history.iterrows():
            outsource = row.get('linkedin_company_outsource')
            attr_value = row.get(self.attribute_name)
            
            if pd.notna(outsource) and pd.notna(attr_value) and attr_value != 'Unknown':
                self.outsource_to_attrs[outsource][attr_value] += 1
        
        # Get all unique attribute values
        self.all_attr_values = set()
        for _, row in self.df_history.iterrows():
            val = row.get(self.attribute_name)
            if pd.notna(val) and val != 'Unknown':
                self.all_attr_values.add(val)
        
        print(f"✓ {self.attribute_name}: {len(self.all_attr_values)} unique values")
    
    def _init_embeddings(self):
        """Initialize embeddings for attribute values."""
        if not OPENAI_AVAILABLE:
            self.use_embeddings = False
            return
        
        # Check if we have any attribute values
        if not self.all_attr_values:
            print(f"⚠️  No attribute values found for {self.attribute_name}, skipping embeddings")
            self.use_embeddings = False
            return
        
        print(f"Computing embeddings for {self.attribute_name}...")
        OpenAIEmbedder = get_embedder()
        self.embedder = OpenAIEmbedder(model="text-embedding-3-large")
        
        # Create text representations
        attr_texts = list(self.all_attr_values)
        self.attr_to_idx = {val: i for i, val in enumerate(attr_texts)}
        
        # Compute embeddings
        try:
            self.attr_embeddings = self.embedder.transform(attr_texts)
            print(f"✓ Computed embeddings: {self.attr_embeddings.shape}")
        except Exception as e:
            print(f"⚠️  Error computing embeddings for {self.attribute_name}: {e}")
            self.use_embeddings = False
    
    def recommend_collaborative(
        self, 
        outsource_url: str,
        context_filter: Optional[Dict[str, str]] = None,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Collaborative filtering: recommend based on what similar outsource companies worked with.
        """
        if outsource_url not in self.outsource_to_attrs:
            return {}
        
        # Get this outsource's historical attribute distribution
        user_attrs = self.outsource_to_attrs[outsource_url]
        
        # Find similar outsource companies (based on attribute overlap)
        similar_outsources = []
        for other_url, other_attrs in self.outsource_to_attrs.items():
            if other_url == outsource_url:
                continue
            
            # Calculate Jaccard similarity
            common = set(user_attrs.keys()) & set(other_attrs.keys())
            union = set(user_attrs.keys()) | set(other_attrs.keys())
            
            if union:
                similarity = len(common) / len(union)
                if similarity > 0.1:  # Threshold
                    similar_outsources.append((other_url, similarity))
        
        # Aggregate attributes from similar companies
        attr_scores = defaultdict(float)
        total_weight = 0.0
        
        for other_url, similarity in similar_outsources:
            for attr_val, count in self.outsource_to_attrs[other_url].items():
                # Apply context filter if provided
                if context_filter and not self._matches_context(attr_val, context_filter):
                    continue
                
                attr_scores[attr_val] += similarity * count
                total_weight += similarity
        
        # Normalize scores
        if total_weight > 0:
            attr_scores = {k: v/total_weight for k, v in attr_scores.items()}
        
        # Sort and return top-k
        sorted_attrs = sorted(attr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_attrs)
    
    def recommend_content_based(
        self,
        outsource_url: str,
        outsource_profile: Dict[str, Any],
        context_filter: Optional[Dict[str, str]] = None,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Content-based: recommend based on semantic similarity to user's profile.
        """
        if not self.use_embeddings or self.attr_embeddings is None or len(self.attr_embeddings) == 0:
            return {}
        
        # Create outsource profile text
        profile_parts = []
        for key, val in outsource_profile.items():
            if pd.notna(val) and val != 'Unknown':
                profile_parts.append(f"{key}: {val}")
        
        if not profile_parts:
            return {}
        
        try:
            profile_text = " | ".join(profile_parts)
            profile_emb = self.embedder.transform([profile_text])[0]
            
            # Compute similarity with all attribute values
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(
                profile_emb.reshape(1, -1),
                self.attr_embeddings
            )[0]
            
            # Create scores dict
            attr_scores = {}
            for attr_val, idx in self.attr_to_idx.items():
                # Apply context filter if provided
                if context_filter and not self._matches_context(attr_val, context_filter):
                    continue
                
                attr_scores[attr_val] = float(similarities[idx])
            
            # Sort and return top-k
            sorted_attrs = sorted(attr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            return dict(sorted_attrs)
        except Exception as e:
            print(f"⚠️  Error in content-based recommendation for {self.attribute_name}: {e}")
            return {}
    
    def recommend_popularity(
        self,
        context_filter: Optional[Dict[str, str]] = None,
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Popularity-based: recommend most common attribute values in dataset.
        """
        attr_counts = defaultdict(int)
        
        for _, row in self.df_history.iterrows():
            attr_val = row.get(self.attribute_name)
            if pd.notna(attr_val) and attr_val != 'Unknown':
                # Apply context filter if provided
                if context_filter and not self._matches_context(attr_val, context_filter):
                    continue
                
                attr_counts[attr_val] += 1
        
        # Normalize to probabilities
        total = sum(attr_counts.values())
        if total > 0:
            attr_scores = {k: v/total for k, v in attr_counts.items()}
        else:
            attr_scores = {}
        
        # Sort and return top-k
        sorted_attrs = sorted(attr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_attrs)
    
    def _matches_context(self, attr_val: str, context_filter: Dict[str, str]) -> bool:
        """Check if attribute value matches context filter (e.g., specific industry)."""
        # For now, simple implementation - can be extended
        # Context filter format: {'industry': 'Software', 'client_size': '11-50 Employees'}
        
        # Check if this attribute value appears in rows matching the context
        matching_rows = self.df_history
        
        for filter_col, filter_val in context_filter.items():
            if filter_col in matching_rows.columns:
                matching_rows = matching_rows[matching_rows[filter_col] == filter_val]
        
        # If no matching rows after filtering, return False to skip this attribute
        if len(matching_rows) == 0:
            return False
        
        # Check if attr_val appears in matching rows
        return attr_val in matching_rows[self.attribute_name].values
    
    def recommend_ensemble(
        self,
        outsource_url: str,
        outsource_profile: Dict[str, Any],
        context_filter: Optional[Dict[str, str]] = None,
        top_k: int = 5,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Ensemble recommendation combining multiple methods.
        """
        if weights is None:
            weights = {
                'collaborative': 0.4,
                'content': 0.4,
                'popularity': 0.2
            }
        
        # Get predictions from each method
        collab_scores = self.recommend_collaborative(outsource_url, context_filter, top_k*2)
        content_scores = self.recommend_content_based(outsource_url, outsource_profile, context_filter, top_k*2)
        pop_scores = self.recommend_popularity(context_filter, top_k*2)
        
        # If all methods fail with context filter, try without context filter
        if not collab_scores and not content_scores and not pop_scores:
            if context_filter:
                print(f"⚠️  No data for {self.attribute_name} with context {context_filter}, trying without context...")
                collab_scores = self.recommend_collaborative(outsource_url, None, top_k*2)
                content_scores = self.recommend_content_based(outsource_url, outsource_profile, None, top_k*2)
                pop_scores = self.recommend_popularity(None, top_k*2)
        
        # If still no results, return global popularity
        if not collab_scores and not content_scores and not pop_scores:
            print(f"⚠️  All methods failed for {self.attribute_name}, using global popularity")
            return self.recommend_popularity(None, top_k)
        
        # Combine scores
        all_attrs = set(collab_scores.keys()) | set(content_scores.keys()) | set(pop_scores.keys())
        ensemble_scores = {}
        
        for attr_val in all_attrs:
            score = 0.0
            score += weights['collaborative'] * collab_scores.get(attr_val, 0.0)
            score += weights['content'] * content_scores.get(attr_val, 0.0)
            score += weights['popularity'] * pop_scores.get(attr_val, 0.0)
            ensemble_scores[attr_val] = score
        
        # Sort and return top-k
        sorted_attrs = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_attrs)


class MultiStageProfileRecommender:
    """
    Multi-stage recommender that generates client profiles using advanced ensemble at each stage.
    """
    
    def __init__(self, df_history: pd.DataFrame, use_embeddings: bool = True):
        """
        Args:
            df_history: Historical interaction data
            use_embeddings: Whether to use OpenAI embeddings
        """
        self.df_history = df_history.copy()
        self.use_embeddings = use_embeddings
        
        # Normalize columns
        self._normalize_columns()
        
        # Initialize stage recommenders
        print("Initializing Multi-Stage Profile Recommender...")
        
        # Stage 1: Industry (use existing advanced ensemble)
        print("Stage 1: Industry Recommender")
        self.industry_recommender = None  # Will use integrate_advanced_ensemble
        
        # Stage 2: Client Size
        print("Stage 2: Client Size Recommender")
        self.client_size_recommender = AttributeEnsembleRecommender(
            'client_size', df_history, use_embeddings
        )
        
        # Stage 3: Location
        print("Stage 3: Location Recommender")
        self.location_recommender = AttributeEnsembleRecommender(
            'location', df_history, use_embeddings
        )
        
        # Stage 4: Services
        print("Stage 4: Services Recommender")
        self.services_recommender = AttributeEnsembleRecommender(
            'services', df_history, use_embeddings
        )
        
        # Stage 5: LLM (initialized when needed)
        if OPENAI_AVAILABLE:
            self.llm_client = OpenAI()
        else:
            self.llm_client = None
        
        print("✓ Multi-Stage Profile Recommender initialized!")
    
    def _normalize_columns(self):
        """Standardize column names."""
        rename_map = {
            'Industry': 'industry',
            'Location': 'location',
            'Client size': 'client_size',
            'Services': 'services',
            'linkedin Company Outsource': 'linkedin_company_outsource'
        }
        self.df_history.rename(columns=rename_map, inplace=True)
    
    def recommend_profile(
        self,
        outsource_url: str,
        outsource_profile: Dict[str, Any],
        industry_recommendations: List[Tuple[str, float]],
        top_profiles: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate complete client profile recommendations.
        
        Args:
            outsource_url: LinkedIn URL of outsource company
            outsource_profile: Profile features of outsource company
            industry_recommendations: [(industry, score), ...] from Stage 1
            top_profiles: Number of profiles to generate
            
        Returns:
            List of profile dicts with industries, sizes, locations, services, examples
        """
        profiles = []
        
        # For each top industry, generate full profile
        for industry, industry_score in industry_recommendations[:top_profiles]:
            print(f"\n{'='*60}")
            print(f"Generating profile for industry: {industry}")
            print(f"{'='*60}")
            
            context = {'industry': industry}
            
            # Stage 2: Recommend client sizes for this industry
            print("  → Stage 2: Client Size...")
            try:
                client_sizes = self.client_size_recommender.recommend_ensemble(
                    outsource_url,
                    outsource_profile,
                    context_filter=context,
                    top_k=3
                )
                if not client_sizes:
                    # Fallback to popularity without context
                    client_sizes = self.client_size_recommender.recommend_popularity(None, top_k=3)
                print(f"     Top sizes: {list(client_sizes.keys())[:3]}")
            except Exception as e:
                print(f"     ⚠️ Error: {e}, using fallback")
                # Ultimate fallback: global popularity
                try:
                    client_sizes = self.client_size_recommender.recommend_popularity(None, top_k=3)
                except Exception:
                    client_sizes = {'11-50 Employees': 1.0}  # Hard-coded fallback
            
            # Take top client size with safe default
            top_size = list(client_sizes.keys())[0] if client_sizes else '11-50 Employees'
            context['client_size'] = top_size
            
            # Stage 3: Recommend locations
            print("  → Stage 3: Location...")
            try:
                locations = self.location_recommender.recommend_ensemble(
                    outsource_url,
                    outsource_profile,
                    context_filter=context,
                    top_k=3
                )
                if not locations:
                    # Fallback: get most common locations in this industry+size
                    filtered = self.df_history[
                        (self.df_history['industry'] == industry) &
                        (self.df_history['client_size'] == top_size)
                    ]
                    if len(filtered) > 0:
                        loc_counts = filtered['location'].value_counts().head(3)
                        locations = {loc: float(count) / len(filtered) for loc, count in loc_counts.items()}
                    else:
                        locations = self.location_recommender.recommend_popularity(None, top_k=3)
                print(f"     Top locations: {list(locations.keys())[:3]}")
            except Exception as e:
                print(f"     ⚠️ Error: {e}, using fallback")
                try:
                    locations = self.location_recommender.recommend_popularity(None, top_k=3)
                except Exception:
                    locations = {'United States': 1.0}
            
            top_location = list(locations.keys())[0] if locations else 'United States'
            context['location'] = top_location
            
            # Stage 4: Recommend services
            print("  → Stage 4: Services...")
            try:
                services = self.services_recommender.recommend_ensemble(
                    outsource_url,
                    outsource_profile,
                    context_filter=context,
                    top_k=5
                )
                if not services:
                    # Fallback: get most common services in this profile
                    filtered = self.df_history[
                        (self.df_history['industry'] == industry) &
                        (self.df_history['client_size'] == top_size)
                    ]
                    if len(filtered) > 0:
                        # Count service occurrences
                        from collections import Counter
                        all_services = []
                        for svc in filtered['services'].dropna():
                            all_services.extend([s.strip() for s in str(svc).split()])
                        svc_counts = Counter(all_services).most_common(5)
                        total = sum(c for _, c in svc_counts)
                        services = {s: c / total for s, c in svc_counts}
                    else:
                        services = self.services_recommender.recommend_popularity(None, top_k=5)
                print(f"     Top services: {list(services.keys())[:5]}")
            except Exception as e:
                print(f"     ⚠️ Error: {e}, using fallback")
                try:
                    services = self.services_recommender.recommend_popularity(None, top_k=5)
                except Exception:
                    services = {'Custom Software Development': 1.0}
            
            # Stage 5: Find example companies matching this profile
            print("  → Stage 5: Finding example companies...")
            service_keys = list(services.keys())[:3] if services else []
            example_companies = self._find_example_companies(
                industry, top_size, top_location, service_keys
            )
            print(f"     Found {len(example_companies)} examples")
            
            # Create profile with safe defaults
            top_services_list = list(services.keys())[:5] if services else []
            profile = {
                'industry': industry,
                'industry_score': float(industry_score),
                'client_size': top_size,
                'client_size_score': float(client_sizes.get(top_size, 0.0)) if client_sizes else 0.0,
                'location': top_location,
                'location_score': float(locations.get(top_location, 0.0)) if locations else 0.0,
                'top_services': top_services_list,
                'service_scores': [float(services.get(s, 0.0)) for s in top_services_list] if services else [],
                'example_companies': example_companies,
                'n_examples': len(example_companies)
            }
            
            profiles.append(profile)
        
        return profiles
    
    def _find_example_companies(
        self,
        industry: str,
        client_size: str,
        location: str,
        services: List[str],
        max_examples: int = 5
    ) -> List[Dict[str, str]]:
        """
        Find real companies from history that match the recommended profile.
        """
        # Filter by exact matches
        matching = self.df_history[
            (self.df_history['industry'] == industry) &
            (self.df_history['client_size'] == client_size)
        ].copy()
        
        # Score by location and service match
        def score_match(row):
            score = 0.0
            
            # Location match (exact or partial)
            if row.get('location') == location:
                score += 2.0
            elif location in str(row.get('location', '')):
                score += 1.0
            
            # Service overlap
            row_services = str(row.get('services', '')).lower()
            for service in services:
                if service.lower() in row_services:
                    score += 1.0
            
            return score
        
        if len(matching) > 0:
            matching['match_score'] = matching.apply(score_match, axis=1)
            matching = matching.sort_values('match_score', ascending=False)
            
            # Get top unique companies
            examples = []
            seen_companies = set()
            
            for _, row in matching.iterrows():
                company_name = row.get('reviewer_company', 'Unknown')
                
                if company_name not in seen_companies and company_name != 'Unknown':
                    examples.append({
                        'company': company_name,
                        'role': row.get('reviewer_role', 'Unknown'),
                        'location': row.get('location', 'Unknown'),
                        'services_used': row.get('services', 'Unknown'),
                        'project_size': row.get('project_size', 'Unknown')
                    })
                    seen_companies.add(company_name)
                    
                    if len(examples) >= max_examples:
                        break
            
            return examples
        
        return []
    
    def generate_llm_explanation(
        self,
        outsource_profile: Dict[str, Any],
        recommended_profile: Dict[str, Any]
    ) -> str:
        """
        Use LLM to generate explanation for why this profile is recommended.
        """
        if not self.llm_client:
            return "LLM not available for generating explanations."
        
        prompt = f"""You are an expert B2B sales consultant specializing in outsourcing recommendations.

Given an outsource company profile:
{self._format_profile(outsource_profile)}

We recommend targeting the following client profile:
- Industry: {recommended_profile['industry']}
- Client Size: {recommended_profile['client_size']}
- Location: {recommended_profile['location']}
- Preferred Services: {', '.join(recommended_profile['top_services'][:3])}

Example companies matching this profile:
{self._format_examples(recommended_profile['example_companies'])}

Please provide:
1. A concise explanation (2-3 sentences) of WHY this client profile is a good fit
2. Key selling points this outsource company should emphasize
3. Potential project value range

Keep the response professional and actionable."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a B2B sales intelligence expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating explanation: {e}"
    
    def _format_profile(self, profile: Dict[str, Any]) -> str:
        """Format profile dict to readable string."""
        parts = []
        for key, val in profile.items():
            if pd.notna(val) and val != 'Unknown':
                parts.append(f"  - {key}: {val}")
        return "\n".join(parts) if parts else "  (No profile information)"
    
    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format example companies to readable string."""
        if not examples:
            return "  (No examples found)"
        
        formatted = []
        for i, ex in enumerate(examples[:3], 1):
            formatted.append(
                f"  {i}. {ex['company']} ({ex['location']}) - {ex['services_used'][:100]}"
            )
        return "\n".join(formatted)


def integrate_profile_recommendations(
    df_history: pd.DataFrame,
    df_test: pd.DataFrame,
    ground_truth: Dict[str, List[str]],
    top_k_industries: int = 10,
    top_k_profiles: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Integration function for complete profile recommendation pipeline.
    
    Returns:
        Tuple of (industry_results, profile_results)
    """
    print("="*80)
    print("MULTI-STAGE CLIENT PROFILE RECOMMENDATION PIPELINE")
    print("="*80)
    
    # Stage 1: Industry recommendation using advanced ensemble
    print("\n[STAGE 1] Industry Recommendation (Advanced Ensemble)")
    print("-"*80)
    integrate_advanced_ensemble = get_advanced_ensemble()
    industry_results = integrate_advanced_ensemble(
        df_history, df_test, ground_truth, top_k=top_k_industries
    )
    
    # Initialize profile recommender
    print("\n[STAGE 2-5] Profile Attribute Recommendation")
    print("-"*80)
    profile_recommender = MultiStageProfileRecommender(
        df_history, use_embeddings=True
    )
    
    # Generate profile recommendations for each test user
    profile_results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        if pd.isna(user_id) or user_id in seen_users:
            continue
        seen_users.add(user_id)
        
        print(f"\n{'='*80}")
        print(f"Processing user: {user_id}")
        print(f"{'='*80}")
        
        # Get user's industry recommendations
        user_industries = industry_results[
            industry_results['linkedin_company_outsource'] == user_id
        ].sort_values('score', ascending=False).head(top_k_industries)
        
        if len(user_industries) == 0:
            print(f"  ⚠ No industry recommendations found, skipping...")
            continue
        
        industry_recs = [
            (row['industry'], row['score']) 
            for _, row in user_industries.iterrows()
        ]
        
        # Create outsource profile
        user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
        outsource_profile = {
            'services_provided': ', '.join(user_history['services'].dropna().unique()[:5]),
            'industries_served': ', '.join(user_history['industry'].dropna().unique()[:5]),
            'avg_project_size': user_history['project_size'].mode()[0] if len(user_history) > 0 and 'project_size' in user_history.columns else 'Unknown'
        }
        
        # Generate profiles
        try:
            profiles = profile_recommender.recommend_profile(
                user_id,
                outsource_profile,
                industry_recs,
                top_profiles=top_k_profiles
            )
            
            # Generate LLM explanations
            for i, profile in enumerate(profiles):
                print(f"\n  → Generating LLM explanation for profile {i+1}...")
                explanation = profile_recommender.generate_llm_explanation(
                    outsource_profile, profile
                )
                profile['llm_explanation'] = explanation
                
                # Add to results
                profile_results.append({
                    'linkedin_company_outsource': user_id,
                    'rank': i + 1,
                    **profile
                })
        
        except Exception as e:
            print(f"  ❌ Error generating profile: {e}")
            continue
    
    profile_df = pd.DataFrame(profile_results)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED")
    print("="*80)
    print(f"✓ Industry recommendations: {len(industry_results)} records")
    print(f"✓ Profile recommendations: {len(profile_df)} profiles")
    
    return industry_results, profile_df
