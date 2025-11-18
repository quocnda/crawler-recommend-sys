"""
Execute Multi-Stage Client Profile Recommendation Pipeline
============================================================

This script runs the complete profile recommendation system that:
1. Uses advanced ensemble to recommend industries (existing best approach)
2. Recommends client size, location, services for each industry
3. Finds example companies matching the profile
4. Uses LLM to generate explanations

Author: Advanced Recommendation System
Date: November 2025
"""

import pandas as pd
from preprocessing_data import full_pipeline_preprocess_data
import warnings
warnings.filterwarnings('ignore')


def main_profile_recommendation_experiment(
    top_k_industries: int = 10,
    top_k_profiles: int = 3
):
    """
    Run complete profile recommendation experiment.
    
    Args:
        top_k_industries: Number of top industries to recommend (Stage 1)
        top_k_profiles: Number of complete profiles to generate per user
    """
    print("="*80)
    print("MULTI-STAGE CLIENT PROFILE RECOMMENDATION EXPERIMENT")
    print("="*80)
    
    # Data paths
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    # Load and preprocess data
    print("\n[DATA LOADING] Loading & preprocessing data...")
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    print(df_hist.iloc[1])
    print(f"✓ Training data: {len(df_hist)} records")
    print(f"✓ Test data: {len(df_test)} records")
    
    # Create ground truth for evaluation
    print("\n[GROUND TRUTH] Creating ground truth mapping...")
    ground_truth = {}
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        if pd.isna(user_id):
            continue
        if user_id not in ground_truth:
            ground_truth[user_id] = []
        ground_truth[user_id].append(row['industry'])
    
    print(f"✓ Ground truth created for {len(ground_truth)} users")
    
    # Run profile recommendation pipeline
    print("\n" + "="*80)
    print("RUNNING PROFILE RECOMMENDATION PIPELINE")
    print("="*80)
    
    from solution.profile_recommender import integrate_profile_recommendations
    
    industry_results, profile_results = integrate_profile_recommendations(
        df_hist,
        df_test,
        ground_truth,
        top_k_industries=top_k_industries,
        top_k_profiles=top_k_profiles
    )
    
    # Save results
    print("\n[SAVING RESULTS] Saving to disk...")
    out_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
    
    # Save industry recommendations (for comparison with previous approach)
    industry_results.to_csv(
        out_dir + "industry_recommendations_from_profile_pipeline.csv",
        index=False
    )
    print(f"✓ Saved industry results: {len(industry_results)} records")
    
    # Save profile recommendations
    profile_results.to_csv(
        out_dir + "client_profile_recommendations.csv",
        index=False
    )
    print(f"✓ Saved profile results: {len(profile_results)} profiles")
    
    # Display sample results
    print("\n" + "="*80)
    print("SAMPLE PROFILE RECOMMENDATIONS")
    print("="*80)
    
    if len(profile_results) > 0:
        # Show first profile
        first_profile = profile_results.iloc[0]
        print(f"\nUser: {first_profile['linkedin_company_outsource']}")
        print(f"Rank: {first_profile['rank']}")
        print(f"\nRecommended Profile:")
        print(f"  Industry: {first_profile['industry']} (score: {first_profile['industry_score']:.4f})")
        print(f"  Client Size: {first_profile['client_size']} (score: {first_profile['client_size_score']:.4f})")
        print(f"  Location: {first_profile['location']} (score: {first_profile['location_score']:.4f})")
        print(f"  Top Services: {first_profile['top_services']}")
        print(f"  Example Companies: {first_profile['n_examples']} found")
        
        if 'llm_explanation' in first_profile:
            print(f"\nLLM Explanation:")
            print(f"{first_profile['llm_explanation']}")
        
        # Show example companies if available
        if first_profile['n_examples'] > 0:
            print(f"\nExample Companies:")
            examples = eval(first_profile['example_companies'])  # Convert string to list
            for i, ex in enumerate(examples[:3], 1):
                print(f"  {i}. {ex.get('company', 'Unknown')} ({ex.get('location', 'Unknown')})")
                print(f"     Services: {ex.get('services_used', 'Unknown')[:80]}...")
    profile_results = pd.read_csv('/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/client_profile_recommendations.csv')
    industry_results = pd.read_csv('/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/industry_recommendations_from_profile_pipeline.csv')
    # Evaluation summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    # Calculate some basic metrics
    n_users_with_profiles = profile_results['linkedin_company_outsource'].nunique()
    avg_profiles_per_user = len(profile_results) / n_users_with_profiles if n_users_with_profiles > 0 else 0
    avg_examples_per_profile = profile_results['n_examples'].mean() if len(profile_results) > 0 else 0
    
    print(f"Users with profile recommendations: {n_users_with_profiles}")
    print(f"Average profiles per user: {avg_profiles_per_user:.2f}")
    print(f"Average example companies per profile: {avg_examples_per_profile:.2f}")
    
    # Industry-level accuracy (can reuse existing metrics)
    print(f"\nIndustry recommendations: {len(industry_results)} records")
    print(f"(Industry-level metrics can be evaluated using existing BenchmarkOutput)")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return industry_results, profile_results


if __name__ == "__main__":
    print('BRANCH: Multi-Stage Client Profile Recommendation')
    print('Approach: Hierarchical Ensemble (Industry → Size → Location → Services → LLM)')
    print('\n' + '='*80)
    
    try:
        industry_results, profile_results = main_profile_recommendation_experiment(
            top_k_industries=10,
            top_k_profiles=3
        )
        print(industry_results.iloc[1])
        print('------------------>')
        print(profile_results.iloc[1]['example_companies'])
        print('----------------->')
        print("\n✅ SUCCESS: Profile recommendation pipeline completed!")
        print(f"   - Industry recommendations: {len(industry_results)}")
        print(f"   - Profile recommendations: {len(profile_results)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
