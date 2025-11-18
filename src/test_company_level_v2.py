"""
Test Advanced Company-Level Recommender

Tests the new architecture that leverages the full ensemble pipeline.
"""

import pandas as pd
from preprocessing_data import full_pipeline_preprocess_data
from solution.company_level_recommender_v2 import AdvancedCompanyLevelRecommender


def test_basic_functionality():
    """Test basic recommendation functionality."""
    print("="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    
    # Ensure required columns
    if 'background' in df_test.columns and 'project_description' not in df_test.columns:
        df_test['project_description'] = df_test['background']
    
    print(f"History: {len(df_hist)} rows")
    print(f"Test: {len(df_test)} rows")
    
    # Initialize recommender
    print("\nInitializing Advanced Company-Level Recommender...")
    recommender = AdvancedCompanyLevelRecommender(
        df_history=df_hist,
        df_candidates=df_test,
        industry_top_k=5,
        company_fanout=10,
        use_ensemble=True
    )
    
    # Test recommendation
    test_user = df_test['linkedin_company_outsource'].iloc[0]
    print(f"\nTesting with user: {test_user}")
    
    recommendations = recommender.recommend_companies(
        test_user,
        top_k=10,
        enable_diversity=True
    )
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print(recommendations.to_string())
    
    # Validate output
    assert not recommendations.empty, "Should return recommendations"
    assert 'company_name' in recommendations.columns, "Should have company_name"
    assert 'industry' in recommendations.columns, "Should have industry"
    assert 'score' in recommendations.columns, "Should have score"
    assert len(recommendations) <= 10, "Should not exceed top_k"
    
    print("\n✓ Test PASSED!")
    return recommendations


def test_multiple_users():
    """Test recommendations for multiple users."""
    print("\n" + "="*80)
    print("TEST 2: Multiple Users")
    print("="*80)
    
    # Load data
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    
    if 'background' in df_test.columns:
        df_test['project_description'] = df_test['background']
    
    # Initialize recommender
    recommender = AdvancedCompanyLevelRecommender(
        df_history=df_hist,
        df_candidates=df_test,
        industry_top_k=5,
        company_fanout=8
    )
    
    # Test with 3 different users
    test_users = df_test['linkedin_company_outsource'].unique()[:3]
    
    all_results = []
    for user in test_users:
        print(f"\n{'='*80}")
        print(f"Testing user: {user}")
        print('='*80)
        
        recs = recommender.recommend_companies(user, top_k=5)
        
        if not recs.empty:
            print(f"\nTop 3 recommendations:")
            print(recs[['company_name', 'industry', 'score']].head(3).to_string())
            all_results.append(recs)
        else:
            print("No recommendations found")
    
    print(f"\n✓ Successfully tested {len(all_results)}/{len(test_users)} users")
    return all_results


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*80)
    print("TEST 3: Edge Cases")
    print("="*80)
    
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    
    recommender = AdvancedCompanyLevelRecommender(
        df_history=df_hist,
        df_candidates=df_test,
        industry_top_k=5,
        company_fanout=5
    )
    
    # Test 1: Non-existent user
    print("\nTest 3.1: Non-existent user")
    try:
        recs = recommender.recommend_companies(
            "https://www.linkedin.com/company/nonexistent",
            top_k=10
        )
        print(f"  Result: {len(recs)} recommendations (expected: empty or fallback)")
    except Exception as e:
        print(f"  ✓ Handled gracefully: {e}")
    
    # Test 2: Very small top_k
    print("\nTest 3.2: top_k=1")
    test_user = df_test['linkedin_company_outsource'].iloc[0]
    recs = recommender.recommend_companies(test_user, top_k=1)
    assert len(recs) <= 1, "Should return at most 1 recommendation"
    print(f"  ✓ Returned {len(recs)} recommendation(s)")
    
    # Test 3: Large top_k
    print("\nTest 3.3: top_k=50")
    recs = recommender.recommend_companies(test_user, top_k=50)
    print(f"  ✓ Returned {len(recs)} recommendations")
    
    print("\n✓ All edge cases handled!")


def benchmark_company_level():
    """
    Benchmark company-level recommendations.
    
    Note: This is different from industry-level benchmarking.
    We need to define what "correct" company recommendation means.
    """
    print("\n" + "="*80)
    print("BENCHMARK: Company-Level Recommendations")
    print("="*80)
    
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    
    recommender = AdvancedCompanyLevelRecommender(
        df_history=df_hist,
        df_candidates=df_test,
        industry_top_k=8,
        company_fanout=10
    )
    
    # Get unique users
    test_users = df_test['linkedin_company_outsource'].unique()[:20]  # Test first 20
    
    results = []
    for user in test_users:
        print(f"\nProcessing: {user[:50]}...")
        
        try:
            recs = recommender.recommend_companies(user, top_k=10)
            
            if not recs.empty:
                # Get ground truth (companies the outsource actually worked with)
                ground_truth = set(df_test[
                    df_test['linkedin_company_outsource'] == user
                ]['reviewer_company'].unique())
                
                # Check how many recommended companies are in ground truth
                recommended_companies = set(recs['company_name'].values)
                hits = len(recommended_companies & ground_truth)
                
                results.append({
                    'user': user,
                    'n_recommendations': len(recs),
                    'n_ground_truth': len(ground_truth),
                    'hits': hits,
                    'hit_rate': hits / len(ground_truth) if ground_truth else 0.0,
                    'industries_covered': recs['industry'].nunique()
                })
                
                print(f"  Recommended: {len(recs)}, Ground truth: {len(ground_truth)}, Hits: {hits}")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Summary
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY:")
        print("="*80)
        print(f"Users tested: {len(results)}")
        print(f"Avg recommendations per user: {results_df['n_recommendations'].mean():.1f}")
        print(f"Avg hit rate: {results_df['hit_rate'].mean():.3f}")
        print(f"Avg industries covered: {results_df['industries_covered'].mean():.1f}")
        print(f"\nHit rate distribution:")
        print(results_df['hit_rate'].describe())
        
        # Save results
        results_df.to_csv(
            '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/company_level_benchmark.csv',
            index=False
        )
        print("\n✓ Results saved to data/benchmark/company_level_benchmark.csv")
        
        return results_df
    else:
        print("\n! No results collected")
        return None


if __name__ == "__main__":
    try:
        print("ADVANCED COMPANY-LEVEL RECOMMENDER TEST SUITE")
        print("="*80)
        
        # Run tests
        test_basic_functionality()
        test_multiple_users()
        test_edge_cases()
        benchmark_company_level()
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        import sys
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
