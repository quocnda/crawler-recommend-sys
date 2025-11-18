"""
Test script for Company-Level Recommender System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from preprocessing_data import full_pipeline_preprocess_data
from solution.company_level_recommender import CompanyLevelRecommender


def test_basic_functionality():
    """Test basic functionality of company-level recommender."""
    print("="*80)
    print("TESTING COMPANY-LEVEL RECOMMENDER")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    
    print(f"   History: {len(df_hist)} records")
    print(f"   Test: {len(df_test)} records")
    
    # Initialize recommender
    print("\n2. Initializing CompanyLevelRecommender...")
    recommender = CompanyLevelRecommender(df_hist, df_test)
    
    # Test with one user
    print("\n3. Testing with sample outsource company...")
    test_user = df_test['linkedin_company_outsource'].dropna().iloc[0]
    print(f"   Test user: {test_user}")
    
    recommendations = recommender.recommend_companies(
        outsource_url=test_user,
        top_k=10,
        return_details=True
    )
    
    print("\n4. Results:")
    print(recommendations.to_string(index=False))
    
    # Analyze recommendation diversity
    print("\n5. Recommendation Analysis:")
    print(f"   Total recommendations: {len(recommendations)}")
    print(f"   Unique industries: {recommendations['industry'].nunique()}")
    print(f"   Unique locations: {recommendations['location'].nunique()}")
    print(f"   Unique sizes: {recommendations['client_size'].nunique()}")
    print(f"   Score range: {recommendations['score'].min():.3f} - {recommendations['score'].max():.3f}")
    
    # Industry distribution
    print("\n6. Industry Distribution:")
    print(recommendations['industry'].value_counts().head(5))
    
    return recommendations


def test_multiple_users(num_users: int = 5):
    """Test with multiple users to see diversity."""
    print("\n" + "="*80)
    print(f"TESTING WITH {num_users} DIFFERENT USERS")
    print("="*80)
    
    # Load data
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    
    # Initialize recommender
    recommender = CompanyLevelRecommender(df_hist, df_test)
    
    # Test with multiple users
    test_users = df_test['linkedin_company_outsource'].dropna().unique()[:num_users]
    
    all_results = []
    for i, user in enumerate(test_users, 1):
        print(f"\nUser {i}/{num_users}: {user[:50]}...")
        
        try:
            recs = recommender.recommend_companies(
                outsource_url=user,
                top_k=10,
                return_details=False
            )
            
            if len(recs) > 0:
                print(f"  ✓ Generated {len(recs)} recommendations")
                print(f"  Top 3 companies: {', '.join(recs['company_name'].head(3).tolist())}")
                all_results.append(recs)
            else:
                print(f"  ✗ No recommendations generated")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        print(f"Total recommendations: {len(combined)}")
        print(f"Unique companies: {combined['company_name'].nunique()}")
        print(f"Unique industries: {combined['industry'].nunique()}")
        print(f"Average score: {combined['score'].mean():.3f}")
        print(f"Score std: {combined['score'].std():.3f}")


def compare_with_industry_level():
    """Compare company-level output with industry-level."""
    print("\n" + "="*80)
    print("COMPARISON: Company-Level vs Industry-Level")
    print("="*80)
    
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    
    # Company-level
    print("\n1. Company-Level Recommender:")
    company_recommender = CompanyLevelRecommender(df_hist, df_test)
    test_user = df_test['linkedin_company_outsource'].dropna().iloc[0]
    
    company_recs = company_recommender.recommend_companies(test_user, top_k=10)
    print(f"   Output: {len(company_recs)} specific companies")
    print(f"   Industries covered: {company_recs['industry'].nunique()}")
    
    # Industry-level (existing approach)
    from solution.content_base_for_item import ContentBaseBasicApproach
    print("\n2. Industry-Level Recommender (existing):")
    industry_app = ContentBaseBasicApproach(df_hist, df_test)
    industry_recs = industry_app.recommend_items(test_user, top_k=10)
    print(f"   Output: {len(industry_recs)} industries")
    
    # Compare
    print("\n3. Comparison:")
    print(f"   Company-level specificity: {len(company_recs)} unique companies")
    print(f"   Industry-level specificity: {len(industry_recs)} unique industries (max 67)")
    print(f"   Granularity improvement: {len(company_recs) / max(len(industry_recs), 1):.1f}x more specific")
    
    # Check if company recommendations cover similar industries
    company_industries = set(company_recs['industry'].tolist())
    industry_set = set(industry_recs['industry'].tolist())
    overlap = company_industries & industry_set
    
    print(f"\n4. Industry Coverage Overlap:")
    print(f"   Company recs cover industries: {len(company_industries)}")
    print(f"   Industry recs suggest: {len(industry_set)}")
    print(f"   Overlap: {len(overlap)} industries")
    if industry_set:
        print(f"   Coverage: {len(overlap)/len(industry_set)*100:.1f}% of top industries")


if __name__ == "__main__":
    print("\n" + "🚀 "*20)
    print("COMPANY-LEVEL RECOMMENDATION SYSTEM TEST")
    print("🚀 "*20 + "\n")
    
    # Run tests
    try:
        # Test 1: Basic functionality
        test_basic_functionality()
        
        # Test 2: Multiple users
        test_multiple_users(num_users=3)
        
        # Test 3: Comparison
        compare_with_industry_level()
        
        print("\n" + "✅ "*20)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("✅ "*20 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
