"""
Execute Company-Level Recommendation Pipeline

Full pipeline execution for company-level recommendations,
leveraging the advanced ensemble architecture.
"""

import pandas as pd
from tqdm import tqdm
from preprocessing_data import full_pipeline_preprocess_data
from solution.company_level_recommender_v2 import AdvancedCompanyLevelRecommender
from benchmark_data import BenchmarkOutput


def get_company_recommendations(
    df_test: pd.DataFrame,
    recommender: AdvancedCompanyLevelRecommender,
    top_k: int = 10,
    enable_diversity: bool = True
) -> pd.DataFrame:
    """
    Generate company-level recommendations for all test users.
    
    Args:
        df_test: Test DataFrame
        recommender: Trained recommender instance
        top_k: Number of recommendations per user
        enable_diversity: Enable diversity in recommendations
        
    Returns:
        DataFrame with columns: [linkedin_company_outsource, company_name, industry, score]
    """
    results = []
    seen_users = set()
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Generating recommendations"):
        user = row.get('linkedin_company_outsource')
        
        if pd.isna(user) or user in seen_users:
            continue
        seen_users.add(user)
        
        try:
            # Get recommendations
            recs = recommender.recommend_companies(
                outsource_url=user,
                top_k=top_k,
                enable_diversity=enable_diversity
            )
            
            if not recs.empty:
                # Add user column
                recs['linkedin_company_outsource'] = user
                results.append(recs[['linkedin_company_outsource', 'company_name', 'industry', 'score']])
                
        except Exception as e:
            print(f"Error processing user {user[:50]}: {e}")
            continue
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame(columns=['linkedin_company_outsource', 'company_name', 'industry', 'score'])


def evaluate_company_level(
    recommendations: pd.DataFrame,
    df_test: pd.DataFrame,
    top_k: int = 10
):
    """
    Evaluate company-level recommendations.
    
    Args:
        recommendations: Recommendations DataFrame
        df_test: Ground truth DataFrame
        top_k: K for evaluation metrics
        
    Returns:
        summary_df, per_user_df
    """
    # For company-level, we evaluate exact company matches
    benchmark = BenchmarkOutput(
        data_output=recommendations.rename(columns={'company_name': 'item'}),
        data_ground_truth=df_test.rename(columns={'reviewer_company': 'item'})
    )
    
    summary, per_user = benchmark.evaluate_topk(k=top_k, item_col='item')
    
    return summary, per_user


def evaluate_industry_level(
    recommendations: pd.DataFrame,
    df_test: pd.DataFrame,
    top_k: int = 10
):
    """
    Evaluate at industry level (for comparison with previous approach).
    
    Args:
        recommendations: Recommendations DataFrame (with industry column)
        df_test: Ground truth DataFrame
        top_k: K for evaluation metrics
        
    Returns:
        summary_df, per_user_df
    """
    # Aggregate by industry for fair comparison
    industry_recs = (
        recommendations
        .groupby(['linkedin_company_outsource', 'industry'])
        .agg({'score': 'max'})  # Take max score per industry
        .reset_index()
    )
    
    benchmark = BenchmarkOutput(
        data_output=industry_recs,
        data_ground_truth=df_test
    )
    
    summary, per_user = benchmark.evaluate_topk(k=top_k, item_col='industry')
    
    return summary, per_user


def main_company_level_pipeline(
    top_k: int = 10,
    industry_top_k: int = 8,
    company_fanout: int = 10,
    enable_diversity: bool = True,
    use_ensemble: bool = True
):
    """
    Main execution pipeline for company-level recommendations.
    
    Args:
        top_k: Number of final recommendations
        industry_top_k: Number of industries to consider
        company_fanout: Number of companies per industry
        enable_diversity: Enable diversity in recommendations
        use_ensemble: Use ensemble methods
    """
    print("="*80)
    print("COMPANY-LEVEL RECOMMENDATION PIPELINE")
    print("="*80)
    
    # Load and preprocess data
    print("\n[1/6] Loading and preprocessing data...")
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)

    df_test['project_description'] = df_test['background']
    
    print(f"  History records: {len(df_hist):,}")
    print(f"  Test records: {len(df_test):,}")
    print(f"  Unique outsource companies (test): {df_test['linkedin_company_outsource'].nunique():,}")
    print(f"  Unique client companies (test): {df_test['reviewer_company'].nunique():,}")
    
    # Initialize recommender
    print(f"\n[2/6] Initializing Advanced Company-Level Recommender...")
    print(f"  Industry top-k: {industry_top_k}")
    print(f"  Company fanout: {company_fanout}")
    print(f"  Use ensemble: {use_ensemble}")
    
    recommender = AdvancedCompanyLevelRecommender(
        df_history=df_hist,
        df_candidates=df_test,
        industry_top_k=industry_top_k,
        company_fanout=company_fanout,
        use_ensemble=use_ensemble
    )
    
    # Generate recommendations
    print(f"\n[3/6] Generating company recommendations (top_k={top_k})...")
    recommendations = get_company_recommendations(
        df_test=df_test,
        recommender=recommender,
        top_k=top_k,
        enable_diversity=enable_diversity
    )
    
    print(f"  Total recommendations generated: {len(recommendations):,}")
    print(f"  Users with recommendations: {recommendations['linkedin_company_outsource'].nunique():,}")
    
    # Save recommendations
    output_path = '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/company_level_recommendations.csv'
    recommendations.to_csv(output_path, index=False)
    print(f"  Saved to: {output_path}")
    
    # Evaluate at company level
    print(f"\n[4/6] Evaluating at COMPANY level...")
    company_summary, company_per_user = evaluate_company_level(
        recommendations=recommendations,
        df_test=df_test,
        top_k=top_k
    )
    
    print("\n" + "="*80)
    print("COMPANY-LEVEL EVALUATION RESULTS:")
    print("="*80)
    print(company_summary.to_string(index=False))
    
    # Save company-level results
    company_summary.to_csv(
        '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/summary_company_level.csv',
        index=False
    )
    company_per_user.to_csv(
        '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/per_user_company_level.csv',
        index=False
    )
    
    # Evaluate at industry level (for comparison)
    print(f"\n[5/6] Evaluating at INDUSTRY level (for comparison)...")
    industry_summary, industry_per_user = evaluate_industry_level(
        recommendations=recommendations,
        df_test=df_test,
        top_k=top_k
    )
    
    print("\n" + "="*80)
    print("INDUSTRY-LEVEL EVALUATION RESULTS (from company recommendations):")
    print("="*80)
    print(industry_summary.to_string(index=False))
    
    # Save industry-level results
    industry_summary.to_csv(
        '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/summary_company_to_industry.csv',
        index=False
    )
    industry_per_user.to_csv(
        '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/per_user_company_to_industry.csv',
        index=False
    )
    
    # Comparison analysis
    print(f"\n[6/6] Generating comparison analysis...")
    
    # Sample recommendations
    print("\n" + "="*80)
    print("SAMPLE RECOMMENDATIONS (First 3 users):")
    print("="*80)
    
    sample_users = recommendations['linkedin_company_outsource'].unique()[:3]
    for user in sample_users:
        user_recs = recommendations[recommendations['linkedin_company_outsource'] == user]
        ground_truth = df_test[df_test['linkedin_company_outsource'] == user]['reviewer_company'].unique()
        
        print(f"\nOutsource: {user[:60]}")
        print(f"Ground truth companies: {len(ground_truth)}")
        print(f"  {', '.join(list(ground_truth)[:3])}{'...' if len(ground_truth) > 3 else ''}")
        print(f"\nTop 5 Recommendations:")
        print(user_recs[['company_name', 'industry', 'score']].head(5).to_string(index=False))
        print("-" * 80)
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nResults saved to:")
    print("  - data/benchmark/company_level_recommendations.csv")
    print("  - data/benchmark/summary_company_level.csv")
    print("  - data/benchmark/per_user_company_level.csv")
    print("  - data/benchmark/summary_company_to_industry.csv")
    print("  - data/benchmark/per_user_company_to_industry.csv")
    
    return {
        'recommendations': recommendations,
        'company_summary': company_summary,
        'company_per_user': company_per_user,
        'industry_summary': industry_summary,
        'industry_per_user': industry_per_user
    }


def compare_with_industry_baseline():
    """
    Compare company-level results with pure industry-level baseline.
    """
    print("\n" + "="*80)
    print("COMPARISON: Company-Level vs Industry-Level Baseline")
    print("="*80)
    
    try:
        # Load company-level results
        company_summary = pd.read_csv(
            '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/summary_company_level.csv'
        )
        company_to_industry = pd.read_csv(
            '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/summary_company_to_industry.csv'
        )
        
        # Load baseline (if exists)
        baseline_files = [
            'summary_advanced_ensemble.csv',
            'summary_improved_ensemble.csv',
            'summary_with_advanced_rerank.csv'
        ]
        
        baseline = None
        for fname in baseline_files:
            try:
                baseline = pd.read_csv(f'/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/{fname}')
                print(f"\nUsing baseline: {fname}")
                break
            except:
                continue
        
        if baseline is not None:
            print("\n" + "="*80)
            print("METRICS COMPARISON:")
            print("="*80)
            
            metrics = ['Precision@10', 'Recall@10', 'F1@10', 'MAP@10', 'nDCG@10', 'HitRate@10']
            
            comparison = pd.DataFrame({
                'Metric': metrics,
                'Company-Level (exact)': [company_summary[m].values[0] if m in company_summary.columns else 0 for m in metrics],
                'Company→Industry': [company_to_industry[m].values[0] if m in company_to_industry.columns else 0 for m in metrics],
                'Industry Baseline': [baseline[m].values[0] if m in baseline.columns else 0 for m in metrics]
            })
            
            print(comparison.to_string(index=False))
            
            # Save comparison
            comparison.to_csv(
                '/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/company_vs_industry_comparison.csv',
                index=False
            )
            print("\n✓ Comparison saved to: data/benchmark/company_vs_industry_comparison.csv")
        else:
            print("\n! No baseline found. Run industry-level experiments first.")
            
    except Exception as e:
        print(f"\n! Error during comparison: {e}")


if __name__ == "__main__":
    import sys
    
    print("="*80)
    print("ADVANCED COMPANY-LEVEL RECOMMENDATION SYSTEM")
    print("="*80)
    
    try:
        # Run main pipeline
        results = main_company_level_pipeline(
            top_k=10,
            industry_top_k=8,
            company_fanout=10,
            enable_diversity=True,
            use_ensemble=True
        )
        
        # Compare with baseline
        compare_with_industry_baseline()
        
        print("\n" + "="*80)
        print("✓ ALL EXPERIMENTS COMPLETED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
