from typing import Dict, List, Tuple
import pandas as pd
from preprocessing_data import full_pipeline_preprocess_data
from benchmark_data import BenchmarkOutput
from solution.advanced_ensemble import integrate_advanced_ensemble
import numpy as np

def main_advanced_ensemble_experiment(top_k: int = 10):
    """Test advanced ensemble methods."""
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"

    print("Loading & preprocessing data ...")
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)

    ground_truth = {}
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        if pd.isna(user_id):
            continue
        if user_id not in ground_truth:
            ground_truth[user_id] = []
        ground_truth[user_id].append(row['industry'])

    # Apply advanced ensemble
    print("Applying advanced ensemble methods ...")
    readable_results = integrate_advanced_ensemble(
        df_hist, df_test, ground_truth, top_k=top_k
    )

    # Evaluate
    print("Evaluating (Advanced Ensemble) ...")
    benchmark = BenchmarkOutput(readable_results, df_test)
    summary, per_user = benchmark.evaluate_topk(k=top_k)

    print("---------- Evaluation Results (Advanced Ensemble) ----------")
    print(summary)

    # Save results
    out_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
    summary.to_csv(out_dir + "summary_advanced_ensemble.csv", index=False)
    per_user.to_csv(out_dir + "per_user_advanced_ensemble.csv", index=False)


if __name__ == "__main__":
    print('BRANCH RUN THIS EXPERIMENT: Enhanced Recall Optimization')
    print('\n' + '='*80)
    
    experiments = [
        ("Advanced Ensemble", main_advanced_ensemble_experiment)
    ]
    
    results_summary = []
    
    for exp_name, exp_function in experiments:
        print(f'RUNNING {exp_name.upper()} EXPERIMENT')
        print('='*80)
        
        try:
            exp_function(top_k=10)
            results_summary.append(f"✅ {exp_name}: Completed successfully")
        except Exception as e:
            print(f"❌ Error in {exp_name}: {e}")
            results_summary.append(f"❌ {exp_name}: Failed - {e}")
        
        print('\n')
    
    print('='*80)
    print('EXPERIMENT SUMMARY')
    print('='*80)
    for result in results_summary:
        print(result)
    