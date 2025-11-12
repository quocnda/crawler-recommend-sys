#!/usr/bin/env python3
"""
Test Improved Ensemble with Advanced Feature Engineering
=======================================================

This script tests the new improved ensemble approach that integrates
advanced feature engineering with multi-stage pipeline.
"""

import sys
import os

# Add source path
sys.path.append('/home/ubuntu/crawl/crawler-recommend-sys/src')

import pandas as pd
from preprocessing_data import full_pipeline_preprocess_data
from benchmark_data import BenchmarkOutput

def test_improved_ensemble():
    """Test the improved ensemble approach."""
    print("="*60)
    print("TESTING IMPROVED ENSEMBLE WITH ADVANCED FEATURES")
    print("="*60)
    
    try:
        from solution.improved_ensemble import main_improved_ensemble_experiment
        
        # Run the experiment
        results = main_improved_ensemble_experiment(top_k=10)
        
        # Load test data for evaluation
        data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
        df_test = full_pipeline_preprocess_data(data_test_path)
        
        # Evaluate
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        benchmark = BenchmarkOutput(results, df_test)
        summary, per_user = benchmark.evaluate_topk(k=10)
        
        print(summary)
        
        # Compare with existing benchmarks
        print("\n" + "="*60)
        print("COMPARISON WITH EXISTING METHODS")
        print("="*60)
        
        # Try to load existing results for comparison
        benchmark_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
        
        comparison_files = [
            ("Advanced Ensemble", "summary_advanced_ensemble.csv"),
            ("Enhanced Fusion", "summary_enhanced_fusion_balanced_content.csv"),
            ("Content-Based OpenAI", "summary_content_base_openai.csv"),
            ("Collaborative", "summary_collaborative.csv")
        ]
        
        current_recall = summary['Recall@10'].iloc[0]
        print(f"🚀 Improved Ensemble Recall@10: {current_recall:.4f}")
        
        for method_name, filename in comparison_files:
            try:
                filepath = os.path.join(benchmark_dir, filename)
                if os.path.exists(filepath):
                    existing = pd.read_csv(filepath)
                    existing_recall = existing['Recall@10'].iloc[0]
                    improvement = (current_recall - existing_recall) / existing_recall * 100
                    
                    if improvement > 0:
                        print(f"📈 vs {method_name}: {existing_recall:.4f} → +{improvement:.1f}% improvement")
                    else:
                        print(f"📊 vs {method_name}: {existing_recall:.4f} → {improvement:.1f}% change")
                else:
                    print(f"⚠️  {method_name}: Benchmark file not found")
            except Exception as e:
                print(f"❌ Error comparing with {method_name}: {e}")
        
        # Save results
        output_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
        os.makedirs(output_dir, exist_ok=True)
        
        summary.to_csv(os.path.join(output_dir, "summary_improved_ensemble_test.csv"), index=False)
        per_user.to_csv(os.path.join(output_dir, "per_user_improved_ensemble_test.csv"), index=False)
        
        print(f"\n✅ Results saved to {output_dir}")
        
        return summary
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required modules are available.")
        return None
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_feature_engineering():
    """Test feature engineering separately."""
    print("\n" + "="*60)
    print("TESTING FEATURE ENGINEERING COMPONENTS")
    print("="*60)
    
    try:
        from solution.feature_engineering import AdvancedFeatureEngineer
        
        # Load sample data
        data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
        df_history = full_pipeline_preprocess_data(data_path)
        
        print(f"Loaded {len(df_history)} history records")
        
        # Initialize feature engineer
        feature_engineer = AdvancedFeatureEngineer()
        feature_engineer.fit(df_history)
        
        # Test on a sample user
        sample_users = df_history['linkedin_company_outsource'].unique()[:5]
        
        for user_id in sample_users:
            user_history = df_history[df_history['linkedin_company_outsource'] == user_id]
            
            # Extract user features
            user_features = feature_engineer.extract_user_features(user_id, user_history)
            
            print(f"\nUser {user_id[:8]}... ({len(user_history)} interactions):")
            print(f"  Tech preference: {user_features.get('tech_preference', 0):.3f}")
            print(f"  Service complexity: {user_features.get('avg_service_complexity', 0):.3f}")
            print(f"  Industry diversity: {user_features.get('industry_diversity', 0):.3f}")
            print(f"  Domestic preference: {user_features.get('domestic_preference', 0):.3f}")
        
        print("✅ Feature engineering test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Improved Ensemble Test")
    
    # Test feature engineering first
    if test_feature_engineering():
        print("\n🎯 Feature engineering working, proceeding to full ensemble test...")
        
        # Test full ensemble
        summary = test_improved_ensemble()
        
        if summary is not None:
            recall = summary['Recall@10'].iloc[0]
            if recall > 0.67:  # Better than current best
                print(f"\n🏆 SUCCESS! New best recall achieved: {recall:.4f}")
                print("🎉 This approach successfully improves upon the current 0.67 recall!")
            else:
                print(f"\n📊 Result: {recall:.4f} recall (baseline: 0.67)")
                print("🔍 Consider further tuning or different approach combinations.")
        else:
            print("\n❌ Ensemble test failed")
    else:
        print("\n❌ Feature engineering test failed, skipping ensemble test")
    
    print("\n✅ Test completed!")