"""
Comprehensive Recommendation System Comparison
=============================================

This script runs all recommendation approaches and provides detailed comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing_data import full_pipeline_preprocess_data
from benchmark_data import BenchmarkOutput
from solution.content_base_for_item import ContentBaseBasicApproach
from solution.collborative_for_item import CollaborativeIndustryRecommender
from solution.advanced_reranker import integrate_advanced_reranking
from solution.enhanced_embeddings import EnhancedContentBasedRecommender
from solution.cold_start_solver import ColdStartSolver


class RecommendationSystemComparison:
    """
    Comprehensive comparison of all recommendation approaches.
    """
    
    def __init__(self, data_path: str, test_path: str):
        self.data_path = data_path
        self.test_path = test_path
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        self.df_hist = full_pipeline_preprocess_data(data_path)
        self.df_test = full_pipeline_preprocess_data(test_path)
        self.df_test["project_description"] = self.df_test["background"]
        
        # Results storage
        self.results = {}
        self.summaries = {}
        
    def run_content_based_openai(self, top_k: int = 10):
        """Run content-based with OpenAI embeddings."""
        print("\n" + "="*60)
        print("RUNNING: Content-Based with OpenAI Embeddings")
        print("="*60)
        
        try:
            # Build content-based approach
            content_app = ContentBaseBasicApproach(self.df_hist, self.df_test)
            
            # Get recommendations
            results = []
            seen_users = set()
            
            for _, row in self.df_test.iterrows():
                user_id = row.get("linkedin_company_outsource")
                if pd.isna(user_id) or user_id in seen_users:
                    continue
                seen_users.add(user_id)
                
                try:
                    recs = content_app.recommend_items(user_id, top_k=top_k)
                    for _, rec in recs.iterrows():
                        results.append({
                            'linkedin_company_outsource': user_id,
                            'industry': rec['industry'],
                            'score': rec['score']
                        })
                except Exception as e:
                    print(f"Error processing user {user_id}: {e}")
                    continue
            
            results_df = pd.DataFrame(results)
            
            # Evaluate
            benchmark = BenchmarkOutput(results_df, self.df_test)
            summary, per_user = benchmark.evaluate_topk(k=top_k)
            
            self.results['content_based_openai'] = results_df
            self.summaries['content_based_openai'] = summary
            
            print("Content-Based OpenAI Results:")
            print(summary)
            
        except Exception as e:
            print(f"Error in content-based approach: {e}")
            self.summaries['content_based_openai'] = pd.DataFrame([{
                f'MAP@{top_k}': 0, f'nDCG@{top_k}': 0, f'Precision@{top_k}': 0,
                f'Recall@{top_k}': 0, f'HitRate@{top_k}': 0, 'users_evaluated': 0
            }])
    
    def run_collaborative_filtering(self, top_k: int = 10):
        """Run collaborative filtering approach."""
        print("\n" + "="*60)
        print("RUNNING: Collaborative Filtering")
        print("="*60)
        
        try:
            # Build collaborative model
            collab = CollaborativeIndustryRecommender(
                n_components=128,
                min_user_interactions=1,
                min_item_interactions=1,
                use_tfidf_weighting=True,
                random_state=42,
            ).fit(df_history=self.df_hist, df_candidates=self.df_test)
            
            # Get recommendations
            results = []
            seen_users = set()
            
            for _, row in self.df_test.iterrows():
                user_id = row.get("linkedin_company_outsource")
                if pd.isna(user_id) or user_id in seen_users:
                    continue
                seen_users.add(user_id)
                
                try:
                    recs = collab.recommend_items(user_id, top_k=top_k)
                    recs["linkedin_company_outsource"] = user_id
                    for _, rec in recs.iterrows():
                        results.append({
                            'linkedin_company_outsource': user_id,
                            'industry': rec['industry'],
                            'score': rec['score']
                        })
                except Exception as e:
                    print(f"Error processing user {user_id}: {e}")
                    continue
            
            results_df = pd.DataFrame(results)
            
            # Evaluate
            benchmark = BenchmarkOutput(results_df, self.df_test)
            summary, per_user = benchmark.evaluate_topk(k=top_k)
            
            self.results['collaborative'] = results_df
            self.summaries['collaborative'] = summary
            
            print("Collaborative Filtering Results:")
            print(summary)
            
        except Exception as e:
            print(f"Error in collaborative approach: {e}")
            self.summaries['collaborative'] = pd.DataFrame([{
                f'MAP@{top_k}': 0, f'nDCG@{top_k}': 0, f'Precision@{top_k}': 0,
                f'Recall@{top_k}': 0, f'HitRate@{top_k}': 0, 'users_evaluated': 0
            }])
    
    def run_fusion_baseline(self, top_k: int = 10):
        """Run fusion baseline (simple weighted combination)."""
        print("\n" + "="*60)
        print("RUNNING: Fusion Baseline (60% Content + 40% Collaborative)")
        print("="*60)
        
        try:
            # Build both models
            content_app = ContentBaseBasicApproach(self.df_hist, self.df_test)
            collab = CollaborativeIndustryRecommender(
                n_components=128,
                min_user_interactions=1,
                min_item_interactions=1,
                use_tfidf_weighting=True,
                random_state=42,
            ).fit(df_history=self.df_hist, df_candidates=self.df_test)
            
            # Use existing fusion function
            from excute import get_recommendations_output_fusion
            results_df = get_recommendations_output_fusion(
                self.df_test, content_app, collab, 
                top_k=top_k, weight_content=0.6, weight_collab=0.4
            )
            
            # Evaluate
            benchmark = BenchmarkOutput(results_df, self.df_test)
            summary, per_user = benchmark.evaluate_topk(k=top_k)
            
            self.results['fusion_baseline'] = results_df
            self.summaries['fusion_baseline'] = summary
            
            print("Fusion Baseline Results:")
            print(summary)
            
        except Exception as e:
            print(f"Error in fusion approach: {e}")
            self.summaries['fusion_baseline'] = pd.DataFrame([{
                f'MAP@{top_k}': 0, f'nDCG@{top_k}': 0, f'Precision@{top_k}': 0,
                f'Recall@{top_k}': 0, f'HitRate@{top_k}': 0, 'users_evaluated': 0
            }])
    
    def run_advanced_reranking(self, top_k: int = 10):
        """Run advanced reranking approach."""
        print("\n" + "="*60)
        print("RUNNING: Advanced Reranking")
        print("="*60)
        
        try:
            # Build base models
            content_app = ContentBaseBasicApproach(self.df_hist, self.df_test)
            collab = CollaborativeIndustryRecommender(
                n_components=128,
                min_user_interactions=1,
                min_item_interactions=1,
                use_tfidf_weighting=True,
                random_state=42,
            ).fit(df_history=self.df_hist, df_candidates=self.df_test)
            
            # Apply advanced reranking
            results_df = integrate_advanced_reranking(
                content_app, collab, self.df_test, self.df_hist, top_k=top_k
            )
            
            # Evaluate
            benchmark = BenchmarkOutput(results_df, self.df_test)
            summary, per_user = benchmark.evaluate_topk(k=top_k)
            
            self.results['advanced_reranking'] = results_df
            self.summaries['advanced_reranking'] = summary
            
            print("Advanced Reranking Results:")
            print(summary)
            
        except Exception as e:
            print(f"Error in advanced reranking: {e}")
            self.summaries['advanced_reranking'] = pd.DataFrame([{
                f'MAP@{top_k}': 0, f'nDCG@{top_k}': 0, f'Precision@{top_k}': 0,
                f'Recall@{top_k}': 0, f'HitRate@{top_k}': 0, 'users_evaluated': 0
            }])
    
    def run_enhanced_embeddings(self, top_k: int = 10):
        """Run enhanced embeddings approach."""
        print("\n" + "="*60)
        print("RUNNING: Enhanced Embeddings (Sentence Transformers)")
        print("="*60)
        
        try:
            # Build enhanced recommender
            enhanced_recommender = EnhancedContentBasedRecommender(
                self.df_hist, self.df_test
            )
            
            # Get recommendations
            results = []
            seen_users = set()
            
            for _, row in self.df_test.iterrows():
                user_id = row.get("linkedin_company_outsource")
                if pd.isna(user_id) or user_id in seen_users:
                    continue
                seen_users.add(user_id)
                
                try:
                    recs = enhanced_recommender.recommend_items(user_id, top_k=top_k)
                    for _, rec in recs.iterrows():
                        results.append({
                            'linkedin_company_outsource': user_id,
                            'industry': rec['industry'],
                            'score': rec['score']
                        })
                except Exception as e:
                    print(f"Error processing user {user_id}: {e}")
                    continue
            
            results_df = pd.DataFrame(results)
            
            # Evaluate
            benchmark = BenchmarkOutput(results_df, self.df_test)
            summary, per_user = benchmark.evaluate_topk(k=top_k)
            
            self.results['enhanced_embeddings'] = results_df
            self.summaries['enhanced_embeddings'] = summary
            
            print("Enhanced Embeddings Results:")
            print(summary)
            
        except Exception as e:
            print(f"Error in enhanced embeddings: {e}")
            self.summaries['enhanced_embeddings'] = pd.DataFrame([{
                f'MAP@{top_k}': 0, f'nDCG@{top_k}': 0, f'Precision@{top_k}': 0,
                f'Recall@{top_k}': 0, f'HitRate@{top_k}': 0, 'users_evaluated': 0
            }])
    
    def run_all_approaches(self, top_k: int = 10):
        """Run all approaches and compare."""
        approaches = [
            self.run_content_based_openai,
            self.run_collaborative_filtering,
            self.run_fusion_baseline,
            self.run_advanced_reranking,
            self.run_enhanced_embeddings
        ]
        
        for approach in approaches:
            approach(top_k)
    
    def create_comparison_report(self, top_k: int = 10) -> pd.DataFrame:
        """Create comprehensive comparison report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("="*80)
        
        comparison_data = []
        
        for approach_name, summary in self.summaries.items():
            if not summary.empty:
                row = {
                    'Approach': approach_name.replace('_', ' ').title(),
                    f'MAP@{top_k}': summary[f'MAP@{top_k}'].iloc[0],
                    f'nDCG@{top_k}': summary[f'nDCG@{top_k}'].iloc[0],
                    f'Precision@{top_k}': summary[f'Precision@{top_k}'].iloc[0],
                    f'Recall@{top_k}': summary[f'Recall@{top_k}'].iloc[0],
                    f'HitRate@{top_k}': summary[f'HitRate@{top_k}'].iloc[0],
                    'Users Evaluated': summary['users_evaluated'].iloc[0]
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(f'MAP@{top_k}', ascending=False)
        
        print("\\nFinal Comparison Results:")
        print("-" * 80)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Calculate improvements
        if len(comparison_df) > 1:
            baseline_map = comparison_df[comparison_df['Approach'].str.contains('Fusion Baseline')][f'MAP@{top_k}'].iloc[0] if any(comparison_df['Approach'].str.contains('Fusion Baseline')) else 0
            
            if baseline_map > 0:
                print(f"\\nImprovements over Fusion Baseline (MAP@{top_k} = {baseline_map:.4f}):")
                print("-" * 60)
                for _, row in comparison_df.iterrows():
                    if 'Fusion Baseline' not in row['Approach']:
                        improvement = ((row[f'MAP@{top_k}'] - baseline_map) / baseline_map) * 100
                        print(f"{row['Approach']}: {improvement:+.2f}%")
        
        return comparison_df
    
    def visualize_results(self, top_k: int = 10):
        """Create visualization of results."""
        try:
            comparison_df = pd.DataFrame([
                {
                    'Approach': approach.replace('_', ' ').title(),
                    f'MAP@{top_k}': summary[f'MAP@{top_k}'].iloc[0],
                    f'nDCG@{top_k}': summary[f'nDCG@{top_k}'].iloc[0],
                    f'Precision@{top_k}': summary[f'Precision@{top_k}'].iloc[0],
                    f'Recall@{top_k}': summary[f'Recall@{top_k}'].iloc[0],
                }
                for approach, summary in self.summaries.items()
                if not summary.empty
            ])
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Recommendation System Performance Comparison', fontsize=16)
            
            metrics = [f'MAP@{top_k}', f'nDCG@{top_k}', f'Precision@{top_k}', f'Recall@{top_k}']
            
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                comparison_df.plot(x='Approach', y=metric, kind='bar', ax=ax, 
                                 color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])
                ax.set_title(f'{metric} Comparison')
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            output_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/comprehensive_comparison.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\\nVisualization saved to: {output_path}")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def save_detailed_results(self):
        """Save all detailed results."""
        output_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
        
        # Save individual results
        for approach, results_df in self.results.items():
            results_df.to_csv(f"{output_dir}detailed_{approach}_results.csv", index=False)
        
        # Save summary comparison
        comparison_df = self.create_comparison_report()
        comparison_df.to_csv(f"{output_dir}comprehensive_comparison_summary.csv", index=False)
        
        print(f"\\nDetailed results saved to: {output_dir}")


def main():
    """Main function to run comprehensive comparison."""
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    # Initialize comparison system
    comparison = RecommendationSystemComparison(data_path, test_path)
    
    # Run all approaches
    comparison.run_all_approaches(top_k=10)
    
    # Create comparison report
    comparison.create_comparison_report(top_k=10)
    
    # Create visualizations
    comparison.visualize_results(top_k=10)
    
    # Save results
    comparison.save_detailed_results()
    
    print("\\n" + "="*80)
    print("COMPREHENSIVE COMPARISON COMPLETED!")
    print("="*80)
    print("Check /data/benchmark/ for detailed results and visualizations.")


if __name__ == "__main__":
    main()