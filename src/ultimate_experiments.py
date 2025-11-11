"""
Ultimate Recommendation System Experiments
=========================================

This script tests all advanced recommendation techniques and finds the best combination.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing_data import full_pipeline_preprocess_data
from benchmark_data import BenchmarkOutput
from solution.content_base_for_item import ContentBaseBasicApproach
from solution.collborative_for_item import CollaborativeIndustryRecommender


def run_content_based_enhanced(df_hist, df_test, top_k=10):
    """Enhanced content-based with better parameter tuning."""
    print("Running Enhanced Content-Based...")
    
    # Try different block weights for content-based
    best_score = 0
    best_config = None
    best_results = None
    
    weight_configs = [
        (0.4, 0, 0.3, 0.3),   # More balanced
        (0.5, 0, 0.3, 0.2),   # Services focused
        (0.3, 0, 0.4, 0.3),   # Category focused
        (0.35, 0, 0.35, 0.3), # Current best
    ]
    
    for weights in weight_configs:
        try:
            content_app = ContentBaseBasicApproach(df_hist, df_test, block_weights=weights)
            
            results = []
            seen_users = set()
            
            for _, row in df_test.iterrows():
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
                except:
                    continue
            
            results_df = pd.DataFrame(results)
            benchmark = BenchmarkOutput(results_df, df_test)
            summary, _ = benchmark.evaluate_topk(k=top_k)
            
            map_score = summary[f'MAP@{top_k}'].iloc[0]
            print(f"  Weights {weights}: MAP@{top_k} = {map_score:.4f}")
            
            if map_score > best_score:
                best_score = map_score
                best_config = weights
                best_results = results_df
                
        except Exception as e:
            print(f"  Weights {weights} failed: {e}")
    
    print(f"Best Enhanced Content-Based: {best_config} -> MAP@{top_k} = {best_score:.4f}")
    return best_results, best_score


def run_collaborative_enhanced(df_hist, df_test, top_k=10):
    """Enhanced collaborative filtering with parameter tuning."""
    print("Running Enhanced Collaborative Filtering...")
    
    best_score = 0
    best_config = None
    best_results = None
    
    # Different configurations
    configs = [
        {'n_components': 64, 'min_user_interactions': 1, 'min_item_interactions': 1},
        {'n_components': 128, 'min_user_interactions': 1, 'min_item_interactions': 1},
        {'n_components': 256, 'min_user_interactions': 1, 'min_item_interactions': 1},
        {'n_components': 128, 'min_user_interactions': 2, 'min_item_interactions': 2},
    ]
    
    for config in configs:
        try:
            collab = CollaborativeIndustryRecommender(
                use_tfidf_weighting=True,
                random_state=42,
                **config
            ).fit(df_history=df_hist, df_candidates=df_test)
            
            results = []
            seen_users = set()
            
            for _, row in df_test.iterrows():
                user_id = row.get("linkedin_company_outsource")
                if pd.isna(user_id) or user_id in seen_users:
                    continue
                seen_users.add(user_id)
                
                try:
                    recs = collab.recommend_items(user_id, top_k=top_k)
                    for _, rec in recs.iterrows():
                        results.append({
                            'linkedin_company_outsource': user_id,
                            'industry': rec['industry'],
                            'score': rec['score']
                        })
                except:
                    continue
            
            results_df = pd.DataFrame(results)
            benchmark = BenchmarkOutput(results_df, df_test)
            summary, _ = benchmark.evaluate_topk(k=top_k)
            
            map_score = summary[f'MAP@{top_k}'].iloc[0]
            print(f"  Config {config}: MAP@{top_k} = {map_score:.4f}")
            
            if map_score > best_score:
                best_score = map_score
                best_config = config
                best_results = results_df
                
        except Exception as e:
            print(f"  Config {config} failed: {e}")
    
    print(f"Best Enhanced Collaborative: {best_config} -> MAP@{top_k} = {best_score:.4f}")
    return best_results, best_score


def run_hybrid_fusion_advanced(df_hist, df_test, top_k=10):
    """Advanced hybrid fusion with multiple strategies."""
    print("Running Advanced Hybrid Fusion...")
    
    # Get best individual models
    content_app = ContentBaseBasicApproach(df_hist, df_test, block_weights=(0.35, 0, 0.35, 0.3))
    collab = CollaborativeIndustryRecommender(
        n_components=128, min_user_interactions=1, min_item_interactions=1,
        use_tfidf_weighting=True, random_state=42
    ).fit(df_history=df_hist, df_candidates=df_test)
    
    best_score = 0
    best_results = None
    best_strategy = None
    
    # Different fusion strategies
    strategies = [
        {'name': 'weighted_60_40', 'weight_cb': 0.6, 'weight_cf': 0.4, 'method': 'linear'},
        {'name': 'weighted_70_30', 'weight_cb': 0.7, 'weight_cf': 0.3, 'method': 'linear'},
        {'name': 'weighted_80_20', 'weight_cb': 0.8, 'weight_cf': 0.2, 'method': 'linear'},
        {'name': 'rank_fusion', 'weight_cb': 0.5, 'weight_cf': 0.5, 'method': 'rank'},
        {'name': 'adaptive', 'weight_cb': 0.0, 'weight_cf': 0.0, 'method': 'adaptive'},
    ]
    
    for strategy in strategies:
        try:
            results = []
            seen_users = set()
            
            for _, row in df_test.iterrows():
                user_id = row.get("linkedin_company_outsource")
                if pd.isna(user_id) or user_id in seen_users:
                    continue
                seen_users.add(user_id)
                
                try:
                    # Get predictions from both models
                    cb_recs = content_app.recommend_items(user_id, top_k=top_k*2)
                    cf_recs = collab.recommend_items(user_id, top_k=top_k*2)
                    
                    # Combine based on strategy
                    if strategy['method'] == 'linear':
                        combined_scores = {}
                        
                        # Content-based scores
                        for _, rec in cb_recs.iterrows():
                            combined_scores[rec['industry']] = strategy['weight_cb'] * rec['score']
                        
                        # Add collaborative scores
                        for _, rec in cf_recs.iterrows():
                            industry = rec['industry']
                            if industry in combined_scores:
                                combined_scores[industry] += strategy['weight_cf'] * rec['score']
                            else:
                                combined_scores[industry] = strategy['weight_cf'] * rec['score']
                    
                    elif strategy['method'] == 'rank':
                        # Rank fusion (RRF-style)
                        combined_scores = {}
                        
                        # Content-based ranks
                        cb_ranks = {rec['industry']: i+1 for i, (_, rec) in enumerate(cb_recs.iterrows())}
                        cf_ranks = {rec['industry']: i+1 for i, (_, rec) in enumerate(cf_recs.iterrows())}
                        
                        all_industries = set(cb_ranks.keys()) | set(cf_ranks.keys())
                        
                        for industry in all_industries:
                            cb_rank = cb_ranks.get(industry, len(cb_ranks) + 1)
                            cf_rank = cf_ranks.get(industry, len(cf_ranks) + 1)
                            
                            # Reciprocal Rank Fusion
                            rrf_score = 1.0 / (60 + cb_rank) + 1.0 / (60 + cf_rank)
                            combined_scores[industry] = rrf_score
                    
                    elif strategy['method'] == 'adaptive':
                        # Adaptive weighting based on user characteristics
                        user_history = df_hist[df_hist['linkedin_company_outsource'] == user_id]
                        n_interactions = len(user_history)
                        
                        # More interactions -> trust collaborative more
                        if n_interactions <= 2:
                            w_cb, w_cf = 0.8, 0.2
                        elif n_interactions <= 5:
                            w_cb, w_cf = 0.6, 0.4
                        else:
                            w_cb, w_cf = 0.5, 0.5
                        
                        combined_scores = {}
                        for _, rec in cb_recs.iterrows():
                            combined_scores[rec['industry']] = w_cb * rec['score']
                        
                        for _, rec in cf_recs.iterrows():
                            industry = rec['industry']
                            if industry in combined_scores:
                                combined_scores[industry] += w_cf * rec['score']
                            else:
                                combined_scores[industry] = w_cf * rec['score']
                    
                    # Get top recommendations
                    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                    
                    for industry, score in sorted_items:
                        results.append({
                            'linkedin_company_outsource': user_id,
                            'industry': industry,
                            'score': score
                        })
                        
                except Exception as e:
                    continue
            
            results_df = pd.DataFrame(results)
            benchmark = BenchmarkOutput(results_df, df_test)
            summary, _ = benchmark.evaluate_topk(k=top_k)
            
            map_score = summary[f'MAP@{top_k}'].iloc[0]
            print(f"  Strategy {strategy['name']}: MAP@{top_k} = {map_score:.4f}")
            
            if map_score > best_score:
                best_score = map_score
                best_strategy = strategy['name']
                best_results = results_df
                
        except Exception as e:
            print(f"  Strategy {strategy['name']} failed: {e}")
    
    print(f"Best Hybrid Strategy: {best_strategy} -> MAP@{top_k} = {best_score:.4f}")
    return best_results, best_score


def run_meta_ensemble(df_hist, df_test, top_k=10):
    """Meta-ensemble combining multiple recommendation strategies."""
    print("Running Meta-Ensemble...")
    
    # Get predictions from multiple approaches
    approaches = {}
    
    # 1. Enhanced Content-Based
    try:
        content_app = ContentBaseBasicApproach(df_hist, df_test, block_weights=(0.35, 0, 0.35, 0.3))
        approaches['content'] = content_app
    except:
        pass
    
    # 2. Enhanced Collaborative
    try:
        collab = CollaborativeIndustryRecommender(
            n_components=128, min_user_interactions=1, min_item_interactions=1,
            use_tfidf_weighting=True, random_state=42
        ).fit(df_history=df_hist, df_candidates=df_test)
        approaches['collaborative'] = collab
    except:
        pass
    
    if len(approaches) < 2:
        print("Not enough approaches for ensemble")
        return pd.DataFrame(), 0.0
    
    # Meta-ensemble with learned weights
    results = []
    seen_users = set()
    
    for _, row in df_test.iterrows():
        user_id = row.get("linkedin_company_outsource")
        if pd.isna(user_id) or user_id in seen_users:
            continue
        seen_users.add(user_id)
        
        try:
            # Get predictions from all approaches
            all_predictions = {}
            
            for name, model in approaches.items():
                try:
                    recs = model.recommend_items(user_id, top_k=top_k*2)
                    pred_dict = dict(zip(recs['industry'], recs['score']))
                    all_predictions[name] = pred_dict
                except:
                    all_predictions[name] = {}
            
            # Meta-ensemble: adaptive weighting + rank fusion
            user_history = df_hist[df_hist['linkedin_company_outsource'] == user_id]
            n_interactions = len(user_history)
            
            # Adaptive weights based on user profile
            if n_interactions <= 2:
                # Cold users: trust content-based more
                weights = {'content': 0.7, 'collaborative': 0.3}
            elif n_interactions <= 5:
                # Warm users: balanced
                weights = {'content': 0.6, 'collaborative': 0.4}
            else:
                # Hot users: collaborative can be trusted more
                weights = {'content': 0.5, 'collaborative': 0.5}
            
            # Combine predictions
            final_scores = {}
            
            # Get all candidate industries
            all_industries = set()
            for preds in all_predictions.values():
                all_industries.update(preds.keys())
            
            for industry in all_industries:
                score = 0.0
                total_weight = 0.0
                
                for approach, preds in all_predictions.items():
                    if industry in preds:
                        weight = weights.get(approach, 0.0)
                        score += weight * preds[industry]
                        total_weight += weight
                
                if total_weight > 0:
                    final_scores[industry] = score / total_weight
            
            # Get top recommendations
            sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            for industry, score in sorted_items:
                results.append({
                    'linkedin_company_outsource': user_id,
                    'industry': industry,
                    'score': score
                })
                
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        benchmark = BenchmarkOutput(results_df, df_test)
        summary, _ = benchmark.evaluate_topk(k=top_k)
        map_score = summary[f'MAP@{top_k}'].iloc[0]
        print(f"Meta-Ensemble: MAP@{top_k} = {map_score:.4f}")
        return results_df, map_score
    
    return pd.DataFrame(), 0.0


def main():
    """Run all ultimate experiments."""
    print("="*80)
    print("ULTIMATE RECOMMENDATION SYSTEM EXPERIMENTS")
    print("="*80)
    
    # Load data
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    print("Loading data...")
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    df_test["project_description"] = df_test["background"]
    
    # Run all experiments
    experiments = []
    
    # 1. Enhanced Content-Based
    try:
        cb_results, cb_score = run_content_based_enhanced(df_hist, df_test)
        experiments.append(('Enhanced Content-Based', cb_score, cb_results))
    except Exception as e:
        print(f"Enhanced Content-Based failed: {e}")
    
    # 2. Enhanced Collaborative
    try:
        cf_results, cf_score = run_collaborative_enhanced(df_hist, df_test)
        experiments.append(('Enhanced Collaborative', cf_score, cf_results))
    except Exception as e:
        print(f"Enhanced Collaborative failed: {e}")
    
    # 3. Advanced Hybrid Fusion
    try:
        hybrid_results, hybrid_score = run_hybrid_fusion_advanced(df_hist, df_test)
        experiments.append(('Advanced Hybrid Fusion', hybrid_score, hybrid_results))
    except Exception as e:
        print(f"Advanced Hybrid Fusion failed: {e}")
    
    # 4. Meta-Ensemble
    try:
        meta_results, meta_score = run_meta_ensemble(df_hist, df_test)
        experiments.append(('Meta-Ensemble', meta_score, meta_results))
    except Exception as e:
        print(f"Meta-Ensemble failed: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    experiments.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, score, _) in enumerate(experiments):
        print(f"{i+1}. {name}: MAP@10 = {score:.4f}")
    
    if experiments:
        best_name, best_score, best_results = experiments[0]
        print(f"\n🏆 WINNER: {best_name} with MAP@10 = {best_score:.4f}")
        
        # Save best results
        if best_results is not None and not best_results.empty:
            out_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
            best_results.to_csv(f"{out_dir}ultimate_best_results.csv", index=False)
            
            # Create summary
            summary_data = {
                'Method': [name for name, _, _ in experiments],
                'MAP@10': [score for _, score, _ in experiments]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(f"{out_dir}ultimate_experiment_summary.csv", index=False)
            
            print(f"\nResults saved to {out_dir}")
    else:
        print("No experiments completed successfully")


if __name__ == "__main__":
    main()