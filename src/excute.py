from typing import Dict, List, Tuple
import pandas as pd
from preprocessing_data import full_pipeline_preprocess_data
from solution.content_base_for_item import ContentBaseBasicApproach
from benchmark_data import BenchmarkOutput
from tqdm import tqdm
from solution.collborative_for_item import CollaborativeIndustryRecommender
import numpy as np
def get_recommendations_output(df_test: pd.DataFrame, approach: ContentBaseBasicApproach,  top_k: int) -> pd.DataFrame:
    results = pd.DataFrame()
    set_url_outsource = set()
    for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        try:
            outsource_url_company = row['linkedin_company_outsource']
            if outsource_url_company in set_url_outsource:
                continue
            print('Processing outsource URL:', outsource_url_company)
            set_url_outsource.add(outsource_url_company)
            recommended_items = approach.recommend_items(outsource_url_company, top_k)
            recommended_items['linkedin_company_outsource'] = outsource_url_company
            results = pd.concat([results, recommended_items], ignore_index=True)
        except Exception as e:
            print(f"Error processing row {idx} with URL {row['linkedin_company_outsource']}: {e}")
            continue
    readable_results = results[['linkedin_company_outsource', 'industry', 'score']]
    
    return readable_results

def main():
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"
    
    data_raw = full_pipeline_preprocess_data(data_path)
    data_test = full_pipeline_preprocess_data(data_test_path)
    data_test['project_description'] = data_test['background']
    print('---------- Check points handle data base ----------')
    print(data_raw.columns)
    approach_content_base = ContentBaseBasicApproach(data_raw,data_test)
    print('---------- Check points build feature ----------')
    readable_results = get_recommendations_output(data_test, approach_content_base, top_k=10)
    
    benchmark = BenchmarkOutput(readable_results, data_test)
    print('---------- Evaluation Results ----------')
    summary, per_user = benchmark.evaluate_topk(k=10)
    print(summary)
    summary.to_csv('/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/summary_with_improve_weight.csv', index=False)
    print('---------- Per User Results ----------')
    print(per_user)
    per_user.to_csv('/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/per_user_with_improve_weight.csv', index=False)



def get_recommendations_output_collab(
    df_test: pd.DataFrame,
    approach: CollaborativeIndustryRecommender,
    top_k: int
) -> pd.DataFrame:
    results = pd.DataFrame()
    seen_users = set()

    for _, row in df_test.iterrows():
        user = row.get("linkedin_company_outsource")
        if pd.isna(user) or user in seen_users:
            continue
        seen_users.add(user)

        rec = approach.recommend_items(user, top_k=top_k)
        rec["linkedin_company_outsource"] = user
        results = pd.concat([results, rec[["linkedin_company_outsource", "industry", "score"]]], ignore_index=True)

    return results



def main_1():
     # Adjust these paths to your environment if needed
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"


    print("Loading & preprocessing data ...")
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    # Ensure project_description exists on df_test like your content-based script
    if "background" in df_test.columns and "project_description" not in df_test.columns:
        df_test["project_description"] = df_test["background"]

    # Fit collaborative model
    print("Fitting CollaborativeIndustryRecommender ...")
    collab = CollaborativeIndustryRecommender(
        n_components=128,
        min_user_interactions=1,
        min_item_interactions=1,
        use_tfidf_weighting=True,
        random_state=42,
    ).fit(df_history=df_hist, df_candidates=df_test)

    # Inference
    print("Scoring recommendations ...")
    readable_results = get_recommendations_output_collab(df_test, collab, top_k=10)

    # Evaluate with your existing BenchmarkOutput
    print("Evaluating ...")
    benchmark = BenchmarkOutput(readable_results, df_test)
    summary, per_user = benchmark.evaluate_topk(k=10)

    print("---------- Evaluation Results (Collaborative) ----------")
    print(summary)
    print("---------- Per User Results (Collaborative) ----------")
    print(per_user)

    # Save
    out_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
    summary.to_csv(out_dir + "summary_collaborative.csv", index=False)
    per_user.to_csv(out_dir + "per_user_collaborative.csv", index=False)



# ========= Helpers: normalization & fusion =========

def _zscore_dict(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=np.float32)
    mu = float(vals.mean())
    sd = float(vals.std(ddof=0)) + 1e-8
    return {k: (float(v) - mu) / sd for k, v in d.items()}

def _rank_from_scores(d: Dict[str, float]) -> Dict[str, int]:
    # rank 1 = tốt nhất
    if not d:
        return {}
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    return {k: i+1 for i, (k, _) in enumerate(items)}

def _rrf_fuse(rank_cb: Dict[str, int], rank_cf: Dict[str, int], k: int = 60) -> Dict[str, float]:
    # Reciprocal Rank Fusion: score = 1/(k + rank_cb) + 1/(k + rank_cf)
    keys = set(rank_cb) | set(rank_cf)
    out = {}
    for key in keys:
        r1 = rank_cb.get(key, 10**9)
        r2 = rank_cf.get(key, 10**9)
        out[key] = 1.0/(k + r1) + 1.0/(k + r2)
    return out

def _choose_weights_by_hist(n_hist: int) -> Tuple[float, float]:
    # Gating đơn giản: user ít lịch sử → tin CB; user dày lịch sử → tăng CF
    if n_hist <= 2:
        return 0.85, 0.15  # (w_cb, w_cf)
    if n_hist > 5:
        return 0.75, 0.25
    return 0.60, 0.40


def _fuse_for_user(
    cb_scores: Dict[str, float],
    cf_scores: Dict[str, float],
    n_hist: int,
    top_k: int = 10,
    use_rrf: bool = False,
    base_w_cb: float = 0.7,
    base_w_cf: float = 0.3,
    per_user_zscore: bool = True,
) -> List[Tuple[str, float]]:
    """
    Trả về list [(industry, fused_score)] đã sort desc.
    - cb_scores / cf_scores: dict industry -> score (thô)
    - n_hist: số record lịch sử của user trong df_hist
    """
    
    keys = set(cb_scores) | set(cf_scores)
    if not keys:
        return []

    # 1) Chuẩn hoá per-user (z-score) để cùng thang đo
    if per_user_zscore:
        cbz = _zscore_dict(cb_scores)
        cfz = _zscore_dict(cf_scores)
    else:
        cbz, cfz = cb_scores.copy(), cf_scores.copy()

    # 2) Gating: chọn w_cb, w_cf theo n_hist (override nhẹ base weights)
    g_w_cb, g_w_cf = _choose_weights_by_hist(n_hist)
    # Kết hợp base với gating (bạn có thể bỏ 2 dòng này nếu muốn dùng gating hoàn toàn)
    w_cb = 0.5 * base_w_cb + 0.5 * g_w_cb
    w_cf = 0.5 * base_w_cf + 0.5 * g_w_cf

    fused = {}

    if use_rrf:
        # Rank-fusion (ít nhạy scale, thường tăng recall)
        r_cb = _rank_from_scores(cb_scores)
        r_cf = _rank_from_scores(cf_scores)
        rrf = _rrf_fuse(r_cb, r_cf, k=60)
        # có thể “blend” thêm z-score để giữ thông tin cường độ:
        for k in keys:
            fused[k] = 0.7 * rrf.get(k, 0.0) + 0.3 * (w_cb * cbz.get(k, 0.0) + w_cf * cfz.get(k, 0.0))
    else:
        # Linear score-fusion trên z-score
        for k in keys:
            fused[k] = w_cb * cbz.get(k, 0.0) + w_cf * cfz.get(k, 0.0)

    # 3) Sort & lấy top_k
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return ranked

def _scores_from_models_for_user(
    user: str,
    content_app: ContentBaseBasicApproach,
    collab: CollaborativeIndustryRecommender,
    fanout: int
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """
    Trả về:
      - cb_scores: dict industry->score
      - cf_scores: dict industry->score
      - n_hist: số record lịch sử của user (để gating)
    """
    # CB candidates
    cb_df = content_app.recommend_items(user, top_k=fanout)
    cb_scores = dict(zip(cb_df["industry"].tolist(), cb_df["score"].astype(float).tolist()))

    # CF candidates
    cf_df = collab.recommend_items(user, top_k=fanout)
    cf_scores = dict(zip(cf_df["industry"].tolist(), cf_df["score"].astype(float).tolist()))

    # n_hist: đếm trên data_raw của content_app (đã có đầy đủ lịch sử)
    n_hist = int((content_app.data_raw["linkedin_company_outsource"] == user).sum())

    return cb_scores, cf_scores, n_hist


def get_recommendations_output_fusion(
    df_test: pd.DataFrame,
    content_app: ContentBaseBasicApproach,
    collab: CollaborativeIndustryRecommender,
    top_k: int = 10,
    weight_content: float = 0.7,
    weight_collab: float = 0.3,
    use_rrf: bool = False,
    per_user_zscore: bool = True,
    fanout_mult: int = 3,   # lấy rộng mỗi bên để hợp nhất (top_k * fanout_mult)
) -> pd.DataFrame:
    results = []
    seen_users = set()
    fanout = max(top_k * fanout_mult, top_k)

    for _, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        user = row.get("linkedin_company_outsource")
        if pd.isna(user) or user in seen_users:
            continue
        seen_users.add(user)

        # Lấy điểm từ 2 model & n_hist
        cb_scores, cf_scores, n_hist = _scores_from_models_for_user(user, content_app, collab, fanout=fanout)
        # Trường hợp cold-start một bên
        if not cb_scores and not cf_scores:
            continue
        if not cb_scores:
            fused_rank = sorted(cf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        elif not cf_scores:
            fused_rank = sorted(cb_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        else:
            
            fused_rank = _fuse_for_user(
                cb_scores, cf_scores,
                n_hist=n_hist, top_k=top_k,
                use_rrf=use_rrf,
                base_w_cb=weight_content,
                base_w_cf=weight_collab,
                per_user_zscore=per_user_zscore,
            )
        for industry, fused_score in fused_rank:
            results.append({
                "linkedin_company_outsource": user,
                "industry": industry,
                "score": float(fused_score)
            })

    return pd.DataFrame(results, columns=["linkedin_company_outsource", "industry", "score"])




def main_fusion(weight_content: float = 0.7, weight_collab: float = 0.3, top_k: int = 10):
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"

    print("Loading & preprocessing data ...")
    df_hist = full_pipeline_preprocess_data(data_path)
    df_test = full_pipeline_preprocess_data(data_test_path)
    if "background" in df_test.columns and "project_description" not in df_test.columns:
        df_test["project_description"] = df_test["background"]

    # Build content-based approach
    print("Building ContentBaseBasicApproach ...")
    content_app = ContentBaseBasicApproach(df_hist, df_test)

    # Fit collaborative model
    print("Fitting CollaborativeIndustryRecommender ...")
    collab = CollaborativeIndustryRecommender(
        n_components=128,
        min_user_interactions=1,
        min_item_interactions=1,
        use_tfidf_weighting=True,
        random_state=42,
    ).fit(df_history=df_hist, df_candidates=df_test)

    # Inference: fused scores
    print("Scoring fused recommendations ...")
    readable_results = get_recommendations_output_fusion(
        df_test, content_app, collab, top_k=top_k, weight_content=weight_content, weight_collab=weight_collab, fanout_mult=10,use_rrf=True
    )
    # Evaluate with your existing BenchmarkOutput
    print("Evaluating (Fused) ...")
    benchmark = BenchmarkOutput(readable_results, df_test)
    summary, per_user = benchmark.evaluate_topk(k=top_k)

    print("---------- Evaluation Results (Fusion) ----------")
    print(summary)
    print("---------- Per User Results (Fusion) ----------")
    print(per_user)

    # Save
    out_dir = "/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark/"
    summary.to_csv(out_dir + f"summary_fusion_v2_{int(weight_content*100)}_{int(weight_collab*100)}.csv", index=False)
    per_user.to_csv(out_dir + f"per_user_fusion_v2_{int(weight_content*100)}_{int(weight_collab*100)}.csv", index=False)


if __name__ == "__main__":
    print('BRANCH RUN THIS EXPERIMENT: with fusion of Content-Based and Collaborative Filtering: feat/content-base-openai-build-outsource-profile')
    # Run fusion by default (0.6 content, 0.4 collaborative)
    main_fusion(weight_content=0.6, weight_collab=0.4, top_k=10)
    # main()