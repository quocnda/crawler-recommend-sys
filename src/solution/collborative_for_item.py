from __future__ import annotations

import math
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

# =====================
# Utilities
# =====================

def _row_l2_normalize(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    if not sparse.isspmatrix_csr(mat):
        mat = sparse.csr_matrix(mat)
    norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A1 + 1e-12
    inv = 1.0 / norms
    return mat.multiply(inv[:, None])


def _cosine_scores(user_vec: np.ndarray, item_mat: np.ndarray) -> np.ndarray:
    # user_vec: (k,), item_mat: (n_items, k) — both L2-normalized row-wise beforehand
    return item_mat @ user_vec
def bm25_weight(X, K1=1.2, B=0.75):
    # X: csr (users x items), values = counts
    X = X.tocsr().astype(np.float32)
    # idf
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    N = X.shape[0]
    idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
    # length norm
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    avgdl = row_sums.mean() + 1e-6
    # apply
    X = X.tocsr().copy()
    for i in range(X.shape[0]):
        start, end = X.indptr[i], X.indptr[i+1]
        dl = row_sums[i]
        denom = X.data[start:end] + K1 * (1 - B + B * dl / avgdl)
        X.data[start:end] = idf[X.indices[start:end]] * (X.data[start:end] * (K1 + 1)) / (denom + 1e-8)
    return X


# =====================
# Collaborative Recommender for Industry
# - User  : linkedin_company_outsource
# - Item  : industry
# - Signal: implicit (counts per (user, item)), TF-IDF style weighting + SVD
# Output columns match your BenchmarkOutput: ['linkedin_company_outsource', 'industry', 'score']
# =====================

class CollaborativeIndustryRecommender:
    def __init__(
        self,
        n_components: int = 128,
        min_user_interactions: int = 1,
        min_item_interactions: int = 1,
        use_tfidf_weighting: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.use_tfidf_weighting = use_tfidf_weighting
        self.random_state = random_state

        # Learned artifacts
        self.user_map: Dict[str, int] = {}
        self.item_map: Dict[str, int] = {}
        self.users_: List[str] = []
        self.items_: List[str] = []

        self.user_factors: Optional[np.ndarray] = None   # (n_users, k)
        self.item_factors: Optional[np.ndarray] = None   # (n_items, k)

        self.item_pop_: Optional[np.ndarray] = None      # popularity for cold-start items
        self.item_df_: Optional[pd.DataFrame] = None     # metadata per item from df_test

        self._svd: Optional[TruncatedSVD] = None

    # ---------- Build interaction matrix ----------
    def _build_interactions(self, df: pd.DataFrame) -> sparse.csr_matrix:
        # Aggregate implicit counts per (user, item)
        grp = (
            df.groupby(["linkedin_company_outsource", "industry"], dropna=True)
              .size()
              .reset_index(name="cnt")
        )

        # Filter sparse tail if desired
        if self.min_user_interactions > 1:
            user_cnt = grp.groupby("linkedin_company_outsource")["cnt"].sum()
            keep_users = set(user_cnt[user_cnt >= self.min_user_interactions].index)
            grp = grp[grp["linkedin_company_outsource"].isin(keep_users)]
        if self.min_item_interactions > 1:
            item_cnt = grp.groupby("industry")["cnt"].sum()
            keep_items = set(item_cnt[item_cnt >= self.min_item_interactions].index)
            grp = grp[grp["industry"].isin(keep_items)]

        # Rebuild lists & maps
        self.users_ = sorted(grp["linkedin_company_outsource"].unique().tolist())
        self.items_ = sorted(grp["industry"].unique().tolist())
        self.user_map = {u: i for i, u in enumerate(self.users_)}
        self.item_map = {it: j for j, it in enumerate(self.items_)}

        # Build CSR
        rows = grp["linkedin_company_outsource"].map(self.user_map).astype(int).to_numpy()
        cols = grp["industry"].map(self.item_map).astype(int).to_numpy()
        vals = grp["cnt"].astype(float).to_numpy()

        # Basic implicit weight (log1p to compress large counts)
        vals = np.log1p(vals)

        mat = sparse.coo_matrix((vals, (rows, cols)), shape=(len(self.users_), len(self.items_))).tocsr()

        if self.use_tfidf_weighting:
            # Column-wise IDF: log((N_users + 1) / (df_item + 1)) + 1
            df_per_item = np.diff(mat.indptr)  # number of non-zeros per row is for rows; we need per column
            # A simpler way: compute df (document frequency) per item with mat.sign().sum(axis=0)
            df_item = np.asarray(mat.sign().sum(axis=0)).ravel()  # how many users interacted with item
            idf = np.log((mat.shape[0] + 1.0) / (df_item + 1.0)) + 1.0
            # Apply IDF per column
            mat = mat @ sparse.diags(idf, offsets=0, shape=(len(self.items_), len(self.items_)))

        return mat

    def fit(self, df_history: pd.DataFrame, df_candidates: pd.DataFrame) -> "CollaborativeIndustryRecommender":
        """
        df_history: full interaction-like table with columns at least:
            - linkedin_company_outsource
            - industry
        df_candidates: used to build per-industry metadata (location, example_services, example_project) for display.
        """
        # 1) Build interactions
        R = self._build_interactions(df_history)
        if R.shape[0] == 0 or R.shape[1] == 0:
            # Degenerate case: nothing to learn
            self.user_factors = np.zeros((0, self.n_components), dtype=np.float32)
            self.item_factors = np.zeros((0, self.n_components), dtype=np.float32)
            self.item_pop_ = np.array([])
            self.item_df_ = pd.DataFrame(columns=["industry", "location", "example_services", "example_project"])
            return self
        R = bm25_weight(R)
        # 2) TruncatedSVD to get latent factors (pure collaborative signal)
        svd = TruncatedSVD(n_components=min(256, R.shape[1]-1), random_state=42)
        svd.fit(R)
        cum = np.cumsum(svd.explained_variance_ratio_)
        k = int(np.searchsorted(cum, 0.9) + 1)
        self._svd = TruncatedSVD(n_components=k, random_state=42)
        U = self._svd.fit_transform(R)           # (n_users, k)
        V = self._svd.components_.T             # (n_items, k)

        # 3) L2-normalize user & item latent vectors for cosine similarity
        U_norm = np.linalg.norm(U, axis=1, keepdims=True) + 1e-12
        V_norm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        self.user_factors = (U / U_norm).astype(np.float32)
        self.item_factors = (V / V_norm).astype(np.float32)

        # 4) Popularity (for cold-start users)
        item_pop = np.asarray(R.sum(axis=0)).ravel()
        self.item_pop_ = item_pop / (item_pop.max() + 1e-12)

        # 5) Prepare candidate item metadata (one row per industry with display fields)
        #    We pull a representative location/services/project_description from df_candidates.
        meta = (
            df_candidates.groupby("industry", dropna=True)
            .agg(
                location=("location", "first"),
                example_services=("services", "first"),
                example_project=("project_description", "first"),
            )
            .reset_index()
        )
        # Keep only items we know about (post-filtering)
        meta = meta[meta["industry"].isin(self.items_)].copy()
        self.item_df_ = meta

        return self

    # ---------- Recommend ----------
    def recommend_items(self, outsource_url: str, top_k: int = 10) -> pd.DataFrame:
        if self.item_factors is None or self.user_factors is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Candidate list from learned items
        items = self.items_
        item_index = self.item_map

        # Cold-start user: not seen in history
        if outsource_url not in self.user_map:
            # Recommend by popularity
            order = np.argsort(-self.item_pop_)
            chosen = [items[i] for i in order[:top_k]]
            scores = self.item_pop_[order[:top_k]]
        else:
            uidx = self.user_map[outsource_url]
            u = self.user_factors[uidx]  # (k,)
            scores_all = self.item_factors @ u  # cosine since both are L2-normalized
            order = np.argsort(-scores_all)[:top_k]
            chosen = [items[i] for i in order]
            scores = scores_all[order]

        # Build output with metadata (best-effort merge)
        out = pd.DataFrame({"industry": chosen, "score": scores.astype(float)})
        if self.item_df_ is not None and not self.item_df_.empty:
            out = out.merge(self.item_df_, on="industry", how="left")

        # Fallback columns for display consistency
        for col in ["location", "example_services", "example_project"]:
            if col not in out.columns:
                out[col] = None

        # Format like your content-based output (without adding the user yet)
        return out.sort_values("score", ascending=False).reset_index(drop=True)


# =====================
# Helper to match your existing evaluation function
# =====================

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


# =====================
# Example main (optional)
# This mirrors your content-based runner: reads CSVs, preprocesses, trains CF, evaluates.
# =====================

def main():
    # Adjust these paths to your environment if needed
    data_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample.csv"
    data_test_path = "/home/ubuntu/crawl/crawler-recommend-sys/data/sample_test.csv"

    # Import your existing helpers
    from preprocessing_data import full_pipeline_preprocess_data
    from benchmark_data import BenchmarkOutput

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
    out_dir = Path("/home/ubuntu/crawl/crawler-recommend-sys/data/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary_collaborative.csv", index=False)
    per_user.to_csv(out_dir / "per_user_collaborative.csv", index=False)
