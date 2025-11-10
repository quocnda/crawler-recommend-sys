from __future__ import annotations
import os
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import numpy as np
import pandas as pd
from typing import List

class OpenAIEmbedder:
    """
    - fit(texts): no-op (stateless)
    - transform(texts): np.ndarray [n_samples, dim], L2-normalized
    Chú ý:
      * Tự sanitize input để tránh 400: rỗng / quá dài / kiểu dữ liệu lạ.
    """
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 256,
        normalize: bool = True,
        max_chars: int = 8000,          # truncate an toàn theo ký tự
        replace_empty_with: str = "[EMPTY]",
        debug_fallback: bool = False    # bật để thử embed từng item khi batch lỗi
    ):
        self.model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_chars = max_chars
        self.replace_empty_with = replace_empty_with
        self.debug_fallback = debug_fallback
        self._client = OpenAI()

    def fit(self, texts: List[str] | pd.Series) -> "OpenAIEmbedder":
        return self

    def _sanitize_texts(self, texts: List[str]) -> List[str]:
        out = []
        for t in texts:
            try:
                s = "" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t)
            except Exception:
                s = ""
            s = s.strip()
            if not s:
                s = self.replace_empty_with
            if self.max_chars is not None and len(s) > self.max_chars:
                s = s[: self.max_chars]
            out.append(s)
        return out

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=batch)
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        if self.normalize:
            denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / denom
        return arr

    def transform(self, texts: List[str] | pd.Series) -> np.ndarray:
        # Chuyển về list
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        # Sanitize toàn bộ
        texts = self._sanitize_texts(texts)

        out = []
        n = len(texts)
        for i in range(0, n, self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                out.append(self._embed_batch(batch))
            except Exception as e:
                # Nếu batch fail do 1 phần tử bẩn, thử fallback xử lý từng phần tử
                if not self.debug_fallback:
                    raise
                # Fallback từng item để tìm cái lỗi và vẫn tiếp tục
                embs = []
                for j, item in enumerate(batch):
                    try:
                        resp = self._client.embeddings.create(model=self.model, input=[item])
                        arr = np.array([resp.data[0].embedding], dtype=np.float32)
                        if self.normalize:
                            denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                            arr = arr / denom
                        embs.append(arr)
                    except Exception as e_single:
                        # Nếu vẫn lỗi, trả vector 0 để không vỡ shape, đồng thời bạn có thể log item
                        dim = 3072 if self.model.endswith("3-large") else 1536
                        embs.append(np.zeros((1, dim), dtype=np.float32))
                out.append(np.vstack(embs))

        if not out:
            dim = 3072 if self.model.endswith("3-large") else 1536
            return np.zeros((0, dim), dtype=np.float32)
        return np.vstack(out)

# =============== Content-based (Embedding + Cat + Num) ===============
class ContentBaseBasicApproach:
    def __init__(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame,
        embedding_model: str = "text-embedding-3-large",
        block_weights: Tuple[float, float, float, float] = (0.35, 0, 0.35, 0.3)  # (services, desc, cat, num)
    ):
        """
        block_weights: trọng số cho (services_emb, description_emb, onehot_cat, numeric_scaled)
        """
        self.data_raw = df.copy()
        self.df_test = df_test.copy()
        self.embedding_model = embedding_model
        self.block_weights = block_weights
        print('Build feature vectors for candidate items ...')
        self.vector_feature = self.build_features_transform_for_item(self.data_raw)
        print('Transform candidate items to feature matrix ...')
        self.X_candidate = self.transform_for_item(self.df_test, self.vector_feature)

    # ---- Utils ----
    def mean_value(self, a: float | None, b: float | None) -> float | None:
        a = np.nan if a is None else a
        b = np.nan if b is None else b
        if pd.isna(a) and pd.isna(b):
            return np.nan
        if pd.isna(a):
            return float(b)
        if pd.isna(b):
            return float(a)
        return float((a + b) / 2.0)

    def _add_mid_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["client_size_mid"] = [self.mean_value(a, b) for a, b in zip(df.get("client_min"), df.get("client_max"))]
        df["project_budget_mid"] = [self.mean_value(a, b) for a, b in zip(df.get("project_min"), df.get("project_max"))]
        df[["client_size_mid", "project_budget_mid"]] = (
            df[["client_size_mid", "project_budget_mid"]].astype(float).fillna(0.0)
        )
        return df

    # ---- Fit feature transformers (replaces TF-IDF by OpenAI embeddings) ----
    def build_features_transform_for_item(
        self, df: pd.DataFrame | None
    ) -> Dict[str, Any]:
        if df is None:
            df = self.data_raw
        df = self._add_mid_columns(df)
        # Embeddings cho services & project_description
        embedder_services = OpenAIEmbedder(model=self.embedding_model, batch_size=1024, normalize=True).fit(
            df["services"].fillna("")
        )
        # embedder_description = OpenAIEmbedder(model=self.embedding_model, batch_size=1024, normalize=True).fit(
        #     df["project_description"].fillna("")
        # )
        # OneHot cho categorical
        cat_cols = ["industry", "location"]
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        _ = ohe.fit(df[cat_cols].fillna(""))

        # Scaler cho numeric
        scaler = StandardScaler(with_mean=True)
        _ = scaler.fit(df[["client_size_mid"]])

        return {
            "embedder_services": embedder_services,
            # "embedder_description": embedder_description,
            "ohe": ohe,
            "scaler": scaler,
            "cat_cols": cat_cols,
            "num_cols": ["client_size_mid"],
        }

    # ---- Transform DF -> sparse feature matrix ----
    def transform_for_item(self, df: pd.DataFrame, vec: Dict[str, Any]) -> sparse.csr_matrix:
        df = self._add_mid_columns(df)
        # Embedding blocks (dense → sparse)
        services_emb = vec["embedder_services"].transform(df["services"].fillna(""))
        # desc_emb = vec["embedder_description"].transform(df["project_description"].fillna(""))

        S = sparse.csr_matrix(services_emb)
        # D = sparse.csr_matrix(desc_emb)
        # Categorical block (sparse)
        C = vec["ohe"].transform(df[vec["cat_cols"]].fillna(""))

        # Numeric block (dense → sparse)
        num_scaled = vec["scaler"].transform(df[vec["num_cols"]].fillna(0.0))
        N = sparse.csr_matrix(num_scaled)

        # Optional: apply block weights before hstack (tuning chất lượng)
        wS, wD, wC, wN = self.block_weights
        if wS != 1.0: S = S.multiply(wS)
        # if wD != 1.0: D = D.multiply(wD)
        if wC != 1.0: C = C.multiply(wC)
        if wN != 1.0: N = N.multiply(wN)

        X = sparse.hstack([S, C, N], format="csr")
        return X

    # ---- Build outsource (user) profile: mean pooling lịch sử ----
    def build_outsource_profile(
        self,
        vector_feature: Dict[str, Any],
        outsource_url: str
    ) -> Tuple[sparse.csr_matrix | None, pd.DataFrame]:
        df = self.data_raw
        mask = df["linkedin_company_outsource"] == outsource_url
        hist = df[mask].copy()
        if hist.empty:
            return None, hist
        X_hist = self.transform_for_item(hist, vector_feature)
        profile = X_hist.mean(axis=0)  # mean pooling
        if not isinstance(profile, sparse.csr_matrix):
            profile = sparse.csr_matrix(profile)
        return profile, hist

    # ---- Recommend ----
    def recommend_items(
        self,
        outsource_url: str,
        top_k: int = 5
    ) -> pd.DataFrame:
        candidate_score = self.df_test.copy()
        profile, hist = self.build_outsource_profile(self.vector_feature, outsource_url)
        if profile is None:
            return pd.DataFrame(columns=["reviewer_company", "score"])
        sim = cosine_similarity(self.X_candidate, profile).ravel()

        candidate_score = candidate_score.assign(score=sim)
        agg = (
            candidate_score.groupby("industry", dropna=True)
            .agg(
                score=("score", "max"),
                # industry=("industry", "first"),
                location=("location", "first"),
                example_services=("services", "first"),
                example_project=("project_description", "first"),
            )
            .reset_index()
            .sort_values("score", ascending=False)
            .head(top_k)
        )
        return agg