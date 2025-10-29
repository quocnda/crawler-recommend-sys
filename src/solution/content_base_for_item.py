import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from scipy import sparse

class ContentBaseBasicApproach():
    def __init__(self, df: pd.DataFrame,  df_test: pd.DataFrame):
        self.data_raw = df
        self.df_test = df_test
        self.vector_feature = self.build_features_transform_for_item(self.data_raw)
    def mean_value(self, a: float | None, b: float | None) -> float | None:
        a = np.nan if a is None else a
        b = np.nan if b is None else b
        if pd.isna(a) and pd.isna(b):
            return np.nan
        if pd.isna(a):
            return b
        if pd.isna(b):
            return a
        return (a + b) / 2.0

    def build_features_transform_for_item(self, df: pd.DataFrame | None) -> dict[str, TfidfVectorizer | OneHotEncoder | StandardScaler | list[str]]:
        if df is None:
            df = self.data_raw
        df["client_size_mid"] = [
            self.mean_value(a, b) for a, b in zip(df.get("client_min"), df.get("client_max"))
        ]
        df["project_budget_mid"] = [
            self.mean_value(a, b) for a, b in zip(df.get("project_min"), df.get("project_max"))
        ]
        df[["client_size_mid", "project_budget_mid"]] = df[["client_size_mid", "project_budget_mid"]].astype(float).fillna(0.0) 
        
        
        tf_idf_services = TfidfVectorizer()
        _ = tf_idf_services.fit_transform(df['services'].fillna(''))
        
        tf_idf_description = TfidfVectorizer()
        _ = tf_idf_description.fit_transform(df['project_description'].fillna(''))
        
        cat_cols = ['industry', 'location']
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        _ = ohe.fit_transform(df[cat_cols].fillna(''))
        
        scaler = StandardScaler(with_mean=True)
        _ = scaler.fit_transform(df[['client_size_mid', 'project_budget_mid']].fillna(0.0))
        
        return {
            'tf_idf_services': tf_idf_services,
            'tf_idf_description': tf_idf_description,
            'ohe': ohe,
            'scaler': scaler,
            'cat_cols': cat_cols,
            "num_cols": ['client_size_mid', 'project_budget_mid']
        }
        
    def transform_for_item(self, df: pd.DataFrame, vec: dict[str, TfidfVectorizer | OneHotEncoder | StandardScaler]):
        S = vec['tf_idf_services'].transform(df['services'].fillna(''))
        D = vec['tf_idf_description'].transform(df['project_description'].fillna(''))
        C = vec['ohe'].transform(df[vec['cat_cols']].fillna(''))
        df["client_size_mid"] = [
            self.mean_value(a, b) for a, b in zip(df.get("client_min"), df.get("client_max"))
        ]
        df["project_budget_mid"] = [
            self.mean_value(a, b) for a, b in zip(df.get("project_min"), df.get("project_max"))
        ]
        df[["client_size_mid", "project_budget_mid"]] = df[["client_size_mid", "project_budget_mid"]].astype(float).fillna(0.0) 
        num_sp = sparse.csr_matrix(vec["scaler"].transform(df[vec["num_cols"]].fillna(0.0)))
        X = sparse.hstack([S, D, C, num_sp], format="csr")
        return X


#--------Out source Profile Building ---------
    def build_outsource_profile(self, 
                                vector_feature: dict[str, TfidfVectorizer | OneHotEncoder | StandardScaler],
                                outsource_url: str) -> sparse.csr.csr_matrix:
        df = self.data_raw
        mask = df["linkedin_company_outsource"] == outsource_url
        hist = df[mask].copy()
        if hist.empty:
            return None, hist
        X_hist = self.transform_for_item(hist, vector_feature)
        profile = X_hist.mean(axis=0)
        if not isinstance(profile, sparse.csr.csr_matrix):
            profile = sparse.csr_matrix(profile)
        return profile, hist

    def recommend_items(self, df_test: pd.DataFrame,
                        outsource_url: str, 
                        top_k:int = 5 ) ->  pd.DataFrame:
        candidate_score = df_test.copy()        
        profile, hist = self.build_outsource_profile(self.vector_feature, outsource_url)
        if profile is None:
            return pd.DataFrame(columns=["reviewer_company", "score"])
        X_candidate = self.transform_for_item(df_test, self.vector_feature)
        sim = cosine_similarity(X_candidate, profile).ravel()
        candidate_score.assign(score=sim)
        agg = (
            candidate_score.groupby("reviewer_company", dropna=True)
                .agg(
                    score=("score", "max"),
                    industry=("industry", "first"),
                    location=("location", "first"),
                    example_services=("services", "first"),
                    example_project=("project_description", "first"),
                )
                .reset_index()
                .sort_values("score", ascending=False)
                .head(top_k)
        )
        return agg
        