import re
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Helpers ----------
def parse_country(loc: str) -> str:
    if not isinstance(loc, str) or not loc.strip():
        return "Unknown"
    # lấy quốc gia là phần sau cùng, ví dụ "Athens, Greece" -> "Greece"
    parts = [p.strip() for p in loc.split(",")]
    return parts[-1] if parts else "Unknown"

def parse_client_size(size: str) -> str:
    # Chuẩn hoá bucket (ví dụ "501-1,000 Employees" -> "501-1000")
    if not isinstance(size, str):
        return "Unknown"
    m = re.search(r"(\d[\d,]*)\s*-\s*(\d[\d,]*)", size)
    if m:
        lo = int(m.group(1).replace(",", ""))
        hi = int(m.group(2).replace(",", ""))
        return f"{lo}-{hi}"
    if "1-10" in size: return "1-10"
    if "11-50" in size: return "11-50"
    if "51-200" in size: return "51-200"
    if "201-500" in size: return "201-500"
    if "1001" in size or "1,001" in size: return "1001+"
    return "Unknown"

def money_midpoint(s: str) -> float:
    # "$50,000 to $199,999" -> 125000; "Confidential" -> NaN
    if not isinstance(s, str): return np.nan
    m = re.findall(r"\$([\d,]+)", s)
    if len(m) >= 2:
        a, b = int(m[0].replace(",", "")), int(m[1].replace(",", ""))
        return (a + b) / 2
    return np.nan

def months_from_range(s: str) -> float:
    # "July - Nov. 2018" -> 5 tháng (xấp xỉ); "Aug. - Dec. 2024" -> 5
    if not isinstance(s, str): return np.nan
    months = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]
    ss = s.lower()
    found = [i for i, m in enumerate(months) if m in ss]
    if len(found) >= 2:
        # khoảng tháng (xấp xỉ, +1 để tính đủ số tháng)
        return (found[1] - found[0] + 1)
    return np.nan

def split_services(s: str) -> list[str]:
    # "IT Managed Services IT Staff Augmentation" -> ["IT Managed Services","IT Staff Augmentation"]
    if not isinstance(s, str): return []
    # tách theo "  " hoặc " · " hoặc dấu phẩy, fallback = theo 2+ spaces
    if "·" in s: parts = [p.strip() for p in s.split("·")]
    elif "," in s: parts = [p.strip() for p in s.split(",")]
    else:
        parts = re.split(r"\s{2,}", s.strip())
        if len(parts) == 1:  # fallback: tách theo single spaces nhưng giữ cụm
            parts = re.split(r"\s{1,}(?=[A-Z])", s.strip())  # heuristic
            parts = [p.strip() for p in parts if p.strip()]
    return [p for p in parts if p]

# ---------- Core ----------
def build_features(df: pd.DataFrame):
    # Chuẩn hoá cột cho phía client
    X = df.copy()
    X["Country"] = X["Location"].map(parse_country)
    X["ClientSizeBucket"] = X["Client size"].map(parse_client_size)
    X["ProjectBudgetMid"] = X["Project size"].map(money_midpoint)
    X["ProjectLengthMonths"] = X["Project length"].map(months_from_range)
    X["ServicesList"] = X["Services"].map(split_services).apply(lambda lst: lst if lst else ["(none)"])
    return X

def fit_vectorizer(train_clients: pd.DataFrame):
    # 1) MultiLabelBinarizer cho Services
    mlb = MultiLabelBinarizer(sparse_output=True)
    S = mlb.fit_transform(train_clients["ServicesList"])

    # 2) TF-IDF cho mô tả
    tfidf = TfidfVectorizer(min_df=1, max_features=3000, ngram_range=(1,2))
    D = tfidf.fit_transform(train_clients["Project description"].fillna(""))

    # 3) One-hot cho categorical
    cat_cols = ["Industry", "Country", "ClientSizeBucket", "Review type"]
    ohe = OneHotEncoder(handle_unknown="ignore")
    C = ohe.fit_transform(train_clients[cat_cols])

    # 4) Numeric
    num = train_clients[["ProjectBudgetMid", "ProjectLengthMonths"]].fillna(0.0).to_numpy()

    # Trả về “bộ biến đổi thủ công”
    return {
        "mlb": mlb,
        "tfidf": tfidf,
        "ohe": ohe,
        "cat_cols": cat_cols,
        "num_cols": ["ProjectBudgetMid", "ProjectLengthMonths"]
    }

def transform(df_clients: pd.DataFrame, vec):
    S = vec["mlb"].transform(df_clients["ServicesList"])
    D = vec["tfidf"].transform(df_clients["Project description"].fillna(""))
    C = vec["ohe"].transform(df_clients[vec["cat_cols"]])
    num = df_clients[vec["num_cols"]].fillna(0.0).to_numpy()

    # Ghép sparse + dense (đưa dense về sparse)
    from scipy import sparse
    num_sp = sparse.csr_matrix(num)
    X = sparse.hstack([S, D, C, num_sp], format="csr")
    return X

def make_profile(X_clients):
    # trung bình vector (chuẩn hoá theo L2 trước khi lấy mean để tránh scale lệch)
    from sklearn.preprocessing import normalize
    Xn = normalize(X_clients)  # row-wise
    profile = Xn.mean(axis=0)  # 1 x d (sparse)
    return profile

def domain_boost(row_client, row_candidate) -> float:
    boost = 0.0
    if row_client["Industry"] == row_candidate["Industry"]:
        boost += 0.10
    if row_client["Country"] == row_candidate["Country"]:
        boost += 0.05

    # giao nhau Services
    s1 = set(row_client["ServicesList"])
    s2 = set(row_candidate["ServicesList"])
    if s1 & s2:
        boost += 0.05

    # ngân sách & độ dài tương tự (±50%)
    b1, b2 = row_client["ProjectBudgetMid"], row_candidate["ProjectBudgetMid"]
    if pd.notna(b1) and pd.notna(b2) and b1 > 0 and abs(b1 - b2) / b1 <= 0.5:
        boost += 0.05
    l1, l2 = row_client["ProjectLengthMonths"], row_candidate["ProjectLengthMonths"]
    if pd.notna(l1) and pd.notna(l2) and l1 > 0 and abs(l1 - l2) / l1 <= 0.5:
        boost += 0.05

    return min(boost, 0.30)

def recommend_clients(history_df: pd.DataFrame, candidates_df: pd.DataFrame, outsource_url: str, top_k: int = 10):
    # Lọc lịch sử theo 1 outsource
    df_h = history_df[history_df["linkedin Company Outsource"] == outsource_url].copy()
    if df_h.empty:
        raise ValueError("Outsource chưa có lịch sử (cold start). Hãy nạp trước 3–5 case studies để học hồ sơ.")
    df_train = df_h.iloc[:7,]
    df_test = df_h.iloc[7:,]
    
    # Build features
    H = build_features(df_train)
    C = build_features(candidates_df)
    vec = fit_vectorizer(H)
    Xh = transform(H, vec)
    Xc = transform(C, vec)  

    # Hồ sơ năng lực
    profile = make_profile(Xh)
    profile = np.asarray(profile)

    # Cosine similarity
    sim = cosine_similarity(Xc, profile).ravel()

    # Domain boost (dùng trung bình hàng đầu trong lịch sử để tham chiếu — hoặc bạn có thể boost so với từng case rồi lấy max)
    # Ở đây đơn giản: so với "median row" của lịch sử
    ref = H.iloc[[0]]  # chọn 1 dòng đại diện; có thể thay bằng “mode”/“majority”
    boosts = []
    for _, cand in C.iterrows():
        boosts.append(domain_boost(ref.iloc[0], cand))
    boosts = np.array(boosts)

    final_score = 0.7 * sim + 0.3 * boosts
    C_out = C.copy()
    C_out["score"] = final_score

    # Sắp xếp & trả top-K
    cols_show = ["reviewer_company", "Industry", "Country", "ClientSizeBucket", "Services", "ProjectBudgetMid", "ProjectLengthMonths", "score"]
    C_out = C_out.assign(reviewer_company=candidates_df["reviewer_company"])  # đảm bảo có tên công ty
    print('df_test :',df_test)
    print('------------->')
    c_score_test = C_out[C_out["reviewer_company"].isin(df_test["reviewer_company"])]
    print('c_score_test :',c_score_test[cols_show])
    return C_out.sort_values("score", ascending=False)[cols_show].head(top_k)


def main():
    data_history = pd.read_csv("/home/quoc/crawl-company/out_2.csv")
    data_candidates = data_history.iloc[7:,].copy()
    outsource_url = "https://www.linkedin.com/company/instinctoolscompany/"
    top_k = 5

    recommendations = recommend_clients(data_history, data_candidates, outsource_url, top_k)
    print('--------------->')
    print(recommendations)

main()