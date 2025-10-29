from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import pandas as pd
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ========= Utilities =========
def _norm_lower_strip(x: Optional[str]) -> str:
    return (x or "").strip().lower()

def _parse_budget_to_number(s: Optional[str]) -> float:
    """
    Chuyển 'Project size' dạng text -> số:
      - "$50,000 to $199,999" => midpoint ~ 125000
      - "Under $10,000"      => 10000
      - None/khác            => 0
    """
    if not s or not isinstance(s, str):
        return 0.0
    ss = s.replace("$", "").replace(",", "").strip().lower()
    try:
        if "to" in ss:
            lo, hi = ss.split("to")
            lo_v = float(lo.strip())
            hi_v = float(hi.strip())
            return max(0.0, (lo_v + hi_v) / 2.0)
        if "under" in ss:
            val = ss.split("under")[1].strip()
            return max(0.0, float(val))
    except Exception:
        return 0.0
    return 0.0

def _safe_get(row: pd.Series, key: str) -> str:
    v = row.get(key, "")
    return "" if pd.isna(v) else str(v)


# ========= Data Contracts =========
REQUIRED_COLUMNS = {
    "reviewer_company",
    "verified_status",
    "Industry",
    "Location",
    "Client size",
    "Review type",
    "Services",
    "Project size",
    "Project length",
    "linkedin Company Outsource",
}

@dataclass
class RecItem:
    company: str
    score: float
    evidence: List[str]


# ========= Recommender =========
class RecommendSystem:
    """
    Recommend reviewer_company (client) cho 1 vendor (qua LinkedIn URL).
    Hợp nhất:
      - Baseline: thống kê lịch sử vendor->client với trọng số
      - Content-based: TF-IDF profile vendor vs client
      - Fusion: score = alpha*baseline + (1-alpha)*content
    """

    def __init__(self, alpha: float = 0.6):
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha

        # Raw and preprocessed
        self.df: Optional[pd.DataFrame] = None

        # Baseline store
        self._weights: Dict[str, Counter] = defaultdict(Counter)  # vendor -> Counter(client->weight)

        # Content-based artifacts
        self._vendors_df: Optional[pd.DataFrame] = None  # columns: linkedin_vendor, vendor_text
        self._clients_df: Optional[pd.DataFrame] = None  # columns: client, client_text
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._V = None  # vendor matrix
        self._C = None  # client matrix
        self._vendor_index: Dict[str, int] = {}

    # ---------- Public API ----------
    @classmethod
    def from_csv(cls, path: str, alpha: float = 0.6) -> "RecommendSystem":
        rs = cls(alpha=alpha)
        rs.fit(pd.read_csv(path))
        return rs

    def fit(self, df: pd.DataFrame) -> None:
        # 1) Validate & copy
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")
        df = df.copy()

        # 2) Normalize columns commonly used
        df["linkedin_vendor"] = df["linkedin Company Outsource"].map(_norm_lower_strip)
        df["client"] = df["reviewer_company"].fillna("").map(lambda x: str(x).strip())
        df["verified_flag"] = df["verified_status"].fillna("").map(lambda x: str(x).strip().lower() == "verified")
        df["budget_num"] = df["Project size"].map(_parse_budget_to_number)

        # 3) Build baseline weights
        weights = defaultdict(Counter)
        for _, r in df.iterrows():
            ven = r["linkedin_vendor"]
            cli = r["client"]
            if not ven or not cli:
                continue

            w = 1.0
            if bool(r["verified_flag"]):
                w += 0.3
            # ngân sách chuẩn hoá: cap 200k => max +1.0
            w += min(r["budget_num"] / 200_000.0, 1.0)

            # nhẹ tay với recency nếu có 'Project length' dạng "July - Nov. 2018"
            # (POC: không parse phức tạp, chỉ cộng thêm nếu có chuỗi non-empty)
            if str(_safe_get(r, "Project length")).strip():
                w += 0.1

            # Review type bonus nhỏ
            if _safe_get(r, "Review type").lower().strip() == "online review":
                w += 0.05

            weights[ven][cli] += w

        self._weights = weights

        # 4) Build content profiles
        df["vendor_text"] = (
            df["Services"].fillna("") + " | " +
            df["Location"].fillna("") + " | " +
            df["Industry"].fillna("")
        )

        df["client_text"] = (
            df["Industry"].fillna("") + " | " +
            df["Location"].fillna("") + " | " +
            df["Client size"].fillna("")
        )

        vendors = (
            df.groupby("linkedin_vendor")["vendor_text"]
            .apply(lambda x: " ".join(sorted(set(map(str, x)))))
            .reset_index()
        )
        clients = (
            df.groupby("client")["client_text"]
            .apply(lambda x: " ".join(sorted(set(map(str, x)))))
            .reset_index()
        )

        # Remove empty keys if any
        vendors = vendors[vendors["linkedin_vendor"] != ""].reset_index(drop=True)
        clients = clients[clients["client"] != ""].reset_index(drop=True)

        # 5) Fit TF-IDF
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        V = vectorizer.fit_transform(vendors["vendor_text"])
        C = vectorizer.transform(clients["client_text"])

        self.df = df
        self._vendors_df = vendors
        self._clients_df = clients
        self._vectorizer = vectorizer
        self._V = V
        self._C = C
        self._vendor_index = {v: i for i, v in enumerate(vendors["linkedin_vendor"])}

    def recommend(self, linkedin_vendor_url: str, k: int = 5) -> List[RecItem]:
        """
        Trả về top-k reviewer_company cho vendor (qua LinkedIn URL).
        Fusion điểm: alpha * baseline + (1-alpha) * content.
        """
        key = _norm_lower_strip(linkedin_vendor_url)
        if not key:
            return []

        base_list = self._baseline_topk(key, k * 3)
        content_list = self._content_topk(key, k * 3)

        # Normalize từng nguồn về 0..1
        b_map = {c: s for c, s in base_list}
        if b_map:
            bmax = max(b_map.values())
            if bmax > 0:
                for c in list(b_map.keys()):
                    b_map[c] = b_map[c] / bmax

        c_map = {c: s for c, s in content_list}
        if c_map:
            cmax = max(c_map.values())
            if cmax > 0:
                for c in list(c_map.keys()):
                    c_map[c] = c_map[c] / cmax

        # Hợp nhất
        all_clients = set(b_map) | set(c_map)
        fused: List[Tuple[str, float]] = []
        for cl in all_clients:
            score = self.alpha * b_map.get(cl, 0.0) + (1.0 - self.alpha) * c_map.get(cl, 0.0)
            fused.append((cl, score))

        fused.sort(key=lambda x: x[1], reverse=True)

        # Giải thích ngắn gọn (evidence)
        out: List[RecItem] = []
        for cl, sc in fused[:k]:
            ev = []
            if cl in b_map and b_map[cl] > 0:
                ev.append("baseline: có lịch sử hợp tác/điểm cao")
            if cl in c_map and c_map[cl] > 0:
                ev.append("content: hồ sơ vendor ↔ client tương đồng")
            out.append(RecItem(company=cl, score=float(sc), evidence=ev or ["no-evidence"]))
        return out

    # ---------- Optional maintenance ----------
    def add_review(self, row: Dict[str, str | float]) -> None:
        """
        Cập nhật nhanh 1 review mới (POC). Gọi fit lại để cập nhật content-based.
        """
        if self.df is None:
            self.fit(pd.DataFrame([row]))
            return
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        # cập nhật baseline tối thiểu
        ven = _norm_lower_strip(str(row.get("linkedin Company Outsource", "")))
        cli = str(row.get("reviewer_company", "")).strip()
        if ven and cli:
            w = 1.0
            if str(row.get("verified_status", "")).strip().lower() == "verified":
                w += 0.3
            w += min(_parse_budget_to_number(str(row.get("Project size", ""))) / 200_000.0, 1.0)
            self._weights[ven][cli] += w
        # muốn cập nhật content-based ngay: gọi self.fit(self.df)

    # ---------- Internals ----------
    def _baseline_topk(self, vendor_key: str, k: int) -> List[Tuple[str, float]]:
        counter = self._weights.get(vendor_key, Counter())
        return counter.most_common(k)

    def _content_topk(self, vendor_key: str, k: int) -> List[Tuple[str, float]]:
        if not self._vendors_df is not None or not self._clients_df is not None:
            return []
        vidx = self._vendor_index.get(vendor_key)
        if vidx is None:
            # Vendor chưa có trong profile => cold-start hoàn toàn
            return []
        sims = cosine_similarity(self._V[vidx], self._C).ravel()
        if not len(sims):
            return []
        # loại các client trống nếu có
        top_idx = sims.argsort()[::-1][:k]
        res = []
        for i in top_idx:
            client = self._clients_df["client"].iloc[i]
            res.append((client, float(sims[i])))
        return res


# ========= Example usage =========
if __name__ == "__main__":
    # 1) Khởi tạo từ CSV
    rs = RecommendSystem.from_csv("/home/quoc/crawl-company/out_2.csv", alpha=0.6)

    # 2) Gọi recommend
    vendor_url = "https://www.linkedin.com/company/instinctoolscompany/"
    recommendations = rs.recommend(vendor_url, k=5)

    # 3) In kết quả
    for r in recommendations:
        print(f"- {r.company:40s} | score={r.score:.3f} | {'; '.join(r.evidence)}")
