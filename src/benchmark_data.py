from __future__ import annotations

import pandas as pd
from math import log2
from collections import OrderedDict
from typing import Optional, Callable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

USER_COL = "linkedin_company_outsource"
ITEM_COL = "industry"
TRIPLET_COL = "triplet"

class BenchmarkOutput():
    def __init__(
        self, 
        data_output: pd.DataFrame, 
        data_ground_truth: pd.DataFrame,
        similarity_fn: Optional[Callable[[str, str], float]] = None
    ):
        """
        Initialize BenchmarkOutput.
        
        Args:
            data_output: Recommendations DataFrame 
                         Must have columns: [user_col, item_col/triplet_col, score]
            data_ground_truth: Ground truth DataFrame
                              Must have columns: [user_col, item_col/triplet_col]
            similarity_fn: Optional function to calculate similarity between items/triplets
                          Used for partial match scoring. Should return float in [0, 1]
        """
        self.data_output = data_output
        self.data_ground_truth = data_ground_truth
        self.similarity_fn = similarity_fn
    
    def _unique_preserve(self,seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _precision_at_k(self, hits: list, k: int) -> float:
        return sum(hits[:k]) / k

    def _recall_at_k(self, hits: list, num_true: int, k: int) -> float:
        if num_true == 0:
            return 0.0
        return sum(hits[:k]) / num_true

    def _ap_at_k(self, hits, k, num_true):
        if num_true == 0:
            return 0.0
        ap, hit_cnt = 0.0, 0
        for i in range(min(k, len(hits))):
            if hits[i] == 1:
                hit_cnt += 1
                ap += hit_cnt / (i + 1)
        return ap / min(num_true, k)

    def _ndcg_at_k(self, hits, k, num_true):
        dcg = 0.0
        for i in range(min(k, len(hits))):
            if hits[i] == 1:
                dcg += 1.0 / log2(i + 2)  # i=0 -> 1/log2(2)=1
        idcg = sum(1.0 / log2(i + 2) for i in range(min(num_true, k)))
        if idcg == 0:
            return 0.0
        return dcg / idcg

    def evaluate_topk(self,
        k: int = 10,
        user_col: str = USER_COL,
        item_col: str = ITEM_COL,
        use_partial_match: bool = False,
        partial_match_threshold: float = 0.5
    ):
        """
        Evaluate recommendations at top-k.
        
        Args:
            k: Top-k for evaluation
            user_col: Column name for user identifier
            item_col: Column name for item/triplet identifier
            use_partial_match: If True, use similarity_fn for partial matches
            partial_match_threshold: Minimum similarity for partial match
        
        Returns:
            (summary_df, per_user_df)
        """
        # Auto-detect if using triplets
        if TRIPLET_COL in self.data_output.columns:
            item_col = TRIPLET_COL
        
        # Gom list đề xuất cho mỗi user (giữ thứ tự xuất hiện, loại trùng)
        data_output = self.data_output
        data_ground_truth = self.data_ground_truth
        pred_lists = (
            data_output
            .groupby(user_col)[item_col]
            .apply(list)
            .apply(self._unique_preserve)
            .to_dict()
        )

        # Gom set ground-truth cho mỗi user
        gt_sets = (
            data_ground_truth
            .groupby(user_col)[item_col]
            .apply(set)
            .to_dict()
        )
        users = sorted(set(pred_lists.keys()) | set(gt_sets.keys()))
        print()
        rows = []
        for u in users:
            preds = pred_lists.get(u, [])
            gts = gt_sets.get(u, set())

            preds_k = preds[:k]
            
            if use_partial_match and self.similarity_fn is not None:
                # Use partial matching with similarity function
                hits = self._compute_partial_hits(
                    preds_k, gts, partial_match_threshold
                )
            else:
                # Exact matching
                hits = [1 if p in gts else 0 for p in preds_k]

            num_true = len(gts)
            precision = self._precision_at_k(hits, k)
            recall = self._recall_at_k(hits, num_true, k)
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            ap = self._ap_at_k(hits, k, num_true)
            ndcg = self._ndcg_at_k(hits, k, num_true)
            hitrate = 1.0 if sum(hits) > 0 else 0.0

            rows.append({
                user_col: u,
                "num_pred": len(preds_k),
                "num_true": num_true,
                "hits": sum(hits),
                f"Precision@{k}": precision,
                f"Recall@{k}": recall,
                f"F1@{k}": f1,
                f"MAP@{k}": ap,
                f"nDCG@{k}": ndcg,
                f"HitRate@{k}": hitrate,
            })

        per_user = pd.DataFrame(rows)

        # Trung bình macro
        summary = {
            "users_evaluated": len(per_user),
            "match_type": "partial" if use_partial_match else "exact",
            f"Precision@{k}": per_user[f"Precision@{k}"].mean() if len(per_user) else 0.0,
            f"Recall@{k}": per_user[f"Recall@{k}"].mean() if len(per_user) else 0.0,
            f"F1@{k}": per_user[f"F1@{k}"].mean() if len(per_user) else 0.0,
            f"MAP@{k}": per_user[f"MAP@{k}"].mean() if len(per_user) else 0.0,
            f"nDCG@{k}": per_user[f"nDCG@{k}"].mean() if len(per_user) else 0.0,
            f"HitRate@{k}": per_user[f"HitRate@{k}"].mean() if len(per_user) else 0.0,
            "median_recall": per_user[f"Recall@{k}"].median() if len(per_user) else 0.0,
            "p90_recall": per_user[f"Recall@{k}"].quantile(0.9) if len(per_user) else 0.0,
        }
        summary_df = pd.DataFrame([summary])
        return summary_df, per_user
    
    def _compute_partial_hits(
        self,
        predictions: list,
        ground_truth: set,
        threshold: float
    ) -> list:
        """
        Compute hits using partial matching with similarity function.
        
        For each prediction, find the best matching ground truth item.
        If similarity >= threshold, count as a hit with weighted score.
        """
        hits = []
        
        for pred in predictions:
            # Find best match in ground truth
            max_similarity = 0.0
            
            for gt_item in ground_truth:
                similarity = self.similarity_fn(pred, gt_item)
                max_similarity = max(max_similarity, similarity)
            
            # If similarity meets threshold, count as weighted hit
            if max_similarity >= threshold:
                hits.append(max_similarity)  # Weighted by similarity
            else:
                hits.append(0.0)
        
        return hits