"""The Eval Harness — Lesson 45 DEMO.

Grades the course's dense retrieval against exact-search ground truth
on the golden query set: recall@10 against the full relevant set,
nDCG@10 against the graded judgments, and a 95% confidence interval
on mean recall by the normal approximation — the reporting shape
Lesson 46 turns into decisions.

Run:  python -m proveit.harness
"""
from __future__ import annotations

import math

from veccommon import QUERIES, brute_force, doc_vectors, embed, ndcg_at_k, normalize


def run() -> dict[str, float]:
    vectors = doc_vectors()
    recalls, ndcgs = [], []
    for query, grades in QUERIES:
        ranked = [d for d, _ in brute_force(normalize(embed(query)), vectors, k=12)]
        relevant = {d for d, g in grades.items() if g >= 2}
        hit = len(set(ranked[:10]) & relevant)
        recalls.append(hit / len(relevant))
        ndcgs.append(ndcg_at_k(ranked, grades, 10))
    mean_r = sum(recalls) / len(recalls)
    var = sum((r - mean_r) ** 2 for r in recalls) / max(len(recalls) - 1, 1)
    half = 1.96 * math.sqrt(var / len(recalls)) if len(recalls) > 1 else 0.0
    return {"mean_recall": mean_r, "ci_low": max(0.0, mean_r - half),
            "ci_high": min(1.0, mean_r + half),
            "mean_ndcg": sum(ndcgs) / len(ndcgs)}


def main() -> None:
    r = run()
    print(f"golden queries: {len(QUERIES)}")
    print(f"mean recall@10: {r['mean_recall']:.3f} "
          f"(95% CI {r['ci_low']:.3f}..{r['ci_high']:.3f})")
    print(f"mean ndcg@10: {r['mean_ndcg']:.3f}")
    print("proveit demo complete")


if __name__ == "__main__":
    main()
