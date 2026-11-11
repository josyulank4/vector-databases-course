"""The Eval Harness — Lesson 45 DEMO.

START SCAFFOLD (tag l45-start): the plumbing is here — the golden
queries load, exact search ranks them, and the table prints. But
read the run the way Lesson 43 taught: the harness counts EVERY
graded document (grade >= 1, mentions included) as a needle, so
recall grades against the wrong denominator, and the interval is
a placeholder constant instead of a spread measured from the
per-query recalls. The finished version lives at tag l45-finish.

Lesson task — write, in this file:
  1. the relevance threshold: relevant means grade >= 2 (Lesson
     43's stated course definition), so recall's denominator is
     the needle set, not every graded mention.
  2. the confidence interval: the sample variance of the
     per-query recalls and the normal-approximation half-width,
     1.96 standard errors of the mean, clipped to [0, 1].

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
