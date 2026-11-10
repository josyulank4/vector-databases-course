"""Watching BM25 Score, Then Watching Meaning Win — Lesson 5 DEMO.

Runs BM25 and brute-force cosine over the same support corpus for the
query "locked out of my account" — a query whose best answer shares
almost no vocabulary with it — and prints both rankings so the semantic
gap is visible, not asserted.

Run:  python -m keywordgap.baseline
"""
from __future__ import annotations

from veccommon import QUERIES, bm25_scores, brute_force, doc_vectors, embed, normalize


def rank_bm25(query: str) -> list[str]:
    scores = bm25_scores(query)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def rank_dense(query: str) -> list[str]:
    return [d for d, _ in brute_force(normalize(embed(query)), doc_vectors(), k=12)]


def main() -> None:
    query, grades = QUERIES[0]
    print(f"query: {query!r}")
    print("relevant:", {d: g for d, g in grades.items() if g >= 2})
    bm = rank_bm25(query)
    dn = rank_dense(query)
    print(f"bm25 top hit: {bm[0]}")
    print(f"dense top hit: {dn[0]}")
    print(f"bm25 rank of reset-2fa: {bm.index('reset-2fa') + 1}")
    print(f"dense rank of reset-2fa: {dn.index('reset-2fa') + 1}")
    print("keywordgap demo complete")


if __name__ == "__main__":
    main()
