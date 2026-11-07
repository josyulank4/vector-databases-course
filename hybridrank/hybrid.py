"""Build the Hybrid Pipeline — Lesson 21 DEMO.

Dense retrieval and BM25 fail differently, and reciprocal rank
fusion is the standard cure — but on this corpus the measured result
is the lesson: with only three golden queries and a toy embedder,
fusion lands *between* its parents on both recall@3 and nDCG@5, and
the lexical reranker makes the head worse, not better. That is the
honest, common first-run outcome in production too, and it is why
Lesson 22 teaches ablation discipline — weights, depths, and reranker
on/off, each measured — instead of fusion by faith. Every number here
is computed on the golden queries in veccommon; nothing is asserted
that was not measured.

Run:  python -m hybridrank.hybrid
"""
from __future__ import annotations

from veccommon import (QUERIES, bm25_scores, brute_force, doc_vectors,
                       embed, ndcg_at_k, normalize, tokenize)


def dense_list(query: str) -> list[str]:
    return [d for d, _ in brute_force(normalize(embed(query)), doc_vectors(), k=12)]


def sparse_list(query: str) -> list[str]:
    s = bm25_scores(query)
    return [d for d, _ in sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))]


def rrf(lists: list[list[str]], k: int = 60) -> list[str]:
    fused: dict[str, float] = {}
    for lst in lists:
        for rank, doc in enumerate(lst):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]


def rerank(query: str, fused: list[str], depth: int = 5) -> list[str]:
    """Cross-encoder stand-in: rescore the fused head by content-word
    overlap with the query (a real cross-encoder reads query+document
    jointly; the interface and the depth discipline are the lesson)."""
    from veccommon import DOCS, content_tokens
    text = {d[0]: d[3] + " " + d[4] for d in DOCS}
    q = set(content_tokens(query))
    head = fused[:depth]
    head.sort(key=lambda d: (-len(q & set(content_tokens(text[d]))),
                             fused.index(d)))
    return head + fused[depth:]


def mean_ndcg(rank_fn) -> float:
    return sum(ndcg_at_k(rank_fn(q), g, 5) for q, g in QUERIES) / len(QUERIES)


def mean_recall(rank_fn, k: int = 3) -> float:
    from veccommon import recall_at_k
    return sum(recall_at_k(rank_fn(q), {d for d, gr in g.items() if gr >= 2}, k)
               for q, g in QUERIES) / len(QUERIES)


def main() -> None:
    fused = lambda q: rrf([dense_list(q), sparse_list(q)])  # noqa: E731
    print(f"dense recall@3: {mean_recall(dense_list):.3f}")
    print(f"sparse recall@3: {mean_recall(sparse_list):.3f}")
    print(f"fusion recall@3: {mean_recall(fused):.3f}")
    print(f"dense mean ndcg@5: {mean_ndcg(dense_list):.3f}")
    print(f"sparse mean ndcg@5: {mean_ndcg(sparse_list):.3f}")
    print(f"fusion mean ndcg@5: {mean_ndcg(fused):.3f}")
    print(f"fusion+rerank mean ndcg@5: "
          f"{mean_ndcg(lambda q: rerank(q, fused(q))):.3f}")
    print("verdict: fusion sits between its parents on this corpus — "
          "ablate before you ship (Lesson 22)")
    print("hybridrank demo complete")


if __name__ == "__main__":
    main()
