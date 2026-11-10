"""Build the Hybrid Pipeline — Lesson 21 DEMO.

START SCAFFOLD (tag l21-start): the plumbing is here; the sparse
ordering in sparse_list() and main() are yours to write in the
lesson. The finished version lives at tag l21-finish.

Lesson task — write, in this file:
  1. sparse_list(query): order doc ids by BM25 score, best first —
     highest score at the head of the list.
  2. main(): grade dense, sparse, fusion and fusion+rerank on the
     golden queries and print every line, verdict included.

Run:  python -m hybridrank.hybrid
"""
from __future__ import annotations

from veccommon import (QUERIES, bm25_scores, brute_force, doc_vectors,
                       embed, ndcg_at_k, normalize, tokenize)


def dense_list(query: str) -> list[str]:
    return [d for d, _ in brute_force(normalize(embed(query)), doc_vectors(), k=12)]


def sparse_list(query: str) -> list[str]:
    s = bm25_scores(query)
    return [d for d, _ in sorted(s.items(), key=lambda kv: (kv[1], kv[0]))]


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
    # TODO (Lesson 21): grade dense, sparse, fusion and
    # fusion+rerank on the golden queries; print every line.
    pass


if __name__ == "__main__":
    main()
