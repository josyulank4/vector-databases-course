"""Watching BM25 Score, Then Watching Meaning Win — Lesson 5 DEMO.

START SCAFFOLD (tag l05-start): the plumbing is here; the ranking
functions and main() are yours to write in the lesson. The finished
version lives at tag l05-finish.

Lesson task — write, in this file:
  1. rank_bm25(query): order doc ids by their BM25 score, best first.
  2. rank_dense(query): order doc ids by cosine similarity of the
     normalized query embedding against the document vectors.
  3. main(): run both rankers on the graded locked-out query and
     print both top hits and the rank of reset-2fa in each list.

Run:  python -m keywordgap.baseline
"""
from __future__ import annotations

from veccommon import QUERIES, bm25_scores, brute_force, doc_vectors, embed, normalize

# TODO (Lesson 5): rank_bm25, rank_dense and main are written live in
# the session recording, step by step, from this scaffold.
