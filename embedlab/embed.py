"""Embed and Search: The Deterministic Lab — Lesson 9 DEMO.

START SCAFFOLD (tag l09-start): the plumbing is here; the experiment
constants and main() are yours to write in the lesson. The finished
version lives at tag l09-finish.

Lesson task — write, in this file:
  1. The three experiment phrases: a password reset said two ways,
     plus one unrelated phrase.
  2. main(): normalize before every cosine; print the paraphrase
     cosine, the unrelated cosine, the verdict, and the word-order
     invariance check.

Run:  python -m embedlab.embed
"""
from __future__ import annotations

from veccommon import brute_force, cosine, doc_vectors, embed, normalize

# TODO (Lesson 9): the phrases and main are written live in the
# session recording, step by step, from this scaffold.
