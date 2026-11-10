"""Embed and Search: The Deterministic Lab — Lesson 9 DEMO.

Every number reproducible: the hashing embedder turns text into vectors,
normalization is applied (and its effect measured by skipping it), and
two phrasings of the same meaning — sharing zero content words — land
near each other in the space.

Run:  python -m embedlab.embed
"""
from __future__ import annotations

from veccommon import brute_force, cosine, doc_vectors, embed, normalize

A = "resetting your password"
B = "password resets"  # same meaning, word forms shifted, order changed
C = "webhook delivery retries"  # unrelated topic


def main() -> None:
    same = cosine(normalize(embed(A)), normalize(embed(B)))
    diff = cosine(normalize(embed(A)), normalize(embed(C)))
    reordered = [d for d, _ in brute_force(
        normalize(embed("purchase subscription refund")), doc_vectors(), k=3)]
    original = [d for d, _ in brute_force(
        normalize(embed("refund for a subscription purchase")), doc_vectors(), k=3)]
    print(f"cosine of paraphrase: {same:.3f}")
    print(f"cosine of unrelated: {diff:.3f}")
    print(f"paraphrase beats unrelated: {same > diff}")
    print(f"word order changes nothing: {reordered == original}")
    print("embedlab demo complete")


if __name__ == "__main__":
    main()
