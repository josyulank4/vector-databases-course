"""IVF and PQ: Divide, Conquer, Compress — Lesson 13 DEMO.

Deterministic k-means (IVF) and product quantization (PQ) over a fixed
synthetic corpus, graded against brute-force ground truth. The recall
cost of each structure is printed from measurement, never asserted.

Run:  python -m indexlab.indexes
"""
from __future__ import annotations

import random

from veccommon import cosine, normalize

_rng = random.Random(42)
N, DIM, CLUSTERS = 2000, 32, 8
CENTERS = [normalize([_rng.gauss(0, 1) for _ in range(DIM)]) for _ in range(CLUSTERS)]
POINTS: list[tuple[int, list[float]]] = []
for cid, center in enumerate(CENTERS):
    for _ in range(N // CLUSTERS):
        v = normalize([c + _rng.gauss(0, 0.35) for c in center])
        POINTS.append((cid, v))
QUERIES = [normalize([c + _rng.gauss(0, 0.3) for c in CENTERS[i % CLUSTERS]])
           for i in range(12)]


def ground_truth(q: list[float], k: int = 10) -> set[int]:
    scored = sorted(range(len(POINTS)), key=lambda i: -cosine(q, POINTS[i][1]))
    return set(scored[:k])


def train_ivf(iters: int = 12) -> list[list[float]]:
    centroids = [POINTS[i * (len(POINTS) // CLUSTERS)][1] for i in range(CLUSTERS)]
    for _ in range(iters):
        buckets: list[list[list[float]]] = [[] for _ in range(CLUSTERS)]
        for _, v in POINTS:
            best = max(range(CLUSTERS), key=lambda c: cosine(v, centroids[c]))
            buckets[best].append(v)
        centroids = [normalize([sum(col) / max(len(b), 1) for col in zip(*b)])
                     if b else centroids[i] for i, b in enumerate(buckets)]
    return centroids


def assign(v: list[float], centroids: list[list[float]]) -> int:
    return max(range(len(centroids)), key=lambda c: cosine(v, centroids[c]))


def ivf_search(q: list[float], centroids: list[list[float]], nprobe: int,
               cells: dict[int, list[int]], k: int = 10) -> set[int]:
    order = sorted(range(len(centroids)), key=lambda c: -cosine(q, centroids[c]))
    cand: list[int] = []
    for c in order[:nprobe]:
        cand.extend(cells[c])
    cand.sort(key=lambda i: -cosine(q, POINTS[i][1]))
    return set(cand[:k])


class PQ:
    """Product quantizer: M subvectors, each snapped to 16 centroids."""

    def __init__(self, m: int = 8, iters: int = 10):
        self.m = m
        self.sub = DIM // m
        self.books: list[list[list[float]]] = []
        for s in range(m):
            pts = [v[s * self.sub:(s + 1) * self.sub] for _, v in POINTS]
            cents = [pts[i * (len(pts) // 16)] for i in range(16)]
            for _ in range(iters):
                buckets: list[list[list[float]]] = [[] for _ in range(16)]
                for p in pts:
                    j = min(range(16), key=lambda c: sum(
                        (a - b) ** 2 for a, b in zip(p, cents[c])))
                    buckets[j].append(p)
                cents = [[sum(col) / max(len(b), 1) for col in zip(*b)]
                         if b else cents[i] for i, b in enumerate(buckets)]
            self.books.append(cents)

    def encode(self, v: list[float]) -> list[int]:
        codes = []
        for s in range(self.m):
            p = v[s * self.sub:(s + 1) * self.sub]
            codes.append(min(range(16), key=lambda c: sum(
                (a - b) ** 2 for a, b in zip(p, self.books[s][c]))))
        return codes

    def decode(self, codes: list[int]) -> list[float]:
        out: list[float] = []
        for s, c in enumerate(codes):
            out.extend(self.books[s][c])
        return out


def main() -> None:
    centroids = train_ivf()
    cells: dict[int, list[int]] = {}
    for i, (_, v) in enumerate(POINTS):
        cells.setdefault(assign(v, centroids), []).append(i)
    pq = PQ()
    codes = [pq.encode(v) for _, v in POINTS]
    approx = [pq.decode(c) for c in codes]
    rec1 = rec4 = recpq = 0.0
    for q in QUERIES:
        gt = ground_truth(q)
        rec1 += len(ivf_search(q, centroids, 1, cells) & gt) / 10
        rec4 += len(ivf_search(q, centroids, 4, cells) & gt) / 10
        top = set(sorted(range(len(POINTS)),
                         key=lambda i: -cosine(q, approx[i]))[:10])
        recpq += len(top & gt) / 10
    n = len(QUERIES)
    print(f"corpus: {N} vectors x {DIM} dims, {CLUSTERS} ivf cells")
    print(f"ivf recall@10 (nprobe=1): {rec1 / n:.3f}")
    print(f"ivf recall@10 (nprobe=4): {rec4 / n:.3f}")
    print(f"pq recall@10 (32 floats -> 8 bytes): {recpq / n:.3f}")
    print("indexlab demo complete")


if __name__ == "__main__":
    main()
