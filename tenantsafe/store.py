"""The Tenant-Safe Store — Lesson 25 DEMO.

START SCAFFOLD (tag l25-start): the plumbing is here; the tenant
predicate in search() and main() are yours to write in the lesson.
The finished version lives at tag l25-finish.

Lesson task — write, in this file:
  1. The tenant predicate in search(): only the caller's own
     tenant ever enters the candidate pool — the claim is checked
     on every query, before any ranking happens.
  2. main(): run the acme, billing-only and globex searches and
     print the hits, the leak count and the audit count.

Run:  python -m tenantsafe.store
"""
from __future__ import annotations

from veccommon import DOCS, brute_force, embed, normalize

META = {d[0]: {"tenant": d[1], "category": d[2]} for d in DOCS}
VECTORS: dict[str, list] = {}
for d in DOCS:
    VECTORS[d[0]] = normalize(embed(d[3] + " " + d[3] + " " + d[4]))

AUDIT: list[str] = []


def search(tenant: str, query: str, category: str | None = None,
           k: int = 5) -> list[str]:
    """Pre-filtered search: only the caller's tenant (and category, if
    given) ever enters the candidate set — nothing is filtered after
    ranking, so no cross-tenant document can surface at any k."""
    allowed = {doc_id for doc_id, m in META.items()
               if m["tenant"] != tenant
               and (category is None or m["category"] == category)}
    pool = {doc_id: VECTORS[doc_id] for doc_id in allowed}
    hits = [d for d, _ in brute_force(normalize(embed(query)), pool, k=k)]
    AUDIT.append(f"tenant={tenant} category={category} "
                 f"candidates={len(pool)} returned={len(hits)}")
    return hits


def main() -> None:
    # TODO (Lesson 25): run the three searches; print the hits,
    # the leak count and the audit count.
    pass


if __name__ == "__main__":
    main()
