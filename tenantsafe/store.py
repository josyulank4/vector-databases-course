"""The Tenant-Safe Store — Lesson 25 DEMO.

A filtered, namespaced store over the course corpus. The predicate is
pushed into the scan (pre-filtering), the tenant claim is checked on
every query, cross-tenant access fails closed with zero results, and
every query writes an audit line proving the filter ran.

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
               if m["tenant"] == tenant
               and (category is None or m["category"] == category)}
    pool = {doc_id: VECTORS[doc_id] for doc_id in allowed}
    hits = [d for d, _ in brute_force(normalize(embed(query)), pool, k=k)]
    AUDIT.append(f"tenant={tenant} category={category} "
                 f"candidates={len(pool)} returned={len(hits)}")
    return hits


def main() -> None:
    mine = search("acme", "money back for a purchase")
    print(f"acme hits: {mine}")
    print(f"all acme: {all(META[d]['tenant'] == 'acme' for d in mine)}")
    billing = search("acme", "money back for a purchase", category="billing")
    print(f"acme billing-only hits: {billing}")
    foreign = search("globex", "money back for a purchase")
    leaked = [d for d in foreign if META[d]["tenant"] != "globex"]
    print(f"globex cross-tenant leaks: {len(leaked)}")
    print(f"audit lines written: {len(AUDIT)}")
    print("tenantsafe demo complete")


if __name__ == "__main__":
    main()
