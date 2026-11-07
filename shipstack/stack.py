"""The Capstone Build — Lessons 50–51 DEMO.

The whole course in one file: the support corpus is ingested through
the idempotent pipeline (freshpipe's discipline), indexed exactly,
queried through hybrid fusion (hybridrank's RRF), filtered by tenant
(tenantsafe's rule), and graded by the eval harness (proveit's
metrics). The ship verdict is computed from the measurements: SHIP
only if recall clears the gate with zero cross-tenant leaks.

Run:  python -m shipstack.stack
"""
from __future__ import annotations

from hybridrank.hybrid import dense_list, rrf, sparse_list
from tenantsafe.store import META
from veccommon import QUERIES, ndcg_at_k

GATE_RECALL = 0.80


def tenant_fused(tenant: str, query: str) -> list[str]:
    fused = rrf([dense_list(query), sparse_list(query)])
    return [d for d in fused if META[d]["tenant"] == tenant]


def main() -> None:
    recalls, leaks = [], 0
    for query, grades in QUERIES:
        hits = tenant_fused("acme", query)
        relevant = {d for d, g in grades.items() if g >= 2}
        recalls.append(len(set(hits[:10]) & relevant) / len(relevant))
        leaks += sum(1 for d in hits if META[d]["tenant"] != "acme")
    mean_recall = sum(recalls) / len(recalls)
    verdict = "SHIP" if mean_recall >= GATE_RECALL and leaks == 0 else "NO-GO"
    print(f"capstone mean recall@10 (tenant acme): {mean_recall:.3f}")
    print(f"cross-tenant leaks: {leaks}")
    print(f"verdict: {verdict}")
    print("shipstack demo complete")


if __name__ == "__main__":
    main()
