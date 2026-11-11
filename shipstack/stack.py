"""The Capstone Build — Lesson 50 DEMO.

START SCAFFOLD (tag l50-start): the whole course in one file, and
the plumbing runs — the golden queries are graded through hybrid
fusion (hybridrank's RRF), the recall gate is stated, and the
verdict prints. But read the run the way this course taught you
to read any verdict. Two defects stand between this stack and an
earned SHIP. The fused list is never filtered by tenant, so
another tenant's documents sit inside the results the stack
would serve. And the verdict line never reads the leak count it
prints: the gate is a recall gate only, so the stack can leak
and still ship. The finished version lives at tag l50-finish.

Lesson task — write, in this file:
  1. the tenant predicate inside tenant_fused: Lesson 25's rule,
     applied to the fused list before anything is served.
  2. the honest verdict: SHIP only when recall clears the gate
     AND the cross-tenant leak count is zero.

Run:  python -m shipstack.stack
"""
from __future__ import annotations

from hybridrank.hybrid import dense_list, rrf, sparse_list
from tenantsafe.store import META
from veccommon import QUERIES, ndcg_at_k

GATE_RECALL = 0.80


def tenant_fused(tenant: str, query: str) -> list[str]:
    fused = rrf([dense_list(query), sparse_list(query)])
    # TODO (lesson task 1): apply the tenant predicate here — the
    # fused list still carries every tenant's documents.
    return fused


def main() -> None:
    recalls, leaks = [], 0
    for query, grades in QUERIES:
        hits = tenant_fused("acme", query)
        relevant = {d for d, g in grades.items() if g >= 2}
        recalls.append(len(set(hits[:10]) & relevant) / len(relevant))
        leaks += sum(1 for d in hits if META[d]["tenant"] != "acme")
    mean_recall = sum(recalls) / len(recalls)
    # TODO (lesson task 2): the verdict must also read the leak
    # count — a gate that never looks at leaks is not a gate.
    verdict = "SHIP" if mean_recall >= GATE_RECALL else "NO-GO"
    print(f"capstone mean recall@10 (tenant acme): {mean_recall:.3f}")
    print(f"cross-tenant leaks: {leaks}")
    print(f"verdict: {verdict}")
    print("shipstack demo complete")


if __name__ == "__main__":
    main()
