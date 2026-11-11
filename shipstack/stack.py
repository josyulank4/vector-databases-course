"""Capstone Review: Ship the Store — Lesson 51 DEMO.

START SCAFFOLD (tag l51-start): the capstone stack from Lesson
50 stands finished — fusion, the tenant predicate, the gated
verdict — and below it sits the readiness review, sketched: five
items named, none graded. main() already reads the review's
blocker list, but the sketch returns every item ungraded, so the
blocker list is empty by construction and the verdict prints
SHIP over a review that never ran. A review that has not run
cannot approve anything. The finished version lives at tag
l51-finish.

Lesson task — write, in this file:
  the evidence under each readiness item: the harness's measured
  recall for quality, the capstone run's own leak count for
  isolation, a replayed batch for idempotency — and an honest
  ESCALATED, with its reason, for each item this stack carries
  no evidence for.

Run:  python -m shipstack.stack
"""
from __future__ import annotations

import sys

from hybridrank.hybrid import dense_list, rrf, sparse_list
from tenantsafe.store import META
from veccommon import QUERIES, ndcg_at_k

GATE_RECALL = 0.80


def tenant_fused(tenant: str, query: str) -> list[str]:
    fused = rrf([dense_list(query), sparse_list(query)])
    return [d for d in fused if META[d]["tenant"] == tenant]


REVIEW_ITEMS = ["retrieval quality", "tenant isolation",
                "ingestion idempotency", "durability evidence",
                "load evidence"]


def readiness(mean_recall: float, leaks: int) -> list[tuple[str, str, str]]:
    from freshpipe.ingest import BATCH1, Store
    from proveit.harness import run as eval_run

    rows = []
    r = eval_run()
    rows.append(("retrieval quality",
                 "APPROVED" if r["mean_recall"] >= GATE_RECALL else "BLOCKED",
                 f"harness mean recall@10 {r['mean_recall']:.3f} against the {GATE_RECALL:.2f} gate"))
    rows.append(("tenant isolation",
                 "APPROVED" if leaks == 0 else "BLOCKED",
                 f"{leaks} cross-tenant leaks in the capstone run above"))
    store = Store()
    store.apply("b1", BATCH1)
    store.apply("b1", BATCH1)
    rows.append(("ingestion idempotency",
                 "APPROVED" if len(store.vectors) == 3 else "BLOCKED",
                 f"a replayed batch leaves the store at {len(store.vectors)} documents"))
    if hasattr(sys.modules[__name__], "restore"):
        rows.append(("durability evidence", "APPROVED",
                     "a restore entry point exists to drill"))
    else:
        rows.append(("durability evidence", "ESCALATED",
                     "no restore entry point exists in the stack: Lesson 30's drill has nothing to run"))
    if hasattr(sys.modules[__name__], "load_report"):
        rows.append(("load evidence", "APPROVED",
                     "a measured load report ships with the stack"))
    else:
        rows.append(("load evidence", "ESCALATED",
                     "no measured load run ships with the stack: sizing stays arithmetic (Lessons 29 and 33)"))
    return rows


def main() -> None:
    recalls, leaks = [], 0
    for query, grades in QUERIES:
        hits = tenant_fused("acme", query)
        relevant = {d for d, g in grades.items() if g >= 2}
        recalls.append(len(set(hits[:10]) & relevant) / len(relevant))
        leaks += sum(1 for d in hits if META[d]["tenant"] != "acme")
    mean_recall = sum(recalls) / len(recalls)
    print(f"capstone mean recall@10 (tenant acme): {mean_recall:.3f}")
    print(f"cross-tenant leaks: {leaks}")
    rows = readiness(mean_recall, leaks)
    for name, status, evidence in rows:
        print(f"readiness: {name}: {status} - {evidence}")
    blockers = [n for n, s, _ in rows if s == "BLOCKED"]
    escalations = [n for n, s, _ in rows if s == "ESCALATED"]
    print(f"blockers: {', '.join(blockers) if blockers else 'none'}")
    if escalations:
        print(f"escalations: {len(escalations)} ({', '.join(escalations)})")
    else:
        print("escalations: none")
    verdict = "SHIP" if mean_recall >= GATE_RECALL and leaks == 0 and not blockers else "NO-GO"
    print(f"verdict: {verdict}")
    print("shipstack demo complete")


if __name__ == "__main__":
    main()
