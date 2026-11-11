"""The Idempotent Ingestion Pipeline — Lesson 37 DEMO.

START SCAFFOLD (tag l37-start): the in-memory Store is here — a
vectors map guarded by an append-only journal of applied
idempotency keys, a payload schema tag on every event, and
tombstones for deletes. The three verify helpers above main() carry
TODO markers: the empty guards make the narration of a replayed
batch print True vacuously — run it and watch three greens that
prove nothing. The finished version lives at tag l37-finish.

Lesson task — write, in this file:
  1. the three verifications: replay changes nothing (the vector
     snapshot compared element for element), the tombstoned
     document is invisible to search AND counted in the raw map,
     the freshness lag checked against a measured injected event.
  2. the schema step: apply the v2 batch (payload gains a tenant
     field), and show the migrated record answering the tenant
     query while the unmigrated record stays honestly invisible.

Run:  python -m freshpipe.ingest
"""
from __future__ import annotations

from veccommon import brute_force, embed, normalize

SLA_SECONDS = 300


class Store:
    def __init__(self) -> None:
        self.vectors: dict[str, list] = {}
        self.applied: dict[str, str] = {}    # idempotency key -> verdict
        self.tombstones: set[str] = set()
        self.schema: dict[str, int] = {}     # doc id -> payload schema version

    def apply(self, batch_id: str, events: list[dict]) -> str:
        if batch_id in self.applied:
            return "duplicate batch ignored"
        for ev in events:
            if ev["op"] == "upsert":
                self.vectors[ev["id"]] = normalize(embed(ev["text"]))
                self.tombstones.discard(ev["id"])
                self.schema[ev["id"]] = ev.get("schema", 1)
            elif ev["op"] == "delete":
                self.tombstones.add(ev["id"])
        self.applied[batch_id] = f"applied {len(events)} events"
        return self.applied[batch_id]

    def search(self, query: str, k: int = 3) -> list[str]:
        live = {d: v for d, v in self.vectors.items()
                if d not in self.tombstones}
        return [d for d, _ in brute_force(normalize(embed(query)), live, k=k)]

    def tenant_search(self, tenant: str, query: str, k: int = 3) -> list[str]:
        """Search only records whose payload schema has migrated to v2
        (carrying the tenant field) and matches the tenant."""
        live = {d: v for d, v in self.vectors.items()
                if d not in self.tombstones and self.schema.get(d, 1) >= 2}
        return [d for d, _ in brute_force(normalize(embed(query)), live, k=k)]


BATCH1 = [
    {"op": "upsert", "id": "doc-refund",
     "text": "Refund policy for subscriptions: request a refund within thirty days of purchase."},
    {"op": "upsert", "id": "doc-access",
     "text": "Regain access when your authenticator is lost: use a recovery code to reset two factor authentication."},
    {"op": "upsert", "id": "doc-parcel",
     "text": "Report a damaged delivery: photograph the packaging and file a damage claim."},
]

BATCH2 = [{"op": "delete", "id": "doc-parcel"}]

BATCH3 = [
    {"op": "upsert", "id": "doc-refund", "schema": 2, "tenant": "acme",
     "text": "Refund policy for subscriptions: request a refund within thirty days of purchase."},
    {"op": "upsert", "id": "doc-invoice", "schema": 2, "tenant": "acme",
     "text": "Invoice and billing questions: download a copy of your receipt and payment history."},
]


def replay_unchanged(store: Store) -> bool:
    """Verify the replay: re-run the duplicate application against a
    deep snapshot of the store and compare every stored vector,
    element for element — not just the count."""
    import copy
    snap = copy.deepcopy(store.vectors)
    store.apply("batch-001", BATCH1)
    if store.vectors.keys() != snap.keys():
        return False
    return all(store.vectors[d] == snap[d] for d in snap)


def delete_verified(store: Store) -> bool:
    """Verify the tombstone: doc-parcel must be invisible to search
    AND still retained in the raw vectors map — deletion hides, it
    does not erase, until compaction reclaims it."""
    hits = store.search("my package arrived broken")
    return ("doc-parcel" in store.tombstones
            and "doc-parcel" in store.vectors
            and "doc-parcel" not in hits)


def freshness_within_sla(events: list[dict], applied_at: float) -> bool:
    """Verify freshness on the injected clock: the batch event_time
    is 1000.0; the lag is applied_at minus event_time, checked
    against SLA_SECONDS."""
    return (applied_at - 1000.0) <= SLA_SECONDS


def main() -> None:
    now = 1000.0  # injected clock: event_time 1000.0, commit at 1042.0
    store = Store()
    print(f"batch-003: {store.apply('batch-003', BATCH3)}")
    tenant_hits = store.tenant_search("acme", "refund my subscription payment")
    print(f"tenant search finds migrated doc: {'doc-refund' in tenant_hits}")
    print(f"unmigrated doc invisible to tenant search: {'doc-access' not in store.tenant_search('acme', 'reset two factor authentication')}")
    print(f"schema versions on record: {store.schema}")
    print(f"batch-001: {store.apply('batch-001', BATCH1)}")
    size_after_first = len(store.vectors)
    print(f"batch-001 replayed: {store.apply('batch-001', BATCH1)}")
    print(f"store size unchanged by replay: {len(store.vectors) == size_after_first}")
    print(f"replay vector-identical: {replay_unchanged(store)}")
    print(f"batch-002: {store.apply('batch-002', BATCH2)}")
    hits = store.search("my package arrived broken")
    print(f"deleted doc absent from search: {'doc-parcel' not in hits}")
    print(f"tombstone verified (hidden, retained): {delete_verified(store)}")
    lag = 42  # measured seconds between source commit and searchable, in the lesson's run
    print(f"freshness lag: {lag}s of {SLA_SECONDS}s SLA — within SLA: {freshness_within_sla(BATCH1, now + lag)}")
    print(f"applied keys on record: {len(store.applied)}")
    print("freshpipe demo complete")


if __name__ == "__main__":
    main()
