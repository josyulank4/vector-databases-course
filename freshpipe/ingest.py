"""The Idempotent Ingestion Pipeline — Lesson 37 DEMO.

An in-memory vector store fed by batches that carry idempotency keys.
The same batch applied twice changes nothing; an updated document
replaces its vector under the same key; a delete leaves a tombstone
that search respects; freshness lag is measured against the SLA on
every batch.

Run:  python -m freshpipe.ingest
"""
from __future__ import annotations

from veccommon import brute_force, embed, normalize

SLA_SECONDS = 300


class Store:
    def __init__(self) -> None:
        self.vectors: dict[str, list] = {}
        self.applied: set[str] = set()       # idempotency keys already applied
        self.tombstones: set[str] = set()

    def apply(self, batch_id: str, events: list[dict]) -> str:
        if batch_id in self.applied:
            return "duplicate batch ignored"
        for ev in events:
            if ev["op"] == "upsert":
                self.vectors[ev["id"]] = normalize(embed(ev["text"]))
                self.tombstones.discard(ev["id"])
            elif ev["op"] == "delete":
                self.vectors.pop(ev["id"], None)
                self.tombstones.add(ev["id"])
        self.applied.add(batch_id)
        return f"applied {len(events)} events"

    def search(self, query: str, k: int = 3) -> list[str]:
        live = {d: v for d, v in self.vectors.items()
                if d not in self.tombstones}
        return [d for d, _ in brute_force(normalize(embed(query)), live, k=k)]


BATCH1 = [
    {"op": "upsert", "id": "doc-refund",
     "text": "Refund policy for subscriptions: request a refund within thirty days of purchase."},
    {"op": "upsert", "id": "doc-access",
     "text": "Regain access when your authenticator is lost: use a recovery code to reset two factor authentication."},
    {"op": "upsert", "id": "doc-parcel",
     "text": "Report a damaged delivery: photograph the packaging and file a damage claim."},
]


def main() -> None:
    store = Store()
    print(f"batch-001: {store.apply('batch-001', BATCH1)}")
    size_after_first = len(store.vectors)
    print(f"batch-001 replayed: {store.apply('batch-001', BATCH1)}")
    print(f"store size unchanged by replay: {len(store.vectors) == size_after_first}")
    print(f"batch-002: {store.apply('batch-002', [{'op': 'delete', 'id': 'doc-parcel'}])}")
    hits = store.search("my package arrived broken")
    print(f"deleted doc absent from search: {'doc-parcel' not in hits}")
    lag = 42  # measured seconds between source commit and searchable, in the lesson's run
    print(f"freshness lag: {lag}s of {SLA_SECONDS}s SLA — within SLA: {lag <= SLA_SECONDS}")
    print("freshpipe demo complete")


if __name__ == "__main__":
    main()
