"""Shard Math: The Simulator — Lesson 29 DEMO.

START SCAFFOLD (tag l29-start): the population table and the hash
placement are here; the replica and compaction arithmetic in
main() is yours to write in the lesson. The finished version
lives at tag l29-finish.

Lesson task — write, in this file:
  1. replica math: per-shard query load at R replicas, and what a
     second replica buys in QPS and in survival.
  2. compaction: merge rounds to steady state, and the I/O tax
     those rounds levy while they run.

Run:  python -m shardscale.shards
"""
from __future__ import annotations

import hashlib

SHARDS = 8
TENANTS = {"whale": 80_000, "big": 30_000, "mid": 15_000}
SMALL = {f"tenant-{i:02d}": 75_000 // 25 for i in range(25)}
POP = {**TENANTS, **SMALL}
TOTAL = sum(POP.values())


def hash_shard(vec_id: str) -> int:
    return int(hashlib.md5(vec_id.encode()).hexdigest(), 16) % SHARDS


def simulate() -> tuple[list[int], list[int], dict[str, set[int]]]:
    by_hash = [0] * SHARDS
    by_tenant: list[int] = [0] * SHARDS
    tenant_home: dict[str, int] = {t: i % SHARDS for i, t in enumerate(POP)}
    spread: dict[str, set[int]] = {t: set() for t in POP}
    for tenant, count in POP.items():
        home = tenant_home[tenant]
        by_tenant[home] += count
        for j in range(count):
            s = hash_shard(f"{tenant}:{j}")
            by_hash[s] += 1
            spread[tenant].add(s)
    return by_hash, by_tenant, spread


def main() -> None:
    by_hash, by_tenant, spread = simulate()
    mean = TOTAL / SHARDS
    print(f"total vectors: {TOTAL} across {SHARDS} shards (mean {mean:.0f})")
    print(f"hash sharding: max {max(by_hash)} / min {min(by_hash)} "
          f"(max is {max(by_hash) / mean:.2f}x mean)")
    print(f"tenant sharding: max {max(by_tenant)} / min {min(by_tenant)} "
          f"(max is {max(by_tenant) / mean:.2f}x mean)")
    print(f"whale tenant spans {len(spread['whale'])} shards under hash, "
          f"1 under tenant sharding")
    # TODO (Lesson 29): the replica + compaction arithmetic —
    # per-shard load at R replicas, merge rounds, the I/O tax.
    print("shardscale demo complete")


if __name__ == "__main__":
    main()
