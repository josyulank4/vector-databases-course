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
    hot_qps = 8000        # illustrative fleet query load, queries/sec
    capacity = 400        # illustrative per-copy capacity in budget
    hot = max(by_tenant) / TOTAL * hot_qps
    print(f"hot shard load: {hot:.0f} qps against {capacity} qps per shard-copy")
    for r in (1, 3):
        print(f"R={r}: hot shard serves {hot / r:.0f} qps per copy "
              f"({hot / r / capacity:.2f}x capacity)")
    print(f"R=3, one copy lost: survivors carry {hot / 2:.0f} qps "
          f"({hot / 2 / capacity:.2f}x capacity)")
    print("the first replica buys survival; the third buys quorum")
    segments, fan_in = 40, 4   # awaiting compaction; merged per group
    rounds, written, cur = 0, 0, segments
    while cur > fan_in:
        written += cur
        rounds += 1
        cur = -(-cur // fan_in)
    written += cur
    print(f"compaction: {segments} segments merge to {cur} in {rounds} rounds")
    print(f"compaction writes {written / segments:.2f}x the corpus while it runs")
    print("shardscale demo complete")


if __name__ == "__main__":
    main()
