"""Shard Math: The Simulator — Lesson 29 DEMO.

200,000 synthetic vectors with a realistic tenant-size skew (one whale
tenant holds 40%) placed two ways across 8 shards: hash sharding,
which balances vectors and scatters every tenant, and tenant
sharding, which colocates tenants and inherits the whale. Skew and
hot shards come out of the simulation, not out of a slide.

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
    print("shardscale demo complete")


if __name__ == "__main__":
    main()
