"""Price the Workload — Lesson 33 DEMO.

RAM-first cost arithmetic for a vector workload, managed vs self-host,
plus the dimension lever: truncating 1536 -> 512 dimensions moves both
the memory bill and the fleet size. Prices are illustrative planning
figures (labelled as such in the lesson); the arithmetic shape — RAM
dominates, replicas multiply, dimensions divide — is the content.

Run:  python -m costclock.cost
"""
from __future__ import annotations

RAM_GB_PRICE = 4.0        # illustrative $ per GB-month, managed premium tier
NODE_GB, NODE_PRICE = 64, 420.0   # illustrative self-host node: 64GB, $/month
OVERHEAD = 1.35           # index + OS + headroom multiplier on raw vector bytes


def raw_gb(vectors: int, dims: int) -> float:
    return vectors * dims * 4 / 1e9   # float32


def managed_monthly(vectors: int, dims: int, replicas: int = 2) -> float:
    return raw_gb(vectors, dims) * OVERHEAD * replicas * RAM_GB_PRICE


def selfhost_monthly(vectors: int, dims: int, replicas: int = 2) -> float:
    import math
    nodes = math.ceil(raw_gb(vectors, dims) * OVERHEAD * replicas / NODE_GB)
    return nodes * NODE_PRICE


def report(vectors: int, dims: int) -> None:
    print(f"{vectors:,} vectors x {dims} dims: raw {raw_gb(vectors, dims):.1f} GB, "
          f"managed ${managed_monthly(vectors, dims):,.0f}/mo, "
          f"self-host ${selfhost_monthly(vectors, dims):,.0f}/mo")


def main() -> None:
    report(10_000_000, 1536)
    report(100_000_000, 1536)
    report(100_000_000, 512)
    cut = managed_monthly(100_000_000, 1536) - managed_monthly(100_000_000, 512)
    print(f"dimension lever at 100M vectors: saves ${cut:,.0f}/mo managed")
    print("costclock demo complete")


if __name__ == "__main__":
    main()
