"""The Selection Scorecard — Lesson 17 DEMO.

Three fictional teams, three workloads, one weighted scorecard over
three candidate shapes (dedicated managed store, pgvector on the
existing Postgres, self-hosted open source). The arithmetic is on
screen; the verdicts differ because the workloads differ — which is
the entire lesson.

Run:  python -m pickfighter.select
"""
from __future__ import annotations

# criteria scores 0..5 per candidate (higher = better fit)
CANDIDATES = {
    "managed-dedicated": {"ops_ease": 5, "filtering": 4, "scale_headroom": 5,
                          "cost_at_scale": 2, "existing_stack_fit": 2},
    "pgvector": {"ops_ease": 3, "filtering": 5, "scale_headroom": 2,
                 "cost_at_scale": 4, "existing_stack_fit": 5},
    "self-hosted-oss": {"ops_ease": 1, "filtering": 4, "scale_headroom": 4,
                        "cost_at_scale": 5, "existing_stack_fit": 3},
}

# (team, weights, security gate: minimum filtering score to be eligible)
TEAMS = [
    ("tiny-saas (40k vectors, postgres already)", 
     {"ops_ease": 3, "filtering": 3, "scale_headroom": 1, "cost_at_scale": 2,
      "existing_stack_fit": 4}, 3),
    ("search-platform (900M vectors, dedicated team)",
     {"ops_ease": 2, "filtering": 3, "scale_headroom": 5, "cost_at_scale": 4,
      "existing_stack_fit": 1}, 3),
    ("regulated-health (2M vectors, strict tenancy)",
     {"ops_ease": 2, "filtering": 5, "scale_headroom": 2, "cost_at_scale": 2,
      "existing_stack_fit": 3}, 5),
]


def score(weights: dict[str, int]) -> dict[str, int]:
    return {name: sum(weights[k] * v[k] for k in weights)
            for name, v in CANDIDATES.items()}


def verdict(team: str, weights: dict[str, int], gate: int) -> str:
    ranked = sorted(score(weights).items(), key=lambda kv: (-kv[1], kv[0]))
    for name, total in ranked:
        if CANDIDATES[name]["filtering"] >= gate:
            return f"{team}: verdict {name} ({total} pts)"
    return f"{team}: NO candidate passes the security gate"


def main() -> None:
    for team, weights, gate in TEAMS:
        print(verdict(team, weights, gate))
    print("pickfighter demo complete")


if __name__ == "__main__":
    main()
