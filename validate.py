"""End-to-end validation for the Course 8 code repository.
Runs every lesson demo as a subprocess (exactly how a student runs them)
and checks expected output markers. Run from the repo root: python validate.py
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
FAILURES = []
COUNT = [0]
def check(name, condition, detail=""):
    COUNT[0] += 1
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition: FAILURES.append(name)
def expect(module, markers):
    proc = subprocess.run([sys.executable, "-m", module], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
    check(f"{module}: runs end-to-end (exit 0)", proc.returncode == 0, proc.stderr.strip()[-300:])
    for m in markers: check(f"{module}: output contains {m!r}", m in proc.stdout)
def main():
    print("== module demos (subprocess) ==")
    expect("keywordgap.baseline", ["bm25 top hit: shipping-delay", "dense top hit: reset-2fa", "dense rank of reset-2fa: 1", "keywordgap demo complete"])
    expect("embedlab.embed", ["paraphrase beats unrelated: True", "word order changes nothing: True", "embedlab demo complete"])
    expect("indexlab.indexes", ["corpus: 2000 vectors x 32 dims, 8 ivf cells", "indexlab demo complete"])
    expect("pickfighter.select", ["tiny-saas (40k vectors, postgres already): verdict pgvector", "search-platform (900M vectors, dedicated team): verdict managed-dedicated", "pickfighter demo complete"])
    expect("hybridrank.hybrid", ["fusion recall@3: 0.667", "hybridrank demo complete"])
    expect("tenantsafe.store", ["all acme: True", "globex cross-tenant leaks: 0", "audit lines written: 3", "tenantsafe demo complete"])
    expect("shardscale.shards", ["total vectors: 200000 across 8 shards (mean 25000)", "shardscale demo complete"])
    expect("costclock.cost", ["dimension lever at 100M vectors: saves $4,424/mo managed", "costclock demo complete"])
    expect("freshpipe.ingest", ["duplicate batch ignored", "store size unchanged by replay: True", "deleted doc absent from search: True", "freshpipe demo complete"])
    expect("proveit.harness", ["mean recall@10: 1.000", "proveit demo complete"])
    expect("shipstack.stack", ["cross-tenant leaks: 0", "verdict: SHIP", "shipstack demo complete"])
    print()
    print("== direct assertions ==")
    from keywordgap.baseline import rank_bm25, rank_dense
    check("dense ranks reset-2fa first for the locked-out query", rank_dense("locked out of my account")[0] == "reset-2fa")
    check("bm25 buries reset-2fa below rank 5 for the locked-out query", rank_bm25("locked out of my account").index("reset-2fa") >= 5)
    from tenantsafe.store import search, META
    check("tenant search returns only the caller's tenant", all(META[d]["tenant"] == "globex" for d in search("globex", "api quota exceeded")))
    from freshpipe.ingest import Store, BATCH1
    s = Store(); s.apply("b1", BATCH1); s.apply("b1", BATCH1)
    check("idempotent replay keeps store size at 3", len(s.vectors) == 3)
    from proveit.harness import run
    r = run()
    check("harness mean recall@10 is 1.000 on the golden set", abs(r["mean_recall"] - 1.0) < 1e-9)
    from indexlab.indexes import train_ivf, assign, POINTS
    cents = train_ivf()
    check("ivf trains 8 centroids", len(cents) == 8)
    check("every point assigns to a valid cell", all(0 <= assign(v, cents) < 8 for _, v in POINTS[:50]))
    print()
    if FAILURES:
        print(f"OVERALL: FAIL ({len(FAILURES)} failures)"); return 1
    print(f"OVERALL: PASS ({COUNT[0]} checks)")
    return 0
if __name__ == "__main__":
    sys.exit(main())
