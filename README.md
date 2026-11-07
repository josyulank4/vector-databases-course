# Vector Databases in Production — Companion Code (Udemy Course 8)

Working, stdlib-only Python demos for every hands-on lesson of the course.
No API keys, no third-party packages: every mechanic runs offline so students
see what the production tools actually compute. One shared module,
`veccommon.py`, carries the corpus, the deterministic character-trigram
embedder, BM25, and the ranking metrics every package grades itself with.

## Run the full validation
```
python validate.py
```
Expected final line: `OVERALL: PASS`.

## Packages (one per DEMO lesson)
| Package | Lesson | What it shows |
|---|---|---|
| keywordgap | 5 | BM25 vs brute-force cosine on the query keywords cannot answer |
| embedlab | 9 | Deterministic embeddings: paraphrase geometry, normalization, order invariance |
| indexlab | 13 | IVF + product quantization graded against brute-force ground truth |
| pickfighter | 17 | The selection scorecard: three teams, three defended verdicts |
| hybridrank | 21 | BM25 + dense fusion with RRF and a reranker — measured, not asserted |
| tenantsafe | 25 | Pre-filtered, namespaced, audited multi-tenant search |
| shardscale | 29 | Hash vs tenant sharding over a skewed 200k-vector population |
| costclock | 33 | RAM-first cost model: managed vs self-host, the dimension lever |
| freshpipe | 37 | Idempotent ingestion: replay-safe batches, tombstones, freshness SLA |
| proveit | 45 | The eval harness: recall@10, nDCG@10, confidence intervals |
| shipstack | 50/51 | The capstone reference stack, end to end, with its ship verdict |

## Lesson tags
Each DEMO lesson is tagged in this repo: `lXX-start` (code before the lesson's
build) and `lXX-finish` (the exact code the lesson ships). Tags are added as
each DEMO lesson is produced; the pilot (Lesson 1) is a theory lesson and
carries no tag.
