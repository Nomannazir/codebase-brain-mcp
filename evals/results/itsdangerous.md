## Retrieval eval: itsdangerous @ `672971d66a`

32 queries · embedder `fastembed:BAAI/bge-small-en-v1.5` · top 10 results per query · 95% bootstrap CIs over queries (2000 resamples)

| Mode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 (95% CI) | nDCG@10 | Recall@10 | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|
| **hybrid** | 56% | 88% | 88% | 91% | 0.697 (0.562–0.822) | 0.735 | 88% | 5 / 6 ms |
| **keyword** | 56% | 75% | 78% | 84% | 0.667 (0.521–0.798) | 0.699 | 81% | 0 / 1 ms |
| **vector** | 56% | 72% | 75% | 94% | 0.658 (0.511–0.798) | 0.701 | 88% | 4 / 5 ms |

**By query type** (Hit@5 / MRR)

| Mode | behaviour (14) | docs (7) | identifier (11) |
|---|---|---|---|
| hybrid | 71% / 0.61 | 100% / 0.64 | 100% / 0.85 |
| keyword | 64% / 0.55 | 71% / 0.61 | 100% / 0.85 |
| vector | 57% / 0.51 | 71% / 0.56 | 100% / 0.91 |

**Paired, per query** (by reciprocal rank)

- hybrid vs keyword: 8 better, 20 same, 4 worse
- hybrid vs vector: 8 better, 17 same, 7 worse

**Hybrid misses (not in top 5)**

- `bh-rotation-verify` "which secret keys are tried when checking a token after keys were rotated": top results `docs/concepts.rst:101-154`, `docs/concepts.rst:51-110`, `src/itsdangerous/signer.py:76-128`
- `bh-future-timestamp` "reject a token that claims it was created in the future": top results `src/itsdangerous/exc.py:36-57`, `README.md:3-15`, `src/itsdangerous/exc.py:22-33`
- `bh-fallback-signers` "fall back to older signing settings when the current ones can't verify": top results `docs/timed.rst:1-28`, `CHANGES.rst:101-160`, `src/itsdangerous/signer.py:244-256`
- `bh-debug-decode` "read the contents of a token even when its signature is invalid": top results `src/itsdangerous/exc.py:22-33`, `src/itsdangerous/exc.py:36-57`, `docs/serializer.rst:51-110`
