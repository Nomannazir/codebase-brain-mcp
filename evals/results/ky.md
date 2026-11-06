## Retrieval eval: ky @ `3541888878`

32 queries · embedder `fastembed:BAAI/bge-small-en-v1.5` · top 10 results per query · 95% bootstrap CIs over queries (2000 resamples)

| Mode | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 (95% CI) | nDCG@10 | Recall@10 | p50 / p95 latency |
|---|---|---|---|---|---|---|---|---|
| **hybrid** | 47% | 66% | 69% | 78% | 0.563 (0.413–0.714) | 0.576 | 70% | 6 / 7 ms |
| **keyword** | 38% | 44% | 53% | 62% | 0.436 (0.285–0.593) | 0.455 | 58% | 1 / 2 ms |
| **vector** | 47% | 69% | 75% | 78% | 0.594 (0.456–0.737) | 0.604 | 71% | 5 / 8 ms |

**By query type** (Hit@5 / MRR)

| Mode | behaviour (14) | docs (7) | identifier (11) |
|---|---|---|---|
| hybrid | 36% / 0.18 | 100% / 0.93 | 91% / 0.82 |
| keyword | 7% / 0.05 | 86% / 0.74 | 91% / 0.73 |
| vector | 43% / 0.24 | 100% / 0.93 | 100% / 0.83 |

**Paired, per query** (by reciprocal rank)

- hybrid vs keyword: 12 better, 20 same, 0 worse
- hybrid vs vector: 5 better, 18 same, 9 worse

**Hybrid misses (not in top 5)**

- `id-hooks-type` "Hooks type beforeRetry afterResponse": top results `test/hooks.ts:4120-4179`, `test-d/hooks.ts:8-17`, `test/hooks.ts:170-229`
- `be-network-error-messages` "recognise a connection failure from the browser's error message text": top results `readme.md:1411-1432`, `source/utils/is-network-error.ts:1-13`, `test/errors.ts:230-289`
- `be-late-response-discarded` "a reply that arrives after the deadline has passed is thrown away": top results `test/response-size.ts:501-560`, `test/resilience.ts:466-525`, `test/resilience.ts:968-1027`
- `be-wait-cancellable` "sleep between attempts but stop waiting if the caller cancels": top results `test/retry.ts:5312-5371`, `test/resilience.ts:566-625`, `test/retry.ts:4762-4821`
- `be-delete-header` "setting a header to undefined removes it when combining defaults with per-call headers": top results `test/headers.ts:816-875`, `readme.md:1008-1097`, `test/headers.ts:766-825`
- `be-unknown-options-forwarded` "pass options this library does not recognise straight through to fetch": top results `source/types/options.ts:451-513`, `test/fetch.ts:901-960`, `readme.md:93-149`
- `be-response-too-big` "stop reading when the downloaded body grows past a byte cap": top results `test/stream.ts:1397-1456`, `test/response-handling.ts:82-141`, `test/stream.ts:1247-1306`
- `be-retry-limit-reached` "give up and rethrow once the maximum number of attempts is used": top results `test/resilience.ts:616-675`, `test/resilience.ts:566-625`, `test/resilience.ts:266-325`
- `be-total-deadline` "one overall deadline across every attempt including the waits between them": top results `test/resilience.ts:566-625`, `test/resilience.ts:466-525`, `test/resilience.ts:516-575`
- `be-custom-json-parser` "use the caller's own function to decode JSON instead of the built-in one": top results `test/validation.ts:56-115`, `source/core/Ky.ts:220-232`, `test/validation.ts:106-165`
