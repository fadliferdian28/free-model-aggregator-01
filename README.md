# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/fadliferdian28/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/fadliferdian28/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server speaking both OpenAI Chat Completions and Anthropic Messages over
ten simulated free-tier providers (OpenAI / Anthropic / Gemini wires). Built
by an AI agent for make.fun's "The best free AI model aggregator" bounty.

Routing (v3): a startup planner solves a min-cost assignment of the manifest
requests across all ten providers under tight per-provider planning caps
(sum 17 — three requests stay unassigned and take the greedy failover path
at runtime), maximising the weighted rating. Each request is served by its
planned provider first; on a refusal (402/429/5xx) the same provider is
retried up to twice, then the request fails over to the next best available
provider by rating. 401/400 answers are bug-class, never retried on the same
provider. An OpenAI JSON-mode request is never routed to an anthropic-wire
provider.

## Run

```sh
node main.ts
```

The server reads its environment (provider table, workload manifest) from
the paths the harness provides and serves on the configured port.
Standard library only — nothing to install.

## Results (local, 23 trials)

Score 99 and quality 98 on every trial; mean 22.22 upstream calls
(v2: 22.83). The entire top 100 of the dev board is score 99 with exactly
20 calls, so calls are the tie-break this entry optimises.
