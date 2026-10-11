# Results: BIRD mini-dev (SQLite), execution accuracy, test split

**150 of 500 questions**, stratified (same difficulty and database mix as the full set). Held-out: none of these questions were looked at while building the agent. Model: `gemini-3.5-flash-lite` (free tier, no fallback), temperature 0. Generated 2026-10-11.

| Variant | What is switched on | EX | simple | moderate | challenging | exec errors | LLM calls / q | fixes used | sec / q |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | bare schema, one shot | **60.7%** | 69.6% | 59.5% | 50.0% | 2.0% | 1.00 | 0.0% | 1.2 |
| `linked` | + schema linking, descriptions, example values | **64.7%** | 73.9% | 67.6% | 43.3% | 2.7% | 1.00 | 0.0% | 1.3 |
| `linked+fix` | + self-correction on errors / empty results | **66.7%** | 73.9% | 71.6% | 43.3% | 0.7% | 1.07 | 5.3% | ≈ `linked` + fixes |
| `agent` | + literal-value check (full agent) | **66.7%** | 73.9% | 71.6% | 43.3% | 0.7% | 1.09 | 6.0% | ≈ `linked` + fixes |

## What each step adds (paired bootstrap, 10,000 resamples)

| Step | Δ EX | 95% CI | questions fixed | questions broken |
|---|---|---|---|---|
| `baseline` → `linked` | +4.0% | [-0.7%, +8.7%] | 9 | 3 |
| `linked` → `linked+fix` | +2.0% | [+0.0%, +4.7%] | 3 | 0 |
| `linked+fix` → `agent` | +0.0% | [+0.0%, +0.0%] | 0 | 0 |
| `baseline` → `agent` | +6.0% | [+1.3%, +11.3%] | 12 | 3 |

A step is a clear improvement when its whole confidence interval is above zero. sec / q is API + database time per question, excluding free-tier rate-limit waiting; the fix variants reuse the first call from the cache, so their time is the `linked` time plus about one call per fix.

Caveat: an audit of BIRD mini-dev found annotation errors in a share of the gold queries, so a few "wrong" answers are actually right. All runs share the same gold queries, so the *differences* between variants are still a fair comparison.
