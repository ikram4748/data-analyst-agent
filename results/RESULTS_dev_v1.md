# Results v1 (before error analysis): BIRD mini-dev dev split, execution accuracy

> Historical: the first version of the agent, measured on the dev split. Its failures were analysed and led to
> three changes (date example values, a ROUND rule, flagging COUNT = 0 + a filter-value probe). The headline
> numbers are in `RESULTS_test.md`, measured on held-out questions.

Stratified subset: **150 of 500 questions** (same difficulty and database mix as the full set, seed 42). Model: `gemini-3.5-flash-lite` (free tier, no fallback), temperature 0. Generated 2026-10-11.

| Variant | What is switched on | EX | simple | moderate | challenging | exec errors | LLM calls / q | fixes used | sec / q |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | bare schema, one shot | **65.3%** | 81.0% | 62.3% | 51.6% | 2.0% | 1.00 | 0.0% | 9.5 |
| `linked` | + schema linking, descriptions, example values | **65.3%** | 85.7% | 58.4% | 54.8% | 0.7% | 1.00 | 0.0% | 8.4 |
| `linked+fix` | + self-correction on errors / empty results | **66.7%** | 88.1% | 59.7% | 54.8% | 0.0% | 1.05 | 3.3% | 0.3 |
| `agent` | + literal-value check (full agent) | **68.0%** | 90.5% | 61.0% | 54.8% | 0.0% | 1.07 | 5.3% | 0.4 |

## What each step adds (paired bootstrap, 10,000 resamples)

| Step | Δ EX | 95% CI | questions fixed | questions broken |
|---|---|---|---|---|
| `baseline` → `linked` | +0.0% | [-4.7%, +4.7%] | 6 | 6 |
| `linked` → `linked+fix` | +1.3% | [+0.0%, +3.3%] | 2 | 0 |
| `linked+fix` → `agent` | +1.3% | [+0.0%, +3.3%] | 2 | 0 |
| `baseline` → `agent` | +2.7% | [-2.7%, +8.0%] | 10 | 6 |

A step is a clear improvement when its whole confidence interval is above zero.

Caveat: an audit of BIRD mini-dev found annotation errors in a share of the gold queries, so a few "wrong" answers are actually right. All runs share the same gold queries, so the *differences* between variants are still a fair comparison.
