# Data Analyst Agent

Ask a question about a database in plain English. The agent finds the relevant tables and columns, writes SQL,
runs it on a **read-only** connection, **checks the result for silent mistakes**, fixes its own query using the
database's feedback, and answers in plain English with a chart.

Built as a **LangGraph** state machine around a small, free model (Gemini 3.5 Flash-Lite), and **measured
component by component** on the [BIRD](https://bird-bench.github.io/) text-to-SQL benchmark, on held-out questions.

```
question ─► link_schema ─► write_sql ─► execute ──ok──► analyze ─► answer + chart
                                          │   ▲
          error · 0 rows · COUNT = 0 ·    │   │ corrected query
          all NULL · unknown text value   ▼   │
                                         fix_sql   (at most 3 times)
```

## Results

Held-out test split: **150 BIRD mini-dev questions never looked at while building the agent**, all answered by
`gemini-3.5-flash-lite` (free tier) at temperature 0. Execution accuracy (EX) = the query returns the same rows as
the gold query.

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

**The full agent is 6.0 points better than the one-shot baseline (60.7% → 66.7%), with a 95% confidence interval
of [+1.3, +11.3]**, so the gain is not noise. It fixes 12 questions and breaks 3. Individually, each step is too
small to prove with 150 questions: schema linking gives the most (+4.0) and self-correction breaks nothing
(+2.0, 3 fixed, 0 broken). The literal-value check fired on only one test question and did not change its
outcome; on the dev split it fixed 2 questions and broke none. Full report: [`results/RESULTS_test.md`](results/RESULTS_test.md);
every per-question answer is in [`results/runs/test/`](results/runs/test/).

## What each part does

| Node | What it does | Why |
|---|---|---|
| **link_schema** | Embeds every column ("table.column: description; value meanings; example values") with a local `bge-small` model and gives the 20 most relevant columns detailed notes in the prompt. No table is ever dropped. | Focuses the model on the right columns without risking the loss of a table the query needs. Runs on CPU in milliseconds, costs no API calls. |
| **write_sql** | One Gemini call: schema + question + business definitions → `{reasoning, sql}`. | |
| **execute** | SQL guard (`sqlglot`: one read-only statement, no writes, no `ATTACH`/`PRAGMA`, no file functions) → read-only connection with a timeout → result checks. A request to change data is refused, not "fixed". | Two independent safety layers. |
| **result checks** | Flags errors, empty results, a single `COUNT = 0`, all-NULL results, and **text values that do not exist in their column** (`'East Bohemia'` vs stored `'east Bohemia'`), with the closest stored values. | Catches the *silent* failures: queries that run fine and return the wrong number. No LLM call. |
| **fix_sql** | Gets the exact problem plus the real stored values of every filtered column, and rewrites the query, or keeps it if it is actually right. | Feedback from the database, not from the model's own guesses. |
| **analyze** | Plain-English answer + a chart **spec** (`{type, x, y, title}`) validated against the result columns. | The model never writes code that gets executed. |

## Quick start

**Windows:** double-click `run.bat`. **macOS:** double-click `run.command`. On first run each launcher creates a
virtual environment, installs the packages, asks for your API key, downloads the benchmark databases
(~800 MB download, 1.4 GB on disk), and opens the app at <http://localhost:8501>.

Manual setup (Python 3.11 or 3.12):

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # add GEMINI_API_KEY (free: https://aistudio.google.com/apikey)
python scripts/setup_data.py                             # BIRD mini-dev databases + questions
streamlit run app.py
```

In the app, pick a BIRD database (or upload your own CSV files; `samples/` has synthetic e-commerce data),
ask a question, and open **"How the agent got there"** to see every query attempt. The **Variant** selector
switches parts of the agent off, so you can compare them on the same question.

## Reproducing the evaluation

```bash
python -m evaluation.run_eval                 # held-out test split, all 4 variants (~350 API calls)
python -m evaluation.run_eval --split dev     # the development split used for error analysis
python -m pytest                              # 35 offline tests: no API key or download needed
```

- **Metric:** execution accuracy (EX), BIRD's official metric. Predicted and gold SQL are both executed, and the
  prediction counts as correct when the result sets are equal.
- **Splits:** BIRD mini-dev has 500 questions. **Dev** (150, seed 42) was used for error analysis while
  building the agent. **Test** (150 *different* questions, seed 7) was never looked at; the headline numbers
  come from it. Both splits keep the full set's difficulty and database mix.
- **Fair comparison:** every variant answers the same questions with the same model (no fallback; the cache key
  includes the model), at temperature 0. Gains come with paired-bootstrap 95% confidence intervals.
- **Resumable:** every answer is appended to `results/runs/<split>/<variant>.jsonl`, and LLM replies are cached
  locally, so an interrupted run (free-tier quota) continues where it stopped and a re-run costs nothing.
- **Caveat:** an audit of BIRD mini-dev found annotation errors in some gold queries, so a few answers scored as
  wrong are right. All variants share the same gold queries, so the differences between them remain fair.

## How it was built

1. **v1** was measured on the dev split. Self-correction and the literal check helped a little; schema linking
   was net zero (fixed 6 questions, broke 6). See [`results/RESULTS_dev_v1.md`](results/RESULTS_dev_v1.md).
2. **Error analysis** of the 48 dev failures found three fixable causes. (1) Column descriptions that disagree with
   the data: a date "in the form YYMMDD" is stored as `1995-03-24`, so the model wrote `LIKE '93%'`. (2) "Give
   the answer to 3 decimal places" was ignored. (3) A wrong filter that returns `COUNT = 0` looks like a valid
   answer, so the fix loop never ran. Fixes: example values for date columns, a `ROUND` rule, and
   `COUNT = 0` is now flagged and the fixer is shown real stored values of the filtered columns.
3. **v2** was then measured once on the untouched test split.

Bugs found along the way:
- The first benchmark run silently mixed three models through rate-limit fallback. Now the client waits for the
  preferred model, falls back only on real errors, and the cache key includes the model.
- An unclosed quote in generated SQL raised sqlglot's `TokenError` (not `ParseError`) and crashed the guard. All
  parser errors are caught now, and the runner records a crash on one question instead of stopping.
- A request to "delete all the accounts" was blocked by the guard, then "repaired" by the fix loop into
  `SELECT * FROM account`. Write requests are now refused; only malformed SQL goes to the fixer.
- CSV dates loaded as text through pandas; DuckDB's CSV sniffer is used instead.

## Project layout

```
analyst_agent/
  graph.py          LangGraph state machine and the evaluated variants
  db.py             SQLite (benchmark) and DuckDB (uploaded CSV) behind one interface; schema text for prompts
  schema_linker.py  local embeddings (fastembed / ONNX) over column descriptions
  guard.py          SQL guard: one read-only statement or nothing
  checks.py         literal-value check and filter-value probe (no LLM calls)
  llm.py            Gemini client: two keys, strict model preference, rate limiting, local cache
  prompts.py        every prompt in one file
  charts.py         chart spec → Plotly figure
evaluation/run_eval.py   execution accuracy, dev/test splits, paired bootstrap, report
app.py                   Streamlit UI
tests/                   offline tests with a scripted fake LLM
results/                 per-question outputs of every run, and the reports
```

## Stack

Python · LangGraph · Gemini 3.5 Flash-Lite (`google-genai`) · sqlglot · SQLite · DuckDB · fastembed
(BAAI/bge-small-en-v1.5, ONNX) · Streamlit · Plotly · pytest. Runs on a laptop CPU with a free API key.

## License

MIT. BIRD mini-dev is CC BY-SA 4.0 and is downloaded by `scripts/setup_data.py`, not redistributed here.
