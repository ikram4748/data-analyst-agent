"""Execution-accuracy benchmark on BIRD mini-dev.

    python -m evaluation.run_eval                              # all variants on the held-out test split (150 questions)
    python -m evaluation.run_eval --split dev                  # the development split used for error analysis
    python -m evaluation.run_eval --variants agent             # one variant
    python -m evaluation.run_eval --report                     # rebuild results/RESULTS_<split>.md from saved runs

Metric: execution accuracy (EX), BIRD's official definition. The predicted and gold queries are both run on
the database, and the prediction counts as correct when the two result sets are equal as sets of rows.

Every answered question is appended to results/runs/<split>/<variant>.jsonl immediately, so an interrupted run
(free-tier quota!) resumes where it stopped. LLM replies are cached, so a re-run costs no quota.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analyst_agent.config import get_settings  # noqa: E402
from analyst_agent.db import open_bird  # noqa: E402
from analyst_agent.graph import VARIANTS, build_agent, run  # noqa: E402
from analyst_agent.llm import Gemini, LLMUnavailable  # noqa: E402
from analyst_agent.schema_linker import SchemaLinker  # noqa: E402

RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"          # set per split in main(): results/runs/<split>/
ORDER = ["baseline", "linked", "linked+fix", "agent"]
DIFFICULTIES = ["simple", "moderate", "challenging"]


def load_questions() -> list[dict]:
    return json.loads((get_settings().bird_dir / "mini_dev_sqlite.json").read_text(encoding="utf-8"))


def split(questions: list[dict], name: str, n: int = 150) -> list[dict]:
    """dev  = 150 questions used for error analysis while building the agent (seed 42)
    test = 150 *different* questions, never looked at while building; the headline numbers come from here."""
    dev = stratified_subset(questions, n, seed=42)
    if name == "dev":
        return dev
    dev_ids = {q["question_id"] for q in dev}
    return stratified_subset([q for q in questions if q["question_id"] not in dev_ids], n, seed=7)


def stratified_subset(questions: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Same difficulty mix as the full set, and every database represented in proportion."""
    if n >= len(questions):
        return questions
    rng = random.Random(seed)
    groups: dict[tuple, list] = defaultdict(list)
    for q in questions:
        groups[(q["difficulty"], q["db_id"])].append(q)
    quota = {k: len(v) * n / len(questions) for k, v in groups.items()}
    take = {k: int(x) for k, x in quota.items()}
    # hand out the remaining slots to the groups with the largest rounding remainder
    for k in sorted(quota, key=lambda k: quota[k] - take[k], reverse=True)[: n - sum(take.values())]:
        take[k] += 1
    picked = []
    for k, v in sorted(groups.items()):
        picked += rng.sample(v, take[k])
    return sorted(picked, key=lambda q: q["question_id"])


_gold_cache: dict[int, tuple] = {}
_gold_lock = threading.Lock()


def score(db, gold_sql: str, pred_sql: str, qid: int) -> tuple[bool, str | None]:
    with _gold_lock:
        gold = _gold_cache.get(qid)
    if gold is None:
        g = db.execute(gold_sql, timeout_s=120)
        gold = (set(g.rows), g.error)
        with _gold_lock:
            _gold_cache[qid] = gold
    if gold[1]:
        return False, f"gold query failed: {gold[1]}"
    p = db.execute(pred_sql, timeout_s=60)
    if not p.ok:
        return False, p.error
    return set(p.rows) == gold[0], None


def done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["question_id"] for line in path.read_text().splitlines() if line.strip()}


def evaluate(variant_name: str, questions: list[dict], llm: Gemini, workers: int) -> None:
    variant = dataclasses.replace(VARIANTS[variant_name], analyze=False)   # SQL only: saves one call per question
    out = RUNS / f"{variant_name.replace('+', '_')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    todo = [q for q in questions if q["question_id"] not in done_ids(out)]
    print(f"\n== {variant_name}: {len(questions) - len(todo)} done, {len(todo)} to go")
    dbs, linkers, agents, lock = {}, {}, {}, threading.Lock()

    def agent_for(db_id: str):
        with lock:
            if db_id not in agents:
                dbs[db_id] = open_bird(db_id)
                linkers[db_id] = SchemaLinker(dbs[db_id]) if variant.rich_schema else None
                agents[db_id] = build_agent(dbs[db_id], llm, variant, linkers[db_id])
            return dbs[db_id], agents[db_id]

    def one(q: dict) -> dict:
        db, agent = agent_for(q["db_id"])
        s = run(agent, q["question"], q.get("evidence", ""))
        correct, err = score(db, q["SQL"], s.get("sql", ""), q["question_id"])
        return {"question_id": q["question_id"], "db_id": q["db_id"], "difficulty": q["difficulty"],
                "question": q["question"], "gold_sql": q["SQL"], "pred_sql": s.get("sql", ""),
                "correct": correct, "error": err, "attempts": s.get("attempts", []), "fixes": s.get("fixes", 0),
                "llm_calls": s.get("llm_calls", 0), "models": s.get("models", []), "seconds": round(s["seconds"], 2),
                "work_seconds": round(s.get("llm_seconds", 0) + sum(a["seconds"] for a in s.get("attempts", [])), 2)}

    write_lock = threading.Lock()
    n_ok = n_done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, q): q for q in todo}
        for f in as_completed(futures):
            try:
                row = f.result()
            except LLMUnavailable as e:
                print(f"\nstopping: {e}\nRe-run the same command later; finished questions are kept.")
                pool.shutdown(cancel_futures=True)
                return
            except Exception as e:  # noqa: BLE001 - a bug on one question must not end a long run
                q = futures[f]
                print(f"\n  question {q['question_id']} crashed: {type(e).__name__}: {e}")
                continue
            with write_lock:
                with out.open("a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
                n_ok += row["correct"]
                print(f"\r  {n_done}/{len(todo)}  EX so far {n_ok / n_done:.1%}", end="", flush=True)
    print()


def load_runs() -> dict[str, list[dict]]:
    runs = {}
    for name in ORDER:
        p = RUNS / f"{name.replace('+', '_')}.jsonl"
        if p.exists():
            runs[name] = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return runs


def summarize(runs: dict[str, list[dict]], subset_ids: set[int]) -> dict:
    summary = {}
    for name, rows in runs.items():
        rows = [r for r in rows if r["question_id"] in subset_ids]
        if not rows:
            continue
        by = {d: [r for r in rows if r["difficulty"] == d] for d in DIFFICULTIES}
        # Variants share their first prompt, so later ones reuse it from the cache. Latency is only reported
        # when most questions were answered without the cache, otherwise it would look unrealistically fast.
        timed = [r for r in rows if r["models"] and "cache" not in r["models"]]
        summary[name] = {
            "n": len(rows),
            "ex": sum(r["correct"] for r in rows) / len(rows),
            **{f"ex_{d}": (sum(r["correct"] for r in v) / len(v) if v else None) for d, v in by.items()},
            "exec_error_rate": sum(1 for r in rows if r["error"]) / len(rows),
            "avg_llm_calls": sum(r["llm_calls"] for r in rows) / len(rows),
            "fix_rate": sum(1 for r in rows if r["fixes"]) / len(rows),
            # API + database time per question; excludes time spent waiting on the free-tier rate limit
            "avg_seconds": (sum(r.get("work_seconds", r["seconds"]) for r in timed) / len(timed)
                            if len(timed) > len(rows) / 2 else None),
            "answered_by": dict(Counter(m.split("@")[0] for r in rows for m in r["models"] if m != "cache")),
        }
    return summary


def paired(runs: dict, a: str, b: str, subset_ids: set[int], iters: int = 10_000, seed: int = 0) -> dict | None:
    """Paired bootstrap: EX(b) - EX(a) over the same questions, with a 95% confidence interval."""
    if a not in runs or b not in runs:
        return None
    ra = {r["question_id"]: r["correct"] for r in runs[a] if r["question_id"] in subset_ids}
    rb = {r["question_id"]: r["correct"] for r in runs[b] if r["question_id"] in subset_ids}
    ids = sorted(ra.keys() & rb.keys())
    if not ids:
        return None
    diffs = [int(rb[i]) - int(ra[i]) for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in ids) / len(ids) for _ in range(iters))
    return {"delta": sum(diffs) / len(ids), "lo": boots[int(0.025 * iters)], "hi": boots[int(0.975 * iters)],
            "won": diffs.count(1), "lost": diffs.count(-1), "n": len(ids)}


def report(subset: list[dict], split_name: str) -> None:
    runs = load_runs()
    ids = {q["question_id"] for q in subset}
    summary = summarize(runs, ids)
    steps = [(a, b, paired(runs, a, b, ids)) for a, b in zip(ORDER, ORDER[1:])]
    steps.append(("baseline", "agent", paired(runs, "baseline", "agent", ids)))
    pct = lambda x: "-" if x is None else f"{x:.1%}"  # noqa: E731
    models = ", ".join(f"`{m}`" for m in sorted({m for s in summary.values() for m in s["answered_by"]})) or "-"
    lines = [
        f"# Results: BIRD mini-dev (SQLite), execution accuracy, {split_name} split",
        "",
        f"**{len(subset)} of 500 questions**, stratified (same difficulty and database mix as the full set). "
        + ("Held-out: none of these questions were looked at while building the agent. "
           if split_name == "test" else "Development set: used for error analysis while building the agent. ")
        + f"Model: {models} (free tier, no fallback), temperature 0. "
        f"Generated {time.strftime('%Y-%m-%d')}.",
        "",
        "| Variant | What is switched on | EX | simple | moderate | challenging | exec errors | LLM calls / q | fixes used | sec / q |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    what = {"baseline": "bare schema, one shot", "linked": "+ schema linking, descriptions, example values",
            "linked+fix": "+ self-correction on errors / empty results", "agent": "+ literal-value check (full agent)"}
    for name in ORDER:
        s = summary.get(name)
        if s:
            lines.append(f"| `{name}` | {what[name]} | **{pct(s['ex'])}** | {pct(s['ex_simple'])} | "
                         f"{pct(s['ex_moderate'])} | {pct(s['ex_challenging'])} | {pct(s['exec_error_rate'])} | "
                         f"{s['avg_llm_calls']:.2f} | {pct(s['fix_rate'])} | "
                         + (f"{s['avg_seconds']:.1f} |" if s["avg_seconds"] is not None else "≈ `linked` + fixes |"))
    lines += ["", "## What each step adds (paired bootstrap, 10,000 resamples)", "",
              "| Step | Δ EX | 95% CI | questions fixed | questions broken |", "|---|---|---|---|---|"]
    for a, b, p in steps:
        if p:
            lines.append(f"| `{a}` → `{b}` | {p['delta']:+.1%} | [{p['lo']:+.1%}, {p['hi']:+.1%}] | {p['won']} | {p['lost']} |")
    lines += ["", "A step is a clear improvement when its whole confidence interval is above zero. "
              "sec / q is API + database time per question, excluding free-tier rate-limit waiting; the fix variants "
              "reuse the first call from the cache, so their time is the `linked` time plus about one call per fix.", "",
              "Caveat: an audit of BIRD mini-dev found annotation errors in a share of the gold queries, so a few "
              "\"wrong\" answers are actually right. All runs share the same gold queries, so the *differences* "
              "between variants are still a fair comparison."]
    (RESULTS / f"RESULTS_{split_name}.md").write_text("\n".join(lines) + "\n")
    (RESULTS / f"summary_{split_name}.json").write_text(json.dumps({"summary": summary, "steps": [
        {"from": a, "to": b, **(p or {})} for a, b, p in steps]}, indent=2))
    print("\n".join(lines))


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    RESULTS.mkdir(exist_ok=True)
    api_log = logging.FileHandler(RESULTS / "api_errors.log")      # rate limits and retries, for debugging quota
    api_log.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger("analyst.llm").addHandler(api_log)
    logging.getLogger("analyst.llm").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--variants", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=get_settings().models[0],
                    help="the one model that answers every question (no fallback, so variants stay comparable)")
    ap.add_argument("--report", action="store_true", help="only rebuild the report from saved runs")
    args = ap.parse_args()
    global RUNS
    RUNS = RESULTS / "runs" / args.split
    subset = split(load_questions(), args.split, args.n)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "question_ids.json").write_text(json.dumps([q["question_id"] for q in subset]))
    if not args.report:
        llm = Gemini(dataclasses.replace(get_settings(), models=[args.model]))
        for v in args.variants:
            evaluate(v, subset, llm, args.workers)
        print(f"\nreal API calls this session: {llm.calls}")
    report(subset, args.split)


if __name__ == "__main__":
    main()
