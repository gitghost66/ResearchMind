"""
Run a benchmark: every question through every system, judged, scored, and
written out (M7).

Each (question, system) pair gets a fresh runtime, so budgets, stores, and
caches are never shared. Pairs run one at a time: free-tier rate limits make
parallel runs slower, not faster, and timings stay comparable.

An output directory holds:

* ``meta.json`` — what was run: suite, systems, models, prompt versions,
  and the eval settings;
* ``details.jsonl`` — one line per pair with everything, including each judged
  sentence and the excerpts the judge read. Appended as each pair finishes,
  so an interrupted benchmark resumes where it stopped. Raw output: large,
  and not committed;
* ``results.json`` — the aggregates plus one summary row per pair;
* ``results.md`` — the comparison tables for the project report;
* ``spot_check.jsonl`` — a sample of judged citations for checking the judge
  by hand: fill in ``human_stance``, then run ``agreement`` on the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchmind_agents.budget import BudgetTracker
from researchmind_agents.config import AgentConfig, LLMConfig
from researchmind_agents.evals.benchmark import BenchmarkQuestion
from researchmind_agents.evals.judge import judge_coverage, judge_support
from researchmind_agents.evals.metrics import (
    HEADLINE_METRICS,
    answer_metrics,
    compare,
    retrieval_metrics,
    summarize_rows,
)
from researchmind_agents.evals.systems import (
    BASELINE_LLM,
    RAG,
    RESEARCHMIND,
    SYSTEM_RUNNERS,
    SYSTEMS,
    SystemAnswer,
    rank_sources,
    system_config,
)
from researchmind_agents.evals.tables import render_results
from researchmind_agents.llm.base import BaseLLMProvider, UsageSink
from researchmind_agents.llm.factory import build_llm_provider
from researchmind_agents.modes import MODES
from researchmind_agents.prompts.baselines import PROMPT_VERSION as CLOSED_BOOK_PROMPT_VERSION
from researchmind_agents.prompts.baselines import RAG_PROMPT_VERSION
from researchmind_agents.prompts.eval_judge import COVERAGE_PROMPT_VERSION
from researchmind_agents.prompts.eval_judge import PROMPT_VERSION as SUPPORT_PROMPT_VERSION
from researchmind_agents.runtime import AgentRuntime, build_runtime

logger = logging.getLogger(__name__)

__all__ = [
    "RuntimeFactory",
    "JudgeFactory",
    "BenchmarkRun",
    "judge_model_name",
    "judge_llm_config",
    "live_runtime_factory",
    "live_judge_factory",
    "run_benchmark",
    "evaluate_one",
    "aggregate",
    "load_results_dir",
    "write_outputs",
    "spot_check_sample",
]

RuntimeFactory = Callable[[AgentConfig], AgentRuntime]
JudgeFactory = Callable[[AgentConfig, UsageSink | None], BaseLLMProvider]

META_FILE = "meta.json"
DETAILS_FILE = "details.jsonl"
RESULTS_JSON = "results.json"
RESULTS_MD = "results.md"
SPOT_CHECK_FILE = "spot_check.jsonl"


@dataclass
class BenchmarkRun:
    out_dir: Path
    meta: dict[str, Any]
    rows: list[dict[str, Any]]
    aggregates: dict[str, Any]
    # Pairs actually run by this call (the rest came from --resume).
    ran: int = 0


def judge_model_name(config: AgentConfig) -> str:
    return config.eval.judge_model or config.llm.model


def judge_llm_config(config: AgentConfig) -> LLMConfig:
    return replace(config.llm, model=judge_model_name(config))


def live_runtime_factory(config: AgentConfig) -> AgentRuntime:
    """Real model, search, pages, and papers; an in-memory store (no database)."""
    return build_runtime(config)


def live_judge_factory(config: AgentConfig, usage_sink: UsageSink | None) -> BaseLLMProvider:
    return build_llm_provider(judge_llm_config(config), usage_sink)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


async def run_benchmark(
    *,
    config: AgentConfig,
    questions: Sequence[BenchmarkQuestion],
    systems: Sequence[str],
    runtime_factory: RuntimeFactory,
    judge_factory: JudgeFactory,
    out_dir: Path,
    suite: str,
    resume: bool = False,
    note: str = "",
    progress: Callable[[str], None] | None = None,
) -> BenchmarkRun:
    """
    Run *questions* through *systems*, judge every answer, and write the
    results to *out_dir*. With *resume*, pairs already in ``details.jsonl``
    are kept and not run again.
    """
    unknown = [s for s in systems if s not in SYSTEM_RUNNERS]
    if unknown:
        raise ValueError(f"unknown system(s) {unknown}; choose from {', '.join(SYSTEMS)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    details = out_dir / DETAILS_FILE
    meta = _new_meta(config, suite=suite, systems=systems, note=note)

    rows: list[dict[str, Any]] = []
    if details.exists() and details.stat().st_size > 0:
        if not resume:
            raise ValueError(f"{out_dir} already holds results; resume it or choose another directory")
        previous, rows = load_results_dir(out_dir)
        for key in ("model", "judge_model"):
            if previous.get(key) != meta[key]:
                raise ValueError(
                    f"cannot resume: {key} was {previous.get(key)!r}, now {meta[key]!r}"
                )
        meta["created"] = previous.get("created", meta["created"])
        meta["systems"] = _ordered_systems([*previous.get("systems", []), *systems])
        meta["judge_usage"] = previous.get("judge_usage", {})
    _write_json(out_dir / META_FILE, meta)

    done = {(row["question_id"], row["system"]) for row in rows}
    todo = [(q, s) for q in questions for s in systems if (q.id, s) not in done]
    say = progress or (lambda _line: None)

    judge_budget = BudgetTracker()
    judge = judge_factory(config, judge_budget.wrap())
    ran = 0
    try:
        for index, (question, system) in enumerate(todo, start=1):
            row = await evaluate_one(config, question, system, runtime_factory, judge)
            with details.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            rows.append(row)
            ran += 1
            say(f"[{index}/{len(todo)}] {_progress_line(row)}")
    finally:
        closer = getattr(judge, "aclose", None)
        if closer is not None:
            await closer()
        meta["judge_usage"] = _add_usage(meta.get("judge_usage") or {}, judge_budget.summary())
        _write_json(out_dir / META_FILE, meta)

    aggregates = write_outputs(out_dir, meta, rows)
    return BenchmarkRun(out_dir=out_dir, meta=meta, rows=rows, aggregates=aggregates, ran=ran)


async def evaluate_one(
    config: AgentConfig,
    question: BenchmarkQuestion,
    system: str,
    runtime_factory: RuntimeFactory,
    judge: BaseLLMProvider,
) -> dict[str, Any]:
    """Run one system on one question, then judge and score its answer."""
    run_id = str(uuid.uuid4())
    runtime = runtime_factory(system_config(config, system))
    answer: SystemAnswer | None = None
    error: str | None = None
    retrieved: list[str] = []
    started = time.monotonic()
    try:
        try:
            answer = await SYSTEM_RUNNERS[system](runtime, question, run_id)
        except Exception as exc:  # noqa: BLE001 - one failed run must not stop the benchmark
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("%s failed on %s: %s", system, question.id, error, exc_info=True)
        seconds = time.monotonic() - started
        usage = runtime.budget.summary()
        if answer is not None and answer.retrieves:
            retrieved = await rank_sources(
                runtime, run_id=run_id, question=question.question, sources=answer.sources
            )
    finally:
        await runtime.aclose()

    completed = (
        error is None
        and answer is not None
        and bool(answer.statements)
        and answer.generation == "llm"
    )
    row: dict[str, Any] = {
        "question_id": question.id,
        "mode": question.mode,
        "system": system,
        "question": question.question,
        "run_id": run_id,
        "completed": completed,
        "error": error,
        "generation": answer.generation if answer else None,
        "prompt_version": answer.prompt_version if answer else "",
        "notes": list(answer.notes) if answer else [],
        "answer": None,
        "coverage": None,
        "retrieval": None,
        "ops": {
            "seconds": round(seconds, 3),
            "tokens": usage["tokens_used"],
            "cost_usd": usage["cost_usd"],
            "calls": usage["calls"],
            "iterations": answer.iterations if answer else 0,
        },
        "judge_errors": [],
        "truncated": 0,
        "detail": {"statements": [], "retrieved": retrieved[:50]},
    }
    if answer is not None and answer.retrieves:
        row["retrieval"] = retrieval_metrics(retrieved, question.gold_sources, config.eval.retrieval_k)
    if not completed or answer is None:
        return row

    # Only completed answers are judged; the others count as not completed.
    support = await judge_support(
        judge,
        question=question.question,
        statements=answer.statements,
        config=config.eval,
        run_id=run_id,
    )
    coverage = await judge_coverage(
        judge,
        question=question.question,
        topics=question.expected_subtopics,
        statements=[s.text for s in answer.statements],
        config=config.eval,
        run_id=run_id,
    )
    row["answer"] = answer_metrics(support.statements)
    row["coverage"] = coverage.as_dict()
    row["judge_errors"] = [*support.errors, *([coverage.error] if coverage.error else [])]
    row["truncated"] = support.truncated
    row["detail"]["statements"] = [s.as_dict() for s in support.statements]
    return row


def _progress_line(row: dict[str, Any]) -> str:
    head = f"{row['question_id']} {row['system']}"
    seconds = row["ops"]["seconds"]
    if row["error"]:
        return f"{head}: error after {seconds:.1f}s: {row['error'][:120]}"
    if not row["completed"]:
        return f"{head}: not completed ({row['generation']}, {'; '.join(row['notes'])}) {seconds:.1f}s"
    answer = row["answer"] or {}
    parts = [f"{answer.get('statements', 0)} sentences"]
    for key, label in (("faithfulness", "faithful"), ("citation_precision", "citation precision")):
        value = answer.get(key)
        if value is not None:
            parts.append(f"{label} {value:.0%}")
    completeness = (row["coverage"] or {}).get("completeness")
    if completeness is not None:
        parts.append(f"complete {completeness:.0%}")
    return f"{head}: {', '.join(parts)}, {seconds:.1f}s, {row['ops']['tokens']} tokens"


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------


def _ordered_systems(names: Sequence[str]) -> list[str]:
    known = [s for s in SYSTEMS if s in names]
    return known + [s for s in dict.fromkeys(names) if s not in known]


def aggregate(rows: Sequence[dict[str, Any]], systems: Sequence[str], ks: Sequence[int]) -> dict[str, Any]:
    """Overall and per-mode summaries per system, and the head-to-head verdicts."""
    systems = _ordered_systems(systems)
    overall = {s: summarize_rows([r for r in rows if r["system"] == s], ks) for s in systems}
    modes = [m for m in MODES if any(r["mode"] == m for r in rows)]
    by_mode = {
        mode: {
            s: summarize_rows([r for r in rows if r["system"] == s and r["mode"] == mode], ks)
            for s in systems
        }
        for mode in modes
    }
    verdicts: dict[str, Any] = {}
    baselines = [s for s in systems if s != RESEARCHMIND]
    if RESEARCHMIND in systems and baselines:
        for metric in HEADLINE_METRICS:
            verdicts[metric] = compare(rows, metric, target=RESEARCHMIND, baselines=baselines)
    return {"overall": overall, "by_mode": by_mode, "verdicts": verdicts}


def write_outputs(out_dir: Path, meta: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Write ``results.json``, ``results.md``, and the spot-check sample; return the aggregates."""
    systems = _ordered_systems([*meta.get("systems", []), *(r["system"] for r in rows)])
    ks = list(meta.get("retrieval_k") or [5, 10])
    questions: dict[str, str] = {}
    for row in rows:
        questions.setdefault(row["question_id"], row["mode"])
    counts = {m: sum(1 for mode in questions.values() if mode == m) for m in MODES}
    shown = {
        **meta,
        "systems": systems,
        "questions": len(questions),
        "modes": {m: n for m, n in counts.items() if n},
    }
    aggregates = aggregate(rows, systems, ks)
    summary_rows = [{k: v for k, v in row.items() if k != "detail"} for row in rows]
    _write_json(out_dir / RESULTS_JSON, {"meta": shown, "aggregates": aggregates, "rows": summary_rows})
    (out_dir / RESULTS_MD).write_text(render_results(shown, aggregates, rows), encoding="utf-8")

    spot = out_dir / SPOT_CHECK_FILE
    if _has_human_labels(spot):
        logger.info("%s has human labels; leaving it unchanged", spot)
    else:
        sample = spot_check_sample(rows, int(meta.get("eval_config", {}).get("spot_check_sample", 30)))
        spot.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in sample), encoding="utf-8"
        )
    return aggregates


def load_results_dir(out_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``meta.json`` and the rows of ``details.jsonl`` (the last line wins per pair)."""
    meta_path = out_dir / META_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    details = out_dir / DETAILS_FILE
    if details.exists():
        for number, line in enumerate(details.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write leaves a partial last line.
                logger.warning("Skipping unreadable line %d of %s", number, details)
                continue
            by_pair[(row["question_id"], row["system"])] = row
    return meta, list(by_pair.values())


def spot_check_sample(rows: Sequence[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """
    Up to *size* judged citations, taken round-robin across systems in a fixed
    pseudo-random order (a hash of each item), so a rerun picks the same ones.
    """
    pools: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for row in rows:
        for index, statement in enumerate(row.get("detail", {}).get("statements", [])):
            if statement.get("judged_by") != "judge":
                continue
            for citation in statement.get("citations", []):
                if citation.get("stance") is None:
                    continue
                key = hashlib.sha1(
                    f"{row['question_id']}|{row['system']}|{index}|{citation['label']}".encode()
                ).hexdigest()
                pools.setdefault(row["system"], []).append(
                    (
                        key,
                        {
                            "question_id": row["question_id"],
                            "system": row["system"],
                            "question": row.get("question", ""),
                            "statement": statement["text"],
                            "source": citation.get("url") or citation["label"],
                            "title": citation.get("title", ""),
                            "excerpt": citation.get("excerpt", ""),
                            "judge_stance": citation["stance"],
                            "judge_verdict": statement.get("verdict"),
                            "judge_reason": statement.get("reason", ""),
                            "human_stance": None,
                            "human_note": "",
                        },
                    )
                )
    queues = [sorted(pools[s]) for s in _ordered_systems(list(pools))]
    picked: list[dict[str, Any]] = []
    rank = 0
    while len(picked) < max(0, size) and any(rank < len(q) for q in queues):
        for queue in queues:
            if rank < len(queue) and len(picked) < size:
                picked.append(queue[rank][1])
        rank += 1
    return [{"id": f"sc-{i:03d}", **item} for i, item in enumerate(picked, start=1)]


def _has_human_labels(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if line.strip() and json.loads(line).get("human_stance"):
                return True
        except json.JSONDecodeError:
            return True  # hand-edited; never overwrite it
    return False


def _new_meta(
    config: AgentConfig, *, suite: str, systems: Sequence[str], note: str
) -> dict[str, Any]:
    return {
        "suite": suite,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "systems": _ordered_systems(systems),
        "provider": config.llm.provider,
        "model": config.llm.model,
        "embedding_model": config.embedding.model,
        "judge_model": judge_model_name(config),
        "judge_prompts": {"support": SUPPORT_PROMPT_VERSION, "coverage": COVERAGE_PROMPT_VERSION},
        "system_prompts": {
            BASELINE_LLM: CLOSED_BOOK_PROMPT_VERSION,
            RAG: RAG_PROMPT_VERSION,
            RESEARCHMIND: "per row (prompt_version)",
        },
        "retrieval_k": list(config.eval.retrieval_k),
        "eval_config": asdict(config.eval),
        "budget": {
            "max_tokens_per_run": config.max_tokens_per_run,
            "max_cost_per_run_usd": config.max_cost_per_run_usd,
        },
        "note": note,
        "judge_usage": {},
    }


def _add_usage(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {
        "calls": int(before.get("calls", 0)) + int(after.get("calls", 0)),
        "tokens_used": int(before.get("tokens_used", 0)) + int(after.get("tokens_used", 0)),
        "cost_usd": round(float(before.get("cost_usd", 0.0)) + float(after.get("cost_usd", 0.0)), 6),
    }


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
