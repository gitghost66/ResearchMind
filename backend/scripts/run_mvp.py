"""
Run research questions end to end against the live APIs (M3).

From the repository root::

    .venv\\Scripts\\python backend\\scripts\\run_mvp.py "Is RAG better than fine-tuning?"
    .venv\\Scripts\\python backend\\scripts\\run_mvp.py --seed            # seed eval
    .venv\\Scripts\\python backend\\scripts\\run_mvp.py --seed --limit 3

Needs GEMINI_API_KEY in backend/.env (embeddings always use Gemini; the LLM may use
LLM_PROVIDER/LLM_API_KEY instead). ``--store db`` persists to DATABASE_URL —
run ``alembic upgrade head`` first; ``--store memory`` needs no database;
``auto`` (default) uses the database when DATABASE_URL is set.

Prints the markdown report, a usage summary, and the grounding check. Exits
non-zero when a run fails, misses the grounding targets, or falls back to an
extractive report (the model did not write it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "agents" / "researchmind_agents" / "evals" / "results"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    from researchmind_agents.modes import SUPPORTED_MODES

    parser = argparse.ArgumentParser(description="Run ResearchMind end to end.")
    parser.add_argument("question", nargs="?", help="research question")
    parser.add_argument(
        "--mode",
        choices=sorted(SUPPORTED_MODES),
        default="quick",
        help="research mode (deep, academic, and decision verify claims and loop)",
    )
    parser.add_argument("--seed", action="store_true", help="run the seed eval instead")
    parser.add_argument("--limit", type=int, default=0, help="seed eval: first N questions only")
    parser.add_argument("--store", choices=["auto", "memory", "db"], default="auto")
    parser.add_argument("--json", type=Path, help="also write results as JSON to this path")
    parser.add_argument("--min-cited", type=float, default=0.8, help="cited-statement target")
    parser.add_argument("--min-resolved", type=float, default=1.0, help="citation target")
    parser.add_argument("-v", "--verbose", action="store_true", help="show INFO logs")
    args = parser.parse_args(argv)
    if not args.seed and not args.question:
        parser.error("give a question, or --seed")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    json_path = args.json.resolve() if args.json else None
    # Settings find .env by absolute path; the chdir keeps relative paths predictable.
    os.chdir(REPO_ROOT)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from app.core.config import get_settings
    from app.core.tracing import configure_tracing

    settings = get_settings()
    configure_tracing(settings)
    problem = _missing_keys(settings)
    if problem:
        print(f"error: {problem} (set it in {REPO_ROOT / 'backend' / '.env'})", file=sys.stderr)
        return 2

    use_db = args.store == "db" or (args.store == "auto" and bool(settings.database_url))
    if args.store == "db" and not settings.database_url:
        print("error: --store db needs DATABASE_URL", file=sys.stderr)
        return 2
    return asyncio.run(_run(args, settings, use_db, json_path))


def _missing_keys(settings: Any) -> str:
    if not settings.gemini_api_key:
        return "GEMINI_API_KEY is not set (needed for embeddings)"
    if settings.llm_provider not in ("gemini", "google") and not settings.llm_api_key:
        return f"LLM_API_KEY is not set for LLM_PROVIDER={settings.llm_provider}"
    return ""


async def _run(args: argparse.Namespace, settings: Any, use_db: bool, json_path: Path | None) -> int:
    from researchmind_agents.evals.check_grounding import load_seed, summarize

    session_factory = None
    engine = None
    if use_db:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        from app.db.session import build_engine

        engine = build_engine(settings.database_url, pool_pre_ping=True)
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    try:
        if not args.seed:
            row = await _one(args.question, "cli", settings, session_factory, args)
            print(row.pop("markdown", ""))
            _print_row(row)
            rows = [row]
        else:
            questions = load_seed()
            if args.limit > 0:
                questions = questions[: args.limit]
            rows = []
            for index, seed in enumerate(questions, start=1):
                print(f"[{index}/{len(questions)}] {seed.id}: {seed.question}", flush=True)
                row = await _one(seed.question, seed.id, settings, session_factory, args)
                row.pop("markdown", None)
                rows.append(row)
            print()
            print(summarize(rows))
            print(_latency_line(rows, args.mode))
    finally:
        if engine is not None:
            await engine.dispose()

    passed = all(not row.get("error") and row["passed"] for row in rows)
    _write_json(rows, json_path, seed=args.seed, store="db" if use_db else "memory")
    print(f"\nstore: {'database' if use_db else 'memory'} | grounding targets met: {passed}")
    return 0 if passed else 1


async def _one(
    question: str,
    row_id: str,
    settings: Any,
    session_factory: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from researchmind_agents.evals.check_grounding import check_grounding

    from app.services.research_runner import execute_research

    started = time.monotonic()
    try:
        outcome = await execute_research(
            question, mode=args.mode, settings=settings, session_factory=session_factory
        )
    except Exception as exc:  # noqa: BLE001 - report and continue with the next question
        logging.getLogger("run_mvp").exception("run failed")
        return {"id": row_id, "question": question, "error": f"{type(exc).__name__}: {exc}",
                "seconds": time.monotonic() - started, "passed": False}

    grounding = check_grounding(outcome.report, outcome.markdown, outcome.stored_source_ids)
    generation = outcome.report.get("generation")
    return {
        "id": row_id,
        "question": question,
        "run_id": outcome.run_id,
        "generation": generation,
        "sources": len(outcome.report.get("sources", [])),
        "sources_stored": len(outcome.stored_source_ids),
        # Includes why the model did not write the report, if it fell back.
        "limitations": outcome.report.get("limitations", []),
        "grounding": grounding,
        # An extractive fallback quotes passages, so it is cited by construction;
        # only a model-written report tests the synthesizer.
        "passed": generation == "llm"
        and grounding.passes(min_resolution=args.min_resolved, min_cited=args.min_cited),
        "tokens": outcome.usage["tokens_used"],
        "cost_usd": outcome.usage["cost_usd"],
        "usage": outcome.usage,
        "seconds": outcome.seconds,
        "research_results": outcome.state.get("research_results", []),
        "mode": args.mode,
        "overall_confidence": outcome.report.get("overall_confidence"),
        "verification": outcome.report.get("verification", {}),
        "markdown": outcome.markdown,
    }


def _print_row(row: dict[str, Any]) -> None:
    if row.get("error"):
        print(f"\nRUN FAILED: {row['error']}")
        return
    usage = row["usage"]
    g = row["grounding"]
    print("--- usage ---")
    print(f"run_id: {row['run_id']}  time: {row['seconds']:.1f}s  "
          f"tokens: {usage['tokens_used']}  cost: ${usage['cost_usd']:.4f}  calls: {usage['calls']}")
    for agent, values in usage["per_agent"].items():
        print(f"  {agent:15} calls={values['calls']} in={values['tokens_in']} "
              f"out={values['tokens_out']} cost=${values['cost_usd']:.4f}")
    print("--- research ---")
    for result in row["research_results"]:
        print(f"  {result['status']:9} sources={result['sources']} passages={result['passages']} "
              f"fetched={result['fetched']} failed={result['fetch_failures']} "
              f"dupes={result['duplicates']} {result['error'] or ''}")
    if row.get("verification"):
        v = row["verification"]
        print("--- verification ---")
        print(f"  claims extracted: {v.get('claims_extracted')}  checked: {v.get('claims_checked')}  "
              f"used: {v.get('claims_used')}  by verdict: {v.get('by_verdict')}")
        print(f"  overall confidence: {row.get('overall_confidence')}")
    print("--- grounding ---")
    print(f"  generation: {row['generation']}  sources cited: {row['sources']} "
          f"(stored: {row['sources_stored']})")
    print(f"  citations resolved: {g.citations_resolved}/{g.citations_total} "
          f"({g.citation_resolution_rate:.0%})")
    print(f"  statements cited:   {g.statements_cited}/{g.statements_total} "
          f"({g.cited_statement_rate:.0%})  dropped: {g.dropped_statements}")


def _latency_line(rows: list[dict[str, Any]], mode: str) -> str:
    """Median run time against the mode's PRD target (M6: each mode at p50)."""
    from statistics import median

    from researchmind_agents.modes import mode_profile

    times = [row["seconds"] for row in rows if not row.get("error")]
    target = mode_profile(mode).target_seconds
    if not times:
        return "p50 latency: no completed runs"
    p50 = median(times)
    verdict = "met" if p50 <= target else "missed"
    return f"p50 latency: {p50:.1f}s over {len(times)} run(s) (target {target:.0f}s for {mode}: {verdict})"


def _write_json(rows: list[dict[str, Any]], path: Path | None, *, seed: bool, store: str) -> None:
    if path is None and seed:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"seed-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    if path is None:
        return
    serialisable = []
    for row in rows:
        item = dict(row)
        if "grounding" in item:
            item["grounding"] = item["grounding"].as_dict()
        serialisable.append(item)
    path.write_text(
        json.dumps({"store": store, "created": datetime.now(UTC).isoformat(), "rows": serialisable},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"results written to {path}")


if __name__ == "__main__":
    sys.exit(main())
