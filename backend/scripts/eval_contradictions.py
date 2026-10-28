"""
M6 contradiction eval against the live model and embeddings.

Runs the contradiction detector on the 15 seeded cases in
``agents/researchmind_agents/evals/contradictions_gold.jsonl`` with the
configured LLM and embedding model, and reports how many of the 14 seeded
contradictions it finds and whether it names their causes. From the
repository root::

    .venv\\Scripts\\python backend\\scripts\\eval_contradictions.py
    .venv\\Scripts\\python backend\\scripts\\eval_contradictions.py --json out.json

Needs the LLM key in backend/.env (GEMINI_API_KEY, or LLM_API_KEY for another
provider) and the Gemini key for embeddings. Makes one embedding call and at
most a few small model calls per case; no database or search is used. The M6
bar is recall >= 70% (a seeded contradiction is found when a detection holds
both of its claims). Candidate recall shows how many seeded pairs cleared
CONTRADICTION_MIN_SIMILARITY; a low value means the threshold is too high.
Exits non-zero when the bar is missed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "agents" / "researchmind_agents" / "evals" / "results"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the contradiction detector on seeded cases.")
    parser.add_argument("--limit", type=int, default=0, help="first N cases only")
    parser.add_argument("--json", type=Path, help="write results to this path")
    parser.add_argument("-v", "--verbose", action="store_true", help="show INFO logs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    json_path = args.json.resolve() if args.json else None
    os.chdir(REPO_ROOT)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from app.core.config import get_settings

    settings = get_settings()
    if not settings.gemini_api_key:
        print("error: GEMINI_API_KEY is not set (backend/.env); embeddings need it", file=sys.stderr)
        return 2
    if settings.llm_provider not in ("gemini", "google") and not settings.llm_api_key:
        print(f"error: LLM_API_KEY is not set for LLM_PROVIDER={settings.llm_provider}", file=sys.stderr)
        return 2
    return asyncio.run(_run(args, settings, json_path))


async def _run(args: argparse.Namespace, settings: Any, json_path: Path | None) -> int:
    from researchmind_agents.evals.check_contradictions import (
        TARGET,
        evaluate_contradictions,
        load_gold,
    )

    from app.services.agent_runtime import build_agent_runtime

    cases = load_gold()
    if args.limit > 0:
        cases = cases[: args.limit]
    # No session factory: an in-memory store, and usage is not written anywhere.
    runtime = build_agent_runtime(settings=settings)
    try:
        result = await evaluate_contradictions(runtime, cases)
    finally:
        await runtime.aclose()

    print(f"{'case':5} {'claims':10} {'expected':12} {'got':12}")
    for outcome in result.outcomes:
        if not outcome.seeded:
            extra = f"{len(outcome.detected)} false positive(s)" if outcome.detected else "none, as expected"
            print(f"{outcome.id:5} {'-':10} {'(none)':12} {extra}")
        for claims, want, got in outcome.seeded:
            mark = "" if got == want else ("  <- missed" if got is None else "  <- cause")
            print(f"{outcome.id:5} {'+'.join(claims):10} {want:12} {str(got):12}{mark}")
    summary = result.summary()
    print()
    print(f"recall:            {summary['found']}/{summary['seeded']} = {summary['recall']:.0%} (target {TARGET:.0%})")
    print(f"cause agreement:   {summary['cause_agreement']:.0%}")
    print(f"precision:         {summary['precision']:.0%}")
    print(f"false positives:   {summary['false_positives']} (in cases with no contradiction)")
    print(
        f"candidate recall:  {summary['candidate_recall']:.0%} "
        f"(min similarity {runtime.config.contradictions.min_similarity})"
    )
    usage = runtime.budget.summary()
    print(f"model: {settings.llm_model}  calls: {usage['calls']}  tokens: {usage['tokens_used']}")
    print(f"passed: {summary['passed']}")

    path = json_path
    if path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"contradictions-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    path.write_text(
        json.dumps(
            {
                "model": settings.llm_model,
                "embedding_model": settings.embedding_model,
                "min_similarity": runtime.config.contradictions.min_similarity,
                "created": datetime.now(UTC).isoformat(),
                **summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"results written to {path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
