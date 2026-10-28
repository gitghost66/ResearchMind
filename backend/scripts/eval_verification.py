"""
M5 verification eval against the live model.

Runs the fact checker on the 20 hand-labelled claims in
``agents/researchmind_agents/evals/claims_gold.jsonl`` with the configured LLM
and reports verdict agreement and confidence error. From the repository root::

    .venv\\Scripts\\python backend\\scripts\\eval_verification.py
    .venv\\Scripts\\python backend\\scripts\\eval_verification.py --json out.json

Needs the LLM key in backend/.env (GEMINI_API_KEY, or LLM_API_KEY for another
provider). Makes one small model call per claim (20 in total); no database,
search, or embeddings are used. The M5 bar is verdict agreement >= 80% and
confidence within +/-0.15 of the label for >= 80% of claims. Exits non-zero
when the bar is missed.
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
    parser = argparse.ArgumentParser(description="Score the fact checker on the gold claims.")
    parser.add_argument("--limit", type=int, default=0, help="first N claims only")
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
    if settings.llm_provider in ("gemini", "google") and not settings.gemini_api_key:
        print("error: GEMINI_API_KEY is not set (backend/.env)", file=sys.stderr)
        return 2
    if settings.llm_provider not in ("gemini", "google") and not settings.llm_api_key:
        print(f"error: LLM_API_KEY is not set for LLM_PROVIDER={settings.llm_provider}", file=sys.stderr)
        return 2
    return asyncio.run(_run(args, settings, json_path))


async def _run(args: argparse.Namespace, settings: Any, json_path: Path | None) -> int:
    from researchmind_agents.evals.check_verification import (
        TARGET,
        evaluate_verification,
        load_gold,
    )

    from app.services.agent_runtime import build_agent_runtime

    items = load_gold()
    if args.limit > 0:
        items = items[: args.limit]
    # No session factory: an in-memory store, and usage is not written anywhere.
    runtime = build_agent_runtime(settings=settings)
    try:
        result = await evaluate_verification(runtime, items)
    finally:
        await runtime.aclose()

    print(f"{'id':5} {'expected':12} {'got':12} {'label':>6} {'conf':>6} {'err':>6}")
    for o in result.outcomes:
        mark = "" if o.verdict_ok else "  <- verdict"
        print(
            f"{o.id:5} {o.expected_verdict:12} {o.verdict:12} {o.expected_confidence:6.2f} "
            f"{o.confidence:6.2f} {o.confidence_error:6.2f}{mark}"
        )
    summary = result.summary()
    print()
    print(f"verdict agreement:            {summary['verdict_agreement']:.0%} (target {TARGET:.0%})")
    print(
        f"confidence within +/-{result.tolerance:.2f}:   "
        f"{summary['confidence_within_tolerance']:.0%} (target {TARGET:.0%})"
    )
    print(f"confidence mean abs. error:   {summary['confidence_mae']:.3f}")
    print(f"stance agreement:             {summary['stance_agreement']:.0%}")
    usage = runtime.budget.summary()
    print(f"model: {settings.llm_model}  calls: {usage['calls']}  tokens: {usage['tokens_used']}")
    print(f"passed: {summary['passed']}")

    path = json_path
    if path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"verification-{datetime.now(UTC):%Y%m%d-%H%M%S}.json"
    path.write_text(
        json.dumps(
            {"model": settings.llm_model, "created": datetime.now(UTC).isoformat(), **summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"results written to {path}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
