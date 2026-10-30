"""
Command line for the evaluation harness (PRD s11, M7). From the repository root::

    python -m researchmind_agents.evals run --suite default
    python -m researchmind_agents.evals run --suite default --modes quick --limit 3
    python -m researchmind_agents.evals run --suite offline     # no network, no API keys
    python -m researchmind_agents.evals run --resume <results dir>
    python -m researchmind_agents.evals table <results dir>
    python -m researchmind_agents.evals agreement <results dir>/spot_check.jsonl

``run`` sends every question through the closed-book baseline, the
single-agent RAG baseline, and ResearchMind, judges each answer, and writes
``results.md`` (the comparison table), ``results.json``, ``details.jsonl``,
and ``spot_check.jsonl`` to a new directory under ``evals/results/``. Results
are appended as each run finishes; ``--resume`` continues an interrupted run.

Live runs need the model and embedding keys, which live in the backend's
settings. This package never imports the backend (AGENTS.md rule 2): the
backend registers ``app.services.agent_runtime:build_agent_config`` as an
entry point in the ``researchmind_agents.config`` group (``backend/pyproject.toml``),
and ``--config auto`` (the default) loads it. ``--config module:function``
names any other function returning an :class:`AgentConfig`.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from researchmind_agents.config import AgentConfig
from researchmind_agents.evals.benchmark import OFFLINE_SUITE, SUITES, load_suite
from researchmind_agents.evals.offline import (
    offline_config,
    offline_judge_factory,
    offline_runtime_factory,
)
from researchmind_agents.evals.runner import (
    META_FILE,
    RESULTS_MD,
    JudgeFactory,
    RuntimeFactory,
    live_judge_factory,
    live_runtime_factory,
    load_results_dir,
    run_benchmark,
    write_outputs,
)
from researchmind_agents.evals.systems import RAG, RESEARCHMIND, SYSTEMS
from researchmind_agents.modes import SUPPORTED_MODES

__all__ = ["CONFIG_GROUP", "RESULTS_DIR", "ConfigError", "load_config", "agreement", "main"]

CONFIG_GROUP = "researchmind_agents.config"
RESULTS_DIR = Path(__file__).with_name("results")
STANCES = ("supports", "contradicts", "neutral")


class ConfigError(RuntimeError):
    """No usable configuration for a live run."""


def load_config(spec: str) -> AgentConfig:
    """
    ``"auto"``: the config provider registered in the ``researchmind_agents.config``
    entry-point group (the backend's, when it is installed). ``"module:function"``:
    that function, called with no arguments.
    """
    if spec == "auto":
        found = sorted(entry_points(group=CONFIG_GROUP), key=lambda ep: (ep.name != "backend", ep.name))
        if not found:
            raise ConfigError(
                "no config provider is installed. Install the backend (pip install -e backend), "
                "pass --config app.services.agent_runtime:build_agent_config, or run "
                "--suite offline"
            )
        source = f"entry point {found[0].name!r} ({found[0].value})"
        try:
            factory: Any = found[0].load()
        except Exception as exc:  # noqa: BLE001 - report a broken install clearly
            raise ConfigError(f"could not load {source}: {exc}") from exc
    else:
        module_name, sep, attr = spec.partition(":")
        if not sep or not module_name or not attr:
            raise ConfigError(f"--config takes module:function, got {spec!r}")
        source = spec
        try:
            factory = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise ConfigError(f"could not load {spec}: {exc}") from exc

    config = factory() if callable(factory) else factory
    if not isinstance(config, AgentConfig):
        raise ConfigError(f"{source} returned {type(config).__name__}, not AgentConfig")
    return config


def _missing_key(config: AgentConfig, systems: Sequence[str]) -> str:
    if config.llm.provider in ("gemini", "google") and not config.llm.api_key:
        return "GEMINI_API_KEY is not set (backend/.env)"
    if any(s in (RAG, RESEARCHMIND) for s in systems) and not config.embedding.api_key:
        return "GEMINI_API_KEY is not set; the RAG baseline and ResearchMind need embeddings"
    return ""


def _names(raw: str | None, allowed: Sequence[str], what: str) -> list[str]:
    if not raw:
        return list(allowed)
    names = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise SystemExit(f"error: unknown {what} {unknown}; choose from {', '.join(allowed)}")
    return list(dict.fromkeys(names))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m researchmind_agents.evals",
        description="Benchmark ResearchMind against a closed-book LLM and single-agent RAG.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a benchmark suite and write the comparison table")
    run.add_argument(
        "--suite",
        help=f"{', '.join([*SUITES, OFFLINE_SUITE])}, or a .jsonl path (default: default; "
        "with --resume, the resumed run's suite)",
    )
    run.add_argument("--systems", help=f"comma-separated subset of {','.join(SYSTEMS)}")
    run.add_argument("--modes", help=f"only these modes ({','.join(sorted(SUPPORTED_MODES))})")
    run.add_argument("--questions", help="only these question ids, comma-separated")
    run.add_argument("--limit", type=int, default=0, help="first N questions after filtering")
    run.add_argument("--out", type=Path, help="results directory (default: a new one under evals/results/)")
    run.add_argument("--resume", type=Path, help="continue the run in this results directory")
    run.add_argument(
        "--config",
        default="auto",
        help="'auto' (the installed backend's settings) or module:function returning an AgentConfig",
    )
    run.add_argument("--note", default="", help="free text recorded with the results")
    run.add_argument("-v", "--verbose", action="store_true", help="show INFO logs")

    table = sub.add_parser("table", help="rebuild results.md and results.json from a results directory")
    table.add_argument("directory", type=Path)

    check = sub.add_parser("agreement", help="score your spot-check labels against the judge")
    check.add_argument("file", type=Path, help="a spot_check.jsonl with human_stance filled in")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "run":
        return _run(args)
    if args.command == "table":
        return _table(args.directory)
    return _agreement(args.file)


def _run(args: argparse.Namespace) -> int:
    resume_dir: Path | None = args.resume
    suite = args.suite
    if resume_dir is not None:
        if not (resume_dir / META_FILE).exists():
            print(f"error: {resume_dir} is not a results directory (no {META_FILE})", file=sys.stderr)
            return 2
        meta = json.loads((resume_dir / META_FILE).read_text(encoding="utf-8"))
        suite = suite or meta.get("suite")
    suite = suite or "default"

    try:
        questions = load_suite(suite)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    modes = _names(args.modes, sorted(SUPPORTED_MODES), "mode(s)")
    questions = [q for q in questions if q.mode in modes]
    if args.questions:
        wanted = {part.strip() for part in args.questions.split(",") if part.strip()}
        questions = [q for q in questions if q.id in wanted]
    if args.limit > 0:
        questions = questions[: args.limit]
    if not questions:
        print("error: no questions left after filtering", file=sys.stderr)
        return 2
    systems = _names(args.systems, SYSTEMS, "system(s)")

    runtime_factory: RuntimeFactory
    judge_factory: JudgeFactory
    if suite == OFFLINE_SUITE:
        config = offline_config()
        runtime_factory, judge_factory = offline_runtime_factory, offline_judge_factory
    else:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        problem = _missing_key(config, systems)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 2
        runtime_factory, judge_factory = live_runtime_factory, live_judge_factory

    out_dir = resume_dir or args.out or RESULTS_DIR / _default_dir_name(suite)
    print(
        f"{len(questions)} question(s) x {len(systems)} system(s) -> {out_dir}\n"
        f"model: {config.llm.model}  judge: {config.eval.judge_model or config.llm.model}",
        flush=True,
    )
    try:
        result = asyncio.run(
            run_benchmark(
                config=config,
                questions=questions,
                systems=systems,
                runtime_factory=runtime_factory,
                judge_factory=judge_factory,
                out_dir=out_dir,
                suite=suite,
                resume=resume_dir is not None,
                note=args.note,
                progress=lambda line: print(line, flush=True),
            )
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\ninterrupted; continue with --resume {out_dir}", file=sys.stderr)
        return 130

    markdown = (result.out_dir / RESULTS_MD).read_text(encoding="utf-8")
    print()
    print(markdown.split("\n## By mode", 1)[0].split("\n## Per question", 1)[0].rstrip())
    print(f"\nresults written to {result.out_dir}")
    return 0


def _default_dir_name(suite: str) -> str:
    slug = Path(suite).stem if suite.endswith(".jsonl") else suite
    prefix = "offline" if suite == OFFLINE_SUITE else "benchmark"
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{slug}-{stamp}" if prefix != slug else f"{prefix}-{stamp}"


def _table(directory: Path) -> int:
    meta, rows = load_results_dir(directory)
    if not meta or not rows:
        print(f"error: no results in {directory}", file=sys.stderr)
        return 2
    write_outputs(directory, meta, rows)
    print(f"rebuilt {directory / RESULTS_MD} from {len(rows)} row(s)")
    return 0


def agreement(path: Path) -> dict[str, Any]:
    """
    How often a human's stance matches the judge's, over the labelled items of
    a spot-check file. ``support_agreement`` collapses the stances to
    supports / does not support, which is what citation precision counts.
    """
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labelled = [i for i in items if i.get("human_stance") in STANCES]
    invalid = [i["id"] for i in items if i.get("human_stance") not in (None, "", *STANCES)]

    def share(flags: list[bool]) -> float | None:
        return sum(flags) / len(flags) if flags else None

    by_system: dict[str, list[bool]] = {}
    for item in labelled:
        by_system.setdefault(item.get("system", "?"), []).append(item["human_stance"] == item["judge_stance"])
    return {
        "items": len(items),
        "labelled": len(labelled),
        "invalid": invalid,
        "agreement": share([i["human_stance"] == i["judge_stance"] for i in labelled]),
        "support_agreement": share(
            [(i["human_stance"] == "supports") == (i["judge_stance"] == "supports") for i in labelled]
        ),
        "by_system": {s: share(flags) for s, flags in by_system.items()},
        "confusion": {
            f"judge {judge} / human {human}": n
            for (judge, human), n in sorted(
                Counter((i["judge_stance"], i["human_stance"]) for i in labelled).items()
            )
        },
    }


def _agreement(path: Path) -> int:
    if path.is_dir():
        path = path / "spot_check.jsonl"
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    result = agreement(path)
    if result["invalid"]:
        print(
            f"warning: ignoring items with an unknown human_stance (use {', '.join(STANCES)}): "
            + ", ".join(result["invalid"]),
            file=sys.stderr,
        )
    if not result["labelled"]:
        print(f"no labelled items yet: set human_stance on the lines of {path}")
        return 1

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.0%}"

    print(f"labelled: {result['labelled']} of {result['items']}")
    print(f"stance agreement:  {pct(result['agreement'])}")
    print(f"support agreement: {pct(result['support_agreement'])} (supports vs not)")
    for system, value in result["by_system"].items():
        print(f"  {system:14} {pct(value)}")
    for pair, count in result["confusion"].items():
        print(f"  {pair}: {count}")
    return 0
