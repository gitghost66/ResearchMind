"""
Markdown for benchmark results: the comparison table for the project report,
a breakdown by mode, one row per question, and the metric definitions.

Rendered from result rows alone (``runner.aggregate``), so ``python -m
researchmind_agents.evals table DIR`` can rebuild it from a results directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from researchmind_agents.evals.metrics import (
    ANSWER_METRICS,
    COVERAGE_METRICS,
    HEADLINE_METRICS,
    LOWER_IS_BETTER,
    metric_value,
)
from researchmind_agents.evals.systems import SYSTEM_NAMES

__all__ = ["render_results", "METRIC_LABELS"]

METRIC_LABELS: dict[str, str] = {
    "faithfulness": "Faithfulness",
    "citation_precision": "Citation precision",
    "fabricated_citation_rate": "Fabricated citations",
    "hallucination_rate": "Hallucination rate",
    "completeness": "Completeness",
    "relevance": "Relevance",
    "mrr": "MRR",
}

_DEFINITIONS = """\
- **Faithfulness**: share of answer sentences fully supported by the evidence they
  cite (an uncited sentence is not supported), averaged over answers.
- **Citation precision**: share of citations whose source supports the sentence citing it.
- **Fabricated citations**: share of citations to sources that do not exist (unknown
  label, not a URL, unresolvable host, HTTP 404/410).
- **Hallucination rate**: share of sentences contradicted by their own evidence, or
  citing only sources that do not exist.
- **Completeness**: share of the question's expected sub-topics the answer covers.
- **Relevance**: share of sentences that help answer the question.
- **Precision@K / Recall@K / MRR**: the system's stored sources, ranked by similarity
  to the question, against the question's hand-labelled gold sources.
- **Task completion**: runs that finished with at least one sentence written by the
  system's model (an extractive fallback does not count).
- Quality and retrieval means are over completed answers (n shown); the verdict uses
  only questions every system completed. Tokens and cost are the system's own calls;
  the judge's are reported separately."""


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _label(metric: str) -> str:
    if metric.startswith(("p@", "r@")):
        kind = "Precision" if metric.startswith("p@") else "Recall"
        return f"{kind}@{metric[2:]}"
    return METRIC_LABELS.get(metric, metric)


def _arrow(metric: str) -> str:
    return "↓" if metric in LOWER_IS_BETTER else "↑"


def _metric_row(metric: str, systems: Sequence[str], summaries: dict[str, Any]) -> str:
    cells = []
    values: dict[str, float] = {}
    for system in systems:
        entry = summaries[system].get(metric) or {}
        value, n = entry.get("mean"), entry.get("n", 0)
        if value is not None:
            values[system] = value
        text = _num(value) if metric == "mrr" else _pct(value)
        cells.append(text if value is None else f"{text} (n={n})")
    if len(values) > 1:
        best = (min if metric in LOWER_IS_BETTER else max)(values.values())
        cells = [
            f"**{cell}**" if values.get(system) == best else cell
            for system, cell in zip(systems, cells, strict=True)
        ]
    return f"| {_label(metric)} {_arrow(metric)} | " + " | ".join(cells) + " |"


def _ops_row(label: str, key: str, systems: Sequence[str], summaries: dict[str, Any], fmt: Any) -> str:
    return f"| {label} | " + " | ".join(fmt(summaries[s].get(key)) for s in systems) + " |"


def render_results(
    meta: dict[str, Any],
    aggregates: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> str:
    systems: list[str] = list(meta["systems"])
    ks: list[int] = list(meta["retrieval_k"])
    summaries = aggregates["overall"]
    names = [SYSTEM_NAMES.get(s, s) for s in systems]
    header = "| Metric | " + " | ".join(names) + " |"
    rule = "|---|" + "---|" * len(systems)

    lines = [
        f"# ResearchMind benchmark: `{meta['suite']}` suite",
        "",
        f"- Questions: {meta['questions']} ({', '.join(f'{m} {n}' for m, n in meta['modes'].items())})",
        f"- Model: `{meta['model']}` · judge: `{meta['judge_model']}` "
        f"(prompts `{meta['judge_prompts']['support']}`, `{meta['judge_prompts']['coverage']}`)",
        f"- Created: {meta['created']}" + (f" · {meta['note']}" if meta.get("note") else ""),
        "",
    ]

    verdicts = aggregates.get("verdicts") or {}
    if verdicts:
        lines += ["## Verdict", ""]
        for metric in HEADLINE_METRICS:
            verdict = verdicts.get(metric)
            if verdict is None:
                continue
            if not verdict["questions"]:
                lines.append(f"- {_label(metric)}: no question was completed by every system.")
                continue
            means = ", ".join(
                f"{SYSTEM_NAMES.get(s, s)} {_pct(v)}" for s, v in verdict["means"].items()
            )
            outcome = "beats both baselines" if verdict["beats"] else "does **not** beat both baselines"
            lines.append(
                f"- {_label(metric)}: ResearchMind {outcome} on the {verdict['questions']} "
                f"question(s) every system completed ({means})."
            )
        lines.append("")

    lines += ["## Comparison", "", header, rule]
    for metric in (*ANSWER_METRICS, *COVERAGE_METRICS):
        lines.append(_metric_row(metric, systems, summaries))
    for k in ks:
        lines.append(_metric_row(f"p@{k}", systems, summaries))
        lines.append(_metric_row(f"r@{k}", systems, summaries))
    lines.append(_metric_row("mrr", systems, summaries))
    lines.append(
        "| Task completion ↑ | "
        + " | ".join(
            f"{_pct(summaries[s]['task_completion'])} "
            f"({summaries[s]['completed']}/{summaries[s]['questions']})"
            for s in systems
        )
        + " |"
    )
    lines.append(_ops_row("Research rounds per run", "iterations", systems, summaries, lambda v: _num(v, 1)))
    lines.append(_ops_row("Latency p50 (s)", "latency_p50_s", systems, summaries, lambda v: _num(v, 1)))
    lines.append(_ops_row("Latency mean (s)", "latency_mean_s", systems, summaries, lambda v: _num(v, 1)))
    lines.append(_ops_row("Tokens per run", "tokens_mean", systems, summaries, lambda v: _num(v, 0)))
    lines.append(_ops_row("Cost per run (USD)", "cost_mean_usd", systems, summaries, lambda v: _num(v, 4)))
    lines.append("")

    by_mode = aggregates.get("by_mode") or {}
    if len(by_mode) > 1:
        lines += ["## By mode", "", "| Mode | Metric | " + " | ".join(names) + " |"]
        lines.append("|---|---|" + "---|" * len(systems))
        for mode, mode_summaries in by_mode.items():
            for metric in (*HEADLINE_METRICS, "completeness"):
                cells = []
                for system in systems:
                    entry = (mode_summaries.get(system) or {}).get(metric) or {}
                    cells.append(_pct(entry.get("mean")))
                lines.append(f"| {mode} | {_label(metric)} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Per question",
        "",
        "Faithfulness / citation precision / completeness for each system.",
        "",
        "| Question | Mode | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(systems),
    ]
    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    modes: dict[str, str] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], {})[row["system"]] = row
        modes[row["question_id"]] = row["mode"]
    for question_id, by_system in by_question.items():
        cells = []
        for system in systems:
            entry = by_system.get(system)
            if entry is None:
                cells.append("")
            elif entry.get("error"):
                cells.append("error")
            elif not entry.get("completed"):
                cells.append(f"not completed ({entry.get('generation') or ''})".replace(" ()", ""))
            else:
                cells.append(
                    " / ".join(
                        _pct(metric_value(entry, m))
                        for m in ("faithfulness", "citation_precision", "completeness")
                    )
                )
        lines.append(f"| {question_id} | {modes[question_id]} | " + " | ".join(cells) + " |")
    lines.append("")

    judge = meta.get("judge_usage") or {}
    if judge:
        lines += [
            "## Judge usage",
            "",
            f"{judge.get('calls', 0)} call(s), {judge.get('tokens_used', 0)} tokens, "
            f"${float(judge.get('cost_usd', 0.0)):.4f}.",
            "",
        ]
    lines += ["## Definitions", "", _DEFINITIONS, ""]
    return "\n".join(lines)
