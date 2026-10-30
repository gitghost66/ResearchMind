"""
M7 acceptance: one command produces the comparison table.

``python -m researchmind_agents.evals run`` on the offline suite exercises the
whole harness with no network or keys: three systems, the judge, the metrics,
the tables, the spot-check sample, and resuming an interrupted run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from researchmind_agents.config import AgentConfig
from researchmind_agents.evals import cli
from researchmind_agents.evals.cli import ConfigError, agreement, load_config, main
from researchmind_agents.evals.offline import offline_runtime_factory
from researchmind_agents.evals.runner import load_results_dir
from researchmind_agents.evals.systems import SYSTEMS
from researchmind_agents.prompts.eval_judge import COVERAGE_PROMPT_VERSION, PROMPT_VERSION


def run(out: Path, *extra: str) -> int:
    return main(["run", "--suite", "offline", "--out", str(out), *extra])


def results(out: Path) -> dict[str, Any]:
    return json.loads((out / "results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bench") / "out"
    assert run(out) == 0
    return out


class TestOfflineRun:
    def test_writes_every_output(self, finished: Path) -> None:
        for name in ("meta.json", "details.jsonl", "results.json", "results.md", "spot_check.jsonl"):
            assert (finished / name).is_file(), name

    def test_markdown_has_the_comparison_table(self, finished: Path) -> None:
        markdown = (finished / "results.md").read_text(encoding="utf-8")
        assert "| Metric | Baseline LLM | Single-agent RAG | ResearchMind |" in markdown
        for heading in ("## Verdict", "## Comparison", "## By mode", "## Per question", "## Definitions"):
            assert heading in markdown
        for metric in ("Faithfulness", "Citation precision", "Hallucination rate", "Completeness",
                       "Precision@5", "Recall@10", "MRR", "Task completion", "Latency p50", "Tokens per run"):
            assert metric in markdown

    def test_one_row_per_question_and_system(self, finished: Path) -> None:
        data = results(finished)
        rows = data["rows"]
        assert len(rows) == 4 * len(SYSTEMS)
        assert {(r["question_id"], r["system"]) for r in rows} == {
            (q, s) for q in ("off-q1", "off-d1", "off-a1", "off-x1") for s in SYSTEMS
        }
        assert all("detail" not in r for r in rows), "results.json holds summaries only"
        assert all(r["completed"] and r["error"] is None for r in rows)

    def test_retrieval_metrics_apply_only_to_systems_that_retrieve(self, finished: Path) -> None:
        for row in results(finished)["rows"]:
            if row["system"] == "baseline_llm":
                assert row["retrieval"] is None
            else:
                assert row["retrieval"]["r@10"] > 0, row["question_id"]

    def test_judge_cost_is_kept_apart_from_the_systems(self, finished: Path) -> None:
        data = results(finished)
        meta = data["meta"]
        assert meta["judge_prompts"] == {"support": PROMPT_VERSION, "coverage": COVERAGE_PROMPT_VERSION}
        assert meta["judge_usage"]["calls"] > 0
        baseline = [r for r in data["rows"] if r["system"] == "baseline_llm"]
        # The closed-book baseline makes exactly one model call per question.
        assert {r["ops"]["calls"] for r in baseline} == {1}

    def test_verdict_on_the_scripted_world(self, finished: Path) -> None:
        # The offline world is scripted so that ResearchMind restates its
        # evidence and the baselines do not: this pins the harness, not the systems.
        verdicts = results(finished)["aggregates"]["verdicts"]
        assert verdicts["faithfulness"]["beats"] is True
        assert verdicts["citation_precision"]["beats"] is True
        assert verdicts["faithfulness"]["questions"] == 4
        baseline = results(finished)["aggregates"]["overall"]["baseline_llm"]
        assert baseline["fabricated_citation_rate"]["mean"] > 0, "the made-up URL answers 404"

    def test_details_keep_each_judged_sentence(self, finished: Path) -> None:
        _, rows = load_results_dir(finished)
        rag = next(r for r in rows if r["system"] == "rag")
        statements = rag["detail"]["statements"]
        assert statements and all("verdict" in s and "citations" in s for s in statements)
        assert any(c["excerpt"] for s in statements for c in s["citations"])
        assert rag["detail"]["retrieved"]

    def test_spot_check_sample_spans_the_systems_and_awaits_labels(self, finished: Path) -> None:
        items = [json.loads(line) for line in (finished / "spot_check.jsonl").read_text(encoding="utf-8").splitlines()]
        assert 0 < len(items) <= 30
        assert [i["id"] for i in items[:2]] == ["sc-001", "sc-002"]
        assert set(i["system"] for i in items) == set(SYSTEMS)
        assert all(i["human_stance"] is None and i["judge_stance"] for i in items)
        assert all(i["statement"] and i["excerpt"] for i in items)

    def test_prints_the_table(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        assert run(tmp_path / "out", "--questions", "off-q1") == 0
        printed = capsys.readouterr().out
        assert "## Comparison" in printed and "[3/3] off-q1 researchmind" in printed
        assert "## Per question" not in printed


class TestResume:
    def test_only_missing_pairs_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "out"
        assert run(out, "--systems", "baseline_llm,rag") == 0
        assert len(results(out)["rows"]) == 8

        built: list[str] = []

        def counting(config: AgentConfig) -> Any:
            built.append("runtime")
            return offline_runtime_factory(config)

        monkeypatch.setattr(cli, "offline_runtime_factory", counting)
        assert main(["run", "--resume", str(out)]) == 0, "the suite comes from meta.json"

        assert len(built) == 4, "only ResearchMind's four runs were missing"
        data = results(out)
        assert len(data["rows"]) == 12
        assert data["meta"]["systems"] == list(SYSTEMS)
        assert data["meta"]["judge_usage"]["calls"] > 0

    def test_existing_results_are_not_overwritten(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        assert run(out, "--limit", "1", "--systems", "baseline_llm") == 0
        assert run(out, "--limit", "1", "--systems", "baseline_llm") == 2

    def test_cannot_resume_with_another_model(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        assert run(out, "--limit", "1", "--systems", "baseline_llm") == 0
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        meta["model"] = "some-other-model"
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        assert main(["run", "--resume", str(out)]) == 2

    def test_resume_needs_a_results_directory(self, tmp_path: Path) -> None:
        assert main(["run", "--resume", str(tmp_path)]) == 2

    def test_a_partial_last_line_is_skipped(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        assert run(out, "--limit", "1", "--systems", "baseline_llm") == 0
        with (out / "details.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"question_id": "off-d1", "sys')
        _, rows = load_results_dir(out)
        assert len(rows) == 1


class TestFilters:
    def test_modes_questions_and_limit(self, tmp_path: Path) -> None:
        assert run(tmp_path / "a", "--modes", "deep,academic", "--systems", "baseline_llm") == 0
        assert {r["mode"] for r in results(tmp_path / "a")["rows"]} == {"deep", "academic"}
        assert run(tmp_path / "b", "--questions", "off-x1", "--systems", "rag") == 0
        assert [(r["question_id"], r["system"]) for r in results(tmp_path / "b")["rows"]] == [("off-x1", "rag")]
        assert run(tmp_path / "c", "--limit", "2", "--systems", "baseline_llm") == 0
        assert len(results(tmp_path / "c")["rows"]) == 2

    def test_nothing_left_to_run(self, tmp_path: Path) -> None:
        assert run(tmp_path / "out", "--questions", "nope") == 2

    def test_unknown_system_or_suite(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="unknown system"):
            run(tmp_path / "out", "--systems", "gpt5")
        assert main(["run", "--suite", "nope", "--out", str(tmp_path / "x")]) == 2


class TestTable:
    def test_rebuilds_the_markdown(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        assert run(out, "--limit", "1") == 0
        (out / "results.md").unlink()
        assert main(["table", str(out)]) == 0
        assert "## Comparison" in (out / "results.md").read_text(encoding="utf-8")

    def test_labelled_spot_checks_are_never_overwritten(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        assert run(out, "--limit", "1") == 0
        path = out / "spot_check.jsonl"
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        items[0]["human_stance"] = "supports"
        labelled = "".join(json.dumps(i) + "\n" for i in items)
        path.write_text(labelled, encoding="utf-8")
        assert main(["table", str(out)]) == 0
        assert path.read_text(encoding="utf-8") == labelled

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert main(["table", str(tmp_path)]) == 2


class TestAgreement:
    def write(self, tmp_path: Path, *pairs: tuple[str, str | None, str]) -> Path:
        path = tmp_path / "spot_check.jsonl"
        path.write_text(
            "".join(
                json.dumps({"id": f"sc-{i}", "system": system, "judge_stance": judge, "human_stance": human}) + "\n"
                for i, (judge, human, system) in enumerate(pairs, start=1)
            ),
            encoding="utf-8",
        )
        return path

    def test_scores_the_labelled_items(self, tmp_path: Path) -> None:
        path = self.write(
            tmp_path,
            ("supports", "supports", "rag"),
            ("supports", "neutral", "rag"),
            ("neutral", "contradicts", "researchmind"),
            ("neutral", None, "researchmind"),
        )
        result = agreement(path)
        assert (result["items"], result["labelled"]) == (4, 3)
        assert result["agreement"] == pytest.approx(1 / 3)
        # Supports-or-not: agree, disagree, agree.
        assert result["support_agreement"] == pytest.approx(2 / 3)
        assert result["by_system"] == {"rag": 0.5, "researchmind": 0.0}
        assert result["confusion"]["judge supports / human neutral"] == 1

    def test_command(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.write(tmp_path, ("supports", "supports", "rag"), ("neutral", "maybe", "rag"))
        assert main(["agreement", str(tmp_path)]) == 0
        captured = capsys.readouterr()
        assert "stance agreement:  100%" in captured.out
        assert "sc-2" in captured.err, "an unknown label is reported"

    def test_nothing_labelled_yet(self, tmp_path: Path) -> None:
        self.write(tmp_path, ("supports", None, "rag"))
        assert main(["agreement", str(tmp_path / "spot_check.jsonl")]) == 1
        assert main(["agreement", str(tmp_path / "missing.jsonl")]) == 2


class TestConfig:
    def test_module_function(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "eval_cfg_provider.py").write_text(
            "from researchmind_agents.config import AgentConfig, LLMConfig\n"
            "def make():\n    return AgentConfig(llm=LLMConfig(model='from-module'))\n"
            "def wrong():\n    return {'llm_model': 'x'}\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        assert load_config("eval_cfg_provider:make").llm.model == "from-module"
        with pytest.raises(ConfigError, match="not AgentConfig"):
            load_config("eval_cfg_provider:wrong")
        with pytest.raises(ConfigError, match="could not load"):
            load_config("eval_cfg_provider:missing")
        sys.modules.pop("eval_cfg_provider", None)

    @pytest.mark.parametrize("spec", ["no-colon", ":make", "module:"])
    def test_malformed_spec(self, spec: str) -> None:
        with pytest.raises(ConfigError, match="module:function"):
            load_config(spec)

    def test_auto_without_a_provider_explains_what_to_do(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "entry_points", lambda group: [])
        with pytest.raises(ConfigError, match="pip install -e backend"):
            load_config("auto")

    def test_a_live_run_without_keys_stops_before_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(cli, "load_config", lambda spec: AgentConfig())
        assert main(["run", "--suite", "default", "--limit", "1", "--out", str(tmp_path / "out")]) == 2
        assert "GEMINI_API_KEY" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()
