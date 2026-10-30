"""
M7: the eval CLI in the agents package reaches the backend's settings through
an entry point, never an import (AGENTS.md rule 2). The declaration must name
the real config builder, and that builder must carry the eval settings.
"""

from __future__ import annotations

import tomllib
from importlib import import_module
from pathlib import Path

from app.core.config import Settings
from app.services.agent_runtime import build_agent_config

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_backend_registers_its_config_builder_for_the_eval_cli() -> None:
    from researchmind_agents.evals.cli import CONFIG_GROUP

    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["entry-points"]
    module, _, attr = declared[CONFIG_GROUP]["backend"].partition(":")
    assert getattr(import_module(module), attr) is build_agent_config


def test_eval_settings_reach_the_agent_config() -> None:
    config = build_agent_config(
        Settings(
            gemini_api_key="k",
            eval_judge_model="judge-model",
            eval_retrieval_k="3,1",
            eval_rag_pages=2,
            eval_max_statements=7,
        )
    )
    assert config.eval.judge_model == "judge-model"
    assert config.eval.retrieval_k == (1, 3)
    assert config.eval.rag_pages == 2
    assert config.eval.max_statements == 7
