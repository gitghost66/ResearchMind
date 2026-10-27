"""
Optional LangSmith tracing.

LangGraph traces itself when LangSmith's environment variables are set; the
graph already names each run's trace and tags it with ``run_id``
(``researchmind_agents.graph.build_run_config``), so a run is one trace with a
span per node. This only exports ``Settings`` into those variables, because
values read from ``.env`` are not in the process environment. Variables
already set in the real environment win.
"""

from __future__ import annotations

import logging
import os

from app.core.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["configure_tracing"]


def configure_tracing(settings: Settings) -> bool:
    """Export the LangSmith settings; returns whether tracing is on."""
    if not settings.langsmith_tracing:
        return False
    if not settings.langsmith_api_key and not os.environ.get("LANGSMITH_API_KEY"):
        logger.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty; tracing stays off")
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_project:
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
    logger.info("LangSmith tracing enabled", extra={"project": settings.langsmith_project})
    return True
