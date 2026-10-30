# syntax=docker/dockerfile:1
#
# ResearchMind image: the API, the research worker, and migrations (M8).
#
#   docker build -f backend/Dockerfile -t researchmind .              # production (default)
#   docker build -f backend/Dockerfile --target dev -t researchmind:dev .
#
# One production image runs every role; the command picks it:
#   api      uvicorn app.main:app                 (the default CMD)
#   worker   python -m app.workers.research_worker
#   migrate  alembic upgrade head
#
# Build from the repository root: the image needs both packages.

ARG PYTHON_VERSION=3.13

# ---------------------------------------------------------------------------
# build: wheels for both packages and all their dependencies
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
COPY agents/pyproject.toml agents/pyproject.toml
COPY agents/researchmind_agents agents/researchmind_agents
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/app backend/app

# Both projects in one call, so the backend's "researchmind-agents"
# requirement resolves to the local package.
RUN pip wheel --wheel-dir /wheels ./agents ./backend

# ---------------------------------------------------------------------------
# dev: editable installs; docker-compose.yml mounts the source and reloads
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS dev

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY agents /app/agents
COPY backend /app/backend
RUN pip install -e /app/agents && pip install -e /app/backend

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--timeout-graceful-shutdown", "5"]

# ---------------------------------------------------------------------------
# prod (default): installed wheels only, no compiler, non-root
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS prod

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 researchmind

COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links=/wheels researchmind-backend \
    && rm -rf /wheels

# Migrations and scripts run from here; the app itself is installed.
WORKDIR /app/backend
COPY --chown=researchmind backend/alembic.ini ./alembic.ini
COPY --chown=researchmind backend/alembic ./alembic
COPY --chown=researchmind backend/scripts ./scripts

USER researchmind
EXPOSE 8000

# uvicorn reads WEB_CONCURRENCY (processes; runs live in the database, so
# several are fine) and FORWARDED_ALLOW_IPS (trust the proxy's X-Forwarded-For).
# SSE streams stay open until their run ends, so shutdown gives them 5 s
# before closing them; the lifespan then hands this process's runs back.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "5"]
