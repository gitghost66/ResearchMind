import { useState } from "react";

import { ApiError, api } from "../api";
import { go, useLoad, useNow } from "../hooks";
import { MODES, isActive, modeName, since } from "../lib";
import type { RunSummary } from "../types";
import { ErrorBanner, Icon, Spinner, StatusText } from "../ui";

const MAX_CHARS = 2000;

export function NewResearch() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("deep");
  const [situation, setSituation] = useState("");
  const [maxSources, setMaxSources] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const current = MODES.find((m) => m.id === mode) ?? MODES[1];

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const created = await api.create({
        question: question.trim(),
        mode,
        ...(mode === "decision" && situation.trim() ? { user_context: { situation: situation.trim() } } : {}),
        ...(maxSources ? { max_sources: Number(maxSources) } : {}),
      });
      go(`/runs/${created.run_id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, "error", String(err)));
      setBusy(false);
    }
  };

  const fieldErrors = error?.status === 422 && Array.isArray(error.details) ? (error.details as { loc: string[]; msg: string }[]) : [];
  const errorFor = (field: string) => fieldErrors.find((d) => d.loc?.includes(field))?.msg;

  return (
    <main className="container page-body" style={{ paddingTop: 40 }}>
      <div className="cols" style={{ gap: 32 }}>
        <section aria-labelledby="new-title" className="col-main stack gap-24">
          <div className="stack gap-8">
            <h1 id="new-title" style={{ fontSize: 30, lineHeight: "38px", fontWeight: 600, letterSpacing: "-0.015em" }}>
              What do you want to know?
            </h1>
            <p className="soft" style={{ fontSize: 15, lineHeight: "24px", maxWidth: 680 }}>
              Agents split your question into sub-questions, search the web and papers in parallel, check every claim against the passage it came from, and write a report in which each statement cites a stored source.
            </p>
          </div>

          <form className="card pad stack gap-24" onSubmit={submit} noValidate>
            <div className="field">
              <div className="row between gap-12">
                <label htmlFor="question">Question</label>
                <span className="mono muted" style={{ fontSize: 12 }}>
                  {question.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}
                </span>
              </div>
              <textarea
                id="question"
                className="textarea"
                rows={3}
                maxLength={MAX_CHARS}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void submit(e);
                }}
                placeholder="e.g. PostgreSQL or MongoDB for an AI SaaS in 2026?"
                style={{ fontSize: 17, lineHeight: "26px" }}
                aria-invalid={errorFor("question") ? true : undefined}
                aria-describedby={errorFor("question") ? "question-error" : undefined}
              />
              {errorFor("question") ? (
                <span id="question-error" className="hint" style={{ color: "#FF8A7A" }}>
                  {errorFor("question")}
                </span>
              ) : null}
            </div>

            <fieldset style={{ margin: 0, padding: 0, border: 0, minWidth: 0 }}>
              <legend style={{ padding: 0, marginBottom: 10, fontWeight: 500 }}>Mode</legend>
              <div className="mode-grid">
                {MODES.map((m) => (
                  <button key={m.id} type="button" className="mode-card" aria-pressed={m.id === mode} onClick={() => setMode(m.id)}>
                    <span className="row between gap-8">
                      <span className="name">{m.name}</span>
                      <span className="row gap-6 mono soft" style={{ fontSize: 12 }}>
                        {m.time}
                        {m.id === mode ? (
                          <span style={{ color: "#3CCFC0" }}>
                            <Icon name="check" size={18} stroke={2.4} />
                          </span>
                        ) : null}
                      </span>
                    </span>
                    <span className="desc">{m.desc}</span>
                  </button>
                ))}
              </div>
            </fieldset>

            {mode === "decision" ? (
              <div className="field">
                <label htmlFor="situation">
                  Your situation <span className="muted" style={{ fontWeight: 400 }}>(optional)</span>
                </label>
                <textarea id="situation" className="textarea" rows={2} value={situation} onChange={(e) => setSituation(e.target.value)} placeholder="Team, budget, deadline, constraints…" />
                <span className="hint">The recommendation is weighed against this. It is sent as the run's user context.</span>
              </div>
            ) : null}

            <div className="row gap-16" style={{ alignItems: "flex-start" }}>
              <div className="field" style={{ flex: "0 1 180px" }}>
                <label htmlFor="max-sources">Max sources</label>
                <input id="max-sources" className="input mono" type="number" min={1} max={50} inputMode="numeric" placeholder="No cap" value={maxSources} onChange={(e) => setMaxSources(e.target.value)} />
                {errorFor("max_sources") ? (
                  <span className="hint" style={{ color: "#FF8A7A" }}>
                    {errorFor("max_sources")}
                  </span>
                ) : null}
              </div>
              <div className="field" style={{ flex: "1 1 220px" }}>
                <span className="label">Rounds</span>
                <span className="soft row" style={{ minHeight: 44 }}>
                  {mode === "quick" ? "One round" : "Up to 3 rounds; gaps start another"}
                </span>
              </div>
            </div>

            {error && error.status !== 422 ? <ErrorBanner error={error} /> : null}

            <div className="row gap-16" style={{ paddingTop: 20, borderTop: "1px solid #232934" }}>
              <button type="submit" className="btn primary" disabled={!question.trim() || busy}>
                {busy ? "Starting…" : "Start research"}
                <Icon name="arrow" size={18} stroke={2.2} />
              </button>
              <span className="hint">
                {current.name} · {current.time.replace("~", "about ")}. The run waits in a queue and keeps going if you close this page.
              </span>
            </div>
          </form>

          <section aria-labelledby="how-title" className="stack gap-12">
            <h2 id="how-title" className="caps">
              How a run works
            </h2>
            <ol className="steps5">
              <li><span className="n">01 Plan</span><span className="t">Four to eight sub-questions, each sent to the right researcher.</span></li>
              <li><span className="n">02 Research</span><span className="t">Web, technical, market, and paper search run in parallel.</span></li>
              <li><span className="n">03 Verify</span><span className="t">Every claim is checked against the passages it came from.</span></li>
              <li><span className="n">04 Reconcile</span><span className="t">Contradictions are explained; gaps start another round.</span></li>
              <li><span className="n">05 Report</span><span className="t">Findings with verdicts, a confidence score, and citations.</span></li>
            </ol>
          </section>
        </section>

        <aside aria-labelledby="recent-title" className="col-side stack gap-12" style={{ flexBasis: 340 }}>
          <div className="row between gap-12" style={{ alignItems: "baseline" }}>
            <h2 id="recent-title" style={{ fontSize: 16, lineHeight: "24px", fontWeight: 600 }}>
              Recent runs
            </h2>
            <a href="#/runs" style={{ fontSize: 13 }}>
              View all
            </a>
          </div>
          <RunList limit={8} compact />
        </aside>
      </div>
    </main>
  );
}

/** Runs, newest first, refreshed while any is active. */
export function RunList({ limit, compact = false, status = "" }: { limit: number; compact?: boolean; status?: string }) {
  const [offset, setOffset] = useState(0);
  const list = useLoad(() => api.list({ limit, offset, status }), [limit, offset, status], {
    intervalMs: 4000,
  });
  const now = useNow(30000);
  const [busyId, setBusyId] = useState<string | null>(null);

  if (list.loading && !list.data) return <Spinner label="Loading runs" />;
  if (list.error && !list.data) return <ErrorBanner error={list.error} onRetry={list.reload} />;
  const items = list.data?.items ?? [];
  if (!items.length) {
    return (
      <div className="empty">
        <strong style={{ color: "#E7EBF1" }}>No runs yet</strong>
        <span>Ask a question to start the first one.</span>
      </div>
    );
  }

  const act = async (run: RunSummary, action: "resume" | "remove") => {
    setBusyId(run.run_id);
    try {
      if (action === "resume") {
        await api.resume(run.run_id);
        go(`/runs/${run.run_id}`);
      } else {
        await api.remove(run.run_id);
        await list.reload();
      }
    } catch (err) {
      window.alert(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  };

  const total = list.data?.total ?? 0;
  return (
    <div className="stack gap-12">
      <ul className="runs">
        {items.map((run) => {
          const target = run.status === "completed" ? `#/runs/${run.run_id}/report` : `#/runs/${run.run_id}`;
          return (
            <li key={run.run_id} className="run-item">
              <a className="q" href={target}>
                {run.question}
              </a>
              <span className="run-meta">
                <span className="chip">{modeName(run.mode)}</span>
                <StatusText status={run.status} />
                <span>{since(run.created_at, now)}</span>
              </span>
              {run.status === "failed" && run.error ? (
                <span className="hint" style={{ fontSize: 12, lineHeight: "18px" }}>
                  {run.error.length > 140 ? `${run.error.slice(0, 140)}…` : run.error}
                </span>
              ) : null}
              {!compact || run.status === "failed" ? (
                <span className="row gap-8">
                  {run.status === "failed" ? (
                    <button type="button" className="btn small" disabled={busyId === run.run_id} onClick={() => act(run, "resume")}>
                      <Icon name="resume" /> Resume
                    </button>
                  ) : null}
                  {!compact && !isActive(run.status) ? (
                    <button
                      type="button"
                      className="btn small"
                      disabled={busyId === run.run_id}
                      onClick={() => {
                        if (window.confirm("Delete this run and everything it stored?")) void act(run, "remove");
                      }}
                    >
                      <Icon name="trash" /> Delete
                    </button>
                  ) : null}
                </span>
              ) : null}
            </li>
          );
        })}
      </ul>
      {!compact && total > limit ? (
        <div className="row between gap-12">
          <span className="hint">
            {offset + 1}–{Math.min(offset + limit, total)} of {total}
          </span>
          <span className="row gap-8">
            <button type="button" className="btn small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>
              Newer
            </button>
            <button type="button" className="btn small" disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)}>
              Older
            </button>
          </span>
        </div>
      ) : null}
    </div>
  );
}

export function RunsPage() {
  const [status, setStatus] = useState("");
  const filters = [
    ["", "All"],
    ["running", "Running"],
    ["queued", "Queued"],
    ["completed", "Completed"],
    ["failed", "Failed"],
    ["cancelled", "Cancelled"],
  ];
  return (
    <main className="container page-body" style={{ paddingTop: 32, maxWidth: 880 }}>
      <div className="stack gap-16">
        <div className="row between gap-12">
          <h1 className="title">Runs</h1>
          <a className="btn" href="#/">
            New research
          </a>
        </div>
        <div className="segmented" role="group" aria-label="Filter by status">
          {filters.map(([id, label]) => (
            <button key={id} type="button" className="seg" aria-pressed={status === id} onClick={() => setStatus(id)}>
              {label}
            </button>
          ))}
        </div>
        <RunList key={status} limit={20} status={status} />
      </div>
    </main>
  );
}
