import { useEffect, useRef, useState } from "react";

import { ApiError, api } from "../api";
import { useLoad, useNow, useRunEvents } from "../hooks";
import { FANOUT, agentName, clock, compact, describeEvent, duration, isActive, modeName, shortId } from "../lib";
import type { RunDetail, RunEvent, StepProgress, UsageResponse } from "../types";
import { ErrorBanner, Icon, Meter, Spinner, StatusChip } from "../ui";
import { GraphView } from "./GraphView";
import { ReportView } from "./ReportView";
import { SourcesView } from "./SourcesView";

type View = "progress" | "report" | "graph" | "sources";

export function RunPage({ id, view }: { id: string; view: View }) {
  const detail = useLoad(() => api.detail(id), [id], {
    intervalMs: 5000,
    poll: (d) => !d || isActive(d.status),
  });
  // Refresh the details as events arrive, at most once a second.
  const pending = useRef<number>();
  const { events, state } = useRunEvents(id, () => {
    if (pending.current) return;
    pending.current = window.setTimeout(() => {
      pending.current = undefined;
      void detail.reload();
    }, 1000);
  });
  useEffect(() => () => window.clearTimeout(pending.current), []);

  if (detail.error && !detail.data) {
    return (
      <main className="container page-body" style={{ paddingTop: 32 }}>
        {detail.error.status === 404 ? (
          <div className="empty">
            <strong style={{ color: "#E7EBF1" }}>No such run</strong>
            <span>It may have been deleted.</span>
            <a href="#/runs">All runs</a>
          </div>
        ) : (
          <ErrorBanner error={detail.error} onRetry={detail.reload} />
        )}
      </main>
    );
  }
  if (!detail.data) {
    return (
      <main className="container page-body" style={{ paddingTop: 32 }}>
        <Spinner label="Loading run" />
      </main>
    );
  }

  const run = detail.data;
  return (
    <>
      <RunHeader run={run} view={view} onChange={detail.reload} />
      {view === "progress" ? <ProgressView run={run} events={events} streamState={state} /> : null}
      {view === "report" ? <ReportView run={run} /> : null}
      {view === "graph" ? <GraphView run={run} /> : null}
      {view === "sources" ? <SourcesView run={run} /> : null}
    </>
  );
}

function RunHeader({ run, view, onChange }: { run: RunDetail; view: View; onChange: () => void }) {
  const active = isActive(run.status);
  const now = useNow(1000, active);
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [copied, setCopied] = useState(false);
  const verified = run.mode !== "quick";
  const base = `#/runs/${run.run_id}`;

  const act = async (action: "cancel" | "resume") => {
    setBusy(true);
    setError(null);
    try {
      if (action === "cancel") await api.cancel(run.run_id);
      else await api.resume(run.run_id);
      setConfirming(false);
      onChange();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, "error", String(err)));
    } finally {
      setBusy(false);
    }
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard blocked: nothing to do.
    }
  };

  return (
    <div className={`container${view === "graph" ? " wide" : ""}`}>
      <div className="page-head">
        <nav aria-label="Breadcrumb" className="crumbs">
          <a href="#/runs">Runs</a>
          <span aria-hidden="true">/</span>
          <span className="mono">{shortId(run.run_id)}</span>
        </nav>
        <div className="row between gap-16" style={{ alignItems: "flex-start" }}>
          <div className="stack gap-12" style={{ flex: "999 1 560px", minWidth: 0 }}>
            <h1 className="title">{run.question}</h1>
            <div className="row gap-8">
              <span className="chip">{modeName(run.mode)} mode</span>
              <StatusChip status={run.status} />
              {run.attempts > 1 ? (
                <span className="chip" style={{ background: "rgba(185,162,255,0.12)", color: "#B9A2FF" }}>
                  <Icon name="resume" size={14} /> Resumed · attempt {run.attempts}
                </span>
              ) : null}
              {verified && run.status !== "queued" ? (
                <span className="chip">
                  Round {run.iteration + 1}
                  {active ? " of up to 3" : ""}
                </span>
              ) : null}
              {run.started_at ? (
                <span className="mono muted" style={{ fontSize: 12 }}>
                  {duration(run.started_at, run.finished_at, now)}
                  {active ? " elapsed" : ""}
                </span>
              ) : null}
            </div>
          </div>
          <div className="row gap-8">
            <button type="button" className="btn" onClick={copy}>
              <Icon name="link" /> {copied ? "Copied" : "Copy link"}
            </button>
            {active ? (
              <button type="button" className="btn danger" onClick={() => setConfirming(true)} disabled={busy}>
                <Icon name="stop" /> Cancel run
              </button>
            ) : null}
            {run.status === "failed" ? (
              <button type="button" className="btn" onClick={() => act("resume")} disabled={busy}>
                <Icon name="resume" /> Resume
              </button>
            ) : null}
          </div>
        </div>

        {run.status === "failed" && run.error ? (
          <div className="banner error" role="status">
            <div className="stack gap-4">
              <strong>The run failed</strong>
              <span className="soft" style={{ fontSize: 13 }}>
                {run.error} Resume continues from the last finished step.
              </span>
            </div>
          </div>
        ) : null}

        {confirming ? (
          <div role="alertdialog" aria-labelledby="cancel-q" className="banner error">
            <div className="stack gap-4">
              <strong id="cancel-q">Cancel this run?</strong>
              <span className="soft" style={{ fontSize: 13 }}>
                Its worker stops within moments. The {run.progress.sources} sources found so far stay stored; no report is written.
              </span>
            </div>
            <div className="row gap-8">
              <button type="button" className="btn" onClick={() => setConfirming(false)}>
                Keep running
              </button>
              <button type="button" className="btn danger-solid" onClick={() => act("cancel")} disabled={busy}>
                Cancel run
              </button>
            </div>
          </div>
        ) : null}
        {error ? <ErrorBanner error={error} /> : null}

        <nav aria-label="Run views" className="tabs">
          <a href={base} aria-current={view === "progress" ? "page" : undefined}>
            Progress
          </a>
          <a href={`${base}/report`} aria-current={view === "report" ? "page" : undefined}>
            Report
            {!run.report_available ? <span className="badge">{active ? "not ready" : "none"}</span> : null}
          </a>
          <a href={`${base}/graph`} aria-current={view === "graph" ? "page" : undefined}>
            Evidence graph
          </a>
          <a href={`${base}/sources`} aria-current={view === "sources" ? "page" : undefined}>
            Sources &amp; usage
          </a>
        </nav>
      </div>
    </div>
  );
}

function stepDetail(step: StepProgress, active: boolean): string {
  const branches = FANOUT.has(step.agent) && step.started > 1;
  const errors = step.errors ? ` · ${step.errors} warning${step.errors === 1 ? "" : "s"}` : "";
  switch (step.status) {
    case "completed":
      return `Done${branches ? ` · ${step.finished} branches` : ""}${errors}`;
    case "running":
      return `Running${branches ? ` · ${step.finished} of ${step.started} branches` : ""}${errors}`;
    case "incomplete":
      return `Stopped${branches ? ` · ${step.finished} of ${step.started} branches` : ""}${errors}`;
    case "skipped":
      return "Skipped";
    default:
      return active ? "Waiting" : "Not run";
  }
}

function ProgressView({ run, events, streamState }: { run: RunDetail; events: RunEvent[]; streamState: string }) {
  const active = isActive(run.status);
  const usage = useLoad(() => api.usage(run.run_id), [run.run_id, run.status], {
    intervalMs: 5000,
    poll: () => active,
  });
  const list = useRef<HTMLOListElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = list.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [events.length]);

  const subq = run.progress.sub_questions;
  return (
    <main className="container page-body">
      <div className="cols" style={{ gap: 24 }}>
        <section aria-labelledby="pipe-title" className="card pad-sm" style={{ flex: "1 1 280px", padding: 20 }}>
          <h2 id="pipe-title" className="caps" style={{ marginBottom: 18 }}>
            Pipeline{run.mode !== "quick" ? ` · round ${run.iteration + 1}` : ""}
          </h2>
          {run.status === "queued" ? (
            <p className="soft">Waiting for a worker to pick this run up.</p>
          ) : (
            <ol className="stepper">
              {run.progress.steps.map((step, index) => (
                <li key={step.agent} className="step" data-status={step.status}>
                  <span className="rail">
                    <span className="node">
                      {step.status === "completed" ? (
                        <span style={{ color: "#6FD3A0", display: "flex" }}>
                          <Icon name="check" size={14} stroke={3} />
                        </span>
                      ) : step.status === "running" ? (
                        <span className="dot live" style={{ background: "#3CCFC0" }} />
                      ) : null}
                    </span>
                    {index < run.progress.steps.length - 1 ? <span className="line" aria-hidden="true" /> : null}
                  </span>
                  <span className="body">
                    <span className="name">{agentName(step.agent)}</span>
                    <span className="detail">{stepDetail(step, active)}</span>
                  </span>
                </li>
              ))}
            </ol>
          )}
        </section>

        <div className="stack gap-24" style={{ flex: "999 1 640px", minWidth: 0 }}>
          <section aria-label="Run totals" className="tiles">
            <div className="tile">
              <span className="caps">Sources</span>
              <span className="value">{run.progress.sources}</span>
              <span className="sub">stored, scored for reliability</span>
            </div>
            <div className="tile">
              <span className="caps">Passages</span>
              <span className="value">{run.progress.passages}</span>
              <span className="sub">embedded and searchable</span>
            </div>
            <div className="tile">
              <span className="caps">Sub-questions</span>
              <span className="value">
                {subq.completed ?? 0}
                <span className="muted" style={{ fontSize: 14 }}> / {subq.total ?? 0}</span>
              </span>
              <span className="sub">researched</span>
            </div>
            <BudgetTile usage={usage.data} />
          </section>

          <section aria-labelledby="sq-title" className="stack gap-12">
            <h2 id="sq-title" style={{ fontSize: 16, lineHeight: "24px", fontWeight: 600 }}>
              Sub-questions
            </h2>
            {run.sub_questions.length ? (
              <div className="table-wrap">
                <table className="table" style={{ minWidth: 560 }}>
                  <thead>
                    <tr>
                      <th scope="col" style={{ width: 32 }}>#</th>
                      <th scope="col">Sub-question</th>
                      <th scope="col">Researcher</th>
                      <th scope="col">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.sub_questions.map((sq, i) => (
                      <tr key={sq.id}>
                        <td className="mono muted">{i + 1}</td>
                        <td>{sq.text}</td>
                        <td className="soft" style={{ textTransform: "capitalize" }}>{sq.domain}</td>
                        <td style={{ color: sq.status === "completed" ? "#6FD3A0" : sq.status === "researching" ? "#3CCFC0" : sq.status === "failed" ? "#FF8A7A" : "#AAB3C0", textTransform: "capitalize" }}>
                          {sq.status}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="soft">The planner has not written its plan yet.</p>
            )}
          </section>

          <section aria-labelledby="events-title" className="stack gap-12">
            <div className="row between gap-8">
              <h2 id="events-title" style={{ fontSize: 16, lineHeight: "24px", fontWeight: 600 }}>
                {active ? "Live events" : "Events"}
              </h2>
              <span className="row gap-8 muted" style={{ fontSize: 12 }}>
                {streamState === "live" && active ? <span className="dot live" style={{ background: "#3CCFC0" }} aria-hidden="true" /> : null}
                {streamState === "retrying" ? "Reconnecting…" : active ? "Streaming" : `${events.length} events`}
                {events.length ? (
                  <span className="mono soft">last #{events[events.length - 1].seq}</span>
                ) : null}
              </span>
            </div>
            <ol
              ref={list}
              className="events"
              aria-label="Run events, oldest first"
              aria-live="off"
              onScroll={(e) => {
                const el = e.currentTarget;
                stick.current = el.scrollTop + el.clientHeight >= el.scrollHeight - 24;
              }}
            >
              {events.length === 0 ? <li className="muted">No events yet.</li> : null}
              {events.map((event) => {
                const what = describeEvent(event);
                return (
                  <li key={event.seq}>
                    <span className="seq">#{event.seq}</span>
                    <span className="ts">{clock(event.ts)}</span>
                    <span className="agent">{event.agent}</span>
                    <span className="type" style={{ color: what.color }}>{event.type}</span>
                    <span className="what">{what.text}</span>
                  </li>
                );
              })}
            </ol>
          </section>
        </div>
      </div>
    </main>
  );
}

function BudgetTile({ usage }: { usage?: UsageResponse }) {
  const used = usage ? usage.totals.tokens_in + usage.totals.tokens_out : 0;
  const max = usage?.limits.max_tokens_per_run ?? 0;
  return (
    <div className="tile" style={{ flexBasis: 200 }}>
      <span className="caps">Token budget</span>
      <span className="value">
        {compact(used)}
        {max ? <span className="muted" style={{ fontSize: 14 }}> / {compact(max)}</span> : null}
      </span>
      {max ? <Meter value={used / max} label={`${Math.round((used / max) * 100)} percent of the token budget used`} /> : <span className="sub">no cap</span>}
      <span className="sub">{usage ? `${usage.totals.calls} model calls` : " "}</span>
    </div>
  );
}
