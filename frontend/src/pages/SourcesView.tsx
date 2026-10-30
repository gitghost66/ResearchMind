import { useMemo, useState } from "react";

import { api } from "../api";
import { useLoad } from "../hooks";
import { agentName, compact, domainOf, humanType, isActive, reliabilityColor } from "../lib";
import type { RunDetail, SourceOut } from "../types";
import { ErrorBanner, Meter, Spinner } from "../ui";

type Sort = "reliability" | "published" | "passages";

export function SourcesView({ run }: { run: RunDetail }) {
  const active = isActive(run.status);
  const sources = useLoad(() => api.sources(run.run_id), [run.run_id], { intervalMs: 6000, poll: () => active });
  const usage = useLoad(() => api.usage(run.run_id), [run.run_id], { intervalMs: 6000, poll: () => active });
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort>("reliability");

  const items = sources.data?.items ?? [];
  const types = useMemo(() => Array.from(new Set(items.map((s) => s.domain_type))).sort(), [items]);
  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const keep = (s: SourceOut) =>
      (filter === "all" || (filter === "cited" ? s.cited : s.domain_type === filter)) &&
      (!q || s.title.toLowerCase().includes(q) || s.url.toLowerCase().includes(q));
    const key = (s: SourceOut) =>
      sort === "reliability" ? s.reliability_score : sort === "passages" ? s.passages : s.published_at ? Date.parse(s.published_at) : 0;
    return items.filter(keep).sort((a, b) => key(b) - key(a));
  }, [items, filter, query, sort]);

  const totals = usage.data?.totals;
  const tokens = totals ? totals.tokens_in + totals.tokens_out : 0;
  const maxTokens = usage.data?.limits.max_tokens_per_run ?? 0;
  const agents = Object.entries(usage.data?.per_agent ?? {})
    .map(([name, b]) => ({ name, calls: b.calls, tokens: b.tokens_in + b.tokens_out }))
    .sort((a, b) => b.tokens - a.tokens);
  const topTokens = Math.max(1, ...agents.map((a) => a.tokens));
  const models = Object.keys(usage.data?.per_model ?? {});

  return (
    <main className="container page-body stack gap-40">
      <section aria-labelledby="sources-title" className="stack gap-12">
        <div className="row between gap-8" style={{ alignItems: "baseline" }}>
          <h2 id="sources-title" className="section-title">
            Sources
          </h2>
          <span className="hint">
            {items.length} stored · {items.filter((s) => s.cited).length} cited · scored by domain type, recency, and methodology
          </span>
        </div>
        {sources.error && !sources.data ? <ErrorBanner error={sources.error} onRetry={sources.reload} /> : null}
        {!sources.data && !sources.error ? <Spinner label="Loading sources" /> : null}
        {sources.data ? (
          <>
            <div className="row gap-12">
              <div className="segmented" role="group" aria-label="Filter sources" style={{ flex: "1 1 360px" }}>
                <button type="button" className="seg" aria-pressed={filter === "all"} onClick={() => setFilter("all")}>
                  All {items.length}
                </button>
                <button type="button" className="seg" aria-pressed={filter === "cited"} onClick={() => setFilter("cited")}>
                  Cited {items.filter((s) => s.cited).length}
                </button>
                {types.map((t) => (
                  <button key={t} type="button" className="seg" aria-pressed={filter === t} onClick={() => setFilter(t)}>
                    {humanType(t)} {items.filter((s) => s.domain_type === t).length}
                  </button>
                ))}
              </div>
              <label className="sr-only" htmlFor="source-q">
                Filter by title or domain
              </label>
              <input id="source-q" className="input" type="search" placeholder="Filter by title or domain" value={query} onChange={(e) => setQuery(e.target.value)} style={{ flex: "0 1 260px", height: 40 }} />
              <label className="row gap-8 soft" style={{ fontSize: 13 }}>
                Sort
                <select className="select" value={sort} onChange={(e) => setSort(e.target.value as Sort)} style={{ height: 40, width: "auto" }}>
                  <option value="reliability">Reliability</option>
                  <option value="published">Published</option>
                  <option value="passages">Passages</option>
                </select>
              </label>
            </div>
            {rows.length ? (
              <div className="table-wrap">
                <table className="table" style={{ minWidth: 820 }}>
                  <thead>
                    <tr>
                      <th scope="col">Source</th>
                      <th scope="col">Type</th>
                      <th scope="col" style={{ width: 180 }}>Reliability</th>
                      <th scope="col">Published</th>
                      <th scope="col" className="num">Passages</th>
                      <th scope="col">Cited</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((s) => (
                      <tr key={s.id}>
                        <td>
                          <span className="stack">
                            <a href={s.url} target="_blank" rel="noreferrer" style={{ color: "#E7EBF1", fontWeight: 500, textDecoration: "none", overflowWrap: "anywhere" }}>
                              {s.title || domainOf(s.url)}
                            </a>
                            <span className="muted" style={{ fontSize: 12 }}>{domainOf(s.url)}</span>
                          </span>
                        </td>
                        <td>
                          <span className="chip" style={{ textTransform: "capitalize" }}>{humanType(s.domain_type)}</span>
                        </td>
                        <td>
                          <span className="row gap-10 mono" style={{ fontSize: 12, flexWrap: "nowrap" }}>
                            <Meter value={s.reliability_score} color={reliabilityColor(s.reliability_score)} />
                            {s.reliability_score.toFixed(2)}
                          </span>
                        </td>
                        <td className="mono soft">{s.published_at ? s.published_at.slice(0, 10) : "—"}</td>
                        <td className="num">{s.passages}</td>
                        <td style={{ color: s.cited ? "#7FE3D8" : "#8892A1" }}>{s.cited ? "Cited" : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty">{items.length ? "No source matches the filter." : "No sources stored yet."}</div>
            )}
          </>
        ) : null}
      </section>

      <section aria-labelledby="usage-title" className="stack gap-16">
        <div className="row between gap-8" style={{ alignItems: "baseline" }}>
          <h2 id="usage-title" className="section-title">
            Usage
          </h2>
          <span className="hint">{models.join(", ") || " "}{run.attempts > 1 ? " · all attempts" : ""}</span>
        </div>
        {usage.error && !usage.data ? <ErrorBanner error={usage.error} onRetry={usage.reload} /> : null}
        {totals ? (
          <>
            <div className="tiles">
              <div className="tile">
                <span className="caps">Model calls</span>
                <span className="value">{totals.calls}</span>
              </div>
              <div className="tile">
                <span className="caps">Tokens</span>
                <span className="value">{compact(tokens)}</span>
                <span className="sub">
                  {compact(totals.tokens_in)} in · {compact(totals.tokens_out)} out
                </span>
              </div>
              <div className="tile">
                <span className="caps">Cost</span>
                <span className="value">${totals.cost_usd.toFixed(2)}</span>
                <span className="sub">{totals.cost_usd === 0 ? "no prices set (free tier)" : `cap $${usage.data?.limits.max_cost_per_run_usd.toFixed(2)}`}</span>
              </div>
              <div className="tile">
                <span className="caps">Model time</span>
                <span className="value">{(totals.latency_ms / 1000).toFixed(0)} s</span>
                <span className="sub">summed call latency</span>
              </div>
            </div>
            {maxTokens ? (
              <div className="card stack gap-8" style={{ padding: "16px 20px", borderRadius: 12 }}>
                <div className="row between gap-8" style={{ fontSize: 13 }}>
                  <span style={{ fontWeight: 500 }}>Run budget</span>
                  <span className="mono soft">
                    {compact(tokens)} of {compact(maxTokens)} tokens
                  </span>
                </div>
                <Meter value={tokens / maxTokens} label={`${Math.round((tokens / maxTokens) * 100)} percent of the token budget used`} />
              </div>
            ) : null}
            {agents.length ? (
              <div className="table-wrap">
                <table className="table" style={{ minWidth: 560 }}>
                  <caption style={{ textAlign: "left", padding: "14px 16px 6px", fontWeight: 600 }}>Tokens per agent</caption>
                  <thead>
                    <tr>
                      <th scope="col" style={{ width: 220 }}>Agent</th>
                      <th scope="col" className="num" style={{ width: 64 }}>Calls</th>
                      <th scope="col">Tokens</th>
                    </tr>
                  </thead>
                  <tbody>
                    {agents.map((a) => (
                      <tr key={a.name}>
                        <td>{agentName(a.name)}</td>
                        <td className="num soft">{a.calls}</td>
                        <td>
                          <span className="row gap-12" style={{ flexWrap: "nowrap" }}>
                            <span aria-hidden="true" style={{ flex: "1 1 auto", height: 8 }}>
                              <span style={{ display: "block", height: 8, borderRadius: 2, width: `${(a.tokens / topTokens) * 100}%`, background: "#79B8FF" }} />
                            </span>
                            <span className="mono" style={{ flex: "0 0 56px", textAlign: "right", fontSize: 12 }}>
                              {compact(a.tokens)}
                            </span>
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty">No model calls recorded yet.</div>
            )}
            {totals.cost_usd === 0 && totals.calls > 0 ? (
              <p className="hint">
                Cost reads $0.00 because LLM_PRICE_PER_1M_INPUT and LLM_PRICE_PER_1M_OUTPUT are 0. Set them on a paid tier so the per-run cost cap applies.
              </p>
            ) : null}
          </>
        ) : null}
      </section>
    </main>
  );
}
