import { useMemo, useState } from "react";

import { api } from "../api";
import { useLoad } from "../hooks";
import { domainOf, humanType, isActive, reliabilityColor, STANCE, verdict } from "../lib";
import type { ReportContent, ReportStatement, RunDetail, SourceOut } from "../types";
import { Cites, Confidence, ErrorBanner, Icon, Meter, Ring, Spinner, VerdictTag } from "../ui";

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}

function Statements({ items }: { items: ReportStatement[] }) {
  return (
    <>
      {items.map((s, i) => (
        <span key={i}>
          {s.text} <Cites labels={s.citations} />{" "}
        </span>
      ))}
    </>
  );
}

export function ReportView({ run }: { run: RunDetail }) {
  const report = useLoad(() => api.report(run.run_id), [run.run_id, run.report_available], {});
  const sources = useLoad(() => api.sources(run.run_id), [run.run_id, run.report_available], {});

  if (!run.report_available) {
    return (
      <main className="container page-body">
        <div className="empty">
          <strong style={{ color: "#E7EBF1" }}>{isActive(run.status) ? "The report is not ready yet" : "This run has no report"}</strong>
          <span>
            {isActive(run.status)
              ? "It appears here when the synthesizer finishes. Follow along on the Progress tab."
              : run.status === "failed"
                ? "The run failed before writing one. Resume it to continue from where it stopped."
                : "The run ended before writing one."}
          </span>
          <a href={`#/runs/${run.run_id}`}>Progress</a>
        </div>
      </main>
    );
  }
  if (report.error && !report.data) {
    return (
      <main className="container page-body">
        <ErrorBanner error={report.error} onRetry={report.reload} />
      </main>
    );
  }
  if (!report.data) {
    return (
      <main className="container page-body">
        <Spinner label="Loading report" />
      </main>
    );
  }
  return (
    <Report
      content={report.data.content}
      markdown={report.data.markdown}
      version={report.data.version}
      runId={run.run_id}
      stored={sources.data?.items ?? []}
      storedTotal={sources.data?.total}
    />
  );
}

function Report({
  content,
  markdown,
  version,
  runId,
  stored,
  storedTotal,
}: {
  content: ReportContent;
  markdown: string;
  version: number;
  runId: string;
  stored: SourceOut[];
  storedTotal?: number;
}) {
  const [filter, setFilter] = useState("all");
  const byId = useMemo(() => new Map(stored.map((s) => [s.id, s])), [stored]);
  const findings = content.key_findings ?? [];
  const verdicts = Array.from(new Set(findings.map((f) => f.verdict)));
  const shown = findings.filter((f) => filter === "all" || f.verdict === filter);
  const byLabel = new Map(findings.map((f) => [f.label, f]));
  const counts = content.verification?.by_verdict ?? {};
  const checked = Object.entries(counts).filter(([v]) => v !== "unverified");
  const checkedTotal = checked.reduce((sum, [, n]) => sum + n, 0);
  const coverage = content.coverage ?? {};
  const slug = content.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 60) || "report";

  return (
    <main className="container page-body">
      <div className="cols">
        <article className="col-main stack gap-28" style={{ flexBasis: 680 }}>
          <div className="row between gap-12">
            <span className="hint">
              {content.title}
              {version > 1 ? ` · version ${version}` : ""}
            </span>
            <span className="row gap-8">
              <button type="button" className="btn small" onClick={() => download(`${slug}.md`, markdown, "text/markdown")}>
                <Icon name="download" /> Markdown
              </button>
              <button type="button" className="btn small" onClick={() => download(`${slug}.json`, JSON.stringify(content, null, 2), "application/json")}>
                <Icon name="download" /> JSON
              </button>
            </span>
          </div>

          {content.generation !== "llm" ? (
            <div className="banner info" role="note">
              <span>
                {content.generation === "extractive"
                  ? "This report was assembled without a model (the budget ran out or the model was unavailable): it quotes stored passages directly."
                  : "The run found nothing to report on."}
              </span>
            </div>
          ) : null}

          {content.recommendation?.length ? (
            <section aria-labelledby="rec-title" className="rec">
              <span className="caps" style={{ color: "#3CCFC0" }}>
                Recommendation
              </span>
              <h2 id="rec-title" className="sr-only">
                Recommendation
              </h2>
              <p className="prose" style={{ color: "#E7EBF1", fontSize: 16, lineHeight: "27px" }}>
                <Statements items={content.recommendation} />
              </p>
            </section>
          ) : null}

          {content.overall_confidence !== null && content.overall_confidence !== undefined ? (
            <section aria-labelledby="conf-title" className="card row gap-24" style={{ padding: "20px 24px" }}>
              <div className="row gap-16">
                <Ring value={content.overall_confidence} />
                <div className="stack gap-4">
                  <h2 id="conf-title" style={{ fontSize: 16, lineHeight: "24px", fontWeight: 600 }}>
                    Overall confidence
                  </h2>
                  <span className="soft" style={{ fontSize: 13 }}>
                    Computed from the claims, never by a model
                  </span>
                </div>
              </div>
              <div className="stack gap-10" style={{ flex: "1 1 320px", minWidth: 0 }}>
                {checkedTotal ? (
                  <span role="img" aria-label={checked.map(([v, n]) => `${n} ${v}`).join(", ")} style={{ display: "flex", height: 8, borderRadius: 4, overflow: "hidden", background: "#1C222C" }}>
                    {checked.map(([v, n]) => (
                      <span key={v} style={{ width: `${(n / checkedTotal) * 100}%`, background: verdict(v).color }} />
                    ))}
                  </span>
                ) : null}
                <span className="row gap-16 soft" style={{ fontSize: 13, rowGap: 6 }}>
                  {checked.map(([v, n]) => (
                    <span key={v} className="row gap-6">
                      <span aria-hidden="true" style={{ width: 10, height: 10, borderRadius: 2, background: verdict(v).color }} />
                      {n} {verdict(v).label.toLowerCase()}
                    </span>
                  ))}
                  {content.verification?.contradictions ? (
                    <span>
                      {content.verification.contradictions} contradiction{content.verification.contradictions === 1 ? "" : "s"}
                      {content.verification.unresolved_contradictions ? `, ${content.verification.unresolved_contradictions} unexplained` : ", explained"}
                    </span>
                  ) : null}
                  {typeof coverage.sub_questions === "number" ? (
                    <span>
                      {String(coverage.sub_questions_with_evidence ?? 0)} of {coverage.sub_questions} sub-questions covered
                    </span>
                  ) : null}
                </span>
                <a href="#method" onClick={(e) => { e.preventDefault(); document.getElementById("method")?.scrollIntoView({ behavior: "smooth" }); }} style={{ fontSize: 13 }}>
                  How this score is computed
                </a>
              </div>
            </section>
          ) : null}

          {content.executive_summary?.length ? (
            <section aria-labelledby="summary-title" className="stack gap-10">
              <h2 id="summary-title" className="section-title">
                Executive summary
              </h2>
              <p className="prose">
                <Statements items={content.executive_summary} />
              </p>
            </section>
          ) : null}

          {findings.length ? (
            <section aria-labelledby="findings-title" className="stack gap-12">
              <div className="row between gap-10">
                <h2 id="findings-title" className="section-title">
                  Key findings
                </h2>
                {verdicts.length > 1 ? (
                  <div role="group" aria-label="Filter findings by verdict" className="segmented">
                    <button type="button" className="seg" aria-pressed={filter === "all"} onClick={() => setFilter("all")}>
                      All {findings.length}
                    </button>
                    {verdicts.map((v) => (
                      <button key={v} type="button" className="seg" aria-pressed={filter === v} onClick={() => setFilter(v)}>
                        {verdict(v).label} {findings.filter((f) => f.verdict === v).length}
                      </button>
                    ))}
                  </div>
                ) : null}
              </div>
              <ol className="stack gap-10" style={{ margin: 0, padding: 0, listStyle: "none" }}>
                {shown.map((f) => (
                  <li key={f.claim_id} className="finding">
                    <div className="row gap-12" style={{ rowGap: 8 }}>
                      <span className="mono muted" style={{ fontSize: 12 }}>{f.label}</span>
                      <VerdictTag value={f.verdict} />
                      <Confidence value={f.confidence} color={verdict(f.verdict).color} />
                    </div>
                    <p>
                      {f.text} <Cites labels={f.citations} />
                    </p>
                    {f.evidence.length ? (
                      <details className="more">
                        <summary>
                          Evidence · {f.evidence.length} passage{f.evidence.length === 1 ? "" : "s"}
                        </summary>
                        <div className="stack gap-8" style={{ marginTop: 10 }}>
                          {f.evidence.map((e) => (
                            <div key={e.passage_id} className="stack gap-6">
                              <span className="row gap-8" style={{ fontSize: 12 }}>
                                <Cites labels={[e.source_label]} />
                                <span className="tag" style={{ color: (STANCE[e.stance] ?? STANCE.neutral).color, background: (STANCE[e.stance] ?? STANCE.neutral).tint, fontWeight: 400 }}>
                                  {e.stance}
                                </span>
                              </span>
                              {e.excerpt ? <span className="excerpt">“{e.excerpt}”</span> : null}
                            </div>
                          ))}
                        </div>
                      </details>
                    ) : null}
                  </li>
                ))}
              </ol>
            </section>
          ) : null}

          {content.contradictory_evidence?.length ? (
            <section aria-labelledby="contra-title" className="stack gap-12">
              <h2 id="contra-title" className="section-title">
                Contradictory evidence
              </h2>
              {content.contradictory_evidence.map((c) => (
                <div key={c.contradiction_id} className="card stack gap-16" style={{ padding: 20, borderColor: "rgba(255,157,92,0.35)", borderRadius: 14 }}>
                  <div className="row gap-8">
                    <span className="tag" style={{ color: "#FF9D5C", background: "rgba(255,157,92,0.12)" }}>
                      Likely cause: {humanType(c.likely_cause)}
                    </span>
                    <span className="tag" style={{ color: "#AAB3C0", background: "#1C222C", fontWeight: 400 }}>
                      {c.resolved ? "Explained · not penalised" : "Unexplained · lowers confidence"}
                    </span>
                  </div>
                  <div className="vs">
                    {c.claims.map((label) => {
                      const claim = byLabel.get(label);
                      return (
                        <div key={label}>
                          <span className="mono muted" style={{ fontSize: 12 }}>
                            {label}
                          </span>
                          <span style={{ fontSize: 14, lineHeight: "22px" }}>
                            {claim ? claim.text : label} {claim ? <Cites labels={claim.citations} /> : null}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <p className="soft" style={{ fontSize: 14 }}>
                    {c.explanation}
                  </p>
                </div>
              ))}
            </section>
          ) : null}

          {content.sections?.map((section, i) =>
            section.statements.length ? (
              <section key={i} className="stack gap-10">
                <h2 className="section-title">{section.heading}</h2>
                <p className="prose">
                  <Statements items={section.statements} />
                </p>
              </section>
            ) : null,
          )}

          {content.limitations?.length ? (
            <section aria-labelledby="limits-title" className="stack gap-10">
              <h2 id="limits-title" className="section-title">
                Limitations
              </h2>
              <ul className="soft stack gap-6" style={{ margin: 0, paddingLeft: 20 }}>
                {content.limitations.map((l, i) => (
                  <li key={i}>{l}</li>
                ))}
              </ul>
            </section>
          ) : null}

          {content.methodology ? (
            <section id="method" aria-labelledby="method-title" className="card stack gap-10" style={{ padding: "20px 24px", scrollMarginTop: 96 }}>
              <h2 id="method-title" style={{ fontSize: 16, lineHeight: "24px", fontWeight: 600 }}>
                Methodology
              </h2>
              <p className="soft" style={{ fontSize: 13, lineHeight: "21px", whiteSpace: "pre-line" }}>
                {content.methodology}
              </p>
            </section>
          ) : null}
        </article>

        <aside aria-labelledby="cited-title" className="col-side stack gap-12">
          <div className="row between gap-12" style={{ alignItems: "baseline" }}>
            <h2 id="cited-title" style={{ fontSize: 16, lineHeight: "24px", fontWeight: 600 }}>
              Sources cited
            </h2>
            <a href={`#/runs/${runId}/sources`} style={{ fontSize: 13 }}>
              All {storedTotal ?? ""}
            </a>
          </div>
          <ol className="stack gap-8" style={{ margin: 0, padding: 0, listStyle: "none" }}>
            {content.sources.map((s) => {
              const info = byId.get(s.source_id);
              return (
                <li key={s.label} id={`src-${s.label}`} tabIndex={-1} className="src-item">
                  <span className="cite" style={{ alignSelf: "flex-start", margin: 0 }}>
                    {s.label}
                  </span>
                  <span className="stack gap-6" style={{ flex: "1 1 auto", minWidth: 0 }}>
                    <a href={s.url} target="_blank" rel="noreferrer" style={{ color: "#E7EBF1", fontSize: 13, lineHeight: "19px", fontWeight: 500, textDecoration: "none", overflowWrap: "anywhere" }}>
                      {s.title || domainOf(s.url)}
                    </a>
                    <span className="muted" style={{ fontSize: 12, lineHeight: "16px" }}>
                      {domainOf(s.url)}
                      {info ? ` · ${humanType(info.domain_type)}` : ""}
                    </span>
                    {info ? (
                      <span className="row gap-8 mono soft" style={{ fontSize: 11, flexWrap: "nowrap" }}>
                        <Meter value={info.reliability_score} color={reliabilityColor(info.reliability_score)} />
                        <span style={{ whiteSpace: "nowrap" }}>reliability {info.reliability_score.toFixed(2)}</span>
                      </span>
                    ) : null}
                  </span>
                </li>
              );
            })}
          </ol>
        </aside>
      </div>
    </main>
  );
}
