import "@xyflow/react/dist/style.css";

import { Background, Controls, Handle, MarkerType, MiniMap, Position, ReactFlow } from "@xyflow/react";
import type { Edge, Node, NodeProps } from "@xyflow/react";
import { useMemo, useState } from "react";

import { api } from "../api";
import { useLoad } from "../hooks";
import { domainOf, humanType, isActive, reliabilityColor, STANCE, verdict } from "../lib";
import type { EvidenceGraph, GraphEdgeOut, GraphNodeOut, RunDetail } from "../types";
import { Confidence, ErrorBanner, Meter, Spinner, VerdictTag } from "../ui";

// ---------------------------------------------------------------------------
// Layout: one column per node type, left to right.
// ---------------------------------------------------------------------------

const WIDTH: Record<string, number> = { question: 200, sub_question: 240, claim: 260, passage: 260, source: 220 };
const HEIGHT: Record<string, number> = { question: 96, sub_question: 80, claim: 88, passage: 88, source: 76 };
const GAP_X = 90;
const GAP_Y = 18;

type GData = { kind: string; head: string; body: string; headColor?: string; selected: boolean };
type GNode = Node<GData, "g">;

function GNodeView({ data }: NodeProps<GNode>) {
  return (
    <div className={`gnode ${data.kind}${data.selected ? " selected" : ""}`} style={{ width: WIDTH[data.kind], height: HEIGHT[data.kind] }}>
      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <Handle type="source" position={Position.Right} style={{ opacity: 0 }} />
      <Handle id="t" type="target" position={Position.Top} style={{ opacity: 0 }} />
      <Handle id="b" type="source" position={Position.Bottom} style={{ opacity: 0 }} />
      <span className="k" style={data.headColor ? { color: data.headColor } : undefined}>
        {data.head}
      </span>
      <span className="clamp">{data.body}</span>
    </div>
  );
}

const nodeTypes = { g: GNodeView };

const str = (v: unknown) => (v === undefined || v === null ? "" : String(v));
const num = (v: unknown) => (typeof v === "number" ? v : 0);

function describe(node: GraphNodeOut): Omit<GData, "selected"> {
  const m = node.meta;
  switch (node.type) {
    case "question":
      return { kind: "question", head: "Question", body: str(m.question) };
    case "sub_question":
      return { kind: "sub_question", head: `${str(m.domain)}${num(m.iteration) > 0 ? ` · round ${num(m.iteration) + 1}` : ""}`, body: str(m.text) };
    case "claim": {
      const v = verdict(str(m.verdict));
      return { kind: "claim", head: `${v.label} · ${num(m.confidence).toFixed(2)}`, headColor: v.color, body: str(m.text) };
    }
    case "source":
      return { kind: "source", head: `${node.label} · ${humanType(str(m.domain_type))} · ${num(m.reliability_score).toFixed(2)}`, headColor: "#7FE3D8", body: str(m.title) || domainOf(str(m.url)) };
    default:
      return { kind: "passage", head: node.label, body: str(m.excerpt) };
  }
}

function layout(graph: EvidenceGraph, showPassages: boolean, selected: string | null) {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const out = (id: string, type: string) => graph.edges.filter((e) => e.from === id && e.type === type);

  // Column order: claims follow their sub-question; passages and sources
  // follow the first claim they bear on.
  const subs = graph.nodes.filter((n) => n.type === "sub_question");
  const claims: GraphNodeOut[] = [];
  const seen = new Set<string>();
  for (const sq of subs) {
    for (const e of out(sq.id, "decomposes")) {
      const target = byId.get(e.to);
      if (target?.type === "claim" && !seen.has(target.id)) {
        claims.push(target);
        seen.add(target.id);
      }
    }
  }
  for (const n of graph.nodes) if (n.type === "claim" && !seen.has(n.id)) claims.push(n);

  const passageClaims = new Map<string, { claim: string; stance: string }[]>();
  for (const e of graph.edges) {
    if ((e.type === "supports" || e.type === "contradicts") && byId.get(e.from)?.type === "passage") {
      const list = passageClaims.get(e.from) ?? [];
      list.push({ claim: e.to, stance: e.type });
      passageClaims.set(e.from, list);
    }
  }
  const passageSource = new Map<string, string>();
  for (const e of graph.edges) if (e.type === "cites") passageSource.set(e.from, e.to);

  const claimRank = new Map(claims.map((c, i) => [c.id, i]));
  const firstClaim = (passageId: string) =>
    Math.min(...(passageClaims.get(passageId) ?? []).map((pc) => claimRank.get(pc.claim) ?? 1e6), 1e6);
  const passages = graph.nodes.filter((n) => n.type === "passage").sort((a, b) => firstClaim(a.id) - firstClaim(b.id));
  const sourceRank = new Map<string, number>();
  passages.forEach((p, i) => {
    const s = passageSource.get(p.id);
    if (s && !sourceRank.has(s)) sourceRank.set(s, i);
  });
  const sources = graph.nodes.filter((n) => n.type === "source").sort((a, b) => (sourceRank.get(a.id) ?? 1e6) - (sourceRank.get(b.id) ?? 1e6));

  const columns: GraphNodeOut[][] = [graph.nodes.filter((n) => n.type === "question"), subs, claims];
  if (showPassages) columns.push(passages);
  columns.push(sources);

  const heights = columns.map((col) => col.reduce((sum, n) => sum + HEIGHT[n.type] + GAP_Y, -GAP_Y));
  const tallest = Math.max(...heights, 0);
  const nodes: GNode[] = [];
  let x = 0;
  columns.forEach((col, ci) => {
    let y = (tallest - heights[ci]) / 2;
    for (const n of col) {
      nodes.push({ id: n.id, type: "g", position: { x, y }, data: { ...describe(n), selected: n.id === selected } });
      y += HEIGHT[n.type] + GAP_Y;
    }
    x += (col[0] ? WIDTH[col[0].type] : 200) + GAP_X;
  });

  const edges: Edge[] = [];
  const add = (id: string, from: string, to: string, kind: string, extra: Partial<Edge> = {}) => {
    const color = kind === "supports" ? "#79B8FF" : kind === "contradicts" ? "#FF9D5C" : kind === "cites" ? "#3E4757" : "#5C6678";
    edges.push({
      id,
      source: from,
      target: to,
      type: "smoothstep",
      style: { stroke: color, strokeWidth: kind === "cites" ? 1.25 : 1.75, ...(kind === "contradicts" ? { strokeDasharray: "5 5" } : {}) },
      ...extra,
    });
  };
  const arrow = (color: string) => ({ type: MarkerType.ArrowClosed, color, width: 16, height: 16 });

  for (const e of graph.edges) {
    const from = byId.get(e.from);
    const to = byId.get(e.to);
    if (!from || !to) continue;
    if (e.type === "decomposes") {
      add(e.id, e.from, e.to, "decomposes", { markerEnd: arrow("#5C6678") });
    } else if (e.type === "contradicts" && from.type === "claim" && to.type === "claim") {
      const [a, b] = (claimRank.get(e.from) ?? 0) <= (claimRank.get(e.to) ?? 0) ? [e.from, e.to] : [e.to, e.from];
      add(e.id, a, b, "contradicts", {
        sourceHandle: "b",
        targetHandle: "t",
        label: `contradicts · ${humanType(str(e.meta.likely_cause))}`,
        labelStyle: { fill: "#FF9D5C", fontSize: 12, fontFamily: "IBM Plex Sans" },
        labelBgStyle: { fill: "#0E1218" },
      });
    } else if (showPassages && (e.type === "supports" || e.type === "contradicts")) {
      // Drawn claim -> passage so the line runs left to right; the arrow sits on the claim.
      add(e.id, e.to, e.from, e.type, { markerStart: arrow(e.type === "supports" ? "#79B8FF" : "#FF9D5C") });
    } else if (showPassages && e.type === "cites") {
      add(e.id, e.from, e.to, "cites");
    }
  }
  if (!showPassages) {
    // Fold passages away: one source -> claim edge per stance.
    const made = new Set<string>();
    for (const [passage, list] of passageClaims) {
      const source = passageSource.get(passage);
      if (!source) continue;
      for (const { claim, stance } of list) {
        const key = `${source}|${claim}|${stance}`;
        if (made.has(key)) continue;
        made.add(key);
        add(`fold:${key}`, claim, source, stance, { markerStart: arrow(stance === "supports" ? "#79B8FF" : "#FF9D5C") });
      }
    }
  }
  return { nodes, edges };
}

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

export function GraphView({ run }: { run: RunDetail }) {
  const graph = useLoad(() => api.graph(run.run_id), [run.run_id], {
    intervalMs: 8000,
    poll: () => isActive(run.status),
  });
  const [showPassages, setShowPassages] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const data = graph.data;
  const flow = useMemo(() => (data ? layout(data, showPassages, selected) : { nodes: [], edges: [] }), [data, showPassages, selected]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const n of data?.nodes ?? []) c[n.type] = (c[n.type] ?? 0) + 1;
    return c;
  }, [data]);

  if (graph.error && !data) {
    return (
      <main className="container wide page-body">
        <ErrorBanner error={graph.error} onRetry={graph.reload} />
      </main>
    );
  }
  if (!data) {
    return (
      <main className="container wide page-body">
        <Spinner label="Loading graph" />
      </main>
    );
  }

  const chosen = data.nodes.find((n) => n.id === selected) ?? null;
  return (
    <main className="container wide page-body" style={{ paddingTop: 20 }}>
      <div className="cols" style={{ gap: 24 }}>
        <section aria-labelledby="graph-title" className="stack gap-12" style={{ flex: "999 1 760px", minWidth: 0 }}>
          <div className="row between gap-12">
            <div className="stack gap-4">
              <h2 id="graph-title" style={{ fontSize: 16, lineHeight: "24px", fontWeight: 600 }}>
                How the evidence connects
              </h2>
              <span className="hint">
                {counts.sub_question ?? 0} sub-questions · {counts.claim ?? 0} claims · {counts.source ?? 0} sources
                {counts.passage ? ` · ${counts.passage} passages` : ""}. Select a node for details.
              </span>
            </div>
            <label className="row gap-8 soft" style={{ fontSize: 13, minHeight: 36, cursor: "pointer" }}>
              <input type="checkbox" checked={showPassages} onChange={(e) => setShowPassages(e.target.checked)} style={{ width: 16, height: 16, accentColor: "#3CCFC0" }} />
              Show passages
            </label>
          </div>
          <div className="legend" aria-label="Legend">
            <span><span aria-hidden="true" style={{ width: 22, height: 2, background: "#5C6678" }} />decomposes</span>
            <span><span aria-hidden="true" style={{ width: 22, height: 2, background: "#79B8FF" }} />supports</span>
            <span><span aria-hidden="true" style={{ width: 22, height: 0, borderTop: "2px dashed #FF9D5C" }} />contradicts</span>
            {["supported", "weak", "mixed", "unsupported"].map((v) => (
              <span key={v}><span aria-hidden="true" style={{ width: 10, height: 10, borderRadius: 2, background: verdict(v).color }} />{v}</span>
            ))}
          </div>
          {data.nodes.length <= 1 ? (
            <div className="empty">
              <strong style={{ color: "#E7EBF1" }}>Nothing to draw yet</strong>
              <span>The graph fills in as the planner and researchers store their work.</span>
            </div>
          ) : (
            <div className="graph-box">
              <ReactFlow
                nodes={flow.nodes}
                edges={flow.edges}
                nodeTypes={nodeTypes}
                colorMode="dark"
                fitView
                fitViewOptions={{ padding: 0.12 }}
                minZoom={0.15}
                nodesConnectable={false}
                onNodeClick={(_, node) => setSelected(node.id)}
                onPaneClick={() => setSelected(null)}
                proOptions={{ hideAttribution: true }}
              >
                <Background color="#1C222C" gap={24} />
                <Controls showInteractive={false} />
                <MiniMap pannable zoomable nodeColor={(n) => ((n.data as GData).headColor ?? "#2E3644")} maskColor="rgba(11,14,19,0.7)" />
              </ReactFlow>
            </div>
          )}
        </section>

        <aside aria-labelledby="detail-title" className="card stack gap-16" style={{ flex: "1 1 320px", minWidth: 0, padding: 20, borderRadius: 14 }}>
          {chosen ? (
            <Details node={chosen} graph={data} onSelect={setSelected} />
          ) : (
            <div className="stack gap-8">
              <span className="caps">Details</span>
              <h2 id="detail-title" style={{ fontSize: 16, lineHeight: "24px", fontWeight: 600 }}>
                Select a node
              </h2>
              <p className="soft" style={{ fontSize: 13, lineHeight: "21px" }}>
                Claims show the passages that support or contradict them and the fact checker's verdict. Sources show their reliability and the claims they back.
              </p>
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}

function Details({ node, graph, onSelect }: { node: GraphNodeOut; graph: EvidenceGraph; onSelect: (id: string) => void }) {
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const m = node.meta;
  const link = (id: string, text: string) => (
    <button type="button" onClick={() => onSelect(id)} style={{ all: "unset", cursor: "pointer", color: "#7FE3D8" }}>
      {text}
    </button>
  );
  const passageSource = (passageId: string) => graph.edges.find((e) => e.type === "cites" && e.from === passageId)?.to;

  if (node.type === "claim") {
    const evidence = graph.edges.filter((e) => (e.type === "supports" || e.type === "contradicts") && e.to === node.id && byId.get(e.from)?.type === "passage");
    const contradictions = graph.edges.filter((e: GraphEdgeOut) => e.type === "contradicts" && byId.get(e.from)?.type === "claim" && (e.from === node.id || e.to === node.id));
    const subq = byId.get(`sub_question:${str(m.sub_question_id)}`);
    const v = verdict(str(m.verdict));
    return (
      <div className="stack gap-16">
        <div className="stack gap-10">
          <span className="caps">Claim</span>
          <div className="row gap-10">
            <VerdictTag value={str(m.verdict)} />
            <Confidence value={num(m.confidence)} color={v.color} />
          </div>
          <h2 id="detail-title" style={{ fontSize: 17, lineHeight: "26px", fontWeight: 600 }}>
            {str(m.text)}
          </h2>
          {subq ? <span className="hint">From “{link(subq.id, str(subq.meta.text))}”</span> : null}
        </div>
        <div className="stack gap-10">
          <h3 style={{ fontSize: 13, lineHeight: "20px", fontWeight: 600 }}>Evidence</h3>
          {evidence.length ? (
            evidence.map((e) => {
              const passage = byId.get(e.from);
              const source = byId.get(passageSource(e.from) ?? "");
              const stance = STANCE[e.type] ?? STANCE.neutral;
              return (
                <div key={e.id} className="stack gap-8" style={{ padding: "12px 14px", background: "#161B23", borderRadius: 10 }}>
                  <span className="row gap-8" style={{ fontSize: 12 }}>
                    {source ? link(source.id, `${source.label} · ${domainOf(str(source.meta.url))}`) : null}
                    <span className="tag" style={{ color: stance.color, background: stance.tint, fontWeight: 400 }}>
                      {e.type}
                    </span>
                  </span>
                  <span style={{ fontSize: 13, lineHeight: "21px", color: "#D3D9E2" }}>“{str(passage?.meta.excerpt)}”</span>
                </div>
              );
            })
          ) : (
            <span className="hint">No passage links.</span>
          )}
        </div>
        {contradictions.map((e) => {
          const other = byId.get(e.from === node.id ? e.to : e.from);
          return (
            <div key={e.id} className="stack gap-8" style={{ padding: "12px 14px", borderRadius: 10, border: "1px dashed rgba(255,157,92,0.5)", background: "rgba(255,157,92,0.06)" }}>
              <span style={{ fontSize: 12, fontWeight: 500, color: "#FF9D5C" }}>
                Contradicts another claim · cause: {humanType(str(e.meta.likely_cause))}
                {e.meta.resolved ? " · explained" : " · unexplained"}
              </span>
              {other ? <span style={{ fontSize: 13, lineHeight: "21px" }}>{link(other.id, str(other.meta.text))}</span> : null}
              {e.meta.explanation ? <span className="soft" style={{ fontSize: 13, lineHeight: "21px" }}>{str(e.meta.explanation)}</span> : null}
            </div>
          );
        })}
      </div>
    );
  }

  if (node.type === "source") {
    const passages = graph.edges.filter((e) => e.type === "cites" && e.to === node.id).map((e) => e.from);
    const claims = new Map<string, string>();
    for (const e of graph.edges) {
      if ((e.type === "supports" || e.type === "contradicts") && passages.includes(e.from)) claims.set(e.to, e.type);
    }
    const score = num(m.reliability_score);
    return (
      <div className="stack gap-14" style={{ gap: 14 }}>
        <span className="caps">Source {node.label}</span>
        <h2 id="detail-title" style={{ fontSize: 17, lineHeight: "26px", fontWeight: 600 }}>
          <a href={str(m.url)} target="_blank" rel="noreferrer" style={{ color: "#E7EBF1" }}>
            {str(m.title) || domainOf(str(m.url))}
          </a>
        </h2>
        <span className="hint">
          {domainOf(str(m.url))} · {humanType(str(m.domain_type))}
          {m.published_at ? ` · ${str(m.published_at).slice(0, 10)}` : ""} · {m.cited ? "cited in the report" : "not cited"}
        </span>
        <span className="row gap-8 mono soft" style={{ fontSize: 12, flexWrap: "nowrap" }}>
          <Meter value={score} color={reliabilityColor(score)} />
          <span style={{ whiteSpace: "nowrap" }}>reliability {score.toFixed(2)}</span>
        </span>
        <div className="stack gap-8">
          <h3 style={{ fontSize: 13, lineHeight: "20px", fontWeight: 600 }}>Bears on</h3>
          {[...claims].map(([id, stance]) => {
            const claim = byId.get(id);
            return claim ? (
              <span key={id} style={{ fontSize: 13, lineHeight: "21px" }}>
                <span style={{ color: (STANCE[stance] ?? STANCE.neutral).color }}>{stance}</span> · {link(id, str(claim.meta.text))}
              </span>
            ) : null;
          })}
          {claims.size === 0 ? <span className="hint">No claim links.</span> : null}
        </div>
      </div>
    );
  }

  if (node.type === "sub_question") {
    const claims = graph.edges.filter((e) => e.type === "decomposes" && e.from === node.id).map((e) => byId.get(e.to)).filter(Boolean) as GraphNodeOut[];
    return (
      <div className="stack gap-12">
        <span className="caps">Sub-question · {str(m.domain)}</span>
        <h2 id="detail-title" style={{ fontSize: 17, lineHeight: "26px", fontWeight: 600 }}>
          {str(m.text)}
        </h2>
        <span className="hint">
          {str(m.status)}
          {num(m.iteration) > 0 ? ` · added in round ${num(m.iteration) + 1}` : ""}
        </span>
        {claims.map((c) => (
          <span key={c.id} className="row gap-8" style={{ fontSize: 13, lineHeight: "21px", alignItems: "flex-start", flexWrap: "nowrap" }}>
            <VerdictTag value={str(c.meta.verdict)} />
            {link(c.id, str(c.meta.text))}
          </span>
        ))}
      </div>
    );
  }

  return (
    <div className="stack gap-12">
      <span className="caps">{node.type === "question" ? "Question" : `Passage ${node.label}`}</span>
      <h2 id="detail-title" style={{ fontSize: 17, lineHeight: "26px", fontWeight: 600 }}>
        {node.type === "question" ? str(m.question) : `“${str(m.excerpt)}”`}
      </h2>
      {node.type === "question" ? <span className="hint">{str(m.mode)} mode · {str(m.status)}</span> : null}
    </div>
  );
}
