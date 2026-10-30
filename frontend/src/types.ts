// Shapes of the ResearchMind API (backend/app/schemas/*.py).

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";
export type StepStatus = "pending" | "running" | "completed" | "incomplete" | "skipped";
export type Mode = "quick" | "deep" | "academic" | "decision";

export interface RunLinks {
  self: string;
  events: string;
  report: string;
  graph: string;
  sources: string;
  usage: string;
  cancel: string;
  resume: string;
}

export interface RunCreated {
  run_id: string;
  status: RunStatus;
  links: RunLinks;
}

export interface RunSummary {
  run_id: string;
  question: string;
  mode: string;
  status: RunStatus;
  error: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface RunList {
  items: RunSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface SubQuestionOut {
  id: string;
  text: string;
  domain: string;
  status: string;
}

export interface StepProgress {
  agent: string;
  status: StepStatus;
  started: number;
  finished: number;
  errors: number;
}

export interface RunDetail extends RunSummary {
  user_context: Record<string, unknown>;
  options: Record<string, unknown>;
  plan: Record<string, unknown>;
  iteration: number;
  attempts: number;
  sub_questions: SubQuestionOut[];
  progress: {
    steps: StepProgress[];
    sub_questions: Record<string, number>;
    sources: number;
    passages: number;
    last_event_seq: number;
  };
  report_available: boolean;
  links: RunLinks;
}

export interface RunEvent {
  seq: number;
  run_id: string;
  ts: string | null;
  agent: string;
  type: string;
  payload: Record<string, unknown>;
}

export interface ReportStatement {
  text: string;
  citations: string[];
  claims?: string[];
}

export interface ReportContent {
  title: string;
  question: string;
  mode: string;
  executive_summary: ReportStatement[];
  sections: { heading: string; statements: ReportStatement[] }[];
  recommendation: ReportStatement[];
  limitations: string[];
  methodology: string;
  sources: { label: string; source_id: string; url: string; title: string; passage_ids: string[] }[];
  coverage: Record<string, number | boolean>;
  key_findings: {
    label: string;
    claim_id: string;
    text: string;
    sub_question_id: string | null;
    verdict: string;
    confidence: number;
    citations: string[];
    evidence: { passage_id: string; source_label: string; stance: string; excerpt: string }[];
  }[];
  contradictory_evidence: {
    label: string;
    contradiction_id: string;
    claims: string[];
    citations: string[];
    likely_cause: string;
    explanation: string;
    resolved: boolean;
  }[];
  verification: {
    claims_extracted?: number;
    claims_checked?: number;
    by_verdict?: Record<string, number>;
    contradictions?: number;
    unresolved_contradictions?: number;
  };
  overall_confidence: number | null;
  generation: "llm" | "extractive" | "empty";
}

export interface ReportResponse {
  run_id: string;
  report_id: string;
  version: number;
  created_at: string | null;
  overall_confidence: number | null;
  content: ReportContent;
  markdown: string;
}

export interface SourceOut {
  id: string;
  url: string;
  title: string;
  domain_type: string;
  reliability_score: number;
  published_at: string | null;
  passages: number;
  cited: boolean;
  created_at: string | null;
}

export interface SourceList {
  run_id: string;
  items: SourceOut[];
  total: number;
}

export interface UsageBucket {
  calls: number;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  latency_ms: number;
}

export interface UsageResponse {
  run_id: string;
  totals: UsageBucket;
  per_agent: Record<string, UsageBucket>;
  per_model: Record<string, UsageBucket>;
  limits: { max_tokens_per_run: number; max_cost_per_run_usd: number };
}

export type GraphNodeType = "question" | "sub_question" | "claim" | "source" | "passage";

export interface GraphNodeOut {
  id: string;
  label: string;
  type: GraphNodeType;
  meta: Record<string, unknown>;
}

export interface GraphEdgeOut {
  id: string;
  from: string;
  to: string;
  type: "decomposes" | "supports" | "contradicts" | "cites";
  meta: Record<string, unknown>;
}

export interface EvidenceGraph {
  run_id: string;
  nodes: GraphNodeOut[];
  edges: GraphEdgeOut[];
}

export interface Ready {
  status: "ready" | "degraded";
  checks: Record<string, string>;
  runs?: { queued: number; running: number };
  worker?: { id: string; running: number; capacity: number };
  listening?: boolean;
}
