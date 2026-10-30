// Labels, colours, and formatting shared by every screen.

import type { RunEvent, RunStatus } from "./types";

export const MODES = [
  { id: "quick", name: "Quick", time: "~45 s", desc: "Web search, one round. A cited answer without fact-checking." },
  { id: "deep", name: "Deep", time: "~5 min", desc: "Web, technical, and market researchers. Claims fact-checked, up to three rounds." },
  { id: "academic", name: "Academic", time: "~6 min", desc: "arXiv and Semantic Scholar first; methodology and datasets weigh more." },
  { id: "decision", name: "Decision", time: "~5 min", desc: "Market and cost first, weighed against your situation. Ends in a recommendation." },
] as const;

export function modeName(mode: string): string {
  return MODES.find((m) => m.id === mode)?.name ?? mode;
}

const AGENTS: Record<string, string> = {
  planner: "Planner",
  web_researcher: "Web researcher",
  technical_researcher: "Technical researcher",
  market_researcher: "Market researcher",
  academic_researcher: "Academic researcher",
  claim_extractor: "Claim extractor",
  fact_checker: "Fact checker",
  contradiction_detector: "Contradiction detector",
  sufficiency_checker: "Sufficiency check",
  synthesizer: "Synthesizer",
  runner: "Runner",
};

export function agentName(agent: string): string {
  return AGENTS[agent] ?? agent.replace(/_/g, " ");
}

export const FANOUT = new Set(["web_researcher", "technical_researcher", "market_researcher", "academic_researcher"]);

export const STATUS: Record<RunStatus, { label: string; color: string; tint: string; live?: boolean }> = {
  queued: { label: "Queued", color: "#aab3c0", tint: "#1c222c" },
  running: { label: "Running", color: "#3ccfc0", tint: "rgba(60,207,192,0.12)", live: true },
  completed: { label: "Completed", color: "#6fd3a0", tint: "rgba(111,211,160,0.12)" },
  failed: { label: "Failed", color: "#ff8a7a", tint: "rgba(255,138,122,0.12)" },
  cancelled: { label: "Cancelled", color: "#aab3c0", tint: "#1c222c" },
};

export const VERDICT: Record<string, { label: string; color: string; tint: string }> = {
  supported: { label: "Supported", color: "#79b8ff", tint: "rgba(121,184,255,0.12)" },
  weak: { label: "Weak", color: "#e2c15b", tint: "rgba(226,193,91,0.12)" },
  mixed: { label: "Mixed", color: "#b9a2ff", tint: "rgba(185,162,255,0.12)" },
  unsupported: { label: "Unsupported", color: "#aab3c0", tint: "#1c222c" },
  unverified: { label: "Unverified", color: "#8892a1", tint: "#1c222c" },
};

export function verdict(v: string) {
  return VERDICT[v] ?? { label: v, color: "#aab3c0", tint: "#1c222c" };
}

export const STANCE: Record<string, { color: string; tint: string }> = {
  supports: { color: "#79b8ff", tint: "rgba(121,184,255,0.12)" },
  contradicts: { color: "#ff9d5c", tint: "rgba(255,157,92,0.12)" },
  neutral: { color: "#aab3c0", tint: "#1c222c" },
};

export function isActive(status: RunStatus): boolean {
  return status === "queued" || status === "running";
}

export function reliabilityColor(score: number): string {
  return score >= 0.7 ? "#3ccfc0" : score >= 0.5 ? "#e2c15b" : "#8892a1";
}

export function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

export function humanType(type: string): string {
  return type.replace(/_/g, " ");
}

export function since(iso: string | null, now = Date.now()): string {
  if (!iso) return "";
  const seconds = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000));
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.round(hours / 24);
  return days === 1 ? "yesterday" : `${days} days ago`;
}

export function duration(fromIso: string | null, toIso: string | null, now = Date.now()): string {
  if (!fromIso) return "";
  const end = toIso ? new Date(toIso).getTime() : now;
  const total = Math.max(0, Math.round((end - new Date(fromIso).getTime()) / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m}:${String(s).padStart(2, "0")}`;
}

export function compact(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
}

export function clock(iso: string | null): string {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
}

export function shortId(id: string): string {
  return id.length > 14 ? `${id.slice(0, 8)}…${id.slice(-4)}` : id;
}

/** One line of plain language for an event, and the colour of its type. */
export function describeEvent(event: RunEvent): { text: string; color: string } {
  const p = event.payload ?? {};
  const s = (key: string) => (p[key] === undefined || p[key] === null ? "" : String(p[key]));
  switch (event.type) {
    case "run_status": {
      const parts = [s("status")];
      if (p.resumed) parts.push("resumed");
      if (p.attempt) parts.push(`attempt ${s("attempt")}`);
      if (p.error) parts.push(s("error"));
      return { text: parts.join(" · "), color: "#3ccfc0" };
    }
    case "source_found": {
      const score = typeof p.reliability_score === "number" ? ` · reliability ${(p.reliability_score as number).toFixed(2)}` : "";
      return { text: `${domainOf(s("url"))} · ${humanType(s("domain_type"))}${score}`, color: "#79b8ff" };
    }
    case "claim_verified": {
      const conf = typeof p.confidence === "number" ? ` · ${(p.confidence as number).toFixed(2)}` : "";
      return { text: `${s("verdict")}${conf} · ${s("text")}`.trim(), color: verdict(s("verdict")).color };
    }
    case "contradiction_found":
      return { text: `cause: ${s("likely_cause")}${p.resolved ? " · explained" : " · unresolved"}`, color: "#ff9d5c" };
    case "error":
      return { text: [s("message"), s("reason")].filter(Boolean).join(": "), color: "#e2c15b" };
    case "agent_finished":
      if (p.status === "interrupted") return { text: "interrupted", color: "#b9a2ff" };
      break;
    case "report_ready":
      return { text: "report written", color: "#6fd3a0" };
  }
  const keys = ["status", "sub_question", "question", "sub_questions", "candidates", "sources", "claims", "contradictions", "decision", "gaps", "iteration"];
  const text = keys
    .filter((k) => p[k] !== undefined && p[k] !== null && typeof p[k] !== "object")
    .slice(0, 3)
    .map((k) => `${k.replace(/_/g, " ")} ${s(k)}`)
    .join(" · ");
  return { text, color: "#aab3c0" };
}
