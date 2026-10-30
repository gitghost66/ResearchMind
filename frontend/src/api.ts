// A small typed client for the ResearchMind API. Requests go to the same
// origin (the Vite dev server or Nginx proxies them to the API).

import type {
  EvidenceGraph,
  Ready,
  ReportResponse,
  RunCreated,
  RunDetail,
  RunEvent,
  RunList,
  SourceList,
  UsageResponse,
} from "./types";

const KEY_STORAGE = "researchmind.apiKey";
const BASE = "/api/v1/research";

export function getApiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setApiKey(value: string): void {
  try {
    if (value) localStorage.setItem(KEY_STORAGE, value);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    // Storage unavailable (private mode): the key lasts until reload.
  }
  memoryKey = value;
}

let memoryKey = "";

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const key = getApiKey() || memoryKey;
  return { ...(key ? { "X-API-Key": key } : {}), ...extra };
}

/** The API's error envelope, `{error: {code, message, details}}`. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details: unknown = null,
  ) {
    super(message);
  }

  get needsKey(): boolean {
    return this.status === 401 || this.status === 403;
  }
}

async function toError(response: Response): Promise<ApiError> {
  try {
    const body = await response.json();
    const error = body?.error ?? {};
    return new ApiError(
      response.status,
      String(error.code ?? "http_error"),
      String(error.message ?? response.statusText),
      error.details ?? null,
    );
  } catch {
    return new ApiError(response.status, "http_error", response.statusText || "Request failed");
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      ...init,
      headers: headers({
        Accept: "application/json",
        ...(init.body ? { "Content-Type": "application/json" } : {}),
      }),
    });
  } catch {
    throw new ApiError(0, "network_error", "The API is not reachable. Is the backend running?");
  }
  if (!response.ok) throw await toError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  ready: () => request<Ready>("/ready"),
  list: (params: { limit?: number; offset?: number; status?: string } = {}) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    return request<RunList>(`${BASE}?${query}`);
  },
  create: (body: {
    question: string;
    mode: string;
    user_context?: Record<string, unknown>;
    max_sources?: number;
  }) => request<RunCreated>(BASE, { method: "POST", body: JSON.stringify(body) }),
  detail: (id: string) => request<RunDetail>(`${BASE}/${id}`),
  report: (id: string) => request<ReportResponse>(`${BASE}/${id}/report`),
  graph: (id: string) => request<EvidenceGraph>(`${BASE}/${id}/graph`),
  sources: (id: string) => request<SourceList>(`${BASE}/${id}/sources`),
  usage: (id: string) => request<UsageResponse>(`${BASE}/${id}/usage`),
  cancel: (id: string) => request<{ run_id: string; status: string }>(`${BASE}/${id}/cancel`, { method: "POST" }),
  resume: (id: string) => request<RunCreated>(`${BASE}/${id}/resume`, { method: "POST" }),
  remove: (id: string) => request<void>(`${BASE}/${id}`, { method: "DELETE" }),
};

/**
 * Stream a run's events (Server-Sent Events) with fetch, since EventSource
 * cannot send the X-API-Key header. Resolves when the server closes the
 * stream (after the run's final status); rejects on network or HTTP errors.
 */
export async function streamEvents(
  id: string,
  options: { after: number; signal: AbortSignal; onEvent: (event: RunEvent) => void },
): Promise<void> {
  const response = await fetch(`${BASE}/${id}/events`, {
    headers: headers({
      Accept: "text/event-stream",
      ...(options.after > 0 ? { "Last-Event-ID": String(options.after) } : {}),
    }),
    signal: options.signal,
  });
  if (!response.ok || !response.body) throw await toError(response);

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value.replace(/\r\n/g, "\n");
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n");
      if (data) options.onEvent(JSON.parse(data) as RunEvent);
    }
  }
}
