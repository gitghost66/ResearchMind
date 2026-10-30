import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, streamEvents } from "./api";
import type { RunEvent } from "./types";

/** Hash routes: #/ , #/runs/<id>[/report|/graph|/sources]. */
export type Route =
  | { page: "new" }
  | { page: "runs" }
  | { page: "run"; id: string; view: "progress" | "report" | "graph" | "sources" };

function parse(hash: string): Route {
  const parts = hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  if (parts[0] === "runs" && parts[1]) {
    const view = parts[2];
    return {
      page: "run",
      id: parts[1],
      view: view === "report" || view === "graph" || view === "sources" ? view : "progress",
    };
  }
  if (parts[0] === "runs") return { page: "runs" };
  return { page: "new" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const onChange = () => {
      setRoute(parse(window.location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  return route;
}

export function go(path: string): void {
  window.location.hash = path;
}

/** Load something, optionally re-loading every `intervalMs` while `poll` holds. */
export function useLoad<T>(
  load: () => Promise<T>,
  deps: unknown[],
  options: { intervalMs?: number; poll?: (value: T | undefined) => boolean } = {},
) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(true);
  const latest = useRef(load);
  latest.current = load;
  const pollRef = useRef(options.poll);
  pollRef.current = options.poll;
  const dataRef = useRef<T>();

  const reload = useCallback(async () => {
    try {
      const value = await latest.current();
      dataRef.current = value;
      setData(value);
      setError(null);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, "error", String(err)));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer: number | undefined;
    setLoading(true);
    setData(undefined);
    dataRef.current = undefined;
    const tick = async () => {
      await reload();
      if (stopped || !options.intervalMs) return;
      if (pollRef.current && !pollRef.current(dataRef.current)) return;
      timer = window.setTimeout(tick, options.intervalMs);
    };
    void tick();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, loading, reload };
}

/**
 * A run's events, live: replays from the start, then follows the stream,
 * reconnecting with Last-Event-ID after a drop. `onEvent` fires for each
 * new event (used to refresh the run's details).
 */
export function useRunEvents(runId: string, onEvent?: (event: RunEvent) => void) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [state, setState] = useState<"connecting" | "live" | "ended" | "retrying">("connecting");
  const callback = useRef(onEvent);
  callback.current = onEvent;

  useEffect(() => {
    const controller = new AbortController();
    let last = 0;
    let stopped = false;
    setEvents([]);
    setState("connecting");

    const run = async () => {
      while (!stopped) {
        try {
          await streamEvents(runId, {
            after: last,
            signal: controller.signal,
            onEvent: (event) => {
              if (event.seq <= last) return;
              last = event.seq;
              setState("live");
              setEvents((current) => [...current.slice(-499), event]);
              callback.current?.(event);
            },
          });
          setState("ended");
          return;
        } catch (err) {
          if (stopped || controller.signal.aborted) return;
          if (err instanceof ApiError && err.status >= 400 && err.status < 500) {
            setState("ended");
            return;
          }
          setState("retrying");
          await new Promise((resolve) => window.setTimeout(resolve, 3000));
        }
      }
    };
    void run();
    return () => {
      stopped = true;
      controller.abort();
    };
  }, [runId]);

  return { events, state };
}

/** Re-render every `ms` (for clocks and "x min ago"). */
export function useNow(ms = 1000, enabled = true): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!enabled) return;
    const timer = window.setInterval(() => setNow(Date.now()), ms);
    return () => window.clearInterval(timer);
  }, [ms, enabled]);
  return now;
}
