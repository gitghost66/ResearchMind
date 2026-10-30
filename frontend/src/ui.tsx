import type { ReactNode } from "react";

import type { ApiError } from "./api";
import { STATUS, verdict } from "./lib";
import type { RunStatus } from "./types";

type IconName = "arrow" | "sliders" | "link" | "stop" | "resume" | "download" | "check" | "trash" | "close";

const PATHS: Record<IconName, ReactNode> = {
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  sliders: (
    <>
      <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
      <circle cx="16" cy="7" r="2" />
      <circle cx="10" cy="17" r="2" />
    </>
  ),
  link: (
    <>
      <path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1" />
      <path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1" />
    </>
  ),
  stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
  resume: <path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4.5h4.5" />,
  download: <path d="M12 4v11M7 10l5 5 5-5M5 20h14" />,
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  trash: <path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12" />,
  close: <path d="M6 6l12 12M18 6L6 18" />,
};

export function Icon({ name, size = 16, stroke = 2 }: { name: IconName; size?: number; stroke?: number }) {
  return (
    <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke} strokeLinecap="round" strokeLinejoin="round">
      {PATHS[name]}
    </svg>
  );
}

export function Logo() {
  return (
    <svg aria-hidden="true" width="28" height="28" viewBox="0 0 28 28" fill="none" stroke="#3CCFC0" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M7 9 L21 9 M7 9 L14 21 M21 9 L14 21" />
      <circle cx="7" cy="9" r="3" fill="#0E1218" />
      <circle cx="21" cy="9" r="3" fill="#0E1218" />
      <circle cx="14" cy="21" r="3" fill="#3CCFC0" />
    </svg>
  );
}

export function StatusChip({ status, suffix }: { status: RunStatus; suffix?: string }) {
  const s = STATUS[status];
  return (
    <span className="chip" style={{ background: s.tint, color: s.color }}>
      <span className={`dot${s.live ? " live" : ""}`} style={{ background: s.color }} aria-hidden="true" />
      {s.label}
      {suffix ? ` · ${suffix}` : ""}
    </span>
  );
}

export function StatusText({ status, children }: { status: RunStatus; children?: ReactNode }) {
  const s = STATUS[status];
  return (
    <span className="row gap-6" style={{ color: s.color }}>
      <span className={`dot${s.live ? " live" : ""}`} style={{ background: s.color }} aria-hidden="true" />
      {s.label}
      {children}
    </span>
  );
}

export function VerdictTag({ value }: { value: string }) {
  const v = verdict(value);
  return (
    <span className="tag" style={{ color: v.color, background: v.tint }}>
      {v.label}
    </span>
  );
}

export function Meter({ value, color, width, label }: { value: number; color?: string; width?: number; label?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <span className="meter" style={{ width: width ?? "100%" }} role={label ? "img" : undefined} aria-label={label} aria-hidden={label ? undefined : true}>
      <span style={{ width: `${pct}%`, background: color }} />
    </span>
  );
}

export function Confidence({ value, color }: { value: number; color: string }) {
  return (
    <span className="row gap-8 mono soft" style={{ fontSize: 12 }}>
      <Meter value={value} color={color} width={64} />
      confidence {value.toFixed(2)}
    </span>
  );
}

export function Ring({ value }: { value: number }) {
  const circumference = 2 * Math.PI * 34;
  return (
    <svg role="img" aria-label={`Overall confidence ${value.toFixed(2)} out of 1`} width="88" height="88" viewBox="0 0 88 88">
      <circle cx="44" cy="44" r="34" fill="none" stroke="#1C222C" strokeWidth="8" />
      <circle cx="44" cy="44" r="34" fill="none" stroke="#3CCFC0" strokeWidth="8" strokeLinecap="round" strokeDasharray={`${value * circumference} ${circumference}`} transform="rotate(-90 44 44)" />
      <text x="44" y="50" textAnchor="middle" fill="#E7EBF1" fontFamily="IBM Plex Mono, monospace" fontSize="18" fontWeight="500">
        {value.toFixed(2)}
      </text>
    </svg>
  );
}

export function Cites({ labels }: { labels: string[] }) {
  return (
    <>
      {labels.map((label) => (
        <a key={label} className="cite" href={`#src-${label}`} onClick={(e) => jumpTo(e, `src-${label}`)}>
          {label}
        </a>
      ))}
    </>
  );
}

/** In-page jump that leaves the hash route alone. */
function jumpTo(event: React.MouseEvent, id: string) {
  event.preventDefault();
  const target = document.getElementById(id);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "center" });
  target.classList.add("flash");
  target.focus({ preventScroll: true });
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span className="row gap-8 soft" role="status">
      <span className="spinner" aria-hidden="true" />
      {label}
    </span>
  );
}

export function ErrorBanner({ error, onRetry }: { error: ApiError; onRetry?: () => void }) {
  return (
    <div className="banner error" role="alert">
      <div className="stack gap-4">
        <strong>{error.needsKey ? "This API needs a key" : "Something went wrong"}</strong>
        <span className="soft" style={{ fontSize: 13 }}>
          {error.needsKey ? "Open Settings (top right) and paste the API key." : error.message}
        </span>
      </div>
      {onRetry ? (
        <button type="button" className="btn small" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}
