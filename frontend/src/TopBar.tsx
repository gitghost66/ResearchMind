import { useEffect, useRef, useState } from "react";

import { api, getApiKey, setApiKey } from "./api";
import { useLoad } from "./hooks";
import { Icon, Logo } from "./ui";

export function TopBar({ section }: { section: "new" | "runs" }) {
  const ready = useLoad(() => api.ready(), [], { intervalMs: 10000 });
  const [open, setOpen] = useState(false);

  let dot = "#8892A1";
  let text = "Connecting…";
  if (ready.error) {
    dot = "#FF8A7A";
    text = "API unreachable";
  } else if (ready.data) {
    const ok = ready.data.status === "ready";
    dot = ok ? "#6FD3A0" : "#E2C15B";
    const parts = [ok ? "API ready" : "API degraded"];
    if (ready.data.runs) parts.push(`${ready.data.runs.running} running`);
    if (ready.data.runs?.queued) parts.push(`${ready.data.runs.queued} queued`);
    text = parts.join(" · ");
  }

  return (
    <header className="topbar">
      <a className="brand" href="#/">
        <Logo />
        ResearchMind
      </a>
      <nav aria-label="Main" className="mainnav">
        <a href="#/" aria-current={section === "new" ? "page" : undefined}>
          New research
        </a>
        <a href="#/runs" aria-current={section === "runs" ? "page" : undefined}>
          Runs
        </a>
        <a href="/docs" target="_blank" rel="noreferrer">
          API docs
        </a>
      </nav>
      <div className="topbar-end">
        <span className="pill" role="status">
          <span className="dot" style={{ background: dot }} aria-hidden="true" />
          {text}
        </span>
        <button type="button" className="icon-btn" aria-label="Settings: API key" onClick={() => setOpen(true)}>
          <Icon name="sliders" size={20} stroke={1.8} />
        </button>
      </div>
      {open ? <SettingsDialog onClose={() => setOpen(false)} /> : null}
    </header>
  );
}

function SettingsDialog({ onClose }: { onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);
  const [key, setKey] = useState(getApiKey());

  useEffect(() => {
    ref.current?.showModal();
  }, []);

  return (
    <dialog
      ref={ref}
      onClose={onClose}
      aria-labelledby="settings-title"
      style={{ width: "min(480px, calc(100vw - 32px))", padding: 0, border: "1px solid #2E3644", borderRadius: 16, background: "#11151C", color: "#E7EBF1" }}
    >
      <form
        method="dialog"
        className="stack gap-16 pad"
        onSubmit={() => {
          setApiKey(key.trim());
          window.location.reload();
        }}
      >
        <div className="row between gap-8">
          <h2 id="settings-title" style={{ fontSize: 18, lineHeight: "26px" }}>
            Settings
          </h2>
          <button type="button" className="icon-btn" aria-label="Close" onClick={() => ref.current?.close()}>
            <Icon name="close" />
          </button>
        </div>
        <div className="field">
          <label htmlFor="api-key">API key</label>
          <input id="api-key" className="input mono" type="password" autoComplete="off" value={key} onChange={(e) => setKey(e.target.value)} placeholder="Empty when the API has no API_KEY" />
          <span className="hint">Sent as the X-API-Key header. Stored in this browser only.</span>
        </div>
        <div className="row gap-8" style={{ justifyContent: "flex-end" }}>
          <button type="button" className="btn" onClick={() => ref.current?.close()}>
            Cancel
          </button>
          <button type="submit" className="btn primary" style={{ minHeight: 44 }}>
            Save
          </button>
        </div>
      </form>
    </dialog>
  );
}
