"""
interface/components/pipeline_runner.py
========================================
Runs pipeline scripts as subprocesses with an animated progress bar.

Progress protocol (scripts print this to stdout):
    PROGRESS:current/total:label text

UI updates are throttled to once every UPDATE_INTERVAL seconds so
intermediate log noise (tqdm lines etc.) never reaches the screen.
All non-progress lines are silently buffered and shown in a collapsed
expander only after the run completes.
"""
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent

# How often the animated bar redraws (seconds)
UPDATE_INTERVAL = 30

# Matches our progress protocol:  PROGRESS:200/5000:Fetching TMDB metadata
_PROGRESS_RE = re.compile(r"^PROGRESS:(\d+)/(\d+):(.+)$")

# Matches tqdm noise:  88%|████  | 4405/5000 [2:54:32<22:49, 2.30s/it]
_TQDM_RE = re.compile(r"^\s*\d+%\s*\|")

# Matches other noisy progress lines (carriage-return redraws, blank lines)
_NOISE_RE = re.compile(r"^\s*$|^\r")


# ── Session state ──────────────────────────────────────────────────────────────

def _init(key: str) -> None:
    if "_runs" not in st.session_state:
        st.session_state["_runs"] = {}
    if key not in st.session_state["_runs"]:
        st.session_state["_runs"][key] = {
            "status":     "idle",
            "started":    None,
            "elapsed":    None,
            "lines":      [],
            "progress":   None,
            "returncode": None,
        }


def _state(key: str) -> dict:
    _init(key)
    return st.session_state["_runs"][key]


def _enqueue(stream, q: queue.Queue) -> None:
    try:
        for line in iter(stream.readline, ""):
            q.put(line)
    finally:
        q.put(None)


# ── Animated progress bar (pure HTML/CSS) ──────────────────────────────────────

def _progress_html(current: int, total: int, label: str, elapsed: float) -> str:
    pct      = min(current / total, 1.0) if total > 0 else 0.0
    pct_disp = f"{pct * 100:.1f}"
    bar_w    = f"{pct * 100:.2f}%"
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"

    # Estimate remaining time
    if pct > 0.01 and elapsed > 1:
        remaining = (elapsed / pct) * (1 - pct)
        rm, rs = divmod(int(remaining), 60)
        eta_str = f"~{rm}m {rs}s left" if rm else f"~{rs}s left"
    else:
        eta_str = "estimating…"

    return f"""
<style>
@keyframes shimmer {{
  0%   {{ background-position: -400px 0; }}
  100% {{ background-position:  400px 0; }}
}}
.cineml-bar-wrap {{
  background: #e2e8f0;
  border-radius: 8px;
  height: 22px;
  width: 100%;
  overflow: hidden;
  margin: 6px 0 4px;
}}
.cineml-bar-fill {{
  height: 100%;
  width: {bar_w};
  min-width: 18px;
  border-radius: 8px;
  background: linear-gradient(90deg, #6366f1 0%, #818cf8 50%, #6366f1 100%);
  background-size: 400px 100%;
  animation: shimmer 1.6s infinite linear;
  transition: width 0.4s ease;
}}
.cineml-meta {{
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #64748b;
  margin-top: 2px;
}}
.cineml-label {{
  font-size: 13px;
  font-weight: 500;
  color: #1e293b;
  margin-bottom: 2px;
}}
</style>
<div>
  <div class="cineml-label">{label}</div>
  <div class="cineml-bar-wrap">
    <div class="cineml-bar-fill"></div>
  </div>
  <div class="cineml-meta">
    <span>{current:,} of {total:,} &nbsp;·&nbsp; {pct_disp}%</span>
    <span>{elapsed_str} elapsed &nbsp;·&nbsp; {eta_str}</span>
  </div>
</div>
"""


def _done_bar_html(current: int, total: int, label: str, elapsed: float, ok: bool) -> str:
    pct   = min(current / total, 1.0) if total > 0 else 1.0
    bar_w = f"{pct * 100:.2f}%"
    color = "#10b981" if ok else "#ef4444"
    icon  = "✅" if ok else "❌"
    mins, secs = divmod(int(elapsed), 60)
    elapsed_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    return f"""
<style>
.cineml-bar-wrap-done {{
  background: #e2e8f0; border-radius: 8px;
  height: 22px; width: 100%; overflow: hidden; margin: 6px 0 4px;
}}
.cineml-bar-fill-done {{
  height: 100%; width: {bar_w}; min-width: 18px;
  border-radius: 8px; background: {color};
}}
.cineml-meta {{ display:flex; justify-content:space-between;
  font-size:12px; color:#64748b; margin-top:2px; }}
.cineml-label {{ font-size:13px; font-weight:500; color:#1e293b; margin-bottom:2px; }}
</style>
<div>
  <div class="cineml-label">{icon} {label}</div>
  <div class="cineml-bar-wrap-done"><div class="cineml-bar-fill-done"></div></div>
  <div class="cineml-meta">
    <span>{current:,} of {total:,} &nbsp;·&nbsp; {pct*100:.1f}%</span>
    <span>{elapsed_str}</span>
  </div>
</div>
"""


# ── Main widget ────────────────────────────────────────────────────────────────

def run_script(
    key: str,
    label: str,
    script: str,
    args: list[str] | None = None,
    cwd: Path | None = None,
    description: str = "",
    warn_long: bool = False,
) -> None:
    _init(key)
    run  = _state(key)
    cwd  = cwd or ROOT
    args = args or []
    cmd  = [sys.executable, str(ROOT / script)] + [str(a) for a in args]

    # ── Header row ─────────────────────────────────────────────────────────────
    col_label, col_status, col_btn, col_clear = st.columns([3, 1.4, 1.2, 0.8])

    with col_label:
        st.markdown(f"**{label}**")
        if description:
            st.caption(description)

    with col_status:
        s = run["status"]
        p = run.get("progress")
        if s == "idle":
            st.caption("—")
        elif s == "running":
            elapsed = time.time() - run["started"].timestamp()
            st.caption(f"⏳ {elapsed:.0f}s")
        elif s == "done":
            if p:
                st.caption(f"✅ {p['current']:,}/{p['total']:,}")
            else:
                st.caption(f"✅ {run['elapsed']:.0f}s")
        elif s == "error":
            st.caption(f"❌ code {run['returncode']}")

    with col_btn:
        clicked = st.button(
            "▶ Run" if run["status"] != "running" else "Running…",
            key=f"btn_{key}",
            disabled=(run["status"] == "running"),
            use_container_width=True,
            type="primary" if run["status"] == "idle" else "secondary",
        )

    with col_clear:
        if run["status"] in ("done", "error"):
            if st.button("Clear", key=f"clear_{key}", use_container_width=True):
                st.session_state["_runs"][key] = {
                    "status": "idle", "started": None, "elapsed": None,
                    "lines": [], "progress": None, "returncode": None,
                }
                st.rerun()

    # ── Execute ────────────────────────────────────────────────────────────────
    if clicked and run["status"] != "running":
        run.update(status="running", started=datetime.now(),
                   lines=[], progress=None, returncode=None)

        bar_slot = st.empty()   # animated bar lives here
        # No log shown during run — all lines buffered silently

        try:
            proc = subprocess.Popen(
                cmd, cwd=str(cwd),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
            )
            q: queue.Queue = queue.Queue()
            threading.Thread(target=_enqueue, args=(proc.stdout, q), daemon=True).start()

            t0           = time.time()
            last_render  = 0.0     # timestamp of last UI update
            progress_seen = False

            while True:
                # Drain queue non-blocking
                drained = False
                while True:
                    try:
                        line = q.get_nowait()
                    except queue.Empty:
                        break

                    if line is None:
                        drained = True
                        break

                    stripped = line.strip()

                    # Progress protocol line
                    m = _PROGRESS_RE.match(stripped)
                    if m:
                        run["progress"] = {
                            "current": int(m.group(1)),
                            "total":   int(m.group(2)),
                            "label":   m.group(3),
                        }
                        progress_seen = True
                        continue

                    # Silently drop tqdm noise and blank lines
                    if _TQDM_RE.match(stripped) or _NOISE_RE.match(stripped):
                        continue

                    # Real log line — buffer it (shown only after completion)
                    run["lines"].append(line)

                if drained:
                    break

                # Check if process finished
                if proc.poll() is not None and q.empty():
                    break

                # Throttled UI render — once per UPDATE_INTERVAL seconds
                now = time.time()
                if now - last_render >= UPDATE_INTERVAL:
                    elapsed = now - t0
                    prog = run.get("progress")
                    if prog:
                        bar_slot.markdown(
                            _progress_html(prog["current"], prog["total"],
                                           prog["label"], elapsed),
                            unsafe_allow_html=True,
                        )
                    else:
                        bar_slot.caption(f"⏳ Running… {elapsed:.0f}s")
                    last_render = now

                time.sleep(0.2)   # tight poll — keeps queue drained without busy-wait

            proc.wait()
            elapsed = time.time() - t0
            ok      = proc.returncode == 0
            run.update(status="done" if ok else "error",
                       elapsed=elapsed, returncode=proc.returncode)

            # Final bar render
            prog = run.get("progress")
            if prog:
                bar_slot.markdown(
                    _done_bar_html(prog["current"], prog["total"],
                                   prog["label"], elapsed, ok),
                    unsafe_allow_html=True,
                )
            else:
                bar_slot.caption("✅ Done" if ok else f"❌ Exit {proc.returncode}")

        except FileNotFoundError:
            run["lines"].append(f"ERROR: script not found — {ROOT / script}\n")
            run.update(status="error", elapsed=0.0, returncode=-1)
            bar_slot.caption("❌ Script not found")
        except Exception as exc:
            run["lines"].append(f"ERROR: {exc}\n")
            run.update(status="error", elapsed=0.0, returncode=-1)
            bar_slot.caption(f"❌ {exc}")

        st.rerun()

    # ── Show completed run ──────────────────────────────────────────────────────
    elif run["status"] in ("done", "error"):
        prog    = run.get("progress")
        elapsed = run.get("elapsed") or 0
        ok      = run["status"] == "done"

        if prog:
            st.markdown(
                _done_bar_html(prog["current"], prog["total"],
                               prog["label"], elapsed, ok),
                unsafe_allow_html=True,
            )
        else:
            st.caption("✅ Completed" if ok else f"❌ Failed (code {run['returncode']})")

        # Log only visible if there are real lines (errors, warnings)
        if run["lines"]:
            with st.expander(
                f"{'⚠️ Output log' if ok else '❌ Error log'} ({len(run['lines'])} lines)",
                expanded=not ok,
            ):
                st.code("".join(run["lines"]), language="bash")


# ── Session run history ────────────────────────────────────────────────────────

def render_run_history() -> None:
    runs = st.session_state.get("_runs", {})
    rows = []
    for key, r in runs.items():
        if r["status"] == "idle":
            continue
        p = r.get("progress")
        rows.append({
            "Script":   key.replace("_", " ").title(),
            "Status":   {"running": "⏳", "done": "✅", "error": "❌"}.get(r["status"], "—"),
            "Progress": f"{p['current']:,}/{p['total']:,}" if p else "—",
            "Duration": f"{r['elapsed']:.0f}s" if r["elapsed"] else "—",
            "Exit":     str(r["returncode"]) if r["returncode"] is not None else "—",
        })
    if rows:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No scripts have been run yet this session.")
