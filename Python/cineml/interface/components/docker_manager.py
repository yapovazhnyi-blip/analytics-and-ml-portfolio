"""
interface/components/docker_manager.py
========================================
Controls Docker Compose services from inside Streamlit.

Provides:
    render_docker_panel()  — full panel: status table + per-service controls
    docker_available()     — True if `docker` binary is on PATH

How it works:
    All Docker operations run as subprocesses via subprocess.run() or
    subprocess.Popen(). The compose project is the repo root (where
    docker-compose.yml lives). Service state is polled every time the
    panel renders; Streamlit's rerun loop keeps it live.

Services managed:
    recommender-api   — M2 FastAPI  → localhost:8001
    ab-api            — M3 FastAPI  → localhost:8002
    ab-dashboard      — M3 Streamlit→ localhost:8501
    diffusion-demo    — M5 Gradio   → localhost:7860
"""
import subprocess
import sys
import threading
import queue
import time
from pathlib import Path
from typing import Literal

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
COMPOSE_FILE = ROOT / "docker-compose.yml"

ServiceName = Literal[
    "recommender-api",
    "ab-api",
    "ab-dashboard",
    "diffusion-demo",
]

SERVICE_META: dict[str, dict] = {
    "recommender-api": {
        "label":   "M2 — Recommender API",
        "port":    8001,
        "url":     "http://localhost:8001/health",
        "desc":    "FastAPI serving ALS & Two-Tower recommendations",
        "icon":    "🎯",
    },
    "ab-api": {
        "label":   "M3 — A/B Engine API",
        "port":    8002,
        "url":     "http://localhost:8002/health",
        "desc":    "FastAPI exposing frequentist + Bayesian test endpoints",
        "icon":    "📊",
    },
    "ab-dashboard": {
        "label":   "M3 — A/B Dashboard",
        "port":    8501,
        "url":     "http://localhost:8501",
        "desc":    "Streamlit A/B analysis dashboard",
        "icon":    "📈",
    },
    "diffusion-demo": {
        "label":   "M5 — Diffusion Demo",
        "port":    7860,
        "url":     "http://localhost:7860",
        "desc":    "Gradio UI: generate posters + ViT classifier",
        "icon":    "🎨",
    },
}


# ── Subprocess helpers ─────────────────────────────────────────────────────────

def _run(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a docker/compose command from the repo root."""
    return subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def docker_available() -> bool:
    try:
        r = _run(["docker", "info"])
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def compose_available() -> bool:
    """Check whether docker-compose.yml exists."""
    return COMPOSE_FILE.exists()


# ── Service state queries ──────────────────────────────────────────────────────

def _get_ps() -> dict[str, str]:
    """
    Return {service_name: status_string} for all compose services.
    Status is one of: "running", "exited", "not created", "unknown".
    """
    states: dict[str, str] = {s: "not created" for s in SERVICE_META}
    try:
        r = _run([
            "docker", "compose", "-f", str(COMPOSE_FILE),
            "ps", "--format", "{{.Service}}\t{{.State}}",
        ])
        if r.returncode != 0:
            # Older docker-compose (v1) fallback
            r = _run(["docker-compose", "-f", str(COMPOSE_FILE), "ps"])
        for line in r.stdout.splitlines():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                svc, state = parts[0].strip(), parts[1].strip().lower()
                if svc in states:
                    states[svc] = state
    except Exception:
        pass
    return states


def _get_logs(service: str, tail: int = 60) -> str:
    try:
        r = _run([
            "docker", "compose", "-f", str(COMPOSE_FILE),
            "logs", "--tail", str(tail), "--no-color", service,
        ])
        return r.stdout or r.stderr or "(no output)"
    except Exception as e:
        return f"Error fetching logs: {e}"


def _status_icon(state: str) -> str:
    return {"running": "🟢", "exited": "🔴", "not created": "⚫"}.get(state, "🟡")


# ── Streaming log helper ───────────────────────────────────────────────────────

def _stream_compose_cmd(args: list[str], log_placeholder) -> int:
    """
    Run a docker compose command and stream its output into a Streamlit
    placeholder line-by-line. Returns the process exit code.
    """
    lines: list[str] = []
    try:
        proc = subprocess.Popen(
            args,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        q: queue.Queue = queue.Queue()

        def _reader():
            for line in iter(proc.stdout.readline, ""):
                q.put(line)
            q.put(None)

        threading.Thread(target=_reader, daemon=True).start()

        while True:
            try:
                line = q.get(timeout=0.15)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                log_placeholder.code("".join(lines[-50:]) + "\n⏳ waiting…",
                                      language="bash")
                continue
            if line is None:
                break
            lines.append(line)
            log_placeholder.code("".join(lines[-50:]), language="bash")

        proc.wait()
        return proc.returncode
    except FileNotFoundError:
        lines.append("ERROR: docker not found on PATH.\n")
        log_placeholder.code("".join(lines), language="bash")
        return -1


# ── Main panel ─────────────────────────────────────────────────────────────────

def render_docker_panel() -> None:
    """
    Render the full Docker Compose control panel.
    Call this from any Streamlit page.
    """
    st.subheader("🐳  Docker Compose")

    if not docker_available():
        st.error(
            "Docker is not running or not installed. "
            "Start Docker Desktop and try again.",
            icon="🐳",
        )
        with st.expander("Install Docker"):
            st.markdown("""
**Windows / macOS** — [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Linux**:
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # then log out and back in
```

After installing, start Docker Desktop or run `sudo systemctl start docker`.
""")
        return

    if not compose_available():
        st.error(f"`docker-compose.yml` not found at `{COMPOSE_FILE}`")
        return

    # ── Global controls ────────────────────────────────────────────────────────
    col_up, col_down, col_build, col_refresh = st.columns(4)

    log_area = st.empty()

    with col_up:
        if st.button("▶ Start all", type="primary", use_container_width=True,
                     key="docker_up_all"):
            with st.spinner("Starting all services…"):
                rc = _stream_compose_cmd(
                    ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"],
                    log_area,
                )
            if rc == 0:
                st.success("All services started.")
            else:
                st.error(f"docker compose up exited with code {rc}.")
            st.rerun()

    with col_down:
        if st.button("⏹ Stop all", use_container_width=True, key="docker_down_all"):
            with st.spinner("Stopping all services…"):
                rc = _stream_compose_cmd(
                    ["docker", "compose", "-f", str(COMPOSE_FILE), "down"],
                    log_area,
                )
            if rc == 0:
                st.success("All services stopped.")
            else:
                st.error(f"docker compose down exited with code {rc}.")
            st.rerun()

    with col_build:
        if st.button("🔨 Build & start", use_container_width=True,
                     key="docker_build_all"):
            with st.spinner("Building images (this can take several minutes)…"):
                rc = _stream_compose_cmd(
                    ["docker", "compose", "-f", str(COMPOSE_FILE),
                     "up", "--build", "-d"],
                    log_area,
                )
            if rc == 0:
                st.success("Build complete. All services started.")
            else:
                st.error(f"Build failed with exit code {rc}.")
            st.rerun()

    with col_refresh:
        if st.button("🔄 Refresh", use_container_width=True, key="docker_refresh"):
            st.rerun()

    st.divider()

    # ── Per-service rows ───────────────────────────────────────────────────────
    ps = _get_ps()

    for svc, meta in SERVICE_META.items():
        state = ps.get(svc, "not created")
        icon  = _status_icon(state)

        with st.container(border=True):
            hcol, scol, ucol, bcol1, bcol2, bcol3 = st.columns([2.5, 1, 1.5, 0.8, 0.8, 0.8])

            hcol.markdown(
                f"{meta['icon']}  **{meta['label']}**  \n"
                f"<small style='color:grey'>{meta['desc']}</small>",
                unsafe_allow_html=True,
            )
            scol.markdown(f"{icon} `{state}`")
            ucol.markdown(
                f"[:{meta['port']}]({meta['url']})"
                if state == "running" else f"port {meta['port']}"
            )

            svc_log = st.empty()   # log placeholder for this service

            with bcol1:
                start_disabled = state == "running"
                if st.button("▶", key=f"start_{svc}", disabled=start_disabled,
                              help="Start this service", use_container_width=True):
                    _stream_compose_cmd(
                        ["docker", "compose", "-f", str(COMPOSE_FILE),
                         "up", "-d", svc],
                        svc_log,
                    )
                    st.rerun()

            with bcol2:
                stop_disabled = state != "running"
                if st.button("⏹", key=f"stop_{svc}", disabled=stop_disabled,
                              help="Stop this service", use_container_width=True):
                    _stream_compose_cmd(
                        ["docker", "compose", "-f", str(COMPOSE_FILE),
                         "stop", svc],
                        svc_log,
                    )
                    st.rerun()

            with bcol3:
                if st.button("📋", key=f"logs_{svc}",
                              help="Show recent logs", use_container_width=True):
                    st.session_state[f"show_logs_{svc}"] = (
                        not st.session_state.get(f"show_logs_{svc}", False)
                    )

            # Inline log viewer (toggled by 📋 button)
            if st.session_state.get(f"show_logs_{svc}", False):
                logs = _get_logs(svc, tail=80)
                st.code(logs, language="bash")
                if st.button("Rebuild this service", key=f"rebuild_{svc}"):
                    with st.spinner(f"Rebuilding {svc}…"):
                        _stream_compose_cmd(
                            ["docker", "compose", "-f", str(COMPOSE_FILE),
                             "up", "--build", "-d", svc],
                            st.empty(),
                        )
                    st.rerun()

    # ── System info ────────────────────────────────────────────────────────────
    with st.expander("Docker system info"):
        try:
            r = _run(["docker", "system", "df"])
            st.code(r.stdout, language="bash")
        except Exception as e:
            st.code(f"Error: {e}")

        if st.button("Prune unused images / volumes", key="docker_prune"):
            rc = _stream_compose_cmd(
                ["docker", "system", "prune", "-f"], st.empty()
            )
            st.rerun()
