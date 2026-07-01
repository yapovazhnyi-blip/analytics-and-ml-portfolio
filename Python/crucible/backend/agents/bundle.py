"""
.crucible Bundle Format — portable packaging for fine-tuned agents.

WHAT'S IN A BUNDLE
---------------------
A .crucible file is a ZIP archive with a fixed structure:

  my-agent.crucible
  ├── manifest.json          — metadata: base model, training config, eval results
  ├── adapter/                — LoRA adapter weights (PEFT format)
  │   ├── adapter_config.json
  │   └── adapter_model.safetensors  (or .bin)
  ├── traces_sample.jsonl     — up to 20 example traces used in training,
  │                             for transparency and reproducibility
  └── benchmark_results.json  — pass/fail scores from the agent benchmark
                                 (omitted if the agent has not been benchmarked)

WHY ZIP + JSON MANIFEST, NOT A CUSTOM BINARY FORMAT
------------------------------------------------------
The LoRA adapter itself is already a standard format — PEFT's
`save_pretrained()` produces `adapter_config.json` + safetensors weights
that any `transformers`/`peft` installation can load directly with
`PeftModel.from_pretrained()`. Wrapping it in a ZIP with a manifest adds:
  - Portability: a single file to email, upload, or commit to Git LFS
  - Self-description: the manifest tells you what base model + training
    method + eval scores it has, without loading the adapter first
  - Provenance: the included trace sample shows exactly what kind of
    data the adapter learned from

This is intentionally similar to how Hugging Face Hub model repos work
(config + weights + README), just bundled into one file instead of a
directory, since Crucible's deployment story favours single-artifact
downloads (see deployment/onnx_exporter.py for the same pattern with ONNX).

IMPORT TARGETS
---------------
A .crucible bundle can be imported by:
  - Another Crucible instance (POST /agents/import) — full integration,
    registers it in the Agent Registry, makes it selectable for /agent/run
  - Any PEFT-compatible system — extract adapter/, load with
    `PeftModel.from_pretrained(base_model, "adapter/")`
  - Hugging Face Hub — adapter/ can be pushed directly via
    `huggingface_hub.upload_folder()`
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BUNDLE_VERSION = "1.0"


@dataclass
class AgentConfig:
    """The agent's runtime configuration — system prompt, tools, limits."""
    system_prompt: str = ""
    tool_names: list[str] = field(default_factory=list)
    max_steps: int = 10
    agent_type: str = "react"   # "react" | "multi"

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "tool_names":    self.tool_names,
            "max_steps":     self.max_steps,
            "agent_type":    self.agent_type,
        }


@dataclass
class BundleManifest:
    """Top-level manifest.json contents."""
    name: str
    base_model: str
    training_method: str          # "sft" | "dpo"
    n_training_traces: int
    crucible_version: str
    agent_config: AgentConfig
    created_at: str
    description: str = ""
    eval_results: Optional[dict] = None
    bundle_version: str = BUNDLE_VERSION

    def to_dict(self) -> dict:
        return {
            "name":               self.name,
            "description":        self.description,
            "base_model":         self.base_model,
            "training_method":    self.training_method,
            "n_training_traces":  self.n_training_traces,
            "crucible_version":   self.crucible_version,
            "agent_config":       self.agent_config.to_dict(),
            "created_at":         self.created_at,
            "eval_results":       self.eval_results,
            "bundle_version":     self.bundle_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BundleManifest":
        cfg = data.get("agent_config", {})
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            base_model=data["base_model"],
            training_method=data.get("training_method", "sft"),
            n_training_traces=data.get("n_training_traces", 0),
            crucible_version=data.get("crucible_version", "unknown"),
            agent_config=AgentConfig(
                system_prompt=cfg.get("system_prompt", ""),
                tool_names=cfg.get("tool_names", []),
                max_steps=cfg.get("max_steps", 10),
                agent_type=cfg.get("agent_type", "react"),
            ),
            created_at=data.get("created_at", ""),
            eval_results=data.get("eval_results"),
            bundle_version=data.get("bundle_version", BUNDLE_VERSION),
        )


def export_bundle(
    output_path: str,
    name: str,
    base_model: str,
    adapter_dir: str,
    agent_config: AgentConfig,
    training_method: str = "sft",
    n_training_traces: int = 0,
    description: str = "",
    eval_results: Optional[dict] = None,
    traces_sample: Optional[list[dict]] = None,
) -> str:
    """
    Packages an adapter + config + sample traces into a .crucible bundle.

    Args:
        output_path:   Where to write the .crucible file.
        name:          Agent name (used in manifest, must be filesystem-safe).
        base_model:    HuggingFace model ID the adapter was trained against.
        adapter_dir:   Directory containing adapter_config.json + weights
                       (output of PeftModel.save_pretrained()).
        agent_config:  Tool/prompt configuration for this agent.
        training_method: "sft" or "dpo".
        n_training_traces: Number of traces used in training (for the manifest).
        eval_results:  Optional benchmark results dict.
        traces_sample: Optional list of example trace dicts (max 20 stored).

    Returns:
        The output_path (for chaining).
    """
    manifest = BundleManifest(
        name=name,
        description=description,
        base_model=base_model,
        training_method=training_method,
        n_training_traces=n_training_traces,
        crucible_version="1.0.0",
        agent_config=agent_config,
        created_at=datetime.now(timezone.utc).isoformat(),
        eval_results=eval_results,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # manifest.json
        zf.writestr("manifest.json", json.dumps(manifest.to_dict(), indent=2))

        # adapter/ — copy every file from adapter_dir
        adapter_path = Path(adapter_dir)
        if adapter_path.is_dir():
            for f in adapter_path.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f"adapter/{f.name}")

        # traces_sample.jsonl — up to 20 examples
        if traces_sample:
            lines = "\n".join(json.dumps(t) for t in traces_sample[:20])
            zf.writestr("traces_sample.jsonl", lines)

        # benchmark_results.json
        if eval_results:
            zf.writestr("benchmark_results.json", json.dumps(eval_results, indent=2))

    return output_path


@dataclass
class ImportedBundle:
    manifest: BundleManifest
    adapter_dir: str          # extracted to a temp directory
    traces_sample: list[dict]
    benchmark_results: Optional[dict]


def import_bundle(bundle_path: str, extract_to: Optional[str] = None) -> ImportedBundle:
    """
    Extracts and validates a .crucible bundle.

    Args:
        bundle_path: Path to the .crucible (ZIP) file.
        extract_to:  Directory to extract the adapter into. If None, a
                     temp directory is created.

    Returns:
        ImportedBundle with the parsed manifest and extracted adapter path.

    Raises:
        ValueError: if the file is not a valid .crucible bundle (missing
        manifest.json, corrupt ZIP, or unsupported bundle_version).
    """
    if not zipfile.is_zipfile(bundle_path):
        raise ValueError(f"{bundle_path} is not a valid ZIP/.crucible bundle")

    extract_dir = extract_to or tempfile.mkdtemp(prefix="crucible-import-")

    with zipfile.ZipFile(bundle_path, "r") as zf:
        names = zf.namelist()
        if "manifest.json" not in names:
            raise ValueError("Bundle is missing manifest.json — not a valid .crucible bundle")

        manifest_data = json.loads(zf.read("manifest.json"))
        manifest = BundleManifest.from_dict(manifest_data)

        if manifest.bundle_version != BUNDLE_VERSION:
            raise ValueError(
                f"Unsupported bundle version {manifest.bundle_version!r}; "
                f"this Crucible instance supports {BUNDLE_VERSION!r}"
            )

        # Extract adapter files
        adapter_dir = os.path.join(extract_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        for name in names:
            if name.startswith("adapter/") and not name.endswith("/"):
                target = os.path.join(extract_dir, name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # traces_sample.jsonl
        traces_sample = []
        if "traces_sample.jsonl" in names:
            content = zf.read("traces_sample.jsonl").decode("utf-8")
            for line in content.splitlines():
                if line.strip():
                    traces_sample.append(json.loads(line))

        # benchmark_results.json
        benchmark_results = None
        if "benchmark_results.json" in names:
            benchmark_results = json.loads(zf.read("benchmark_results.json"))

    return ImportedBundle(
        manifest=manifest,
        adapter_dir=adapter_dir,
        traces_sample=traces_sample,
        benchmark_results=benchmark_results,
    )
