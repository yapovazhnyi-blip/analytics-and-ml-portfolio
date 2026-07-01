"""
Experiment lineage DAG for Crucible Phase 3.

Builds a directed acyclic graph (DAG) that captures the full provenance
of an experiment:

  DataVersion ──► PreprocessingStep ──► ModelConfig ──► EvaluationResult

Each node carries its metadata as attributes. The graph is serialised
to a JSON-safe dict that React Flow (+ dagre layout) can render directly
without any frontend transformation.

Design decisions validated in the spike:
  - NetworkX DiGraph for the graph structure (lightweight, no server needed)
  - Adjacency list + node attributes → JSON (no heavy graph DB)
  - Data versioned by SHA-256 hash of file content (already computed at ingest)
  - Config versioned by deterministic hash of the config dict
  - Parallel experiments from the same data/preprocessing share nodes
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx


# ── Node types (mirror the React Flow spike) ───────────────────────────────

class NodeType:
    DATA_VERSION   = "data_version"
    PREPROCESSING  = "preprocessing"
    MODEL_CONFIG   = "model_config"
    EVALUATION     = "evaluation"


@dataclass
class LineageNode:
    node_id: str
    node_type: str
    label: str
    subtitle: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class LineageEdge:
    source: str
    target: str
    label: str = ""


@dataclass
class LineageGraph:
    nodes: list[LineageNode]
    edges: list[LineageEdge]
    experiment_id: int
    root_node_id: str         # the data version node

    def to_react_flow(self) -> dict:
        """
        Serialises to the shape React Flow expects.
        dagre layout is applied client-side (validated in spike).
        """
        rf_nodes = [
            {
                "id": n.node_id,
                "type": "lineage",
                "data": {
                    "nodeType": n.node_type,
                    "label": n.label,
                    "subtitle": n.subtitle,
                    "metadata": n.metadata,
                },
                "position": {"x": 0, "y": 0},   # dagre fills this in
            }
            for n in self.nodes
        ]
        rf_edges = [
            {
                "id": f"{e.source}-{e.target}",
                "source": e.source,
                "target": e.target,
                "label": e.label,
            }
            for e in self.edges
        ]
        return {"nodes": rf_nodes, "edges": rf_edges}


# ── Builder ────────────────────────────────────────────────────────────────

def build_lineage(
    experiment_id: int,
    dataset_name: str,
    dataset_row_count: Optional[int],
    dataset_column_count: Optional[int],
    content_hash: Optional[str],
    source_type: str,
    feature_columns: list[str],
    target_column: str,
    task_type: str,
    training_config: dict,
    best_family: Optional[str],
    best_score: Optional[float],
    scoring_metric: Optional[str],
    holdout_metrics: dict,
    n_trials: Optional[int],
    n_pruned: Optional[int],
) -> LineageGraph:
    """
    Constructs the lineage DAG for a single experiment.

    Node IDs are deterministic hashes of their content so that two
    experiments sharing the same dataset + preprocessing config share
    the same nodes in the graph — the graph is a true provenance record,
    not just per-experiment metadata.
    """
    nodes: list[LineageNode] = []
    edges: list[LineageEdge] = []

    # ── Node 1: Data version ───────────────────────────────────────────────
    dv_id = f"dv_{content_hash[:12] if content_hash else _hash({'name': dataset_name})}"
    nodes.append(LineageNode(
        node_id=dv_id,
        node_type=NodeType.DATA_VERSION,
        label=dataset_name,
        subtitle=_format_shape(dataset_row_count, dataset_column_count, source_type),
        metadata={
            "source_type": source_type,
            "row_count": dataset_row_count,
            "column_count": dataset_column_count,
            "content_hash": content_hash,
        },
    ))

    # ── Node 2: Preprocessing step ─────────────────────────────────────────
    pp_config = {
        "features": sorted(feature_columns),
        "target": target_column,
        "task_type": task_type,
    }
    pp_id = f"pp_{_hash(pp_config)}"
    n_dropped = (dataset_column_count or 0) - len(feature_columns) - 1
    pp_subtitle = f"{len(feature_columns)} features · target: {target_column}"
    if n_dropped > 0:
        pp_subtitle += f" · {n_dropped} col(s) dropped"

    nodes.append(LineageNode(
        node_id=pp_id,
        node_type=NodeType.PREPROCESSING,
        label=f"Features ({task_type})",
        subtitle=pp_subtitle,
        metadata=pp_config,
    ))
    edges.append(LineageEdge(source=dv_id, target=pp_id, label="prepare"))

    # ── Node 3: Model config ───────────────────────────────────────────────
    mc_config = {
        "n_trials": training_config.get("n_trials"),
        "cv_folds": training_config.get("cv_folds"),
        "best_family": best_family,
        "task_type": task_type,
        "experiment_id": experiment_id,
    }
    mc_id = f"mc_exp{experiment_id}"
    n_trials_done = n_trials or training_config.get("n_trials", "?")
    pruned_pct = ""
    if n_trials and n_pruned:
        pruned_pct = f" · {n_pruned} pruned"
    mc_subtitle = f"{n_trials_done} trials{pruned_pct}"

    nodes.append(LineageNode(
        node_id=mc_id,
        node_type=NodeType.MODEL_CONFIG,
        label=_family_label(best_family),
        subtitle=mc_subtitle,
        metadata=mc_config,
    ))
    edges.append(LineageEdge(source=pp_id, target=mc_id, label="train"))

    # ── Node 4: Evaluation result ──────────────────────────────────────────
    ev_id = f"ev_exp{experiment_id}"
    score_str = f"{scoring_metric}: {best_score:.4f}" if best_score is not None and scoring_metric else "pending"
    ev_subtitle_parts = [score_str]
    if holdout_metrics:
        first_key = next(iter(holdout_metrics), None)
        if first_key:
            ev_subtitle_parts.append(f"holdout {first_key.replace('holdout_', '')}: {holdout_metrics[first_key]:.4f}")
    ev_subtitle = " · ".join(ev_subtitle_parts)

    nodes.append(LineageNode(
        node_id=ev_id,
        node_type=NodeType.EVALUATION,
        label="Results",
        subtitle=ev_subtitle,
        metadata={
            "best_score": best_score,
            "scoring_metric": scoring_metric,
            "holdout_metrics": holdout_metrics,
        },
    ))
    edges.append(LineageEdge(source=mc_id, target=ev_id, label="evaluate"))

    return LineageGraph(
        nodes=nodes,
        edges=edges,
        experiment_id=experiment_id,
        root_node_id=dv_id,
    )


def build_multi_experiment_lineage(experiments: list[dict]) -> dict:
    """
    Builds a combined lineage graph across multiple experiments on the
    same dataset. Shared data version and preprocessing nodes are merged
    (same deterministic ID = same node in the graph).

    Returns a React Flow-compatible dict ready to send to the frontend.
    """
    g = nx.DiGraph()
    node_metadata: dict[str, dict] = {}

    for exp in experiments:
        lg = build_lineage(**exp)
        for node in lg.nodes:
            if node.node_id not in g:
                g.add_node(node.node_id, **{
                    "node_type": node.node_type,
                    "label": node.label,
                    "subtitle": node.subtitle,
                    "metadata": node.metadata,
                })
        for edge in lg.edges:
            g.add_edge(edge.source, edge.target, label=edge.label)

    # Serialise graph to React Flow shape
    rf_nodes = [
        {
            "id": nid,
            "type": "lineage",
            "data": {
                "nodeType": data["node_type"],
                "label": data["label"],
                "subtitle": data["subtitle"],
                "metadata": data.get("metadata", {}),
            },
            "position": {"x": 0, "y": 0},
        }
        for nid, data in g.nodes(data=True)
    ]
    rf_edges = [
        {
            "id": f"{u}-{v}",
            "source": u,
            "target": v,
            "label": g.edges[u, v].get("label", ""),
        }
        for u, v in g.edges()
    ]
    return {
        "nodes": rf_nodes,
        "edges": rf_edges,
        "n_experiments": len(experiments),
        "is_dag": nx.is_directed_acyclic_graph(g),
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _format_shape(rows: Optional[int], cols: Optional[int], source: str) -> str:
    parts = []
    if rows is not None:
        parts.append(f"{rows:,} rows")
    if cols is not None:
        parts.append(f"{cols} cols")
    parts.append(source)
    return " · ".join(parts)


def _family_label(family: Optional[str]) -> str:
    labels = {
        "random_forest":       "Random Forest",
        "gradient_boosting":   "Gradient Boosting",
        "logistic_regression": "Logistic Regression",
        "ridge":               "Ridge Regression",
        "svm":                 "SVM",
        "knn":                 "k-Nearest Neighbours",
    }
    if not family:
        return "AutoML Search"
    return labels.get(family, family.replace("_", " ").title())
