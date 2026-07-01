"""
Multi-agent shared state — the data structure that flows through the LangGraph.

STATE DESIGN PRINCIPLES
------------------------
In LangGraph, all agents share a single State object. Every node (agent)
reads from the current state and returns a partial update. LangGraph merges
the update into the accumulated state before passing it to the next node.

The state design determines what agents can know about each other's work:
  - messages:       Full conversation history (supervisor ↔ specialist turns)
  - goal:           The original user request — always visible to all agents
  - active_agent:   Which specialist is currently running
  - context:        Discovered facts (dataset_id, experiment_id, artifact paths)
                    — specialists write here so successors can build on prior work
  - agent_scratchpad: Per-turn working notes (overwritten each turn, not accumulated)
  - step_count:     Guard against infinite loops
  - final_answer:   Set by supervisor when the goal is complete

WHY TypedDict AND NOT A DATACLASS
----------------------------------
LangGraph requires TypedDict for its state because it uses Python type
annotations to detect which fields should be merged vs replaced on each
update. Fields annotated with Annotated[list, add_messages] are accumulated
(new items appended); plain fields are replaced (new value overwrites old).
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AgentContext(TypedDict, total=False):
    """
    Structured context that specialists write and successors read.
    All fields are optional — specialists only populate what they discover.
    """
    dataset_id:      Optional[int]
    dataset_name:    Optional[str]
    experiment_id:   Optional[int]
    task_type:       Optional[str]       # "classification" | "regression" | "anomaly"
    target_column:   Optional[str]
    best_model:      Optional[str]
    best_score:      Optional[float]
    artifact_path:   Optional[str]
    onnx_path:       Optional[str]
    deployment_url:  Optional[str]
    anomaly_report:  Optional[dict]
    fairness_report: Optional[dict]
    profiling_done:  bool
    training_done:   bool
    deployment_done: bool


class MultiAgentState(TypedDict):
    """
    Shared state flowing through the multi-agent LangGraph.

    LangGraph semantics:
      messages        — Annotated with add_messages: each update APPENDS to history
      goal            — Replaced each turn (but supervisor never changes it)
      active_agent    — Replaced: which specialist is currently running
      context         — Replaced: merged dict of discovered facts
      agent_output    — Replaced: the most recent specialist's output text
      step_count      — Replaced: incremented each turn
      final_answer    — Replaced: set once the supervisor decides the goal is complete
      error           — Replaced: set on failures
    """
    # Accumulated conversation history (supervisor ↔ specialist dialogue)
    messages:      Annotated[list, add_messages]

    # Stable across the entire run
    goal:          str

    # Updated each turn by the supervisor
    active_agent:  str           # "supervisor" | "dataset_analyst" | "model_trainer" | "deployer"
    next_agent:    str           # where to route after current node
    step_count:    int

    # Written by specialists, read by supervisor and successors
    context:       AgentContext
    agent_output:  str           # text summary of what the current specialist did

    # Terminal state
    final_answer:  str
    error:         str


# Route names used in conditional edges
ROUTE_SUPERVISOR       = "supervisor"
ROUTE_DATASET_ANALYST  = "dataset_analyst"
ROUTE_MODEL_TRAINER    = "model_trainer"
ROUTE_DEPLOYER         = "deployer"
ROUTE_END              = "__end__"

ALL_SPECIALISTS = [ROUTE_DATASET_ANALYST, ROUTE_MODEL_TRAINER, ROUTE_DEPLOYER]

MAX_STEPS = 15   # hard limit — supervisor loops can be expensive
