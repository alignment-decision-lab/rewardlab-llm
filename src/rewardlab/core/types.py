from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class Step:
    prompt: str
    response: str


@dataclass
class Trajectory:
    steps: List[Step]
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# General supervision / feedback
# -----------------------------
# Add new kinds freely over time. 

FeedbackKind = Literal[
    "scalar",     # one scalar per trajectory/output
    "pairwise",   # preference between two candidates
    "ranking",    # ordered list of candidates
    "trajectory", # per-step or per-episode feedback
    "token",      # token-level feedback (e.g., per-token rewards)
]


@dataclass
class FeedbackSignal:
    """
    Unified container for supervision/feedback signals.

    Common pattern:
      - Put universal info in (total, components).
      - Put kind-specific content in payload.
      - Put uncertainty/provenance in metadata.

    Examples:
      scalar:
        kind="scalar", total=..., components={...}

      pairwise:
        kind="pairwise", payload={"chosen": str, "rejected": str}
        optionally metadata={"confidence": 0.0..1.0, "p_chosen": ...}

      ranking:
        kind="ranking", payload={"ranking": [str, ...]}

      trajectory:
        kind="trajectory", payload={"step_rewards": [float, ...], "final": float}

      token:
        kind="token", payload={"token_rewards": [float, ...], "mask": [0/1, ...]}
    """
    kind: FeedbackKind

    # Optional common scalar summary (useful for logging/comparisons)
    total: Optional[float] = None

    # Optional named breakdown (e.g., length penalty, safety penalty, etc.)
    components: Dict[str, float] = field(default_factory=dict)

    # Kind-specific content (chosen/rejected, ranking list, per-token rewards, etc.)
    payload: Dict[str, Any] = field(default_factory=dict)

    # Uncertainty + provenance (confidence, annotator votes, judge id, seed, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Interfaces (for modularity)
# -----------------------------

class Judge(ABC):
    """
    Produces a FeedbackSignal from a Trajectory.

    Implementations can be:
      - data-backed (offline labels)
      - model-backed (reward model / LLM-as-judge)
      - rule-based
      - ensemble-based
      - noisy wrappers (inject uncertainty)
    """
    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> FeedbackSignal:
        raise NotImplementedError


class Objective(ABC):
    """
    Converts (feedback + model outputs) into a training loss.

    Keep the runner generic: it calls compute_loss() and does not special-case objectives.
    """
    @abstractmethod
    def compute_loss(
        self,
        feedback: FeedbackSignal,
        model_outputs: Dict[str, Any],
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Return a dict with at least:
          - "loss": scalar loss (torch.Tensor or float depending on your stack)
          - "logs": dict of scalar values
        """
        raise NotImplementedError
