from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from rewardlab.core.types import FeedbackSignal, Objective


@dataclass
class SupervisedObjective(Objective):
    """
    Minimal objective for 'toy_math'-style tasks where the dataset provides the target
    (often stored in batch or trajectory.metadata["answer"]).

    This objective does NOT use reward for learning; it is here as a clean baseline.
    """

    def compute_loss(
        self,
        feedback: FeedbackSignal,
        model_outputs: Dict[str, Any],
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Expect the model to return a precomputed loss (common in HF training)
        # If your model doesn't return it, adapt here.
        loss = model_outputs.get("loss", None)
        if loss is None:
            raise ValueError("SupervisedObjective expects model_outputs['loss'] to exist.")

        logs = {
            "loss": float(loss.detach().cpu().item()) if hasattr(loss, "detach") else float(loss),
        }

        # Optional: log reward for analysis (even if not used for training)
        if feedback.kind == "scalar" and feedback.total is not None:
            logs["reward_total"] = float(feedback.total)

        return {"loss": loss, "logs": logs}
