from dataclasses import dataclass
from typing import Any, Dict, Optional

from rewardlab.core.types import FeedbackSignal, Objective

from rewardlab.core.registry import OBJECTIVES
@OBJECTIVES.register("scalar_eval")

@dataclass
class ScalarEvalObjective(Objective):
    """
    Demo-only objective.
    Uses reward as signal and defines loss = -reward.
    """

    def compute_loss(
        self,
        feedback: FeedbackSignal,
        model_outputs: Dict[str, Any],
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        r = float(feedback.total) if (
            feedback.kind == "scalar" and feedback.total is not None
        ) else 0.0

        loss = -r

        return {
            "loss": loss,
            "logs": {
                "reward_total": r,
                "loss": loss,
            },
        }
