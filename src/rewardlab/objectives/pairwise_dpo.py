from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rewardlab.core.types import FeedbackSignal, Objective


def _logsigmoid(x: float) -> float:
    # numerically stable logsigmoid for floats (demo-friendly)
    # for torch tensors, you can replace this with torch.nn.functional.logsigmoid
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))

from rewardlab.core.registry import OBJECTIVES

@OBJECTIVES.register("pairwise_dpo")

@dataclass
class PairwiseDPOObjective(Objective):
    """
    Minimal DPO-style objective.

    Requires model_outputs to provide:
      - logp_chosen, logp_rejected
      - ref_logp_chosen, ref_logp_rejected

    Loss (scalar float version):
      L = -log sigma( beta * [ (logp_c - logp_r) - (ref_logp_c - ref_logp_r) ] )

    If you use torch, pass tensors instead of floats and replace _logsigmoid().
    """
    beta: float = 0.1

    def compute_loss(
        self,
        feedback: FeedbackSignal,
        model_outputs: Dict[str, Any],
        batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if feedback.kind != "pairwise":
            raise ValueError(f"PairwiseDPOObjective expects feedback.kind='pairwise', got {feedback.kind}")

        # Extract log-probs from model_outputs
        required = ["logp_chosen", "logp_rejected", "ref_logp_chosen", "ref_logp_rejected"]
        for k in required:
            if k not in model_outputs:
                raise KeyError(
                    f"Missing '{k}' in model_outputs. "
                    f"PairwiseDPOObjective requires {required}."
                )

        logp_c = model_outputs["logp_chosen"]
        logp_r = model_outputs["logp_rejected"]
        ref_logp_c = model_outputs["ref_logp_chosen"]
        ref_logp_r = model_outputs["ref_logp_rejected"]

        # Support either floats or torch tensors
        is_torch = hasattr(logp_c, "detach")

        delta_pi = logp_c - logp_r
        delta_ref = ref_logp_c - ref_logp_r
        logits = self.beta * (delta_pi - delta_ref)

        if is_torch:
            import torch.nn.functional as F
            loss = -F.logsigmoid(logits)
            logs = {
                "dpo_loss": float(loss.detach().cpu().item()),
                "beta": float(self.beta),
                "logits": float(logits.detach().cpu().item()),
                "delta_pi": float(delta_pi.detach().cpu().item()),
                "delta_ref": float(delta_ref.detach().cpu().item()),
            }
        else:
            loss = -_logsigmoid(float(logits))
            logs = {
                "dpo_loss": float(loss),
                "beta": float(self.beta),
                "logits": float(logits),
                "delta_pi": float(delta_pi),
                "delta_ref": float(delta_ref),
            }

        # Optional: if you store uncertainty/confidence in feedback.metadata, log it
        if "confidence" in feedback.metadata:
            logs["label_confidence"] = float(feedback.metadata["confidence"])

        return {"loss": loss, "logs": logs}
