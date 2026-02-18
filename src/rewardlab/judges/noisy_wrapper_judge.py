from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from rewardlab.core.types import FeedbackSignal, Judge, Trajectory

from rewardlab.core.registry import JUDGES

@JUDGES.register("noisy_judge")

@dataclass
class NoisyJudge(Judge):
    """
    Wrap another Judge and inject controlled noise.

    - If feedback.kind == "scalar":
        * if in [0,1], flips: r -> 1-r
        * else, negates: r -> -r
    - If feedback.kind == "pairwise":
        * swaps chosen/rejected in payload with prob flip_prob

    Useful for robustness experiments.
    """
    base: Judge
    flip_prob: float = 0.1
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)

    def evaluate(self, trajectory: Trajectory) -> FeedbackSignal:
        fb = self.base.evaluate(trajectory)

        flipped = False
        if random.random() < self.flip_prob:
            if fb.kind == "scalar":
                if fb.total is not None:
                    r = float(fb.total)
                    if 0.0 <= r <= 1.0:
                        fb.total = 1.0 - r
                    else:
                        fb.total = -r
                    flipped = True

            elif fb.kind == "pairwise":
                chosen = fb.payload.get("chosen", None)
                rejected = fb.payload.get("rejected", None)
                if chosen is not None and rejected is not None:
                    fb.payload["chosen"], fb.payload["rejected"] = rejected, chosen
                    flipped = True

        fb.metadata["noisy_wrapper"] = True
        fb.metadata["flip_prob"] = self.flip_prob
        fb.metadata["flipped"] = flipped
        return fb

