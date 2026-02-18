
from __future__ import annotations

from typing import Optional

from rewardlab.core.types import FeedbackSignal, Judge, Trajectory
from rewardlab.judges.data_judge import DataJudge


class ExactMatchReward:
    """
    Backward-compatible reward wrapper.
    Internally delegates to a Judge (data-backed by default).
    """
    name = "exact_match"

    def __init__(self, judge: Optional[Judge] = None):
        self.judge = judge or DataJudge(field="answer")

    def score(self, traj: Trajectory) -> FeedbackSignal:
        return self.judge.evaluate(traj)

