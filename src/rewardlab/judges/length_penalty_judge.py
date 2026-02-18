from rewardlab.core.types import FeedbackSignal, Judge, Trajectory


class LengthPenaltyJudge(Judge):
    """
    Penalizes long responses.
    No ground truth required.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def evaluate(self, trajectory: Trajectory) -> FeedbackSignal:
        pred = trajectory.steps[-1].response
        penalty = -self.alpha * len(pred)

        return FeedbackSignal(
            kind="scalar",
            total=penalty,
            components={"length_penalty": penalty},
            payload={"length": len(pred)},
            metadata={"judge_type": "rule", "alpha": self.alpha}
        )
