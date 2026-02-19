from rewardlab.core.types import FeedbackSignal, Judge, Trajectory


class HybridJudge(Judge):
    """
    Combines multiple judges by weighted sum.
    """

    def __init__(self, judges, weights=None):
        self.judges = judges
        self.weights = weights or [1.0] * len(judges)

    def evaluate(self, trajectory: Trajectory) -> FeedbackSignal:
        total = 0.0
        components = {}

        for judge, w in zip(self.judges, self.weights):
            fb = judge.evaluate(trajectory)
            score = fb.total or 0.0
            total += w * score

            components[judge.__class__.__name__] = score

        return FeedbackSignal(
            kind="scalar",
            total=total,
            components=components,
            metadata={"judge_type": "hybrid"}
        )
