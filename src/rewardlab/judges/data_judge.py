from rewardlab.core.types import FeedbackSignal, Judge, Trajectory

from rewardlab.core.registry import JUDGES

@JUDGES.register("data_judge")

class DataJudge(Judge):
    """
    Data-backed judge using ground-truth labels from trajectory.metadata.
    """

    def __init__(self, field: str = "answer"):
        self.field = field

    def evaluate(self, trajectory: Trajectory) -> FeedbackSignal:
        gt = trajectory.metadata[self.field]
        pred = trajectory.steps[-1].response

        val = float(str(gt) == pred)

        return FeedbackSignal(
            kind="scalar",
            total=val,
            components={"exact_match": val},
            payload={"gt": str(gt), "pred": str(pred)},
            metadata={"judge_type": "data"}
        )
