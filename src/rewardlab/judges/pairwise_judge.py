from rewardlab.core.types import FeedbackSignal, Judge, Trajectory


class PairwiseJudge(Judge):
    """
    Example pairwise judge.
    Assumes trajectory.metadata contains:
        {
            "chosen": "...",
            "rejected": "..."
        }
    """

    def evaluate(self, trajectory: Trajectory) -> FeedbackSignal:
        chosen = trajectory.metadata["chosen"]
        rejected = trajectory.metadata["rejected"]

        return FeedbackSignal(
            kind="pairwise",
            payload={
                "chosen": chosen,
                "rejected": rejected
            },
            metadata={"judge_type": "pairwise_data"}
        )
