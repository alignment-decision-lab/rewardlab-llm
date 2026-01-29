from rewardlab.core.types import Reward, Trajectory

class ExactMatchReward:
    name = "exact_match"

    def score(self, traj: Trajectory) -> Reward:
        gt = traj.metadata["answer"]
        pred = traj.steps[-1].response
        val = float(str(gt) == pred)
        return Reward(total=val, components={"exact": val})
