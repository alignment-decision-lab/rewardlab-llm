from rewardlab.rewards.scalar import ExactMatchReward
from rewardlab.tasks.toy_math import ToyMathTask

def run_demo():
    task = ToyMathTask()
    reward = ExactMatchReward()

    trajs = task.rollout(20)
    scores = [reward.score(t).total for t in trajs]
    print("Mean reward:", sum(scores)/len(scores))
