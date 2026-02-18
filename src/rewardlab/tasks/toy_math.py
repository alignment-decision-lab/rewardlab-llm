import random
from rewardlab.core.types import Step, Trajectory

from rewardlab.core.registry import TASKS
@TASKS.register("toy_math")

class ToyMathTask:
    name = "toy_math"

    def rollout(self, n: int = 10):
        trajs = []
        for _ in range(n):
            a, b = random.randint(0,5), random.randint(0,5)
            prompt = f"What is {a}+{b}?"
            response = str(a+b)
            trajs.append(
                Trajectory(
                    steps=[Step(prompt, response)],
                    metadata={"answer": a+b}
                )
            )
        return trajs
