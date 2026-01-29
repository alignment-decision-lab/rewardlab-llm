import random
from rewardlab.core.types import Step, Trajectory

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
