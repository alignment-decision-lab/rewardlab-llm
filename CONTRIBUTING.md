# Contributing to REWARD-Lab

REWARD-Lab is designed to be modular.  
New research ideas should be added by creating new modules, not by modifying the training loop.

This document explains how to extend the framework cleanly.

---

# Core Design Principle

Each component has a single responsibility:

| Component | Role |
|-----------|------|
| Task | Defines data or environment |
| Judge | Produces supervision (`FeedbackSignal`) |
| Objective | Converts supervision into a loss |
| Runner | Orchestrates experiments |

Do not mix responsibilities.

---

# Adding a New Judge

Judges live in: `src/rewardlab/judges/` 


A Judge must:

- Inherit from `Judge`
- Implement `evaluate(self, trajectory) -> FeedbackSignal`
- Be registered in the registry

Example:

```python
from rewardlab.core.types import FeedbackSignal, Judge, Trajectory
from rewardlab.core.registry import JUDGES

@JUDGES.register("my_judge")
class MyJudge(Judge):

    def evaluate(self, trajectory: Trajectory) -> FeedbackSignal:
        score = 0.0  # compute something

        return FeedbackSignal(
            kind="scalar",
            total=score,
            metadata={"judge": "my_judge"}
        )
```
Then use it in YAML:
```yaml 
judge:
  name: my_judge
  params: {}
```
No changes to runner required.

# Adding a New Objective

Objectives live in: `src/rewardlab/objectives/`

An Objective must:

- Inherit from Objective
- Implement compute_loss(...)
- Return {"loss": ..., "logs": {...}}
- Be registered in the registry

Example:

```python
from rewardlab.core.types import FeedbackSignal, Objective
from rewardlab.core.registry import OBJECTIVES

@OBJECTIVES.register("my_objective")
class MyObjective(Objective):

    def compute_loss(self, feedback, model_outputs, batch=None):
        loss = 0.0  # define training loss
        return {"loss": loss, "logs": {"my_metric": loss}}
```
then switch to YAML :

```yaml 
objective:
  name: my_objective
  params: {}
```

# Adding a new task 
tasks live in : `src/rewardlab/tasks/`

Tasks must:
- Implement `rollout(n)`
- Return Trajectory objects
- Be registered via `@TASKS.register("task_name")`

# Adding Uncertainty / Robustness

You can:
- Wrap a judge using `NoisyJudge`
- Build ensemble judges
- Store confidence or variance in `FeedbackSignal.metadata`

Objectives can optionally use uncertainty fields.

# Guidelines

- Do not modify runner.py unless it's necessary.
- Do not mix reward computation with loss computation.
- Keep components independent.
- Use YAML configuration for experiments.
