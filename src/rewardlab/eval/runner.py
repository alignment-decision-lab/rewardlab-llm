from __future__ import annotations

from typing import Any, Dict

from rewardlab.core.registry import TASKS, JUDGES, OBJECTIVES
from rewardlab.rewards.scalar import ExactMatchReward


def run_from_config(cfg: Dict[str, Any]) -> None:
    # Build components from registry
    task_cfg = cfg["task"]
    judge_cfg = cfg["judge"]
    obj_cfg = cfg["objective"]

    task = TASKS.create(task_cfg["name"], **task_cfg.get("params", {}))
    # judge = JUDGES.create(judge_cfg["name"], **judge_cfg.get("params", {}))
    judge = build(JUDGES, judge_cfg)

    objective = OBJECTIVES.create(obj_cfg["name"], **obj_cfg.get("params", {}))

    # Reward wrapper (kept for backward compatibility)
    reward = ExactMatchReward(judge=judge)

    # Run settings
    episodes = int(cfg.get("run", {}).get("episodes", 20))

    trajs = task.rollout(episodes)

    rewards = []
    losses = []

    for t in trajs:
        feedback = reward.score(t)  # FeedbackSignal
        out = objective.compute_loss(feedback=feedback, model_outputs={}, batch=None)
        rewards.append(float(out["logs"].get("reward_total", 0.0)))
        losses.append(float(out["logs"].get("loss", 0.0)))

    print("Mean reward:", sum(rewards) / len(rewards))
    print("Mean loss:", sum(losses) / len(losses))

def build(registry, cfg):
    cls = registry.get(cfg["name"])
    params = dict(cfg.get("params", {}))

    # recursively build nested components
    if "base" in params and isinstance(params["base"], dict):
        params["base"] = build(JUDGES, params["base"])

    return cls(**params)
