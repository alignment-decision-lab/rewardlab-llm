# rewardlab-llm

Research toolkit for studying **reward and objective design** in LLM-based decision-making.

## What this is
- A research tool to compare reward structures under controlled settings.
- Focused on analysis, ablations, and modeling choices.

<p align="center">
  <img src="docs/figures/general-schema.png" alt="SHIFT-Lab overview" width="700">
</p>

**REWARD-Lab** is a research framework for **systematically comparing training objectives for decision and generation models** under controlled conditions.

The core idea is to **isolate the role of the reward or objective function** by keeping tasks, models, and optimization settings fixed, and varying only the form of supervision used during training. Supervision signals are treated in a general way: they may come from **human feedback, other models acting as judges, hand-crafted reward functions**, or any programmable criterion, not only preference-based losses.

REWARD-Lab provides:

- a collection of **toy and synthetic tasks** designed to make objective-level effects observable,

- a modular library of **reward and objective formulations** (scalar rewards, pairwise preferences, DPO-style and surrogate objectives),

- a standardized **training and evaluation loop** that ensures fair comparisons,

- and an analysis layer to study **stability, agreement across signals, and failure modes such as over-optimization or reward hacking**.


## Quickstart
```bash
pip install -e .
python -m rewardlab.cli run --config configs/demo.yaml
