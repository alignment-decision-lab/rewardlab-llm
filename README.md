# rewardlab-llm

Research toolkit for studying **reward and objective design** in LLM-based decision-making.

## What this is
- A research tool to compare reward structures under controlled settings.
- Focused on analysis, ablations, and modeling choices.

## What this is NOT
- Not a production RL framework.
- Not a full LLM training library.

## Quickstart
```bash
pip install -e .
python -m rewardlab.cli run --config configs/demo.yaml
