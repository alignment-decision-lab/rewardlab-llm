import argparse
import yaml



# Ensure registries are populated by importing modules (simple approach for now)
import rewardlab.tasks.toy_math  # noqa: F401
import rewardlab.judges.data_judge  # noqa: F401
import rewardlab.objectives.scalar_eval  # noqa: F401
import rewardlab.judges  # noqa: F401
import rewardlab.objectives  # noqa: F401
import rewardlab.tasks  # noqa: F401  (if you do the same in tasks/__init__.py)

from rewardlab.eval.runner import run_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", nargs="?", help="Run the framework (kept for compatibility)")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_from_config(cfg)


if __name__ == "__main__":
    main()
