#!/usr/bin/env python3
"""Train the DQN controller (RL 1) on the TRAINING cows only.

    python scripts/train_dqn.py                       # main agent
    python scripts/train_dqn.py --run dqn_no_faults --no-sensor-faults
"""
from __future__ import annotations

import argparse
import time

import _paths  # noqa: F401
from agents.dqn_agent import train_dqn
from rumen_twin.utils import DISCLAIMER, banner, load_config, load_seeds


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DQN on RumenTwinEnv")
    ap.add_argument("--run", default="dqn_main", help="run name (also the model folder)")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--weight-set", default=None, help="reward weight set from config.yaml")
    ap.add_argument("--no-sensor-faults", action="store_true",
                    help="Ablation 1: train without scenario-level sensor degradation")
    ap.add_argument("--homogeneous", action="store_true",
                    help="Ablation 2: train without individual cow variability")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg, sds = load_config(), load_seeds()
    print(banner(f"RumenTwin-RL | DQN training | run = {args.run}"))
    print(DISCLAIMER)
    print(f"  training scenarios : {args.scenarios or cfg['training_scenarios']}")
    print(f"  sensor faults      : {not args.no_sensor_faults}")
    print(f"  cow variability    : {not args.homogeneous}")
    print(f"  reward weight set  : {args.weight_set or cfg['reward']['active_weights']}")

    t0 = time.time()
    path = train_dqn(
        run_name=args.run, seed=args.seed, config=cfg, seeds=sds,
        total_timesteps=args.timesteps, weight_set=args.weight_set,
        sensor_faults=not args.no_sensor_faults, homogeneous=args.homogeneous,
        scenarios=args.scenarios, verbose=0 if args.quiet else 1,
    )
    print(f"\nDQN model saved to {path}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
