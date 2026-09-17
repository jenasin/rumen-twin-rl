#!/usr/bin/env python3
"""Reward-weight sensitivity analysis (Section 8).

The reward is a weighted sum of seven terms, and the ranking of controllers
could in principle be an artefact of one particular weight vector. This script
re-evaluates every controller under each weight set defined in
``config.yaml -> reward.weight_sets``.

Two things are varied and reported separately:

* **Scoring weights** — the same trajectories are re-scored under each weight
  set. This isolates the question "is the ranking robust to how we score?"
  (the policies themselves are unchanged).
* The policies are *not* retrained per weight set here; retraining under a
  modified reward is covered by Ablation 3.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import _paths  # noqa: F401
from agents.common import load_controllers
from rumen_twin.cow import CowPopulation
from rumen_twin.environment import RumenTwinEnv
from rumen_twin.evaluation import make_episode_plan, run_plan
from rumen_twin.utils import (DISCLAIMER, PATHS, banner, ensure_dirs, load_config,
                              load_seeds, write_parquet_or_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reward-weight sensitivity analysis")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--weight-sets", nargs="*", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--out", default="sensitivity_results")
    args = ap.parse_args()

    ensure_dirs()
    cfg, sds = load_config(), load_seeds()
    n_ep = int(args.episodes or sds["sensitivity_analysis"]["episodes_per_condition"])
    weight_sets = args.weight_sets or list(cfg["reward"]["weight_sets"])
    scenarios = args.scenarios or ["A_normal", "B_high_concentrate", "D_heat_stress",
                                   "E_gradual_sara", "G_sensor_noise", "I_mixed"]
    controllers = cfg["evaluation"]["controllers"]

    print(banner("RumenTwin-RL | REWARD SENSITIVITY ANALYSIS"))
    print(DISCLAIMER)
    print(f"  weight sets : {weight_sets}")
    print(f"  scenarios   : {scenarios}")
    print(f"  episodes    : {n_ep} per (weight set x controller x scenario)")

    pop = CowPopulation(config=cfg, seeds=sds)
    cow_ids = [c.cow_id for c in pop.subset("test")]
    plan = make_episode_plan(cow_ids, scenarios, cfg["evaluation"]["eval_seeds"],
                             n_ep, sds["sensitivity_analysis"]["seed"])
    ctrls = load_controllers(controllers, cfg)

    frames = []
    for ws in weight_sets:
        env = RumenTwinEnv(config=cfg, seeds=sds, population=pop, split="test",
                           scenarios=scenarios, weight_set=ws)
        for ctrl in ctrls:
            t0 = time.time()
            res = run_plan(env, ctrl, plan, progress=False)
            res["weight_set"] = ws
            frames.append(res)
            print(f"  [{ws:14s}] {ctrl.name:16s} reward {res['cumulative_reward'].mean():+8.2f} "
                  f"| %low-pH {res['pct_time_below_low_pH'].mean():6.2f} "
                  f"| interventions {res['n_interventions'].mean():5.1f}  ({time.time() - t0:.0f}s)")

    df = pd.concat(frames, ignore_index=True)
    path = write_parquet_or_csv(df, PATHS["results_raw"] / args.out)
    print(f"\nSaved {len(df)} records to {path}")

    print("\nController ranking by mean cumulative reward under each weight set:")
    rank = (df.groupby(["weight_set", "controller"])["cumulative_reward"].mean()
              .unstack().round(2))
    rank["best"] = rank.idxmax(axis=1)
    print(rank.to_string())


if __name__ == "__main__":
    main()
