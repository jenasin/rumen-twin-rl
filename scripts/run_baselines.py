#!/usr/bin/env python3
"""Run the two non-learned baselines (no intervention, expert rules).

Useful as a fast sanity check that does not need any trained model:

    python scripts/run_baselines.py --episodes 60
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import _paths  # noqa: F401
from rumen_twin.cow import CowPopulation
from rumen_twin.environment import RumenTwinEnv
from rumen_twin.evaluation import make_episode_plan, run_plan
from rumen_twin.expert_rules import ExpertRuleController, NoInterventionController
from rumen_twin.utils import (DISCLAIMER, PATHS, banner, ensure_dirs, load_config,
                              load_seeds, write_parquet_or_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline controllers")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="baseline_results")
    args = ap.parse_args()

    ensure_dirs()
    cfg, sds = load_config(), load_seeds()
    n_ep = int(args.episodes or cfg["evaluation"]["episodes_per_condition"])
    scenarios = list(cfg["evaluation_scenarios"])

    print(banner("RumenTwin-RL | baseline controllers"))
    print(DISCLAIMER)

    pop = CowPopulation(config=cfg, seeds=sds)
    cow_ids = [c.cow_id for c in pop.subset(args.split)]
    plan = make_episode_plan(cow_ids, scenarios, cfg["evaluation"]["eval_seeds"],
                             n_ep, sds["evaluation"]["episode_seed_offset"])
    print(f"  {len(plan)} episodes ({n_ep} per scenario x {len(scenarios)} scenarios)"
          f" on {len(cow_ids)} {args.split} cows")

    env = RumenTwinEnv(config=cfg, seeds=sds, population=pop, split=args.split,
                       scenarios=scenarios)
    frames = []
    for ctrl in [NoInterventionController(cfg), ExpertRuleController(cfg)]:
        t0 = time.time()
        print(f"\n  -> {ctrl.name}")
        frames.append(run_plan(env, ctrl, plan, tag=ctrl.name))
        print(f"     done in {time.time() - t0:.1f}s")

    df = pd.concat(frames, ignore_index=True)
    path = write_parquet_or_csv(df, PATHS["results_raw"] / args.out)
    print(f"\nSaved {len(df)} episode records to {path}")
    print(df.groupby("controller")[["mean_pH", "pct_time_below_low_pH",
                                    "n_interventions", "cumulative_reward"]].mean().round(3))


if __name__ == "__main__":
    main()
