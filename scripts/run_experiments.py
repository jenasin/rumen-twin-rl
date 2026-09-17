#!/usr/bin/env python3
"""Main experimental protocol (Section 13).

For every (controller x scenario) condition it runs the same deterministic
plan of independent episodes on the HELD-OUT TEST COWS, across several
evaluation seeds, and writes one row per episode to ``results/raw/``.

Nothing is ever entered by hand: every table and figure in the paper is built
from this file.

    python scripts/run_experiments.py                 # trains missing models, then evaluates
    python scripts/run_experiments.py --skip-training # requires models to exist
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from agents.common import load_controllers, model_exists
from rumen_twin.cow import CowPopulation
from rumen_twin.environment import RumenTwinEnv
from rumen_twin.evaluation import make_episode_plan, run_plan
from rumen_twin.utils import (DISCLAIMER, PATHS, banner, ensure_dirs, load_config,
                              load_seeds, save_json, write_parquet_or_csv)


def ensure_models(cfg, sds, skip_training: bool) -> None:
    """Train the RL agents if their checkpoints are missing."""
    needed = {"ppo_main": "ppo", "dqn_main": "dqn"}
    missing = [r for r in needed if not model_exists(r)]
    if not missing:
        return
    if skip_training:
        raise SystemExit(
            f"Missing trained models: {missing}. Run scripts/train_ppo.py and "
            f"scripts/train_dqn.py first, or drop --skip-training.")
    from agents.dqn_agent import train_dqn
    from agents.ppo_agent import train_ppo
    for run in missing:
        print(banner(f"Training missing model: {run}"))
        (train_ppo if run.startswith("ppo") else train_dqn)(
            run_name=run, config=cfg, seeds=sds, verbose=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="RumenTwin-RL main experiments")
    ap.add_argument("--episodes", type=int, default=None,
                    help="episodes per (controller x scenario) condition")
    ap.add_argument("--controllers", nargs="*", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--out", default="main_results")
    args = ap.parse_args()

    ensure_dirs()
    cfg, sds = load_config(), load_seeds()
    controllers = args.controllers or cfg["evaluation"]["controllers"]
    scenarios = args.scenarios or cfg["evaluation_scenarios"]
    n_ep = int(args.episodes or cfg["evaluation"]["episodes_per_condition"])
    seeds_eval = cfg["evaluation"]["eval_seeds"]

    print(banner("RumenTwin-RL | MAIN EXPERIMENTS  (synthetic simulation study)"))
    print(DISCLAIMER)
    print(f"  controllers      : {controllers}")
    print(f"  scenarios        : {len(scenarios)} -> {scenarios}")
    print(f"  episodes/cond    : {n_ep}  across evaluation seeds {seeds_eval}")
    print(f"  cow split        : {args.split} (held out from RL training)")

    ensure_models(cfg, sds, args.skip_training)

    pop = CowPopulation(config=cfg, seeds=sds)
    cow_ids = [c.cow_id for c in pop.subset(args.split)]
    plan = make_episode_plan(cow_ids, scenarios, seeds_eval, n_ep,
                             sds["evaluation"]["episode_seed_offset"])
    print(f"  total episodes   : {len(plan) * len(controllers)} "
          f"({len(plan)} per controller, identical plan -> paired analysis)")

    env = RumenTwinEnv(config=cfg, seeds=sds, population=pop, split=args.split,
                       scenarios=scenarios)
    frames, timings = [], {}
    for ctrl in load_controllers(controllers, cfg):
        t0 = time.time()
        print(f"\n  -> {ctrl.name}")
        frames.append(run_plan(env, ctrl, plan, tag=ctrl.name))
        timings[ctrl.name] = round(time.time() - t0, 1)
        print(f"     done in {timings[ctrl.name]}s")

    df = pd.concat(frames, ignore_index=True)
    path = write_parquet_or_csv(df, PATHS["results_raw"] / args.out)

    save_json({
        "study_type": cfg["project"]["study_type"],
        "disclaimer": DISCLAIMER,
        "controllers": controllers,
        "scenarios": scenarios,
        "episodes_per_condition": n_ep,
        "eval_seeds": seeds_eval,
        "cow_split": args.split,
        "n_episodes_total": int(len(df)),
        "runtime_seconds": timings,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": cfg,
        "seeds": sds,
    }, PATHS["results"] / "run_metadata.json")

    print(f"\nSaved {len(df)} episode records to {path}")
    summary = (df.groupby("controller")[
        ["mean_pH", "pct_time_below_low_pH", "n_sara_like_episodes",
         "physiological_stability", "n_interventions", "total_intervention_cost",
         "mean_milk_yield_proxy", "cumulative_reward", "success"]]
        .mean().round(3))
    print("\nOverall (all scenarios pooled):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
