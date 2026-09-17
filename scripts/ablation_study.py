#!/usr/bin/env python3
"""Ablation experiments (Section 16).

Ablation 1  PPO trained WITHOUT sensor faults vs PPO trained WITH them,
            both evaluated on the sensor-fault scenarios (F, G, H).
            -> does training-time exposure to sensor degradation matter?

Ablation 2  PPO trained on a HOMOGENEOUS population (every cow gets the
            population-mean trait vector) vs the main PPO, both evaluated on
            the heterogeneous test cows.
            -> does individual cow variability during training matter?

Ablation 3  PPO trained with the intervention-cost terms REMOVED from the
            reward (w5 = w6 = w7 = 0) vs the main PPO, both evaluated under
            the standard reward.
            -> what do the cost penalties actually buy?

Ablation 4  Expert rules vs the main PPO and DQN on the two disturbance
            combinations that were never seen during training (I and J).
            -> which approach generalises to an unseen combined disturbance?

Each ablated agent is trained from scratch; the ablation is in the TRAINING
condition, while evaluation is always on the same held-out test cows under the
same standard reward, so the comparison is like-for-like.
"""
from __future__ import annotations

import argparse
import time
from typing import Dict, List

import pandas as pd

import _paths  # noqa: F401
from agents.common import load_controllers, model_exists
from agents.ppo_agent import load_ppo_controller, train_ppo
from rumen_twin.cow import CowPopulation
from rumen_twin.environment import RumenTwinEnv
from rumen_twin.evaluation import make_episode_plan, run_plan
from rumen_twin.utils import (DISCLAIMER, PATHS, banner, ensure_dirs, load_config,
                              load_seeds, write_parquet_or_csv)

# run name -> training-condition overrides
ABLATION_RUNS: Dict[str, Dict] = {
    "ppo_no_sensor_faults":   {"sensor_faults": False},
    "ppo_no_cow_variability": {"homogeneous": True},
    "ppo_no_cost_penalty":    {"weight_set": "cost_free"},
}


def ensure_ablation_models(cfg, sds, timesteps=None, force: bool = False) -> None:
    for run, kw in ABLATION_RUNS.items():
        if model_exists(run) and not force:
            print(f"  [{run}] already trained - skipping")
            continue
        print(banner(f"Training ablation agent: {run}  ({kw})"))
        t0 = time.time()
        train_ppo(run_name=run, config=cfg, seeds=sds, total_timesteps=timesteps,
                  verbose=0, **kw)
        print(f"  [{run}] trained in {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="RumenTwin-RL ablation study")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--out", default="ablation_results")
    args = ap.parse_args()

    ensure_dirs()
    cfg, sds = load_config(), load_seeds()
    n_ep = int(args.episodes or cfg["evaluation"]["episodes_per_condition"])

    print(banner("RumenTwin-RL | ABLATION STUDY  (synthetic simulation study)"))
    print(DISCLAIMER)

    if not model_exists("ppo_main") or not model_exists("dqn_main"):
        raise SystemExit("Train the main agents first (scripts/train_ppo.py, train_dqn.py).")
    ensure_ablation_models(cfg, sds, args.timesteps, args.force_retrain)

    pop = CowPopulation(config=cfg, seeds=sds)
    cow_ids = [c.cow_id for c in pop.subset("test")]
    fault_scenarios = ["F_sensor_dropout", "G_sensor_noise", "H_sensor_delay"]
    clean_scenarios = ["A_normal", "B_high_concentrate", "C_delayed_feeding",
                       "D_heat_stress", "E_gradual_sara"]
    unseen_scenarios = ["I_mixed", "J_stress_test"]

    # (ablation, variant, run_name/controller, scenarios, scenario_group)
    conditions = [
        ("1_sensor_faults", "PPO trained WITH sensor faults", "ppo_main",
         fault_scenarios, "sensor-fault scenarios (F,G,H)"),
        ("1_sensor_faults", "PPO trained WITHOUT sensor faults", "ppo_no_sensor_faults",
         fault_scenarios, "sensor-fault scenarios (F,G,H)"),
        ("2_cow_variability", "PPO trained WITH cow variability", "ppo_main",
         clean_scenarios, "physiological scenarios (A-E)"),
        ("2_cow_variability", "PPO trained WITHOUT cow variability", "ppo_no_cow_variability",
         clean_scenarios, "physiological scenarios (A-E)"),
        ("3_intervention_cost", "PPO with cost penalties", "ppo_main",
         clean_scenarios + fault_scenarios, "all trained scenarios (A-H)"),
        ("3_intervention_cost", "PPO without cost penalties", "ppo_no_cost_penalty",
         clean_scenarios + fault_scenarios, "all trained scenarios (A-H)"),
        ("4_unseen_disturbance", "Expert rules", "expert_rules",
         unseen_scenarios, "unseen combined disturbances (I,J)"),
        ("4_unseen_disturbance", "DQN", "dqn_main",
         unseen_scenarios, "unseen combined disturbances (I,J)"),
        ("4_unseen_disturbance", "PPO", "ppo_main",
         unseen_scenarios, "unseen combined disturbances (I,J)"),
    ]

    env = RumenTwinEnv(config=cfg, seeds=sds, population=pop, split="test",
                       scenarios=cfg["evaluation_scenarios"])
    frames: List[pd.DataFrame] = []
    for ablation, variant, run, scenarios, group in conditions:
        if run == "expert_rules":
            ctrl = load_controllers(["expert_rules"], cfg)[0]
        elif run.startswith("dqn"):
            ctrl = load_controllers(["dqn"], cfg, {"dqn": run})[0]
        else:
            ctrl = load_ppo_controller(run, name=run)
        plan = make_episode_plan(cow_ids, scenarios, cfg["evaluation"]["eval_seeds"],
                                 n_ep, sds["evaluation"]["episode_seed_offset"])
        print(f"\n  [{ablation}] {variant}  ({len(plan)} episodes)")
        t0 = time.time()
        res = run_plan(env, ctrl, plan, progress=False)
        res["ablation"] = ablation
        res["variant"] = variant
        res["run_name"] = run
        res["scenario_group"] = group
        frames.append(res)
        print(f"     {time.time() - t0:.1f}s | %low-pH {res['pct_time_below_low_pH'].mean():.2f} "
              f"| interventions {res['n_interventions'].mean():.1f} "
              f"| reward {res['cumulative_reward'].mean():+.2f}")

    df = pd.concat(frames, ignore_index=True)
    path = write_parquet_or_csv(df, PATHS["results_raw"] / args.out)
    print(f"\nSaved {len(df)} ablation episode records to {path}")


if __name__ == "__main__":
    main()
