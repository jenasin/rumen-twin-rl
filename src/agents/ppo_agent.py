"""RL 2 — Proximal Policy Optimization (Stable-Baselines3).

PPO is an on-policy actor-critic method. It suits this problem because the
reward is dense, the action space is small and discrete, and the entropy bonus
keeps the policy exploring the intervention set instead of collapsing onto
"never intervene" early in training (which is a strong local optimum here:
doing nothing already earns positive reward in the undisturbed scenario).

Model selection: periodic evaluation on the *validation* cows; the best
checkpoint by mean validation return is the one carried into the experiments.
Test cows are never used for selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from rumen_twin.utils import PATHS, load_config, load_seeds
from .common import SB3Controller, make_eval_env, make_train_env


def train_ppo(run_name: str = "ppo_main", seed: Optional[int] = None,
              config: Optional[Dict] = None, seeds: Optional[Dict] = None,
              total_timesteps: Optional[int] = None,
              weight_set: Optional[str] = None, sensor_faults: bool = True,
              homogeneous: bool = False, scenarios: Optional[Sequence[str]] = None,
              eval_freq: int = 20000, n_eval_episodes: int = 40,
              verbose: int = 1) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    cfg = config or load_config()
    sds = seeds or load_seeds()
    hp = cfg["training"]["ppo"]
    seed = int(seed if seed is not None else sds["training"].get(run_name, 3001))
    total_timesteps = int(total_timesteps or hp["total_timesteps"])

    model_dir = PATHS["models"] / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS["logs"] / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_train_env(cfg, sds, seed=seed, weight_set=weight_set,
                               sensor_faults=sensor_faults, homogeneous=homogeneous,
                               scenarios=scenarios, monitor_dir=log_dir)
    eval_env = make_eval_env(cfg, sds, seed=seed + 7777, weight_set=weight_set,
                             sensor_faults=sensor_faults, homogeneous=homogeneous,
                             scenarios=scenarios)

    model = PPO(
        "MlpPolicy", train_env, seed=seed, verbose=verbose,
        n_steps=int(hp["n_steps"]), batch_size=int(hp["batch_size"]),
        learning_rate=float(hp["learning_rate"]), gamma=float(hp["gamma"]),
        gae_lambda=float(hp["gae_lambda"]), clip_range=float(hp["clip_range"]),
        ent_coef=float(hp["ent_coef"]), vf_coef=float(hp["vf_coef"]),
        max_grad_norm=float(hp["max_grad_norm"]), n_epochs=int(hp["n_epochs"]),
        policy_kwargs={"net_arch": list(hp["net_arch"])},
    )
    cb = EvalCallback(eval_env, best_model_save_path=str(model_dir),
                      log_path=str(log_dir), eval_freq=max(1, eval_freq // train_env.num_envs),
                      n_eval_episodes=n_eval_episodes, deterministic=True, verbose=verbose)
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)

    final_path = model_dir / "final_model.zip"
    model.save(str(final_path))
    train_env.close(); eval_env.close()
    best = model_dir / "best_model.zip"
    return best if best.exists() else final_path


def load_ppo_controller(run_name: str = "ppo_main", name: Optional[str] = None,
                        deterministic: bool = True) -> SB3Controller:
    from stable_baselines3 import PPO

    model_dir = PATHS["models"] / run_name
    path = model_dir / "best_model.zip"
    if not path.exists():
        path = model_dir / "final_model.zip"
    if not path.exists():
        raise FileNotFoundError(f"No trained PPO model in {model_dir}. Run scripts/train_ppo.py first.")
    return SB3Controller(PPO.load(str(path), device="cpu"), name or run_name, deterministic)
