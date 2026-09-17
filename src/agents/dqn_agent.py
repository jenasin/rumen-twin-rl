"""RL 1 — Deep Q-Network (Stable-Baselines3).

DQN is the canonical value-based method for a small discrete action space and
is included as the off-policy counterpart to PPO. It learns from a replay
buffer, so it reuses the (expensive) simulated transitions more aggressively
than PPO, but it is more sensitive to the non-stationarity introduced by the
partially observed state.

Model selection: same protocol as PPO — best checkpoint by mean return on the
*validation* cows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

from rumen_twin.utils import PATHS, load_config, load_seeds
from .common import SB3Controller, make_eval_env, make_train_env


def train_dqn(run_name: str = "dqn_main", seed: Optional[int] = None,
              config: Optional[Dict] = None, seeds: Optional[Dict] = None,
              total_timesteps: Optional[int] = None,
              weight_set: Optional[str] = None, sensor_faults: bool = True,
              homogeneous: bool = False, scenarios: Optional[Sequence[str]] = None,
              eval_freq: int = 20000, n_eval_episodes: int = 40,
              verbose: int = 1) -> Path:
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import EvalCallback

    cfg = config or load_config()
    sds = seeds or load_seeds()
    hp = cfg["training"]["dqn"]
    seed = int(seed if seed is not None else sds["training"].get(run_name, 4001))
    total_timesteps = int(total_timesteps or hp["total_timesteps"])

    model_dir = PATHS["models"] / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PATHS["logs"] / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # DQN is off-policy: a modest number of parallel envs is enough.
    train_env = make_train_env(cfg, sds, n_envs=4, seed=seed, weight_set=weight_set,
                               sensor_faults=sensor_faults, homogeneous=homogeneous,
                               scenarios=scenarios, monitor_dir=log_dir)
    eval_env = make_eval_env(cfg, sds, seed=seed + 7777, weight_set=weight_set,
                             sensor_faults=sensor_faults, homogeneous=homogeneous,
                             scenarios=scenarios)

    model = DQN(
        "MlpPolicy", train_env, seed=seed, verbose=verbose,
        learning_rate=float(hp["learning_rate"]), buffer_size=int(hp["buffer_size"]),
        learning_starts=int(hp["learning_starts"]), batch_size=int(hp["batch_size"]),
        tau=float(hp["tau"]), gamma=float(hp["gamma"]),
        train_freq=int(hp["train_freq"]), gradient_steps=int(hp["gradient_steps"]),
        target_update_interval=int(hp["target_update_interval"]),
        exploration_fraction=float(hp["exploration_fraction"]),
        exploration_final_eps=float(hp["exploration_final_eps"]),
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


def load_dqn_controller(run_name: str = "dqn_main", name: Optional[str] = None,
                        deterministic: bool = True) -> SB3Controller:
    from stable_baselines3 import DQN

    model_dir = PATHS["models"] / run_name
    path = model_dir / "best_model.zip"
    if not path.exists():
        path = model_dir / "final_model.zip"
    if not path.exists():
        raise FileNotFoundError(f"No trained DQN model in {model_dir}. Run scripts/train_dqn.py first.")
    return SB3Controller(DQN.load(str(path), device="cpu"), name or run_name, deterministic)
