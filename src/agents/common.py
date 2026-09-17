"""Shared plumbing for the RL agents: environment factories and the wrapper
that lets a trained Stable-Baselines3 policy be used as a controller.

Every controller in this project exposes::

    ctrl.name          str
    ctrl.uses_context  bool   True -> act(ControllerContext), False -> act(obs)
    ctrl.reset(seed)
    ctrl.act(...)      -> int in [0, 6]

so the evaluation runner treats rule-based and learned policies identically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from rumen_twin.environment import RumenTwinEnv
from rumen_twin.utils import load_config, load_seeds


class ControllerProtocol(Protocol):
    name: str
    uses_context: bool

    def reset(self, seed: Optional[int] = None) -> None: ...
    def act(self, x) -> int: ...


# ---------------------------------------------------------------------------
def env_factory(rank: int, seed: int, *, config: Dict, seeds: Dict, split: str,
                scenarios: Sequence[str], weight_set: Optional[str],
                sensor_faults: bool, homogeneous: bool,
                monitor_dir: Optional[Path] = None) -> Callable:
    """Return a thunk creating one (Monitor-wrapped) environment."""
    from stable_baselines3.common.monitor import Monitor

    def _init():
        env = RumenTwinEnv(
            config=config, seeds=seeds, split=split, scenarios=list(scenarios),
            weight_set=weight_set, sensor_faults=sensor_faults,
            homogeneous_population=homogeneous, light_info=True,
        )
        env.reset(seed=int(seed + rank))
        path = str(monitor_dir / f"monitor_{split}_{rank}") if monitor_dir else None
        return Monitor(env, filename=path)

    return _init


def _make_vec(n_envs: int, seed: int, **kwargs):
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([env_factory(i, seed, **kwargs) for i in range(n_envs)])


def make_train_env(config: Optional[Dict] = None, seeds: Optional[Dict] = None,
                   n_envs: Optional[int] = None, seed: int = 0,
                   weight_set: Optional[str] = None, sensor_faults: bool = True,
                   homogeneous: bool = False, scenarios: Optional[Sequence[str]] = None,
                   monitor_dir: Optional[Path] = None):
    """Vectorised TRAINING environment — training cows only."""
    cfg = config or load_config()
    sds = seeds or load_seeds()
    n = int(n_envs if n_envs is not None else cfg["training"]["n_envs"])
    return _make_vec(n, seed + sds["training"]["env_seed_offset"],
                     config=cfg, seeds=sds, split="train",
                     scenarios=scenarios if scenarios is not None else cfg["training_scenarios"],
                     weight_set=weight_set, sensor_faults=sensor_faults,
                     homogeneous=homogeneous, monitor_dir=monitor_dir)


def make_eval_env(config: Optional[Dict] = None, seeds: Optional[Dict] = None,
                  seed: int = 12345, weight_set: Optional[str] = None,
                  sensor_faults: bool = True, homogeneous: bool = False,
                  scenarios: Optional[Sequence[str]] = None, n_envs: int = 1):
    """Vectorised VALIDATION environment — validation cows only.

    Model selection uses this split; the test cows are never touched during
    training or checkpoint selection.
    """
    cfg = config or load_config()
    sds = seeds or load_seeds()
    return _make_vec(n_envs, seed, config=cfg, seeds=sds, split="validation",
                     scenarios=scenarios if scenarios is not None else cfg["training_scenarios"],
                     weight_set=weight_set, sensor_faults=sensor_faults,
                     homogeneous=homogeneous, monitor_dir=None)


# ---------------------------------------------------------------------------
class SB3Controller:
    """Wraps a trained SB3 model so it plugs into the evaluation runner."""

    uses_context = False

    def __init__(self, model, name: str, deterministic: bool = True):
        self.model = model
        self.name = name
        self.deterministic = bool(deterministic)

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(np.asarray(obs, dtype=np.float32),
                                       deterministic=self.deterministic)
        return int(np.asarray(action).reshape(-1)[0])


# ---------------------------------------------------------------------------
def load_controllers(names: Sequence[str], config: Optional[Dict] = None,
                     run_names: Optional[Dict[str, str]] = None) -> List:
    """Instantiate controllers by name for the evaluation scripts.

    ``run_names`` remaps a controller onto a different trained run, which the
    ablation study uses (e.g. ``{"ppo": "ppo_no_sensor_faults"}``).
    """
    from rumen_twin.expert_rules import ExpertRuleController, NoInterventionController
    from .dqn_agent import load_dqn_controller
    from .ppo_agent import load_ppo_controller

    cfg = config or load_config()
    run_names = run_names or {}
    out = []
    for n in names:
        if n == "no_intervention":
            out.append(NoInterventionController(cfg))
        elif n == "expert_rules":
            out.append(ExpertRuleController(cfg))
        elif n.startswith("ppo"):
            out.append(load_ppo_controller(run_names.get(n, "ppo_main"), name=n))
        elif n.startswith("dqn"):
            out.append(load_dqn_controller(run_names.get(n, "dqn_main"), name=n))
        else:
            raise KeyError(f"Unknown controller '{n}'")
    return out


def model_exists(run_name: str) -> bool:
    from rumen_twin.utils import PATHS
    d = PATHS["models"] / run_name
    return (d / "best_model.zip").exists() or (d / "final_model.zip").exists()
