"""``RumenTwinEnv`` — Gymnasium environment closing the digital-twin loop.

    true physiological state  s_t        (physiology.RumenPhysiology)
        -> synthetic sensors             (sensors.SensorArray)
        -> state estimation              (state_estimator.RumenStateEstimator)
        -> decision policy               (RL agent / expert rules)
        -> intervention                  (interventions.InterventionManager)
        -> updated physiological state   s_{t+1}
        -> new observation

The agent NEVER receives ``s_t``. It receives a 22-dimensional feature vector
built only from sensor readings, estimator outputs and the controller's own
bookkeeping. Three state variables (VFA, lactate, health risk) have no sensor
at all, so the problem is partially observable by construction.

SYNTHETIC SIMULATION STUDY - not a veterinary decision tool.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("gymnasium is required: pip install -r requirements.txt") from exc

from .cow import Cow, CowPopulation
from .disturbances import ScenarioInstance, build_scenario
from .expert_rules import ControllerContext
from .interventions import ACTION_NAMES, N_ACTIONS, InterventionManager
from .physiology import STATE_FIELDS, RumenPhysiology, StepInputs
from .reward import RewardFunction
from .sensors import SensorArray
from .state_estimator import RumenStateEstimator
from .utils import clip, episode_seed, halflife_to_decay, load_config, load_seeds

OBS_NAMES: List[str] = [
    "pH_estimate", "pH_available", "pH_trend", "pH_uncertainty",
    "pH_min_recent", "frac_low_recent", "temperature", "rumination",
    "dry_matter_intake", "heat_load", "milk_yield_proxy", "hydration_index",
    "forage_ratio", "planned_forage_ratio", "nominal_buffer_pool",
    "steps_since_pH_obs", "recent_interventions", "cumulative_cost",
    "time_sin", "time_cos",
    # which of the controller's own interventions are still in effect
    "active_forage", "active_reduce_conc", "active_buffer",
    "active_split_feed", "active_obs_boost", "active_alert",
]
OBS_DIM = len(OBS_NAMES)


class RumenTwinEnv(gym.Env):
    """Gymnasium environment for closed-loop control of the rumen digital twin."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[Dict] = None,
        seeds: Optional[Dict] = None,
        population: Optional[CowPopulation] = None,
        split: str = "train",
        scenarios: Optional[Sequence[str]] = None,
        weight_set: Optional[str] = None,
        weight_overrides: Optional[Dict[str, float]] = None,
        episode_steps: Optional[int] = None,
        estimator_method: Optional[str] = None,
        sensor_faults: bool = True,
        homogeneous_population: bool = False,
        render_mode: Optional[str] = None,
        record_trajectory: bool = False,
        light_info: bool = False,
        fixed_cow_id: Optional[int] = None,
        fixed_scenario: Optional[str] = None,
    ):
        super().__init__()
        self.cfg = config or load_config()
        self.seed_cfg = seeds or load_seeds()
        self.split = split
        self.population = population or CowPopulation(
            config=self.cfg, seeds=self.seed_cfg, homogeneous=homogeneous_population)
        self.scenarios = list(scenarios) if scenarios is not None else list(
            self.cfg["training_scenarios"])
        self.weight_set = weight_set
        self.weight_overrides = weight_overrides
        self.reward_fn = RewardFunction(self.cfg, weight_set, weight_overrides)
        self.estimator_method = estimator_method
        # ``sensor_faults=False`` (Ablation 1) strips every scenario-level sensor
        # degradation; nominal sensor noise is kept.
        self.sensor_faults = bool(sensor_faults)
        self.render_mode = render_mode
        self.record_trajectory = bool(record_trajectory)
        self.light_info = bool(light_info)
        self.fixed_cow_id = fixed_cow_id
        self.fixed_scenario = fixed_scenario

        sim = self.cfg["simulation"]
        self.dt = float(sim["dt_minutes"])
        self.steps_per_day = int(sim["steps_per_day"])
        self.n_steps = int(episode_steps if episode_steps is not None
                           else round(sim["episode_hours"] * 60.0 / self.dt))

        self.thr = self.cfg["thresholds"]
        self.hist_window = int(self.cfg["estimator"]["history_window"])
        self.excess_window = int(self.cfg["reward"]["excessive_intervention"]["window_steps"])
        self._nominal_buffer_decay = halflife_to_decay(
            self.cfg["physiology"]["buffer_halflife_h"], self.dt)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)

        self._episode_counter = 0
        self.trajectory: List[Dict[str, Any]] = []
        self.cow: Optional[Cow] = None
        self.scenario: Optional[ScenarioInstance] = None
        self.last_context: Optional[ControllerContext] = None
        self._np_random_seed_used: Optional[int] = None

    # ==================================================================
    # Gymnasium API
    # ==================================================================
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        options = options or {}
        if seed is None:
            seed = episode_seed(self.seed_cfg["evaluation"]["episode_seed_offset"],
                                self.split, self._episode_counter)
        super().reset(seed=seed)
        self._np_random_seed_used = int(seed)
        # Three INDEPENDENT streams spawned from the episode seed. This gives
        # common random numbers across controllers: for a fixed episode seed the
        # cow, the disturbance realisation and the physiological process noise
        # are identical no matter which controller is driving the loop, so
        # controllers can be compared with paired statistics. (The sensor stream
        # can diverge after the first control-dependent draw, because action 5
        # changes the dropout rate - this is stated in the Methods.)
        ss = np.random.SeedSequence(int(seed))
        rng_scen, rng_phys, rng_sens = (np.random.default_rng(c) for c in ss.spawn(3))
        rng = rng_scen
        self._episode_counter += 1

        # --- pick cow and scenario ------------------------------------
        cow_id = options.get("cow_id", self.fixed_cow_id)
        if cow_id is None:
            self.cow = self.population.sample(self.split, rng)
        else:
            self.cow = self.population.get(int(cow_id))
        scen_name = options.get("scenario", self.fixed_scenario)
        if scen_name is None:
            scen_name = self.scenarios[int(rng.integers(len(self.scenarios)))]
        self.scenario_name = scen_name
        self.scenario = build_scenario(scen_name, self.cow, rng, self.n_steps, self.cfg)

        overrides = dict(self.scenario.sensor_overrides) if self.sensor_faults else {}
        if options.get("sensor_overrides"):
            overrides.update(options["sensor_overrides"])

        # --- (re)build the loop components ----------------------------
        self.physio = RumenPhysiology(self.cow, self.cfg, rng_phys)
        self.physio.reset(rng_phys)
        self.sensors = SensorArray(self.cfg, rng_sens, overrides, self.n_steps)
        self.estimator = RumenStateEstimator(self.cfg, self.estimator_method)
        self.estimator.reset({"pH": self.cow.baseline_pH,
                              "milk_yield_proxy": self.cow.milk_potential_kg_day * 0.94})
        self.manager = InterventionManager(self.cfg, self.n_steps)
        self.manager.reset(self.n_steps)
        self.reward_fn = RewardFunction(self.cfg, self.weight_set, self.weight_overrides)

        self.t = 0
        self.nominal_buffer_pool = 0.0
        self.cumulative_reward = 0.0
        self.low_pH_steps = 0
        self.severe_low_pH_steps = 0
        self.trajectory = []
        self._critical_health = False

        obs_raw = self.sensors.observe(self.physio.state, 0)
        est = self.estimator.update(obs_raw)
        self._last_obs_raw = obs_raw
        self._last_est = est
        observation = self._build_observation(est, obs_raw)
        info = self._build_info(action=-1, result=None, components=None, reset=True)
        return observation, info

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.cow is None:
            raise RuntimeError("reset() must be called before step().")
        t = self.t
        action = int(action)

        # --- 1. decision -> intervention ------------------------------
        result = self.manager.apply(action, t)
        mods = self.manager.modifiers(t, float(self.scenario.forage_plan[t]),
                                      float(self.scenario.planned_dmi[t]))
        # the controller's own (cow-independent) belief about the buffer pool
        self.nominal_buffer_pool = (self._nominal_buffer_decay * self.nominal_buffer_pool
                                    + mods.buffer_dose)

        # --- 2. twin dynamics ------------------------------------------
        u = StepInputs(planned_dmi=mods.planned_dmi,
                       forage_ratio=mods.forage_ratio,
                       concentrate_scale=mods.concentrate_scale,
                       buffer_dose=mods.buffer_dose,
                       heat_load=float(self.scenario.heat[t]),
                       health_relief=mods.health_relief)
        state = self.physio.step(u)

        # --- 3. reward --------------------------------------------------
        components = self.reward_fn(state, result, self.manager, t)
        reward = float(components.total)
        self.cumulative_reward += reward

        if state.rumen_pH < self.thr["low_pH"]:
            self.low_pH_steps += 1
        if state.rumen_pH < self.thr["severe_low_pH"]:
            self.severe_low_pH_steps += 1
        if state.health_risk >= 0.95:
            self._critical_health = True

        # --- 4. new observation ----------------------------------------
        self.t = t + 1
        obs_index = min(self.t, self.n_steps - 1)
        obs_raw = self.sensors.observe(state, obs_index,
                                       noise_scale=mods.noise_scale,
                                       dropout_scale=mods.dropout_scale)
        est = self.estimator.update(obs_raw)
        self._last_obs_raw = obs_raw
        self._last_est = est
        observation = self._build_observation(est, obs_raw)

        terminated = False                      # never terminate early: an
        truncated = self.t >= self.n_steps      # early exit would be rewarded
        info = self._build_info(action, result, components)

        if self.record_trajectory:
            self.trajectory.append(self._trajectory_row(action, result, components, obs_raw, est))
        return observation, reward, terminated, truncated, info

    # ==================================================================
    # Observation construction
    # ==================================================================
    def _build_observation(self, est, obs_raw) -> np.ndarray:
        v = est.values
        t = self.t
        plan_idx = min(t, self.n_steps - 1)
        o = np.empty(OBS_DIM, dtype=np.float32)
        o[0] = (est.pH_estimate - 6.20) / 0.60
        o[1] = 1.0 if est.pH_available else 0.0
        o[2] = clip(est.pH_trend * 20.0, -5.0, 5.0)
        o[3] = clip(est.pH_uncertainty * 10.0, 0.0, 5.0)
        o[4] = (est.pH_min_recent - 6.20) / 0.60
        o[5] = est.frac_low_recent * 2.0 - 1.0
        o[6] = (v["temperature"] - 38.90) / 0.80
        o[7] = (v["rumination"] - 5.20) / 2.50
        o[8] = (v["dry_matter_intake"] - 0.24) / 0.30
        o[9] = (v["heat_load"] - 0.30) / 0.35
        o[10] = (v["milk_yield_proxy"] - 31.0) / 8.0
        o[11] = (v["hydration_index"] - 0.85) / 0.15
        o[12] = (self.physio.state.forage_ratio - 0.60) / 0.25
        o[13] = (float(self.scenario.forage_plan[plan_idx]) - 0.60) / 0.25
        o[14] = clip(self.nominal_buffer_pool / 0.60, 0.0, 5.0)
        o[15] = clip(est.steps_since_pH_obs / 8.0, 0.0, 5.0)
        o[16] = clip(self.manager.recent_intervention_count(t, self.excess_window) / 6.0, 0.0, 5.0)
        o[17] = clip(self.manager.cumulative_cost / 20.0, 0.0, 5.0)
        o[18] = float(np.sin(2.0 * np.pi * t / self.steps_per_day))
        o[19] = float(np.cos(2.0 * np.pi * t / self.steps_per_day))
        for j, a in enumerate(range(1, 7)):
            o[20 + j] = 1.0 if self.manager.is_active(a, t) else 0.0
        return np.clip(o, -10.0, 10.0).astype(np.float32)

    def controller_context(self) -> ControllerContext:
        """Physical-unit view of exactly the same information (for the rules)."""
        est, t = self._last_est, self.t
        v = est.values
        plan_idx = min(t, self.n_steps - 1)
        ctx = ControllerContext(
            step=t,
            hour=(t * self.dt / 60.0),
            pH_estimate=est.pH_estimate,
            pH_available=est.pH_available,
            pH_raw=self._last_obs_raw.get("pH"),
            pH_uncertainty=est.pH_uncertainty,
            pH_trend=est.pH_trend,
            pH_min_recent=est.pH_min_recent,
            frac_low_recent=est.frac_low_recent,
            steps_since_pH_obs=est.steps_since_pH_obs,
            temperature=v["temperature"],
            rumination=v["rumination"],
            dry_matter_intake=v["dry_matter_intake"],
            heat_load=v["heat_load"],
            milk_yield_proxy=v["milk_yield_proxy"],
            hydration_index=v["hydration_index"],
            forage_ratio=self.physio.state.forage_ratio,
            planned_forage_ratio=float(self.scenario.forage_plan[plan_idx]),
            nominal_buffer_pool=self.nominal_buffer_pool,
            recent_interventions=self.manager.recent_intervention_count(t, self.excess_window),
            cumulative_cost=self.manager.cumulative_cost,
            alert_active=self.manager.is_active(6, t),
            obs_boost_active=self.manager.is_active(5, t),
            planned_dmi_next=float(self.scenario.planned_dmi[plan_idx]),
            active_actions=tuple(self.manager.is_active(a, t) for a in range(7)),
        )
        self.last_context = ctx
        return ctx

    # ==================================================================
    # Info dictionary (Section 10 of the specification)
    # ==================================================================
    def _build_info(self, action: int, result, components, reset: bool = False) -> Dict[str, Any]:
        s = self.physio.state
        info: Dict[str, Any] = {
            "cow_id": int(self.cow.cow_id),
            "cow_split": self.cow.split,
            "scenario": self.scenario_name,
            "scenario_label": self.scenario.label,
            "step": int(self.t),
            "hour": float(self.t * self.dt / 60.0),
            "pH": float(s.rumen_pH),
            "observed_pH": self._last_obs_raw.get("pH"),
            "estimated_pH": float(self._last_est.pH_estimate),
            "pH_available": bool(self._last_est.pH_available),
            "health_risk": float(s.health_risk),
            "action": int(action),
            "action_name": ACTION_NAMES[action] if action >= 0 else "reset",
            "action_accepted": bool(result.accepted) if result is not None else True,
            "intervention_cost": float(result.cost) if result is not None else 0.0,
            "cumulative_cost": float(self.manager.cumulative_cost),
            "low_pH_duration": int(s.low_pH_run * self.dt),   # minutes in the current run
            "low_pH_steps": int(self.low_pH_steps),
            "severe_low_pH_steps": int(self.severe_low_pH_steps),
            "n_interventions": int(self.manager.n_interventions),
            "n_alerts": int(self.manager.n_alerts),
            "cumulative_reward": float(self.cumulative_reward),
            "reward_components": components.as_dict() if components is not None else None,
        }
        if not self.light_info:
            info["true_state"] = s.as_dict()
            info["observed_state"] = dict(self._last_obs_raw.as_dict())
            info["estimated_state"] = dict(self._last_est.values)
        return info

    def _trajectory_row(self, action, result, components, obs_raw, est) -> Dict[str, Any]:
        s = self.physio.state
        row = {
            "step": self.t - 1,
            "hour": (self.t - 1) * self.dt / 60.0,
            "cow_id": self.cow.cow_id,
            "scenario": self.scenario_name,
            "action": int(action),
            "action_name": ACTION_NAMES[action],
            "action_accepted": bool(result.accepted),
            "intervention_cost": float(result.cost),
            "cumulative_cost": float(self.manager.cumulative_cost),
            "reward": float(components.total),
            "cumulative_reward": float(self.cumulative_reward),
            "observed_pH": obs_raw.get("pH"),
            "estimated_pH": float(est.pH_estimate),
            "pH_available": bool(est.pH_available),
            "planned_forage_ratio": float(self.scenario.forage_plan[min(self.t - 1, self.n_steps - 1)]),
            "buffer_pool": float(s.buffer_pool),
        }
        row.update({f"true_{k}": float(getattr(s, k)) for k in STATE_FIELDS})
        row.update({f"rw_{k}": float(v) for k, v in components.as_dict().items()})
        return row

    # ==================================================================
    def render(self):
        if self.cow is None:
            return ""
        s = self.physio.state
        est = self._last_est
        bar_pos = int(clip((s.rumen_pH - 5.2) / 1.4, 0.0, 1.0) * 28)
        bar = "".join("#" if i == bar_pos else ("|" if i == int((self.thr["low_pH"] - 5.2) / 1.4 * 28)
                                                else "-") for i in range(29))
        pH_obs = self._last_obs_raw.get("pH")
        pH_obs_txt = "  n/a" if pH_obs is None else f"{pH_obs:5.2f}"
        txt = (
            f"[RumenTwin-RL | SYNTHETIC SIMULATION - not a veterinary decision tool]\n"
            f"  cow {self.cow.cow_id:4d} ({self.cow.split:10s})  scenario {self.scenario_name}\n"
            f"  t = {self.t:3d}/{self.n_steps}  ({self.t * self.dt / 60.0:5.2f} h)\n"
            f"  pH      true {s.rumen_pH:5.2f} | obs {pH_obs_txt}"
            f" | est {est.pH_estimate:5.2f}   5.2 {bar} 6.6\n"
            f"  temp {s.rumen_temperature:5.2f} C   rumination {s.rumination_minutes:5.2f} min/step"
            f"   DMI {s.dry_matter_intake:5.3f} kg/step\n"
            f"  VFA  {s.vfa_index:5.2f} (latent)  lactate {s.lactate_index:5.2f} (latent)"
            f"   health risk {s.health_risk:5.2f} (latent)\n"
            f"  milk {s.milk_yield_proxy:5.1f} kg/d    heat {s.heat_load:4.2f}"
            f"   forage {s.forage_ratio:4.2f}   buffer {s.buffer_pool:4.2f}\n"
            f"  interventions {self.manager.n_interventions:3d}  alerts {self.manager.n_alerts:2d}"
            f"  cost {self.manager.cumulative_cost:6.2f}  return {self.cumulative_reward:7.3f}\n"
        )
        if self.render_mode == "human":
            print(txt)
            return None
        return txt

    def close(self):
        pass

    # ==================================================================
    def episode_trajectory(self):
        import pandas as pd
        return pd.DataFrame(self.trajectory)
