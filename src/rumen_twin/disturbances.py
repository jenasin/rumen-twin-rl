"""Feeding plan and disturbance scenarios (the exogenous input e_t).

A :class:`ScenarioInstance` is fully pre-computed at episode reset and holds,
for every step of the episode:

  * ``planned_dmi``   — dry matter offered by the management plan (kg/step)
  * ``forage_plan``   — planned forage ratio of the ration
  * ``heat``          — exogenous thermal load index in [0, 1]
  * ``sensor_overrides`` — per-sensor modifications (noise / bias / dropout / delay)

Scenarios A-E perturb the *physiology* through feed and climate; F-H perturb
the *observation channel* only (on top of a nutritional base disturbance, so
that sensor degradation actually matters); I and J combine both and are held
out of RL training.

SYNTHETIC SIMULATION STUDY.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .cow import Cow
from .utils import load_config


@dataclass
class ScenarioInstance:
    name: str
    label: str
    family: str
    planned_dmi: np.ndarray
    forage_plan: np.ndarray
    heat: np.ndarray
    sensor_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    meta: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.planned_dmi.shape[0])


# ---------------------------------------------------------------------------
# Baseline feeding profile
# ---------------------------------------------------------------------------
def _meal_profile(cfg: Dict, n_steps: int, meal_shift_h: Dict[int, float],
                  meal_spread_scale: Dict[int, float], meal_mass_scale: Dict[int, float],
                  rng: np.random.Generator) -> np.ndarray:
    """Within-day intake weights (sum to 1 per simulated day).

    Three Gaussian meal bumps on a small constant baseline; individual meals
    can be shifted (delayed feeding), narrowed (slug feeding / rebound) or
    enlarged (concentrate bolus).
    """
    spd = int(cfg["simulation"]["steps_per_day"])
    fcfg = cfg["feeding"]
    base_frac = float(fcfg["baseline_intake_fraction"])
    spread0 = float(fcfg["meal_spread_steps"])
    meal_hours = list(fcfg["meal_hours"])
    n_meals = len(meal_hours)

    k = np.arange(spd, dtype=np.float64)
    weights = np.full(spd, base_frac / spd, dtype=np.float64)
    meal_mass = (1.0 - base_frac) / n_meals
    for m, hour in enumerate(meal_hours):
        centre = ((hour + meal_shift_h.get(m, 0.0)) * 60.0 / cfg["simulation"]["dt_minutes"]) % spd
        spread = max(0.8, spread0 * meal_spread_scale.get(m, 1.0))
        # circular distance so a shifted meal wraps correctly around midnight
        d = np.minimum(np.abs(k - centre), spd - np.abs(k - centre))
        bump = np.exp(-0.5 * (d / spread) ** 2)
        bump /= bump.sum()
        weights += meal_mass * meal_mass_scale.get(m, 1.0) * bump

    # tile over the episode with mild day-to-day variation
    n_days = int(np.ceil(n_steps / spd))
    out = np.concatenate([weights * (1.0 + 0.04 * rng.normal()) for _ in range(n_days)])
    return out[:n_steps]


def _diurnal_heat(cfg: Dict, n_steps: int, peak: float, peak_hour: float,
                  width_hours: float, baseline: float) -> np.ndarray:
    dt_h = cfg["simulation"]["dt_minutes"] / 60.0
    hours = (np.arange(n_steps) * dt_h) % 24.0
    d = np.minimum(np.abs(hours - peak_hour), 24.0 - np.abs(hours - peak_hour))
    return np.clip(baseline + (peak - baseline) * np.exp(-0.5 * (d / width_hours) ** 2), 0.0, 1.0)


# ---------------------------------------------------------------------------
def build_scenario(name: str, cow: Cow, rng: np.random.Generator,
                   n_steps: int, config: Optional[Dict] = None) -> ScenarioInstance:
    """Instantiate one stochastic realisation of a named scenario."""
    cfg = config or load_config()
    scfg = cfg["scenarios"][name]
    fcfg = cfg["feeding"]
    spd = int(cfg["simulation"]["steps_per_day"])
    dt_h = cfg["simulation"]["dt_minutes"] / 60.0
    hours = np.arange(n_steps) * dt_h

    base_forage = float(fcfg["baseline_forage_ratio"])
    forage = np.full(n_steps, base_forage, dtype=np.float64)
    heat = _diurnal_heat(cfg, n_steps, peak=0.20, peak_hour=15.0, width_hours=5.0, baseline=0.05)
    meal_shift: Dict[int, float] = {}
    meal_spread: Dict[int, float] = {}
    meal_mass: Dict[int, float] = {}
    sensor_overrides: Dict[str, Dict[str, float]] = {}
    meta: Dict[str, float] = {}

    daily_intake = cow.intake_capacity_kg_day * float(np.clip(1.0 + 0.03 * rng.normal(), 0.85, 1.15))

    def apply_high_concentrate(target_forage: float, onset_h: float, ramp_h: float):
        ramp = np.clip((hours - onset_h) / max(1e-6, ramp_h), 0.0, 1.0)
        return base_forage + (target_forage - base_forage) * ramp

    def apply_delayed_feeding(delay_h: float, affected: int, rebound: float):
        meal_shift[affected] = delay_h * float(np.clip(1.0 + 0.1 * rng.normal(), 0.6, 1.4))
        meal_spread[affected] = 1.0 / max(1.0, rebound)     # narrower -> higher intake rate
        meal_mass[affected] = float(np.clip(1.0 + 0.15 * (rebound - 1.0), 1.0, 1.5))

    def apply_heat(peak: float, peak_hour: float, width: float):
        jitter_peak = float(np.clip(peak * (1.0 + 0.08 * rng.normal()), 0.0, 1.0))
        return _diurnal_heat(cfg, n_steps, jitter_peak, peak_hour, width, baseline=0.10)

    # ---------------- scenario dispatch ----------------
    if name == "A_normal":
        pass

    elif name == "B_high_concentrate":
        forage = apply_high_concentrate(scfg["forage_ratio"], scfg["onset_hour"], scfg["ramp_hours"])

    elif name == "C_delayed_feeding":
        apply_delayed_feeding(scfg["delay_hours"], int(scfg["affected_meal"]), scfg["rebound_factor"])

    elif name == "D_heat_stress":
        heat = apply_heat(scfg["peak_heat"], scfg["peak_hour"], scfg["width_hours"])

    elif name == "E_gradual_sara":
        frac = np.clip(hours / max(1e-6, hours[-1]), 0.0, 1.0)
        forage = scfg["forage_start"] + (scfg["forage_end"] - scfg["forage_start"]) * frac
        for m in range(len(fcfg["meal_hours"])):
            meal_mass[m] = float(scfg["meal_bolus"])

    elif name in ("F_sensor_dropout", "G_sensor_noise", "H_sensor_delay"):
        # sensor scenarios sit on top of a nutritional base disturbance
        base = cfg["scenarios"][scfg["base_disturbance"]]
        forage = apply_high_concentrate(base["forage_ratio"], base["onset_hour"], base["ramp_hours"])
        if name == "F_sensor_dropout":
            mask = _dropout_blocks(n_steps, scfg["block_prob"], scfg["block_len_steps"], rng)
            sensor_overrides["pH"] = {"dropout_mask": mask, "dropout_prob": 0.02}
            meta["dropout_fraction"] = float(mask.mean())
        elif name == "G_sensor_noise":
            sensor_overrides["pH"] = {"noise_sd": float(scfg["noise_sd"]),
                                      "bias": float(scfg["bias"]) * float(np.clip(1 + 0.3 * rng.normal(), 0.2, 1.8)),
                                      "bias_drift_sd": float(scfg["bias_drift_sd"])}
        else:
            lo, hi = scfg["delay_steps"]
            d = int(rng.integers(lo, hi + 1))
            sensor_overrides["pH"] = {"delay_steps": d}
            sensor_overrides["rumination"] = {"delay_steps": d}
            meta["delay_steps"] = float(d)

    elif name == "I_mixed":
        forage = apply_high_concentrate(scfg["forage_ratio"], 6.0, 3.0)
        heat = apply_heat(scfg["peak_heat"], 15.0, 6.0)
        sensor_overrides["pH"] = {"noise_sd": float(scfg["noise_sd"]), "bias": float(scfg["bias"])}

    elif name == "J_stress_test":
        forage = apply_high_concentrate(scfg["forage_ratio"], 4.0, 3.0)
        heat = apply_heat(scfg["peak_heat"], 14.0, 6.5)
        apply_delayed_feeding(scfg["delay_hours"], 1, scfg["rebound_factor"])
        lo, hi = scfg["delay_steps"]
        d = int(rng.integers(lo, hi + 1))
        sensor_overrides["pH"] = {"noise_sd": float(scfg["noise_sd"]),
                                  "bias": float(scfg["bias"]),
                                  "delay_steps": d}
        sensor_overrides["rumination"] = {"delay_steps": d}
        meta["delay_steps"] = float(d)
    else:
        raise KeyError(f"Unknown scenario '{name}'")

    weights = _meal_profile(cfg, n_steps, meal_shift, meal_spread, meal_mass, rng)
    # weights sum to 1 per simulated day -> multiply by daily intake (kg/step)
    planned = np.clip(weights * daily_intake, 0.0, 6.0 * daily_intake / spd)
    forage = np.clip(forage, fcfg["forage_ratio_min"], fcfg["forage_ratio_max"])

    return ScenarioInstance(
        name=name,
        label=scfg.get("label", name),
        family=scfg.get("family", "unknown"),
        planned_dmi=planned.astype(np.float64),
        forage_plan=forage.astype(np.float64),
        heat=heat.astype(np.float64),
        sensor_overrides=sensor_overrides,
        meta=meta,
    )


def _dropout_blocks(n_steps: int, block_prob: float, block_len: List[int],
                    rng: np.random.Generator) -> np.ndarray:
    """Boolean mask: True where the sensor is unavailable."""
    mask = np.zeros(n_steps, dtype=bool)
    lo, hi = int(block_len[0]), int(block_len[1])
    t = 0
    while t < n_steps:
        if rng.random() < block_prob:
            length = int(rng.integers(lo, hi + 1))
            mask[t:t + length] = True
            t += length
        else:
            t += 1
    return mask


def scenario_names(config: Optional[Dict] = None, subset: str = "evaluation") -> List[str]:
    cfg = config or load_config()
    key = "training_scenarios" if subset == "training" else "evaluation_scenarios"
    return list(cfg[key])
