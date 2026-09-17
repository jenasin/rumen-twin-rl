"""Transparent reward function for the RumenTwin-RL control problem.

    R_t = kappa * [  w1 * stability_t
                   + w2 * milk_t / milk_ref
                   - w3 * lowpH_t
                   - w4 * severe_t
                   - w4b * duration_t
                   - w5 * cost_t
                   - w6 * excess_t
                   - w7 * alert_t ]

with

    stability_t = 0.6 g(pH_t; 6.40, sigma_pH)
                + 0.2 g(Rum_t; Rum_ref, 0.35 Rum_ref)
                + 0.2 g(T_t; 38.90, sigma_T)          g(x;mu,s)=exp(-0.5((x-mu)/s)^2)

    lowpH_t     = relu(pH_low - pH_t) / 0.5                      (linear)
    severe_t    = (relu(pH_sev - pH_t) / 0.3)^2                  (quadratic)
    duration_t  = min(cap, (n_low_run,t / n_ref)^2)              (quadratic in the
                  length of the *current uninterrupted* low-pH run: this is the
                  term that specifically penalises sustained exposure rather
                  than isolated dips)
    cost_t      = monetary-equivalent cost of the action taken at t
    excess_t    = (relu(n_int,t-W..t - allowance))^2 / normaliser
    alert_t     = 1 if a veterinary alert was *requested* at t else 0

Guarding against a degenerate "always intervene" policy relies on three
independent mechanisms:
  (i)   the per-action ``cost_t`` term,
  (ii)  the *quadratic* ``excess_t`` term over a rolling 6-h window,
  (iii) the saturating buffer dose implemented in ``interventions.py`` — a
        buffer-spamming policy pays full price for ~18 % of the effect.

All weights are read from config.yaml -> reward.weight_sets, so the
sensitivity analysis only has to switch the active weight set.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

from .interventions import ActionResult, InterventionManager
from .physiology import TwinState
from .utils import clip, gaussian_kernel, load_config, relu


@dataclass
class RewardComponents:
    stability: float = 0.0
    milk: float = 0.0
    low_pH: float = 0.0
    severe_low_pH: float = 0.0
    duration: float = 0.0
    cost: float = 0.0
    excess: float = 0.0
    alert: float = 0.0
    total: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class RewardFunction:
    """Configurable reward with a fully itemised breakdown."""

    def __init__(self, config: Optional[Dict] = None, weight_set: Optional[str] = None,
                 weight_overrides: Optional[Dict[str, float]] = None):
        self.cfg = config or load_config()
        rcfg = self.cfg["reward"]
        name = weight_set or rcfg.get("active_weights", "default")
        if name not in rcfg["weight_sets"]:
            raise KeyError(f"Unknown reward weight set '{name}'. "
                           f"Available: {sorted(rcfg['weight_sets'])}")
        self.weight_set_name = name
        self.w = dict(rcfg["weight_sets"][name])
        if weight_overrides:
            self.w.update(weight_overrides)
        self.scale = float(rcfg.get("scale", 1.0))
        self.excess_cfg = rcfg["excessive_intervention"]
        self.dur_cfg = rcfg["duration_penalty"]
        thr = self.cfg["thresholds"]
        self.pH_low = float(thr["low_pH"])
        self.pH_sev = float(thr["severe_low_pH"])
        self.pH_target = float(thr["target_pH"])
        self.sigma_pH = float(thr["stability_pH_sigma"])
        self.rum_ref = float(thr["stability_rum_ref"])
        self.temp_ref = float(thr["stability_temp_ref"])
        self.sigma_T = float(thr["stability_temp_sigma"])
        self.milk_ref = float(thr["milk_reference"])

    # ------------------------------------------------------------------
    def stability(self, s: TwinState) -> float:
        g_pH = gaussian_kernel(s.rumen_pH, self.pH_target, self.sigma_pH)
        g_rum = gaussian_kernel(s.rumination_minutes, self.rum_ref, 0.35 * self.rum_ref)
        g_T = gaussian_kernel(s.rumen_temperature, self.temp_ref, self.sigma_T)
        return float(0.6 * g_pH + 0.2 * g_rum + 0.2 * g_T)

    # ------------------------------------------------------------------
    def __call__(self, state: TwinState, result: ActionResult,
                 manager: InterventionManager, t: int) -> RewardComponents:
        w = self.w
        c = RewardComponents()
        c.stability = self.stability(state)
        c.milk = clip(state.milk_yield_proxy / self.milk_ref, 0.0, 1.3)
        c.low_pH = relu(self.pH_low - state.rumen_pH) / 0.5
        c.severe_low_pH = (relu(self.pH_sev - state.rumen_pH) / 0.3) ** 2
        c.duration = min(float(self.dur_cfg["cap"]),
                         (state.low_pH_run / float(self.dur_cfg["reference_steps"])) ** 2)
        c.cost = float(result.cost)

        n_recent = manager.recent_intervention_count(t, int(self.excess_cfg["window_steps"]))
        over = relu(n_recent - float(self.excess_cfg["free_allowance"]))
        c.excess = (over ** 2 / float(self.excess_cfg["normaliser"])
                    if self.excess_cfg.get("quadratic", True)
                    else over / float(self.excess_cfg["normaliser"]))
        # a *requested* alert is penalised whether or not it was accepted:
        # a call rejected on cooldown is still a false alarm
        c.alert = 1.0 if result.action == 6 else 0.0

        raw = (w["w1_stability"] * c.stability
               + w["w2_milk"] * c.milk
               - w["w3_low_pH"] * c.low_pH
               - w["w4_severe_low_pH"] * c.severe_low_pH
               - w["w4b_duration"] * c.duration
               - w["w5_intervention_cost"] * c.cost
               - w["w6_excessive_intervention"] * c.excess
               - w["w7_alert"] * c.alert)
        c.total = float(self.scale * raw)
        if not np.isfinite(c.total):
            raise FloatingPointError("Reward became non-finite.")
        return c

    def describe(self) -> Dict[str, float]:
        return {"weight_set": self.weight_set_name, "scale": self.scale, **self.w}
