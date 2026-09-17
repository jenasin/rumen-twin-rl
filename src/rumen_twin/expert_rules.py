"""Baseline 2: transparent expert-derived rule-based controller.

This file is intentionally self-contained and simple to edit — every
threshold lives in ``config.yaml -> expert_rules`` and the decision logic is a
single ordered cascade that can be read top to bottom:

    if pH sensor has been silent too long              -> increase observation frequency
    if pH has been low and persistent for long enough  -> raise a veterinary alert
    if pH < threshold_3                                -> administer buffer
    if pH < threshold_2                                -> reduce concentrate
    if pH < threshold_1                                -> increase forage ratio
    if pH < split_feed_pH and a large meal is arriving  -> split feeding
    otherwise                                          -> no intervention

A management action that is *already in effect* is never re-issued: raising
the forage ratio is a standing order that lasts 6 h, not something a stockman
repeats every 15 minutes. The cascade therefore falls through to the next
applicable rule (and ultimately to "no intervention") whenever its preferred
action is still active. Without this guard the controller would re-order the
same standing change ~10x per effect window and be penalised for it.

The controller is *static*: thresholds never adapt, it has no memory of what
worked, and it reacts to the current pH estimate plus two persistence
counters. It sees exactly the same information as the RL agents
(``ControllerContext`` below is built from the estimator output only — never
from the latent true state).

These rules are a simulated decision-support heuristic, NOT a veterinary
protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .utils import load_config


@dataclass
class ControllerContext:
    """Everything a controller is allowed to see at decision time.

    Built exclusively from sensor observations, the state estimator and the
    controller's own bookkeeping (what it fed / dosed). The latent variables
    ``vfa_index``, ``lactate_index`` and ``health_risk`` are never included.
    """

    step: int
    hour: float
    pH_estimate: float
    pH_available: bool
    pH_raw: Optional[float]
    pH_uncertainty: float
    pH_trend: float
    pH_min_recent: float
    frac_low_recent: float
    steps_since_pH_obs: int
    temperature: float
    rumination: float
    dry_matter_intake: float
    heat_load: float
    milk_yield_proxy: float
    hydration_index: float
    forage_ratio: float
    planned_forage_ratio: float
    nominal_buffer_pool: float
    recent_interventions: int
    cumulative_cost: float
    alert_active: bool
    obs_boost_active: bool
    planned_dmi_next: float
    # Which of the controller's own interventions are still in effect.
    # Index = action id; a controller knows what it has already ordered.
    active_actions: tuple = (False,) * 7

    def is_active(self, action: int) -> bool:
        return bool(self.active_actions[int(action)])


class ExpertRuleController:
    """Ordered threshold cascade. Stateless apart from persistence counters."""

    name = "expert_rules"
    uses_context = True   # acts on ControllerContext, not the RL observation vector

    def __init__(self, config: Optional[Dict] = None):
        self.cfg = config or load_config()
        self.r = self.cfg["expert_rules"]
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self.low1_run = 0          # consecutive steps with pH_est < threshold_1
        self.low2_run = 0          # consecutive steps with pH_est < threshold_2
        self.buffer_cooldown = 0
        self.alert_cooldown = 0

    # ------------------------------------------------------------------
    def act(self, ctx: ControllerContext) -> int:
        r = self.r
        pH = ctx.pH_estimate

        # --- persistence bookkeeping -------------------------------------
        self.low1_run = self.low1_run + 1 if pH < r["threshold_1"] else 0
        self.low2_run = self.low2_run + 1 if pH < r["threshold_2"] else 0
        self.buffer_cooldown = max(0, self.buffer_cooldown - 1)
        self.alert_cooldown = max(0, self.alert_cooldown - 1)

        # --- R0: blind sensor -> buy information -------------------------
        if (ctx.steps_since_pH_obs >= r["missing_pH_steps_for_obs_boost"]
                and not ctx.obs_boost_active):
            return 5

        # --- R1: persistent moderate/low pH -> escalate to a human --------
        if (self.low2_run >= r["alert_persistent_steps"]
                and self.alert_cooldown == 0 and not ctx.alert_active):
            self.alert_cooldown = r["alert_cooldown_steps"]
            return 6

        # --- R2: marked acidification -> buffer ---------------------------
        if pH < r["threshold_3"] and self.buffer_cooldown == 0 and not ctx.is_active(3):
            self.buffer_cooldown = r["buffer_cooldown_steps"]
            return 3

        # --- R3: moderate acidification -> cut fermentable carbohydrate ---
        if pH < r["threshold_2"] and not ctx.is_active(2):
            return 2

        # --- R4: mild acidification (or persistent mild) -> more forage ---
        if (pH < r["threshold_1"] or self.low1_run >= r["persistent_low_steps"]) \
                and not ctx.is_active(1):
            return 1

        # --- R5: pre-emptive meal management before a large meal ----------
        if (pH < r["split_feed_pH"] and ctx.planned_dmi_next >= r["high_intake_dmi"]
                and not ctx.is_active(4)):
            return 4

        return 0


class NoInterventionController:
    """Baseline 1: never intervenes."""

    name = "no_intervention"
    uses_context = True

    def __init__(self, config: Optional[Dict] = None):
        self.cfg = config or load_config()

    def reset(self, seed: Optional[int] = None) -> None:
        pass

    def act(self, ctx: ControllerContext) -> int:
        return 0
