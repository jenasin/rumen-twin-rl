"""Discrete intervention (action) space and its physiological consequences.

    0  no_intervention
    1  increase_forage_ratio            phi += delta for D steps
    2  reduce_concentrate               concentrate scaled by u_c for D steps
    3  administer_buffer                one-shot dose into the buffer pool
    4  split_feeding                    meals truncated, mass deferred to later steps
    5  increase_observation_frequency   sensor noise and dropout reduced for D steps
    6  trigger_alert                    human/veterinary escalation (cooldown-limited)

IMPORTANT: these are *simulated decision-support actions inside a synthetic
model*. They do not represent a real veterinary treatment protocol and must
not be read as one.

Two mechanisms keep the action space from admitting a degenerate "always
buffer" solution:

  * **Saturating buffer response.** The effective dose decays exponentially in
    the number of doses given within ``saturation_window`` steps:
        dose_eff = dose * exp(-k * n_recent)
    so the 2nd dose within 4 h delivers ~43 % and the 3rd ~18 % of the first.
  * **Cost and excess penalties** in the reward (see ``reward.py``).

Action 6 additionally has a hard cooldown: requesting an alert while one is
already active is rejected, but the call is still charged, still counted in the
excess-intervention window and still penalised as a (false) alarm — otherwise
a policy could use rejected alerts as a free no-op.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from .utils import clip, load_config

N_ACTIONS = 7
ACTION_NAMES = [
    "no_intervention",
    "increase_forage_ratio",
    "reduce_concentrate",
    "administer_buffer",
    "split_feeding",
    "increase_observation_frequency",
    "trigger_alert",
]
WASTED_CALL_COST = 0.25


@dataclass
class Modifiers:
    """Effective control input for one step, after all active effects."""

    forage_ratio: float
    concentrate_scale: float
    planned_dmi: float
    buffer_dose: float
    health_relief: float
    noise_scale: float
    dropout_scale: float


@dataclass
class ActionResult:
    action: int
    name: str
    accepted: bool
    cost: float
    is_intervention: bool
    reason: str = ""


class InterventionManager:
    """Tracks active effects, costs, cooldowns and buffer saturation."""

    def __init__(self, config: Optional[Dict] = None, n_steps: int = 192):
        self.cfg = config or load_config()
        self.spec = {int(k): v for k, v in self.cfg["interventions"].items()}
        self.fmin = float(self.cfg["feeding"]["forage_ratio_min"])
        self.fmax = float(self.cfg["feeding"]["forage_ratio_max"])
        self.n_steps = int(n_steps)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, n_steps: Optional[int] = None) -> None:
        if n_steps is not None:
            self.n_steps = int(n_steps)
        # active[action] = expiry step index (exclusive); absent == inactive
        self.active: Dict[int, int] = {}
        self.carry = np.zeros(self.n_steps + 64, dtype=np.float64)  # deferred feed mass
        self.buffer_dose_times: Deque[int] = deque()
        self.alert_times: List[int] = []
        self.alert_cooldown_until: int = -1
        self.pending_buffer_dose: float = 0.0
        self.pending_health_relief: float = 0.0
        self.action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
        self.cumulative_cost: float = 0.0
        self.n_interventions: int = 0
        self.n_alerts: int = 0
        self.intervention_times: List[int] = []

    # ------------------------------------------------------------------
    def apply(self, action: int, t: int) -> ActionResult:
        """Register an action at step ``t``. Returns its cost and acceptance."""
        action = int(action)
        if action < 0 or action >= N_ACTIONS:
            raise ValueError(f"action must be in [0,{N_ACTIONS - 1}], got {action}")
        spec = self.spec[action]
        name = spec["name"]
        self.action_counts[action] += 1

        if action == 0:
            return ActionResult(0, name, True, 0.0, False)

        # --- alert: cooldown-limited escalation ---
        if action == 6:
            if t < self.alert_cooldown_until:
                # A rejected call is still a call: it costs, it counts towards
                # the excess-intervention window and it incurs the alert
                # penalty in the reward. Without this, requesting an alert on
                # cooldown would be an almost-free no-op that a policy can spam.
                self.cumulative_cost += WASTED_CALL_COST
                self.n_interventions += 1
                self.intervention_times.append(t)
                return ActionResult(action, name, False, WASTED_CALL_COST, True,
                                    reason="alert on cooldown")
            self.alert_times.append(t)
            self.alert_cooldown_until = t + int(spec["cooldown_steps"])
            self.active[action] = t + int(spec["duration_steps"])
            self.pending_health_relief += float(spec["health_relief"])
            self.pending_buffer_dose += float(spec["buffer_dose"])
            self.n_alerts += 1

        # --- buffer: saturating dose ---
        elif action == 3:
            win = int(spec["saturation_window"])
            while self.buffer_dose_times and self.buffer_dose_times[0] < t - win:
                self.buffer_dose_times.popleft()
            n_recent = len(self.buffer_dose_times)
            dose_eff = float(spec["dose"]) * float(np.exp(-float(spec["saturation_k"]) * n_recent))
            self.pending_buffer_dose += dose_eff
            self.buffer_dose_times.append(t)
            self.active[action] = t + int(spec["duration_steps"])

        # --- ration / management effects: refresh the expiry window ---
        else:
            self.active[action] = t + int(spec["duration_steps"])

        cost = float(spec["cost"])
        self.cumulative_cost += cost
        self.n_interventions += 1
        self.intervention_times.append(t)
        return ActionResult(action, name, True, cost, True)

    # ------------------------------------------------------------------
    def is_active(self, action: int, t: int) -> bool:
        return t < self.active.get(int(action), -1)

    def modifiers(self, t: int, base_forage: float, base_planned_dmi: float) -> Modifiers:
        """Combine all effects active at step ``t`` into one control input."""
        forage = float(base_forage)
        conc_scale = 1.0
        planned = float(base_planned_dmi)
        noise_scale = 1.0
        dropout_scale = 1.0

        if self.is_active(1, t):
            forage += float(self.spec[1]["forage_delta"])
        if self.is_active(6, t):
            forage += float(self.spec[6]["forage_delta"])
        if self.is_active(2, t):
            conc_scale = min(conc_scale, float(self.spec[2]["concentrate_scale"]))
        if self.is_active(4, t):
            # truncate the offered meal and defer the removed mass to later steps
            scale = float(self.spec[4]["meal_scale"])
            deferred = planned * (1.0 - scale)
            planned = planned * scale
            k = max(1, int(round(4 * float(self.spec[4]["spread_factor"]))))
            end = min(len(self.carry), t + 1 + k)
            if end > t + 1:
                self.carry[t + 1:end] += deferred / (end - (t + 1))
        if self.is_active(5, t):
            noise_scale = float(self.spec[5]["noise_scale"])
            dropout_scale = float(self.spec[5]["dropout_scale"])

        if t < len(self.carry):
            planned += float(self.carry[t])

        dose = self.pending_buffer_dose
        relief = self.pending_health_relief
        self.pending_buffer_dose = 0.0
        self.pending_health_relief = 0.0

        return Modifiers(
            forage_ratio=clip(forage, self.fmin, self.fmax),
            concentrate_scale=conc_scale,
            planned_dmi=max(0.0, planned),
            buffer_dose=dose,
            health_relief=relief,
            noise_scale=noise_scale,
            dropout_scale=dropout_scale,
        )

    # ------------------------------------------------------------------
    def recent_intervention_count(self, t: int, window: int) -> int:
        """Interventions registered in the half-open window ``(t-window, t]``.

        The upper bound matters: the reward queries this at the current step,
        and without it a query for an earlier step would also count later
        interventions.
        """
        return int(sum(1 for s in self.intervention_times if t - window < s <= t))

    def summary(self) -> Dict[str, float]:
        return {
            "n_interventions": int(self.n_interventions),
            "n_alerts": int(self.n_alerts),
            "cumulative_cost": float(self.cumulative_cost),
            **{f"n_action_{i}_{ACTION_NAMES[i]}": int(self.action_counts[i])
               for i in range(N_ACTIONS)},
        }


def intervention_table(config: Optional[Dict] = None):
    """Table 3: intervention definitions (generated from config)."""
    import pandas as pd

    cfg = config or load_config()
    dt = cfg["simulation"]["dt_minutes"]
    effects = {
        0: "None (monitoring only).",
        1: "Forage fraction of the ration raised by +{d:.2f} (more rumination, more saliva buffering).",
        2: "Concentrate fraction scaled by x{s:.2f} (lower fermentable carbohydrate inflow).",
        3: "One-shot buffer dose {b:.2f} into the rumen buffer pool; saturating with repeated use.",
        4: "Offered meal truncated to x{m:.2f}; removed mass deferred to later steps (slug feeding avoided).",
        5: "Sensor noise x{n:.2f} and dropout x{p:.2f} (denser observation regime).",
        6: "Human/veterinary escalation: health-risk relief {h:.2f}, forage +{d:.2f}, buffer {b:.2f}; hard cooldown.",
    }
    rows = []
    for a in range(N_ACTIONS):
        sp = cfg["interventions"][a]
        eff = effects[a].format(d=sp.get("forage_delta", 0.0), s=sp.get("concentrate_scale", 1.0),
                                b=sp.get("dose", sp.get("buffer_dose", 0.0)),
                                m=sp.get("meal_scale", 1.0), n=sp.get("noise_scale", 1.0),
                                p=sp.get("dropout_scale", 1.0), h=sp.get("health_relief", 0.0))
        rows.append({
            "action_id": a,
            "name": sp["name"],
            "physiological_effect": eff,
            "duration_steps": sp.get("duration_steps", 0),
            "duration_hours": round(sp.get("duration_steps", 0) * dt / 60.0, 2),
            "cost": sp.get("cost", 0.0),
            "usage_constraint": ("saturating dose (exp decay, window "
                                 f"{sp.get('saturation_window')} steps)" if a == 3 else
                                 (f"cooldown {sp.get('cooldown_steps')} steps" if a == 6 else
                                  "excess-use penalty in reward" if a > 0 else "-")),
        })
    return pd.DataFrame(rows)
