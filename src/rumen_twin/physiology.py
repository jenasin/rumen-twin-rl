"""Discrete-time dynamical model of the rumen digital twin.

    s_{t+1} = f(s_t, a_t, e_t) + eps_t          (Delta t = 15 min)

``a_t`` enters through :class:`StepInputs` (the intervention layer converts a
discrete action into ration / buffer modifiers), ``e_t`` through the exogenous
heat load and feeding plan produced by ``disturbances.py``.

--------------------------------------------------------------------------
MODEL EQUATIONS (all coefficients live in config.yaml -> physiology)
--------------------------------------------------------------------------
Notation: subscript *i* denotes an individual-cow trait, ``relu(x)=max(x,0)``,
``P`` concentrate pool, ``B`` buffer pool, ``S`` saliva flux, ``phi`` forage
ratio, ``h`` heat load.

(1) Intake modifiers and dry-matter intake
    m_heat   = 1 - k_dh * heatsens_i * h_t
    m_hyd    = 1 - k_dw * (1 - H_t)
    m_health = 1 - k_dr * R_t                        (R = health_risk)
    m_pH     = 1 - k_dp * relu(pH_ref_d - pH_t)/0.5
    D*_t     = plan_t * m_heat * m_hyd * m_health * m_pH
    D_{t+1}  = D_t + a_D (D*_t - D_t) + eps_D

(2) Ration split (phi_t = forage ratio, u_c = concentrate scaling by action)
    c_t = D_t (1 - phi_t) u_c ,   f_t = D_t phi_t

(3) Rapidly fermentable carbohydrate pool (first-order washout, half-life T_P)
    P_{t+1} = rho_P P_t + g_P c_t ksens_i ,     rho_P = 2^(-dt/T_P)

(4) Total VFA index (production scaled by individual acid clearance)
    V*      = (v0 + a_v P_t + b_v D_t/D_ref,i) / clear_i
    V_{t+1} = V_t + a_V (V* - V_t) + eps_V

(5) Lactate index — only accumulates once the pool exceeds a threshold, and is
    amplified at low pH (the positive feedback that produces SARA-like states)
    L*      = g_L relu(P_t - P_thr) (1 + b_L relu(pH_L - pH_t))
    L_{t+1} = L_t + a_L (L* - L_t) - gamma_L clear_i L_t + eps_L

(6) Rumination and saliva buffering
    R*      = R_max ruminEff_i (a_f + b_f phi_t)
                  (1 - k_rh heatsens_i h_t)
                  (1 - k_rp relu(pH_r - pH_t))
                  (1 - k_rr R_t)
    Rum_{t+1} = Rum_t + a_R (R* - Rum_t) + eps_R
    S_t       = s0 Rum_t / R_max

(7) Administered buffer pool (half-life T_B; dose already de-rated for
    repeated administration in interventions.py)
    B_{t+1} = rho_B B_t + dose_t bufresp_i

(8) pH balance — equilibrium pH minus acid loads plus buffering, approached
    with first-order inertia scaled by individual recovery ability
    pH*  = pH0_i - c_V (V - V_ref) - c_L L - c_P (P - P_ref) + c_S (S - S_ref)
                 + c_B B + c_F (phi_t - phi_ref) - c_H relu(h_t - h_ref)

    The balance is written in DEVIATION form around the reference operating
    point (balanced ration, phi = 0.60), so pH* == pH0_i exactly when the cow
    is fed the reference ration and every coefficient is interpretable as
    "pH units per unit deviation".
    pH_{t+1} = pH_t + a_pH recov_i (pH* - pH_t) + eps_pH

(9) Rumen temperature
    T*  = T0 + c_th h + c_tv relu(V - V_ref) + c_tr R - c_tw (H - 0.85)
    T_{t+1} = T_t + a_T (T* - T_t) + eps_T

(10) Hydration
    H*  = clip(h0 - c_hh heatsens_i h_t, 0.3, 1.0)
    H_{t+1} = H_t + a_H (H* - H_t)

(11) Milk yield proxy
    Dbar_{t+1} = Dbar_t + a_Dbar (D_t - Dbar_t)      (smoothed nutrient supply)
    M*  = milkpot_i (f0 + f1 min(1.25, Dbar_t/D_ref,i))(1 - k_mr R_t)
              (1 - k_mp relu(pH_m - pH_t)/0.5)(1 - k_mh h_t)
    M_{t+1} = M_t + a_M (M* - M_t) + eps_M

(12) Health risk accumulator (fast rise, slow individual-dependent recovery)
    acid_t   = relu(pH_low - pH_t)/0.5
    R_{t+1}  = R_t + u_a acid_t + u_l relu(L_t - L_ref) + u_t relu(T_t - T_ref)
                   - u_d recov_i R_t * 1{acid_t == 0}

Every state variable is clipped to the admissible range declared in
config.yaml -> state_variables, so the twin can never produce NaN or
physiologically impossible values.

SYNTHETIC SIMULATION STUDY — these equations are a transparent
proof-of-concept, not a validated biochemical rumen model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from .cow import Cow
from .utils import clip, halflife_to_decay, load_config, relu

STATE_FIELDS = [
    "rumen_pH",
    "rumen_temperature",
    "rumination_minutes",
    "dry_matter_intake",
    "vfa_index",
    "lactate_index",
    "milk_yield_proxy",
    "hydration_index",
    "heat_load",
    "health_risk",
]

AUX_FIELDS = ["concentrate_pool", "buffer_pool", "saliva_buffer", "forage_ratio",
              "intake_ema", "low_pH_run"]


@dataclass
class StepInputs:
    """Everything the twin needs to advance one step.

    Produced by the feeding plan / scenario (``e_t``) and the intervention
    layer (``a_t``).
    """

    planned_dmi: float          # kg DM the management plan offers this step
    forage_ratio: float         # phi_t in [0,1] after interventions
    concentrate_scale: float = 1.0   # u_c multiplier on concentrate fraction
    buffer_dose: float = 0.0    # effective buffer dose added to B
    heat_load: float = 0.0      # exogenous thermal load e_t
    health_relief: float = 0.0  # direct health-risk relief (veterinary alert)


@dataclass
class TwinState:
    """True latent physiological state s_t plus auxiliary pools."""

    rumen_pH: float
    rumen_temperature: float
    rumination_minutes: float
    dry_matter_intake: float
    vfa_index: float
    lactate_index: float
    milk_yield_proxy: float
    hydration_index: float
    heat_load: float
    health_risk: float
    # auxiliary / latent
    concentrate_pool: float = 0.0
    buffer_pool: float = 0.0
    saliva_buffer: float = 0.0
    forage_ratio: float = 0.60
    intake_ema: float = 0.24      # exponentially smoothed DMI (nutrient supply)
    low_pH_run: int = 0
    step_index: int = 0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def core_vector(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in STATE_FIELDS], dtype=np.float64)


class RumenPhysiology:
    """Deterministic-plus-noise rumen dynamics for a single cow."""

    def __init__(self, cow: Cow, config: Optional[Dict] = None,
                 rng: Optional[np.random.Generator] = None,
                 process_noise_scale: float = 1.0):
        self.cfg = config or load_config()
        self.cow = cow
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.p = self.cfg["physiology"]
        self.sv = self.cfg["state_variables"]
        self.thr = self.cfg["thresholds"]
        self.noise = self.cfg["population"]["process_noise"]
        self.noise_scale = float(process_noise_scale)

        dt = float(self.cfg["simulation"]["dt_minutes"])
        self.dt = dt
        self.steps_per_day = int(self.cfg["simulation"]["steps_per_day"])
        self.rho_P = halflife_to_decay(self.p["conc_pool_halflife_h"], dt)
        self.rho_B = halflife_to_decay(self.p["buffer_halflife_h"], dt)
        # Individual reference intake per step
        self.dmi_ref = cow.intake_capacity_kg_day / self.steps_per_day

        self.state: TwinState = self.reset()

    # ------------------------------------------------------------------
    def reset(self, rng: Optional[np.random.Generator] = None) -> TwinState:
        """Initialise the twin near its individual steady state."""
        if rng is not None:
            self.rng = rng
        c, p = self.cow, self.p
        d = {k: self.sv[k]["default"] for k in STATE_FIELDS}
        jitter = self.rng.normal(0.0, 1.0, size=4)
        pH0 = clip(c.baseline_pH - 0.06 + 0.03 * jitter[0],
                   self.sv["rumen_pH"]["min"], self.sv["rumen_pH"]["max"])
        rum0 = clip(p["rumination_max"] * c.rumination_efficiency
                    * (p["rumination_forage_a"] + p["rumination_forage_b"]
                       * self.cfg["feeding"]["baseline_forage_ratio"])
                    + 0.15 * jitter[1], 0.0, self.sv["rumination_minutes"]["max"])
        self.state = TwinState(
            rumen_pH=pH0,
            rumen_temperature=clip(p["temp_base"] + 0.1 + 0.05 * jitter[2], 36.5, 42.5),
            rumination_minutes=rum0,
            dry_matter_intake=max(0.0, self.dmi_ref * 0.9),
            vfa_index=clip(d["vfa_index"] + 0.02 * jitter[3], 0.0, 1.0),
            lactate_index=d["lactate_index"],
            milk_yield_proxy=clip(c.milk_potential_kg_day * 0.94, 0.0, 65.0),
            hydration_index=d["hydration_index"],
            heat_load=d["heat_load"],
            health_risk=max(0.0, d["health_risk"] + 0.01 * self.rng.normal()),
            concentrate_pool=0.22,
            buffer_pool=0.0,
            saliva_buffer=p["saliva_gain"] * rum0 / p["rumination_max"],
            forage_ratio=self.cfg["feeding"]["baseline_forage_ratio"],
            intake_ema=self.dmi_ref * 0.9,
            low_pH_run=0,
            step_index=0,
        )
        return self.state

    # ------------------------------------------------------------------
    def _eps(self, key: str) -> float:
        sd = float(self.noise[key]) * self.noise_scale
        return float(self.rng.normal(0.0, sd)) if sd > 0 else 0.0

    def _clip_state(self, name: str, value: float) -> float:
        meta = self.sv[name]
        return clip(value, meta["min"], meta["max"])

    # ------------------------------------------------------------------
    def step(self, u: StepInputs) -> TwinState:
        """Advance the twin by one 15-minute step. Returns the new state."""
        s, c, p = self.state, self.cow, self.p
        phi = clip(u.forage_ratio, 0.0, 1.0)
        h = clip(u.heat_load, 0.0, 1.0)

        # --- (1) dry-matter intake -----------------------------------
        m_heat = 1.0 - p["dmi_heat_k"] * c.heat_sensitivity * h
        m_hyd = 1.0 - p["dmi_hydration_k"] * (1.0 - s.hydration_index)
        m_health = 1.0 - p["dmi_health_k"] * s.health_risk
        m_pH = 1.0 - p["dmi_pH_k"] * relu(p["dmi_pH_ref"] - s.rumen_pH) / 0.5
        dmi_target = max(0.0, u.planned_dmi * max(0.05, m_heat) * max(0.3, m_hyd)
                         * max(0.3, m_health) * max(0.25, m_pH))
        dmi = s.dry_matter_intake + p["alpha_dmi"] * (dmi_target - s.dry_matter_intake)
        dmi = self._clip_state("dry_matter_intake", dmi + self._eps("dry_matter_intake"))

        # --- (2) ration split ----------------------------------------
        conc = dmi * (1.0 - phi) * clip(u.concentrate_scale, 0.0, 2.0)

        # --- (3) fermentable carbohydrate pool ------------------------
        pool = self.rho_P * s.concentrate_pool + p["conc_pool_gain"] * conc * c.concentrate_sensitivity
        pool = clip(pool, 0.0, 1.0)

        # --- (4) VFA --------------------------------------------------
        rel_dmi = dmi / max(1e-6, self.dmi_ref)
        vfa_target = (p["vfa_intercept"] + p["vfa_from_pool"] * pool
                      + p["vfa_from_intake"] * rel_dmi) / max(0.3, c.acid_clearance)
        vfa_target = clip(vfa_target, 0.0, 1.0)
        vfa = s.vfa_index + p["alpha_vfa"] * (vfa_target - s.vfa_index) + self._eps("vfa")
        vfa = self._clip_state("vfa_index", vfa)

        # --- (5) lactate (positive feedback -> SARA-like states) ------
        lac_drive = relu(pool - p["lactate_pool_threshold"])
        amp = 1.0 + p["lactate_pH_feedback"] * relu(p["lactate_pH_trigger"] - s.rumen_pH)
        lac_target = clip(p["lactate_gain"] * lac_drive * amp, 0.0, 1.0)
        lac = (s.lactate_index + p["alpha_lactate"] * (lac_target - s.lactate_index)
               - p["lactate_clearance"] * c.acid_clearance * s.lactate_index
               + self._eps("lactate"))
        lac = self._clip_state("lactate_index", lac)

        # --- (6) rumination and saliva --------------------------------
        rum_target = (p["rumination_max"] * c.rumination_efficiency
                      * (p["rumination_forage_a"] + p["rumination_forage_b"] * phi)
                      * max(0.15, 1.0 - p["rumination_heat_k"] * c.heat_sensitivity * h)
                      * max(0.2, 1.0 - p["rumination_pH_k"] * relu(p["rumination_pH_ref"] - s.rumen_pH))
                      * max(0.3, 1.0 - p["rumination_health_k"] * s.health_risk))
        rum = s.rumination_minutes + p["alpha_rumination"] * (rum_target - s.rumination_minutes)
        rum = self._clip_state("rumination_minutes", rum + self._eps("rumination"))
        saliva = p["saliva_gain"] * rum / p["rumination_max"]

        # --- (7) administered buffer ----------------------------------
        buf = self.rho_B * s.buffer_pool + max(0.0, u.buffer_dose) * c.buffer_response
        buf = clip(buf, 0.0, 1.6)

        # --- (8) pH balance -------------------------------------------
        pH_eq = (c.baseline_pH
                 - p["pH_vfa"] * (vfa - p["vfa_ref"])
                 - p["pH_lactate"] * lac
                 - p["pH_pool"] * (pool - p["pool_ref"])
                 + p["pH_saliva"] * (saliva - p["saliva_ref"])
                 + p["pH_buffer"] * buf
                 + p["pH_forage"] * (phi - p["pH_forage_ref"])
                 - p["pH_heat"] * relu(h - p["pH_heat_ref"]))
        pH = (s.rumen_pH + p["alpha_pH"] * c.recovery_ability * (pH_eq - s.rumen_pH)
              + self._eps("pH"))
        pH = self._clip_state("rumen_pH", pH)

        # --- (9) temperature ------------------------------------------
        temp_target = (p["temp_base"] + p["temp_heat"] * h
                       + p["temp_vfa"] * relu(vfa - p["temp_vfa_ref"])
                       + p["temp_health"] * s.health_risk
                       - p["temp_hydration"] * (s.hydration_index - 0.85))
        temp = s.rumen_temperature + p["alpha_temperature"] * (temp_target - s.rumen_temperature)
        temp = self._clip_state("rumen_temperature", temp + self._eps("temperature"))

        # --- (10) hydration -------------------------------------------
        hyd_target = clip(p["hydration_base"] - p["hydration_heat"] * c.heat_sensitivity * h, 0.30, 1.0)
        hyd = s.hydration_index + p["alpha_hydration"] * (hyd_target - s.hydration_index)
        hyd = self._clip_state("hydration_index", hyd)

        # --- (11) milk yield proxy ------------------------------------
        # smoothed nutrient supply (half-life ~4 h) drives milk, not the
        # instantaneous 15-min meal spike
        intake_ema = s.intake_ema + p["milk_intake_smoothing"] * (dmi - s.intake_ema)
        rel_ema = intake_ema / max(1e-6, self.dmi_ref)
        milk_target = (c.milk_potential_kg_day
                       * (p["milk_intake_floor"] + p["milk_intake_gain"] * min(1.25, rel_ema))
                       * max(0.2, 1.0 - p["milk_health_k"] * s.health_risk)
                       * max(0.2, 1.0 - p["milk_pH_k"] * relu(p["milk_pH_ref"] - pH) / 0.5)
                       * max(0.2, 1.0 - p["milk_heat_k"] * h))
        milk = s.milk_yield_proxy + p["alpha_milk"] * (milk_target - s.milk_yield_proxy)
        milk = self._clip_state("milk_yield_proxy", milk + self._eps("milk"))

        # --- (12) health risk accumulator ------------------------------
        acid = relu(self.thr["low_pH"] - pH) / 0.5
        hr = (s.health_risk
              + p["health_up_acid"] * acid
              + p["health_up_lactate"] * relu(lac - p["health_lactate_ref"])
              + p["health_up_temp"] * relu(temp - p["health_temp_ref"]))
        if acid <= 0.0:
            hr -= p["health_down"] * c.recovery_ability * (1.0 + 2.0 * s.health_risk)
        hr = self._clip_state("health_risk", hr - max(0.0, u.health_relief))

        low_run = s.low_pH_run + 1 if pH < self.thr["low_pH"] else 0

        self.state = TwinState(
            rumen_pH=pH,
            rumen_temperature=temp,
            rumination_minutes=rum,
            dry_matter_intake=dmi,
            vfa_index=vfa,
            lactate_index=lac,
            milk_yield_proxy=milk,
            hydration_index=hyd,
            heat_load=h,
            health_risk=hr,
            concentrate_pool=pool,
            buffer_pool=buf,
            saliva_buffer=saliva,
            forage_ratio=phi,
            intake_ema=intake_ema,
            low_pH_run=int(low_run),
            step_index=s.step_index + 1,
        )
        return self.state
