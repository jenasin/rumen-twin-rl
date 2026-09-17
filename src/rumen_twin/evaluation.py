"""Episode runner and the 15 evaluation metrics of Section 14.

Nothing here is ever entered by hand: every number in the paper comes out of
:func:`run_episode`, which drives the closed loop and returns one row of
metrics per simulated episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .environment import RumenTwinEnv
from .interventions import ACTION_NAMES, N_ACTIONS
from .utils import episode_seed, stable_hash


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _runs_below(x: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """Return [(start, length), ...] of maximal runs with ``x < threshold``."""
    below = x < threshold
    runs, start = [], None
    for i, b in enumerate(below):
        if b and start is None:
            start = i
        elif not b and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(below) - start))
    return runs


def _recovery_times(pH: np.ndarray, low_thr: float, recover_thr: float,
                    confirm: int, dt: float) -> float:
    """Mean minutes from the start of a low-pH excursion to confirmed recovery.

    An excursion is recovered once pH stays at or above ``recover_thr`` for
    ``confirm`` consecutive steps. Excursions never recovered within the
    episode are censored at the episode end (conservative, i.e. they count as
    the longest possible recovery).
    """
    times = []
    n = len(pH)
    for start, _ in _runs_below(pH, low_thr):
        rec, streak = None, 0
        for i in range(start, n):
            streak = streak + 1 if pH[i] >= recover_thr else 0
            if streak >= confirm:
                rec = i - confirm + 1
                break
        times.append(((rec if rec is not None else n) - start) * dt)
    return float(np.mean(times)) if times else float("nan")


# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    metrics: Dict[str, Any]
    trajectory: Optional[pd.DataFrame] = None


def run_episode(env: RumenTwinEnv, controller, cow_id: Optional[int] = None,
                scenario: Optional[str] = None, seed: Optional[int] = None,
                record: bool = False) -> EpisodeResult:
    """Run one closed-loop episode and compute every Section-14 metric."""
    cfg = env.cfg
    thr = cfg["thresholds"]
    ev = cfg["evaluation"]
    dt = float(cfg["simulation"]["dt_minutes"])

    env.record_trajectory = bool(record)
    options: Dict[str, Any] = {}
    if cow_id is not None:
        options["cow_id"] = int(cow_id)
    if scenario is not None:
        options["scenario"] = scenario
    obs, info0 = env.reset(seed=seed, options=options)
    if hasattr(controller, "reset"):
        controller.reset(seed)

    pH, stability, rumination, milk, health, temp, dmi = [], [], [], [], [], [], []
    rewards, actions, costs, pH_seen = [], [], [], []
    done = False
    info = info0
    while not done:
        action = controller.act(env.controller_context()) if getattr(
            controller, "uses_context", False) else controller.act(obs)
        obs, r, terminated, truncated, info = env.step(int(action))
        st = env.physio.state
        pH.append(st.rumen_pH)
        stability.append(env.reward_fn.stability(st))
        rumination.append(st.rumination_minutes)
        milk.append(st.milk_yield_proxy)
        health.append(st.health_risk)
        temp.append(st.rumen_temperature)
        dmi.append(st.dry_matter_intake)
        rewards.append(r)
        actions.append(int(action))
        costs.append(float(info["intervention_cost"]))
        pH_seen.append(bool(info["pH_available"]))
        done = terminated or truncated

    pH = np.asarray(pH); n = len(pH)
    low_thr = float(thr["low_pH"]); sev_thr = float(thr["severe_low_pH"])
    sara_min_steps = int(round(thr["sara_episode_min_minutes"] / dt))
    sev_min_steps = int(round(ev["severe_episode_min_minutes"] / dt))

    low_runs = _runs_below(pH, low_thr)
    sev_runs = _runs_below(pH, sev_thr)
    n_sara = sum(1 for _, L in low_runs if L >= sara_min_steps)
    n_sev_ep = sum(1 for _, L in sev_runs if L >= sev_min_steps)
    pct_low = 100.0 * float((pH < low_thr).mean())
    # cumulative exposure: integral of (threshold - pH)+ over time, pH-units x minutes
    cum_low_exposure = float(np.clip(low_thr - pH, 0.0, None).sum() * dt)

    action_counts = np.bincount(np.asarray(actions), minlength=N_ACTIONS)
    final_health = float(health[-1])
    sc = ev["success_criteria"]
    success = bool(pct_low <= sc["max_pct_time_below_low_pH"]
                   and n_sev_ep <= sc["max_severe_episodes"]
                   and final_health <= sc["max_final_health_risk"])

    m: Dict[str, Any] = {
        "controller": getattr(controller, "name", "unknown"),
        "scenario": env.scenario_name,
        "scenario_label": env.scenario.label,
        "scenario_family": env.scenario.family,
        "cow_id": int(env.cow.cow_id),
        "cow_split": env.cow.split,
        "seed": int(env._np_random_seed_used),
        "n_steps": int(n),
        # --- 1-2 central tendency / variability -----------------------
        "mean_pH": float(pH.mean()),
        "sd_pH": float(pH.std(ddof=1)) if n > 1 else 0.0,
        "min_pH": float(pH.min()),
        # --- 3-6 low-pH exposure --------------------------------------
        "pct_time_below_low_pH": pct_low,
        "cumulative_low_pH_exposure": cum_low_exposure,
        "n_sara_like_episodes": int(n_sara),
        "n_severe_low_pH_episodes": int(n_sev_ep),
        "pct_time_below_severe_pH": 100.0 * float((pH < sev_thr).mean()),
        "longest_low_pH_run_min": float(max((L for _, L in low_runs), default=0) * dt),
        # --- 7-8 stability and recovery -------------------------------
        "physiological_stability": float(np.mean(stability)),
        "mean_recovery_time_min": _recovery_times(
            pH, low_thr, float(thr["recovery_target_pH"]),
            int(ev["recovery_confirm_steps"]), dt),
        # --- 9-11 interventions ---------------------------------------
        "total_intervention_cost": float(np.sum(costs)),
        "n_interventions": int(info["n_interventions"]),
        "n_alerts": int(info["n_alerts"]),
        # --- 12-14 production, return, success --------------------------
        "mean_milk_yield_proxy": float(np.mean(milk)),
        "cumulative_reward": float(np.sum(rewards)),
        "success": success,
        # --- supporting descriptors -------------------------------------
        "mean_health_risk": float(np.mean(health)),
        "final_health_risk": final_health,
        "mean_rumination_min_per_day": float(np.mean(rumination) * cfg["simulation"]["steps_per_day"]),
        "mean_temperature": float(np.mean(temp)),
        "mean_dmi_kg_per_day": float(np.mean(dmi) * cfg["simulation"]["steps_per_day"]),
        # observation quality actually experienced during the episode
        "pct_steps_pH_unobserved": 100.0 * float(1.0 - np.mean(pH_seen)),
    }
    for a in range(N_ACTIONS):
        m[f"n_action_{a}_{ACTION_NAMES[a]}"] = int(action_counts[a])

    traj = env.episode_trajectory() if record else None
    return EpisodeResult(metrics=m, trajectory=traj)


# ---------------------------------------------------------------------------
def make_episode_plan(cow_ids: Sequence[int], scenarios: Sequence[str],
                      eval_seeds: Sequence[int], episodes_per_condition: int,
                      seed_offset: int) -> pd.DataFrame:
    """Deterministic (cow, scenario, seed) plan shared by ALL controllers.

    Two levels of matching are built in:

    * every controller runs the *identical* plan, so controller comparisons are
      paired at the level of the individual simulated episode;
    * the cow at a given ``(eval_seed, episode_index)`` is the same in every
      scenario, so scenario comparisons (e.g. sensor-fault robustness against
      the clean-sensor reference) are paired too.

    The episode seed itself still depends on the scenario, so each scenario
    gets its own stochastic disturbance and sensor-noise realisation.
    """
    rows = []
    per_seed = int(np.ceil(episodes_per_condition / len(eval_seeds)))
    cow_ids = np.asarray(cow_ids)
    # One fixed shuffle of the test cows, then each evaluation seed takes the
    # next contiguous block of it. With episodes_per_condition >= n_test_cows
    # this covers EVERY test cow instead of resampling a random ~78 % of them,
    # and the cow at a given (eval_seed, episode_index) is the same in every
    # scenario - so episodes are matched across controllers AND scenarios,
    # which is what makes the robustness comparisons in Table 7 paired.
    base = np.random.default_rng(
        stable_hash("cow_order") % (2 ** 32 - 1)).permutation(cow_ids)
    for scen in scenarios:
        k = 0
        for i_s, s in enumerate(eval_seeds):
            offsets = (np.arange(per_seed) + i_s * per_seed) % len(base)
            picks = base[offsets]
            for j, cid in enumerate(picks):
                if k >= episodes_per_condition:
                    break
                rows.append({
                    "scenario": scen,
                    "eval_seed": int(s),
                    "episode_index": int(k),
                    "cow_id": int(cid),
                    "seed": episode_seed(seed_offset, scen, int(s), int(j), int(cid)),
                })
                k += 1
    return pd.DataFrame(rows)


def run_plan(env: RumenTwinEnv, controller, plan: pd.DataFrame,
             progress: bool = True, tag: str = "") -> pd.DataFrame:
    """Run every episode in ``plan`` with one controller."""
    out = []
    total = len(plan)
    for i, row in enumerate(plan.itertuples(index=False)):
        res = run_episode(env, controller, cow_id=int(row.cow_id),
                          scenario=str(row.scenario), seed=int(row.seed))
        rec = res.metrics
        rec["eval_seed"] = int(row.eval_seed)
        rec["episode_index"] = int(row.episode_index)
        out.append(rec)
        if progress and (i + 1) % max(1, total // 10) == 0:
            print(f"    [{tag}] {i + 1}/{total} episodes", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
def select_demo_cow(env, cfg: Dict, pop, scenario: str, seed: int,
                     n_candidates: int = 25, quantile: float = 0.75) -> int:
    """Pick a *representative affected* test cow for the illustrative figure.

    Choosing the demo episode by hand would be cherry-picking, so instead we
    screen candidate test cows under the no-intervention baseline and take the
    cow whose low-pH exposure sits at a fixed quantile of that screen. The
    choice is therefore deterministic and stated, not curated.
    """
    from rumen_twin.expert_rules import NoInterventionController

    base = NoInterventionController(cfg)
    cows = pop.subset("test")[:n_candidates]
    expo = []
    for c in cows:
        m = run_episode(env, base, cow_id=c.cow_id, scenario=scenario, seed=seed).metrics
        expo.append((m["pct_time_below_low_pH"], c.cow_id))
    expo.sort()
    idx = min(len(expo) - 1, int(round(quantile * (len(expo) - 1))))
    return int(expo[idx][1])
