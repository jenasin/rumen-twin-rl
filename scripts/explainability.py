#!/usr/bin/env python3
"""Explainability analysis (Section 18).

Two complementary views of *why and when* a learned policy intervenes:

1. **Episode timeline** — one held-out test episode with the true state, what
   the sensors actually reported, the estimator's belief, the action taken and
   the reward earned at every 15-min step. Written both as Figure 9 and as a
   machine-readable timeline table, so any individual decision can be traced.

2. **Local sensitivity of the policy** — for a large sample of states actually
   visited during evaluation, each observation feature is perturbed by
   +/- 1 SD (of that feature's own distribution across the sample) while all
   others are held fixed, and we record

       * the mean absolute change in the probability the policy assigns to the
         action it originally chose, and
       * the fraction of states where the greedy action *changes*.

   This is a deliberately simple local method: it needs no surrogate model and
   no additivity assumption, and it answers the question actually being asked
   ("which observed signals move this policy's decisions?"). It is a *local*
   diagnostic and does not attribute causal importance.
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import _paths  # noqa: F401
from agents.common import load_controllers
from rumen_twin.cow import CowPopulation
from rumen_twin.environment import OBS_NAMES, RumenTwinEnv
from rumen_twin.evaluation import run_episode, select_demo_cow
from rumen_twin.interventions import ACTION_NAMES
from rumen_twin.plotting import (ACTION_MARKER, add_ph_threshold_bands, controller_colour,
                                 controller_label, save_figure, set_manuscript_mode,
                                 set_style)
from rumen_twin.utils import (DISCLAIMER, PATHS, banner, ensure_dirs, load_config,
                              load_seeds, write_parquet_or_csv)

FEATURE_LABEL = {
    "pH_estimate": "pH estimate", "pH_available": "pH sensor available",
    "pH_trend": "pH trend", "pH_uncertainty": "pH estimate uncertainty",
    "pH_min_recent": "min pH (last 6 h)", "frac_low_recent": "frac. low pH (last 6 h)",
    "temperature": "rumen temperature", "rumination": "rumination",
    "dry_matter_intake": "dry matter intake", "heat_load": "heat load",
    "milk_yield_proxy": "milk yield proxy", "hydration_index": "hydration",
    "forage_ratio": "current forage ratio", "planned_forage_ratio": "planned forage ratio",
    "nominal_buffer_pool": "buffer given (nominal)", "steps_since_pH_obs": "steps since pH reading",
    "recent_interventions": "recent interventions", "cumulative_cost": "cumulative cost",
    "time_sin": "time of day (sin)", "time_cos": "time of day (cos)",
    "active_forage": "forage order active", "active_reduce_conc": "concentrate cut active",
    "active_buffer": "buffer active", "active_split_feed": "split feeding active",
    "active_obs_boost": "obs. boost active", "active_alert": "alert active",
}


# ---------------------------------------------------------------------------
def action_probabilities(model, obs: np.ndarray) -> np.ndarray:
    """Policy action distribution (PPO) or softmax-free greedy one-hot (DQN)."""
    x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
    with torch.no_grad():
        if hasattr(model.policy, "get_distribution"):
            dist = model.policy.get_distribution(x)
            return dist.distribution.probs.cpu().numpy()
        q = model.q_net(x).cpu().numpy()                       # DQN: Q-values
        out = np.zeros_like(q)
        out[np.arange(len(q)), q.argmax(axis=1)] = 1.0
        return out


def q_or_prob_matrix(model, obs: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32)
    with torch.no_grad():
        if hasattr(model.policy, "get_distribution"):
            return model.policy.get_distribution(x).distribution.probs.cpu().numpy()
        return model.q_net(x).cpu().numpy()


# ---------------------------------------------------------------------------
def figure9_timeline(cfg: Dict, seeds: Dict, style: Dict, controller_name: str,
                     scenario: str) -> pd.DataFrame:
    pop = CowPopulation(config=cfg, seeds=seeds)
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=pop, split="test",
                       scenarios=[scenario], record_trajectory=True)
    seed = int(seeds["explainability"]["seed"])

    # deterministic, stated choice of the illustrated episode
    cow_id = select_demo_cow(env, cfg, pop, scenario, seed)

    ctrl = load_controllers([controller_name], cfg)[0]
    res = run_episode(env, ctrl, cow_id=cow_id, scenario=scenario, seed=seed, record=True)
    tr = res.trajectory
    col = controller_colour(controller_name, style)

    fig, axes = plt.subplots(5, 1, figsize=(8.8, 9.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1.1, 1.0, 1.0, 1.0],
                                          "hspace": 0.14})
    # --- pH: truth vs what the controller could see -------------------
    ax = axes[0]
    ax.plot(tr["hour"], tr["observed_pH"], linestyle="none", marker=".", markersize=2.6,
            color="#9a9994", label="observed (sensor)", zorder=2)
    ax.plot(tr["hour"], tr["estimated_pH"], color="#52514e", lw=1.2, ls="--",
            label="estimated (belief)", zorder=3)
    ax.plot(tr["hour"], tr["true_rumen_pH"], color=col, lw=1.9, label="true pH (latent)", zorder=4)
    ax.set_ylim(min(5.3, tr["true_rumen_pH"].min() - 0.12),
                max(6.8, tr["true_rumen_pH"].max() + 0.08))
    add_ph_threshold_bands(ax, cfg, style)
    ax.set_ylabel("Rumen pH")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.115), ncol=3, fontsize=8)
    ax.set_title(f"Figure 9 | Decision timeline: {controller_label(controller_name)} on test cow {cow_id}, "
                 f"{cfg['scenarios'][scenario]['label']}", loc="left", pad=42)

    # --- actions ------------------------------------------------------
    ax = axes[1]
    acted = tr[tr["action"] > 0]
    for a in sorted(acted["action"].unique()):
        sub = acted[acted["action"] == a]
        ax.scatter(sub["hour"], np.full(len(sub), a), marker=ACTION_MARKER.get(int(a), "o"),
                   s=30, color=col, edgecolor=style["surface"], linewidth=0.6, zorder=3)
    for _, r in acted.iterrows():                       # link each action to the pH panel
        axes[0].axvline(r["hour"], color=col, lw=0.5, alpha=0.18, zorder=0)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([ACTION_NAMES[i].replace("_", " ") for i in range(1, 7)], fontsize=7)
    ax.set_ylim(0.4, 6.6)
    ax.set_ylabel("Action", fontsize=8)

    # --- rumination ---------------------------------------------------
    ax = axes[2]
    ax.plot(tr["hour"], tr["true_rumination_minutes"] * cfg["simulation"]["steps_per_day"],
            color=col, lw=1.4)
    ax.set_ylabel("Rumination\n(min/day eq.)", fontsize=8)

    # --- latent drivers the controller never sees ---------------------
    ax = axes[3]
    ax.plot(tr["hour"], tr["true_vfa_index"], color="#52514e", lw=1.4, label="VFA index")
    ax.plot(tr["hour"], tr["true_lactate_index"], color="#9c2b2b", lw=1.4, label="lactate index")
    ax.plot(tr["hour"], tr["true_health_risk"], color=style["status"]["critical"], lw=1.4,
            ls="--", label="health risk")
    ax.set_ylabel("Latent state\n(never observed)", fontsize=8)
    ax.legend(loc="upper left", ncol=3, fontsize=7.2)

    # --- reward -------------------------------------------------------
    ax = axes[4]
    ax.bar(tr["hour"], tr["reward"], width=0.2,
           color=np.where(tr["reward"] >= 0, style["status"]["good"],
                          style["status"]["critical"]), zorder=2)
    ax.axhline(0, color="#c9c8c3", lw=0.8)
    ax.set_ylabel("Reward\nper step", fontsize=8)
    ax.set_xlabel("Time (h)")
    for a in axes:
        a.set_xlim(0, tr["hour"].max())
    save_figure(fig, "figure9_explainability_timeline", cfg,
                extra_footer="Vertical rules in the top panel mark the steps at which the "
                             "controller intervened.")
    return tr


# ---------------------------------------------------------------------------
def local_sensitivity(cfg: Dict, seeds: Dict, controller_name: str, run_name: str,
                      n_states: int = 900, delta_sd: float = 1.0) -> pd.DataFrame:
    """Perturb one observation feature at a time across many visited states."""
    pop = CowPopulation(config=cfg, seeds=seeds)
    scenarios = list(cfg["evaluation_scenarios"])
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=pop, split="test",
                       scenarios=scenarios)
    ctrl = load_controllers([controller_name], cfg,
                            {controller_name: run_name})[0]
    model = ctrl.model

    # --- collect states the policy actually visits --------------------
    rng = np.random.default_rng(int(seeds["explainability"]["seed"]))
    states: List[np.ndarray] = []
    ep = 0
    while len(states) < n_states:
        scen = scenarios[ep % len(scenarios)]
        obs, _ = env.reset(seed=int(rng.integers(1, 2 ** 31)), options={"scenario": scen})
        done = False
        while not done:
            states.append(obs.copy())
            obs, _, te, tr_, _ = env.step(ctrl.act(obs))
            done = te or tr_
        ep += 1
    X = np.asarray(states[:n_states], dtype=np.float32)

    base = q_or_prob_matrix(model, X)
    base_action = base.argmax(axis=1)
    base_p = base[np.arange(len(X)), base_action]
    sd = X.std(axis=0)

    rows = []
    for j, feat in enumerate(OBS_NAMES):
        if sd[j] < 1e-8:                     # constant feature: cannot perturb it
            rows.append({"feature": feat, "label": FEATURE_LABEL.get(feat, feat),
                         "feature_sd": 0.0, "mean_abs_prob_change": 0.0,
                         "action_switch_rate_pct": 0.0})
            continue
        switch, dprob = 0.0, 0.0
        for sign in (-1.0, +1.0):
            Xp = X.copy()
            Xp[:, j] = Xp[:, j] + sign * delta_sd * sd[j]
            out = q_or_prob_matrix(model, Xp)
            switch += float((out.argmax(axis=1) != base_action).mean())
            dprob += float(np.abs(out[np.arange(len(X)), base_action] - base_p).mean())
        rows.append({"feature": feat, "label": FEATURE_LABEL.get(feat, feat),
                     "feature_sd": float(sd[j]),
                     "mean_abs_prob_change": dprob / 2.0,
                     "action_switch_rate_pct": 100.0 * switch / 2.0})
    df = pd.DataFrame(rows).sort_values("action_switch_rate_pct", ascending=False)
    df.insert(0, "controller", controller_name)
    df.insert(1, "n_states", len(X))
    # Absolute switch rates must be read against how often the policy is on a
    # decision boundary at all: it selects "no intervention" in most states, and
    # a +/- 1 SD nudge will not move it off that choice.
    df["pct_states_action_0"] = 100.0 * float((base_action == 0).mean())
    return df


def figure10_sensitivity(df: pd.DataFrame, cfg: Dict, style: Dict,
                         controller_name: str, top_n: int = 14) -> None:
    d = df.head(top_n).iloc[::-1]
    col = controller_colour(controller_name, style)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharey=True)
    y = np.arange(len(d))
    for ax, (c, lab) in zip(axes, [
            ("action_switch_rate_pct", "States where the greedy action changes (%)"),
            ("mean_abs_prob_change", "Mean |change| in P(originally chosen action)")]):
        ax.barh(y, d[c], color=col, height=0.72, zorder=2)
        ax.set_xlabel(lab, fontsize=8.4)
        ax.grid(axis="y", visible=False)
        for yi, v in zip(y, d[c]):
            ax.annotate(f"{v:.3g}", xy=(v, yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7, color="#52514e")
        ax.set_xlim(0, float(d[c].max()) * 1.18)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d["label"], fontsize=8)
    fig.suptitle(f"Figure 10 | Local sensitivity of the {controller_label(controller_name)} policy "
                 f"to each observed feature (+/- 1 SD perturbation, "
                 f"{int(df['n_states'].iloc[0])} visited states)",
                 x=0.012, ha="left", fontsize=9.6, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "figure10_policy_sensitivity", cfg,
                extra_footer="Local diagnostic of decision sensitivity; not a causal "
                             "attribution of importance.")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Explainability analysis")
    ap.add_argument("--controller", default="ppo")
    ap.add_argument("--run-name", default="ppo_main")
    ap.add_argument("--scenario", default="E_gradual_sara")
    ap.add_argument("--states", type=int, default=900)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the cached sensitivity table instead of recomputing it")
    ap.add_argument("--manuscript", action="store_true",
                    help="drop in-image titles/disclaimer and write to figures/manuscript/")
    args = ap.parse_args()

    ensure_dirs()
    cfg, sds = load_config(), load_seeds()
    style = set_style(cfg)
    if args.manuscript:
        set_manuscript_mode(True)
    print(banner("RumenTwin-RL | explainability analysis"))
    print(DISCLAIMER)

    tr = figure9_timeline(cfg, sds, style, args.controller, args.scenario)
    cols = ["step", "hour", "cow_id", "scenario", "true_rumen_pH", "observed_pH",
            "estimated_pH", "pH_available", "true_rumination_minutes", "true_vfa_index",
            "true_lactate_index", "true_health_risk", "true_dry_matter_intake",
            "true_milk_yield_proxy", "action", "action_name", "intervention_cost",
            "cumulative_cost", "reward", "cumulative_reward"]
    timeline = tr[[c for c in cols if c in tr.columns]]
    p = write_parquet_or_csv(timeline, PATHS["results_processed"] / "explainability_timeline")
    timeline.to_csv(PATHS["tables"] / "tableS3_explainability_timeline.csv", index=False)
    print(f"  timeline ({len(timeline)} steps) -> {p.name} and tables/tableS3_...csv")

    cache = PATHS["tables"] / "tableS4_policy_sensitivity.csv"
    if args.reuse and cache.exists():
        print(f"\n  reusing cached sensitivity analysis ({cache.name})")
        sens = pd.read_csv(cache)
    else:
        print("\n  running local sensitivity analysis ...")
        sens = local_sensitivity(cfg, sds, args.controller, args.run_name,
                                 n_states=args.states)
        sens.to_csv(cache, index=False)
    figure10_sensitivity(sens, cfg, style, args.controller)
    print("\n  most decision-relevant features:")
    print(sens.head(8)[["label", "action_switch_rate_pct", "mean_abs_prob_change"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
