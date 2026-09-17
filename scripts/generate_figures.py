#!/usr/bin/env python3
"""Generate Figures 1-9 from the simulation output (Section 19).

Every figure is drawn from ``results/raw/*.parquet`` (or, for Figure 1, from
the architecture itself). Figures are written to ``figures/`` at 300 dpi as PNG
and as vector PDF.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import _paths  # noqa: F401
from rumen_twin.cow import CowPopulation
from rumen_twin.environment import RumenTwinEnv
from rumen_twin.evaluation import run_episode, select_demo_cow
from rumen_twin.interventions import ACTION_NAMES
from rumen_twin.plotting import (ACTION_MARKER, CONTROLLER_HATCH, CONTROLLER_LINESTYLE,
                                 CONTROLLER_MARKER, add_ph_threshold_bands, bar_with_ci,
                                 controller_colour, controller_label, controller_legend,
                                 order_controllers, save_figure, set_manuscript_mode,
                                 set_style)
from rumen_twin.statistics import bootstrap_ci
from rumen_twin.utils import PATHS, banner, ensure_dirs, load_config, load_seeds, read_table


# ===========================================================================
# Figure 1 — architecture
# ===========================================================================
def figure1_architecture(cfg: Dict, style: Dict) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.axis("off")
    ax.grid(False)

    pal = style["palette"]
    #            label                                    x   y   w   h  face      edge
    blocks = [
        ("Cow physiological\nstate  $s_t$",                2, 27, 15, 11, "#e9eef8", pal["expert_rules"]),
        ("Synthetic\nsensors",                            23, 27, 14, 11, "#e9eef8", pal["expert_rules"]),
        ("State\nestimator",                              43, 27, 14, 11, "#e9eef8", pal["expert_rules"]),
        ("Decision policy\n(rules / DQN / PPO)",           63, 27, 21, 11, "#e6f6ef", pal["ppo"]),
        ("Intervention  $a_t$",                            63,  6, 21,  9, "#fdeee6", pal["dqn"]),
        ("Rumen digital twin\n$s_{t+1}=f(s_t,a_t,e_t)+\\epsilon_t$",
                                                           23,  6, 34,  9, "#efedf8", pal["no_intervention"]),
        ("Disturbances $e_t$\n(ration, heat,\nfeed timing)", 2,  4, 15, 11, "#fdf4e2",
         style["status"]["warning"]),
    ]
    box = {}
    for text, x, y, w, h, fc, ec in blocks:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                                    facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4,
                color=style["text_primary"], zorder=3)
        box[text] = (x, y, w, h)

    def h_arrow(a, b, label="", colour="#52514e", fs=7.0):
        """Horizontal arrow; the label sits ABOVE the row so it cannot collide
        with the boxes it connects."""
        xa, ya, wa, ha_ = box[a]; xb, yb, wb, hb = box[b]
        y = ya + ha_ / 2
        ax.add_patch(FancyArrowPatch((xa + wa + 0.4, y), (xb - 0.4, y), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.3, color=colour, zorder=1))
        if label:
            xm = (xa + wa + xb) / 2
            ax.text(xm, ya + ha_ + 2.0, label, ha="center", va="bottom",
                    fontsize=fs, color="#6d6c68", linespacing=1.2)
            ax.plot([xm, xm], [ya + ha_ + 0.4, ya + ha_ + 1.7], color="#c9c8c3",
                    lw=0.7, zorder=0)

    keys = list(box)
    h_arrow(keys[0], keys[1], "true state")
    h_arrow(keys[1], keys[2], "noisy / missing / delayed", fs=6.8)
    h_arrow(keys[2], keys[3], "belief")
    h_arrow(keys[6], keys[5], colour=style["status"]["warning"])
    # policy -> intervention (down), intervention -> twin (left)
    ax.add_patch(FancyArrowPatch((73.5, 27 - 0.4), (73.5, 15 + 0.4), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.3, color="#52514e", zorder=1))
    ax.text(75.2, 21, "action $a_t$", ha="left", va="center", fontsize=7.0, color="#6d6c68")
    ax.add_patch(FancyArrowPatch((63 - 0.4, 10.5), (57 + 0.4, 10.5), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.3, color="#52514e", zorder=1))
    # twin -> cow state (feedback loop, up the left side)
    ax.add_patch(FancyArrowPatch((25, 15 + 0.4), (9.5, 27 - 0.4), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.3, color="#52514e",
                                 connectionstyle="arc3,rad=0.32", zorder=1))
    ax.text(20.5, 22.5, "updated state,\nnew observation", ha="left", va="center",
            fontsize=7.0, color="#6d6c68", linespacing=1.3)

    ax.text(87, 33, "latent, never\nobserved:\nVFA, lactate,\nhealth risk", ha="left",
            va="center", fontsize=7.2, color="#9c2b2b", style="italic", linespacing=1.35)
    ax.set_title("Figure 1 | RumenTwin-RL closed-loop architecture", loc="left", pad=6)
    save_figure(fig, "figure1_architecture", cfg)


# ===========================================================================
# Figure 2 — example trajectory
# ===========================================================================
def figure2_trajectory(cfg: Dict, style: Dict, seeds: Dict,
                       controllers: Sequence[str] = ("no_intervention", "ppo"),
                       scenario: str = "E_gradual_sara") -> None:
    from agents.common import load_controllers

    pop = CowPopulation(config=cfg, seeds=seeds)
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=pop, split="test",
                       scenarios=[scenario], record_trajectory=True)
    seed = int(seeds["explainability"]["seed"])
    cow_id = select_demo_cow(env, cfg, pop, scenario, seed)

    trajs = {}
    for c in load_controllers(list(controllers), cfg):
        res = run_episode(env, c, cow_id=cow_id, scenario=scenario, seed=seed, record=True)
        trajs[c.name] = res.trajectory

    fig, axes = plt.subplots(4, 1, figsize=(8.8, 8.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1.0, 1.0, 1.0], "hspace": 0.15})
    ref_name = controllers[-1]
    ctrl_tr = trajs[ref_name]

    # --- panel 1: pH ------------------------------------------------
    ax = axes[0]
    ax.plot(ctrl_tr["hour"], ctrl_tr["observed_pH"], linestyle="none", marker=".",
            markersize=2.4, color="#9a9994", alpha=0.85, zorder=2,
            label="observed pH (sensor)")
    for name, tr in trajs.items():
        ax.plot(tr["hour"], tr["true_rumen_pH"], color=controller_colour(name, style), lw=1.9,
                ls=CONTROLLER_LINESTYLE.get(name, "-"), label=controller_label(name), zorder=4)
    allv = np.concatenate([t["true_rumen_pH"].to_numpy() for t in trajs.values()])
    ax.set_ylim(min(5.32, allv.min() - 0.12), max(6.75, allv.max() + 0.10))
    add_ph_threshold_bands(ax, cfg, style)
    ax.set_ylabel("Rumen pH")
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.105), ncol=3, fontsize=8)
    ax.set_title(f"Figure 2 | 48-h trajectory, test cow {cow_id}, "
                 f"{cfg['scenarios'][scenario]['label']}", loc="left", pad=42)

    # --- panel 2: intervention raster -------------------------------
    ax = axes[1]
    acted = ctrl_tr[ctrl_tr["action"] > 0]
    for a in sorted(acted["action"].unique()):
        sub = acted[acted["action"] == a]
        ax.scatter(sub["hour"], np.full(len(sub), a), marker=ACTION_MARKER.get(int(a), "o"),
                   s=30, color=controller_colour(ref_name, style),
                   edgecolor=style["surface"], linewidth=0.6, zorder=3)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels([ACTION_NAMES[i].replace("_", " ") for i in range(1, 7)], fontsize=7)
    ax.set_ylim(0.4, 6.6)
    ax.set_ylabel(f"{controller_label(ref_name)}\nactions", fontsize=8)

    # --- panel 3: rumination ----------------------------------------
    ax = axes[2]
    for name, tr in trajs.items():
        ax.plot(tr["hour"], tr["true_rumination_minutes"] * cfg["simulation"]["steps_per_day"],
                color=controller_colour(name, style), lw=1.4,
                ls=CONTROLLER_LINESTYLE.get(name, "-"))
    ax.set_ylabel("Rumination\n(min/day eq.)", fontsize=8)

    # --- panel 4: latent health risk (never observed by any controller)
    ax = axes[3]
    for name, tr in trajs.items():
        ax.plot(tr["hour"], tr["true_health_risk"], color=controller_colour(name, style),
                lw=1.6, ls=CONTROLLER_LINESTYLE.get(name, "-"))
    ax.set_ylabel("Health-risk index\n(latent)", fontsize=8)
    ax.set_xlabel("Time (h)")
    ax.set_ylim(bottom=0)
    for a in axes:
        a.set_xlim(0, ctrl_tr["hour"].max())
    save_figure(fig, "figure2_example_trajectory", cfg,
                extra_footer="Controllers are distinguished by line style as well as colour.")


# ===========================================================================
# helpers for the comparison figures
# ===========================================================================
# vertical nudges (points) that keep direct labels from colliding when two
# series end at nearly the same value
# (dx, dy, ha) offsets keeping the four controller labels apart in Figure 6
ANNOT_OFFSET = {"no_intervention": (11, -13, "left"), "expert_rules": (13, -19, "left"),
                "dqn": (13, 7, "left"), "ppo": (4, 14, "left")}
LABEL_DY = {"no_intervention": 0.0, "expert_rules": -7.0, "dqn": 7.0, "ppo": 0.0}


def _ci_table(df: pd.DataFrame, metric: str, by: List[str], n_boot: int,
              seed: int) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ci = bootstrap_ci(sub[metric].astype(float) * (100.0 if metric == "success" else 1.0),
                          np.mean, n_boot, 0.95, seed)
        rows.append({**dict(zip(by, keys)), "mean": ci["estimate"],
                     "lo": ci["ci_low"], "hi": ci["ci_high"], "n": len(sub)})
    return pd.DataFrame(rows)


# ===========================================================================
# Figure 3 — low-pH exposure comparison
# ===========================================================================
def figure3_low_ph_exposure(df: pd.DataFrame, cfg: Dict, style: Dict,
                            n_boot: int, seed: int) -> None:
    ctrls = order_controllers(df["controller"].unique())
    metrics = [("pct_time_below_low_pH", "Time below pH 5.8 (%)"),
               ("cumulative_low_pH_exposure", "Cumulative low-pH exposure\n(pH-units x min)"),
               ("n_sara_like_episodes", "SARA-like episodes per 48 h (n)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    for ax, (m, lab) in zip(axes, metrics):
        t = _ci_table(df, m, ["controller"], n_boot, seed).set_index("controller").loc[ctrls]
        x = np.arange(len(ctrls))
        rng = np.random.default_rng(seed)
        for i, c in enumerate(ctrls):                      # raw episode cloud
            v = df.loc[df["controller"] == c, m].to_numpy()
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=1.6, alpha=0.10,
                       color=controller_colour(c, style), zorder=1, linewidths=0)
        bar_with_ci(ax, x, t["mean"].to_numpy(), t["lo"].to_numpy(), t["hi"].to_numpy(),
                    [controller_colour(c, style) for c in ctrls],
                    hatches=[CONTROLLER_HATCH[c] for c in ctrls], label_fmt="{:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([controller_label(c) for c in ctrls], rotation=18, ha="right")
        ax.set_title(lab, fontsize=8.8)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("mean ± 95 % bootstrap CI", fontsize=8)
    fig.suptitle("Figure 3 | Low-pH exposure by controller (all evaluation scenarios pooled)",
                 x=0.012, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "figure3_low_pH_exposure", cfg,
                extra_footer="Bars also carry a distinct hatch per controller; dots are individual episodes.")


# ===========================================================================
# Figure 4 — performance by scenario
# ===========================================================================
def figure4_by_scenario(df: pd.DataFrame, cfg: Dict, style: Dict,
                        n_boot: int, seed: int) -> None:
    ctrls = order_controllers(df["controller"].unique())
    scens = [s for s in cfg["evaluation_scenarios"] if s in set(df["scenario"])]
    labels = {s: cfg["scenarios"][s]["label"] for s in scens}
    metrics = [("pct_time_below_low_pH", "Time below pH 5.8 (%)"),
               ("cumulative_reward", "Cumulative reward per episode")]
    fig, axes = plt.subplots(2, 1, figsize=(9.8, 6.6), sharex=True)
    width = 0.8 / len(ctrls)
    for ax, (m, lab) in zip(axes, metrics):
        t = _ci_table(df, m, ["scenario", "controller"], n_boot, seed)
        for i, c in enumerate(ctrls):
            sub = t[t["controller"] == c].set_index("scenario").reindex(scens)
            x = np.arange(len(scens)) + (i - (len(ctrls) - 1) / 2) * width
            err = np.vstack([sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]])
            b = ax.bar(x, sub["mean"], width=width * 0.92, color=controller_colour(c, style),
                       edgecolor=style["surface"], linewidth=0.8,
                       label=controller_label(c), zorder=2)
            for bb in b:
                bb.set_hatch(CONTROLLER_HATCH[c])
            ax.errorbar(x, sub["mean"], yerr=err, fmt="none", ecolor="#3a3a38",
                        elinewidth=0.9, capsize=1.8, zorder=3)
        ax.set_ylabel(lab, fontsize=8.6)
        ax.axhline(0, color="#c9c8c3", lw=0.8)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.17 * (hi_ - lo_))      # headroom for the legend
    # mark the held-out scenarios
    for ax in axes:
        for j, s in enumerate(scens):
            if s not in cfg["training_scenarios"]:
                ax.axvspan(j - 0.5, j + 0.5, color="#fdf4e2", zorder=0)
    axes[-1].set_xticks(np.arange(len(scens)))
    axes[-1].set_xticklabels([labels[s] for s in scens], rotation=24, ha="right", fontsize=7.6)
    axes[0].legend(ncol=4, loc="upper left", fontsize=8)
    axes[0].set_title("Figure 4 | Performance by disturbance scenario "
                      "(shaded = held out from RL training)", loc="left", pad=6)
    fig.tight_layout()
    save_figure(fig, "figure4_performance_by_scenario", cfg)


# ===========================================================================
# Figure 5 — robustness to sensor faults
# ===========================================================================
def figure5_robustness(df: pd.DataFrame, cfg: Dict, style: Dict,
                       n_boot: int, seed: int) -> None:
    ref = cfg["scenarios"]["F_sensor_dropout"]["base_disturbance"]
    faults = [s for s in ["F_sensor_dropout", "G_sensor_noise", "H_sensor_delay"]
              if s in set(df["scenario"])]
    ctrls = order_controllers(df["controller"].unique())
    scens = [ref] + faults
    short = {ref: "clean\nsensors", "F_sensor_dropout": "dropout",
             "G_sensor_noise": "noise + bias", "H_sensor_delay": "delay"}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, (m, lab, better) in zip(axes, [
            ("pct_time_below_low_pH", "Time below pH 5.8 (%)", "lower is better"),
            ("cumulative_reward", "Cumulative reward", "higher is better")]):
        t = _ci_table(df[df["scenario"].isin(scens)], m, ["scenario", "controller"], n_boot, seed)
        x = np.arange(len(scens))
        for c in ctrls:
            sub = t[t["controller"] == c].set_index("scenario").reindex(scens)
            col = controller_colour(c, style)
            ax.plot(x, sub["mean"], color=col, lw=1.8, ls=CONTROLLER_LINESTYLE.get(c, "-"),
                    marker=CONTROLLER_MARKER.get(c, "o"), markersize=6,
                    markeredgecolor=style["surface"], markeredgewidth=0.9, zorder=3)
            ax.fill_between(x, sub["lo"], sub["hi"], color=col, alpha=0.13, lw=0, zorder=1)
            ax.annotate(controller_label(c), xy=(x[-1], sub["mean"].iloc[-1]),
                        xytext=(6, LABEL_DY.get(c, 0)), textcoords="offset points",
                        va="center", fontsize=7.4, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels([short[s] for s in scens], fontsize=8)
        ax.set_title(f"{lab}  ({better})", fontsize=8.8)
        ax.set_xlim(-0.3, len(scens) - 0.3 + 0.9)
    axes[0].set_ylabel("mean ± 95 % bootstrap CI", fontsize=8)
    fig.suptitle("Figure 5 | Robustness to sensor dropout, noise and delay "
                 "(same underlying nutritional disturbance throughout)",
                 x=0.012, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "figure5_sensor_robustness", cfg,
                extra_footer="Series are direct-labelled and carry distinct markers and line styles.")


# ===========================================================================
# Figure 6 — cost vs stability
# ===========================================================================
def figure6_cost_vs_stability(df: pd.DataFrame, cfg: Dict, style: Dict,
                              n_boot: int, seed: int) -> None:
    ctrls = order_controllers(df["controller"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))

    ax = axes[0]
    for c in ctrls:
        sub = df[df["controller"] == c]
        ax.scatter(sub["total_intervention_cost"], sub["physiological_stability"],
                   s=3.5, alpha=0.10, color=controller_colour(c, style), linewidths=0, zorder=1)
    for c in ctrls:
        sub = df[df["controller"] == c]
        cx = bootstrap_ci(sub["total_intervention_cost"], np.mean, n_boot, 0.95, seed)
        cy = bootstrap_ci(sub["physiological_stability"], np.mean, n_boot, 0.95, seed)
        col = controller_colour(c, style)
        ax.errorbar(cx["estimate"], cy["estimate"],
                    xerr=[[cx["estimate"] - cx["ci_low"]], [cx["ci_high"] - cx["estimate"]]],
                    yerr=[[cy["estimate"] - cy["ci_low"]], [cy["ci_high"] - cy["estimate"]]],
                    fmt=CONTROLLER_MARKER.get(c, "o"), color=col, markersize=9,
                    markeredgecolor=style["surface"], markeredgewidth=1.2,
                    ecolor=col, elinewidth=1.6, capsize=3, zorder=4)
        dx, dy, ha = ANNOT_OFFSET.get(c, (9, 8, "left"))
        ax.annotate(controller_label(c), xy=(cx["estimate"], cy["estimate"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8, color=col, ha=ha,
                    fontweight="semibold")
    # a handful of very expensive episodes would otherwise compress the means
    xmax = float(np.percentile(df["total_intervention_cost"], 99.5))
    ax.set_xlim(-1.0, xmax * 1.12)
    ax.set_xlabel("Total intervention cost per 48-h episode\n"
                  "(x-axis truncated at the 99.5th percentile)", fontsize=8.6)
    ax.set_ylabel("Physiological stability score")
    ax.set_title("Cost-benefit position of each controller\n"
                 "(large markers: mean ± 95 % CI; dots: individual episodes)", fontsize=8.8)

    ax = axes[1]
    t = _ci_table(df, "physiological_stability", ["scenario", "controller"], n_boot, seed)
    tc = _ci_table(df, "total_intervention_cost", ["scenario", "controller"], n_boot, seed)
    merged = t.merge(tc, on=["scenario", "controller"], suffixes=("_stab", "_cost"))
    for c in ctrls:
        sub = merged[merged["controller"] == c]
        ax.scatter(sub["mean_cost"], sub["mean_stab"], s=42,
                   marker=CONTROLLER_MARKER.get(c, "o"), color=controller_colour(c, style),
                   edgecolor=style["surface"], linewidth=1.0, zorder=3)
    ax.set_xlabel("Total intervention cost per 48-h episode")
    ax.set_ylabel("Physiological stability score")
    ax.set_title("One point per (controller x scenario) condition", fontsize=8.8)
    controller_legend(ax, ctrls, style, loc="lower right")

    fig.suptitle("Figure 6 | Intervention cost versus physiological stability",
                 x=0.012, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "figure6_cost_vs_stability", cfg)


# ===========================================================================
# Figure 7 — individual variability
# ===========================================================================
def figure7_individual_variability(df: pd.DataFrame, cfg: Dict, style: Dict) -> None:
    ctrls = order_controllers(df["controller"].unique())
    per_cow = (df.groupby(["controller", "cow_id"])
                 .agg(pct_low=("pct_time_below_low_pH", "mean"),
                      reward=("cumulative_reward", "mean")).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))

    ax = axes[0]                                   # ECDF avoids colour cycling issues
    for c in ctrls:
        v = np.sort(per_cow.loc[per_cow["controller"] == c, "pct_low"].to_numpy())
        y = np.arange(1, len(v) + 1) / len(v)
        ax.step(v, y, where="post", color=controller_colour(c, style), lw=1.8,
                ls=CONTROLLER_LINESTYLE.get(c, "-"))
        ax.plot(v[::max(1, len(v) // 10)], y[::max(1, len(y) // 10)], linestyle="none",
                marker=CONTROLLER_MARKER.get(c, "o"), markersize=4,
                color=controller_colour(c, style), markeredgecolor=style["surface"],
                markeredgewidth=0.7)
    ax.set_xlabel("Per-cow mean time below pH 5.8 (%)")
    ax.set_ylabel("Cumulative fraction of test cows")
    ax.set_title(f"Between-cow distribution ({per_cow['cow_id'].nunique()} test cows)",
                 fontsize=8.8)
    controller_legend(ax, ctrls, style, loc="lower right")

    ax = axes[1]                                   # per-cow paired change vs no intervention
    base = per_cow[per_cow["controller"] == "no_intervention"].set_index("cow_id")["pct_low"]
    for c in [x for x in ctrls if x != "no_intervention"]:
        sub = per_cow[per_cow["controller"] == c].set_index("cow_id")["pct_low"]
        joined = pd.concat([base.rename("b"), sub.rename("c")], axis=1).dropna()
        ax.scatter(joined["b"], joined["c"], s=11, alpha=0.55,
                   marker=CONTROLLER_MARKER.get(c, "o"), color=controller_colour(c, style),
                   linewidths=0, zorder=3)
    lim = [0, float(per_cow["pct_low"].max()) * 1.05]
    ax.plot(lim, lim, color="#9a9994", lw=1.0, ls=":", zorder=2)
    ax.annotate("no change", xy=(lim[1] * 0.72, lim[1] * 0.75), fontsize=7.2, color="#6d6c68",
                rotation=38)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("No intervention: time below pH 5.8 (%)")
    ax.set_ylabel("Controller: time below pH 5.8 (%)")
    ax.set_title("Each point is one test cow (points below the line = improved)", fontsize=8.8)
    controller_legend(ax, [x for x in ctrls if x != "no_intervention"], style, loc="upper left")

    fig.suptitle("Figure 7 | Individual variability across the virtual cow population",
                 x=0.012, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "figure7_individual_variability", cfg)


# ===========================================================================
# Figure 8 — generalisation to unseen disturbances
# ===========================================================================
def figure8_generalisation(df: pd.DataFrame, cfg: Dict, style: Dict,
                           n_boot: int, seed: int) -> None:
    ctrls = order_controllers(df["controller"].unique())
    unseen = [s for s in df["scenario"].unique() if s not in cfg["training_scenarios"]]
    d = df.copy()
    d["group"] = np.where(d["scenario"].isin(unseen), "unseen (I, J)", "seen in training (A-H)")

    metrics = [("pct_time_below_low_pH", "Time below pH 5.8 (%)"),
               ("physiological_stability", "Physiological stability"),
               ("cumulative_reward", "Cumulative reward")]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.8))
    groups = ["seen in training (A-H)", "unseen (I, J)"]
    width = 0.8 / len(ctrls)
    for ax, (m, lab) in zip(axes, metrics):
        t = _ci_table(d, m, ["group", "controller"], n_boot, seed)
        for i, c in enumerate(ctrls):
            sub = t[t["controller"] == c].set_index("group").reindex(groups)
            x = np.arange(len(groups)) + (i - (len(ctrls) - 1) / 2) * width
            err = np.vstack([sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]])
            b = ax.bar(x, sub["mean"], width=width * 0.9, color=controller_colour(c, style),
                       edgecolor=style["surface"], linewidth=0.8, zorder=2,
                       label=controller_label(c))
            for bb in b:
                bb.set_hatch(CONTROLLER_HATCH[c])
            ax.errorbar(x, sub["mean"], yerr=err, fmt="none", ecolor="#3a3a38",
                        elinewidth=0.9, capsize=2, zorder=3)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, fontsize=7.8)
        ax.set_title(lab, fontsize=8.8)
        ax.axhline(0, color="#c9c8c3", lw=0.8)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.20 * (hi_ - lo_))
    axes[0].legend(ncol=2, fontsize=7.4, loc="upper left")
    fig.suptitle("Figure 8 | Generalisation to disturbance combinations never seen "
                 "during RL training (no retraining)",
                 x=0.012, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "figure8_generalisation_unseen", cfg)


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all figures")
    ap.add_argument("--results", default="main_results")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--only", nargs="*", default=None, help="subset of figure numbers")
    ap.add_argument("--manuscript", action="store_true",
                    help="drop in-image titles/disclaimer and write to figures/manuscript/")
    args = ap.parse_args()

    ensure_dirs()
    cfg, sds = load_config(), load_seeds()
    style = set_style(cfg)
    if args.manuscript:
        set_manuscript_mode(True)
    n_boot, seed = int(args.bootstrap), int(sds["master_seed"]) % (2 ** 31)
    want = set(args.only) if args.only else None

    def wanted(n: str) -> bool:
        return want is None or n in want

    print(banner("RumenTwin-RL | generating figures"))
    if wanted("1"):
        figure1_architecture(cfg, style)
    if wanted("2"):
        figure2_trajectory(cfg, style, sds)

    try:
        df = read_table(PATHS["results_raw"] / args.results)
    except FileNotFoundError:
        print("  No main results found - run scripts/run_experiments.py first.")
        return
    print(f"  loaded {len(df)} episode records")
    for n, fn in [("3", figure3_low_ph_exposure), ("4", figure4_by_scenario),
                  ("5", figure5_robustness), ("6", figure6_cost_vs_stability),
                  ("8", figure8_generalisation)]:
        if wanted(n):
            fn(df, cfg, style, n_boot, seed)
    if wanted("7"):
        figure7_individual_variability(df, cfg, style)
    print("\nAll figures written to figures/")


if __name__ == "__main__":
    main()
