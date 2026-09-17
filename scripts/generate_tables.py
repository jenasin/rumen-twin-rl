#!/usr/bin/env python3
"""Generate Tables 1-8 (plus supplementary tables) from the simulation output.

Descriptive tables (1-4) are generated from ``config.yaml``; result tables
(5-8) are computed from the episode-level records written by
``run_experiments.py`` / ``ablation_study.py`` / ``sensitivity_analysis.py``.
No number in this file is hard-coded.

Each table is written as CSV (machine-readable) and Markdown (for the paper).
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from rumen_twin.interventions import intervention_table
from rumen_twin.statistics import bootstrap_ci, compare_paired, describe
from rumen_twin.utils import (DISCLAIMER, PATHS, banner, ensure_dirs, load_config,
                              load_seeds, read_table)

# metric -> (display name, decimals, higher_is_better)
METRIC_SPEC: Dict[str, tuple] = {
    "mean_pH": ("Mean rumen pH", 3, True),
    "sd_pH": ("SD of pH", 3, False),
    "pct_time_below_low_pH": ("Time below pH 5.8 (%)", 2, False),
    "cumulative_low_pH_exposure": ("Cumulative low-pH exposure (pH-units x min)", 1, False),
    "n_sara_like_episodes": ("SARA-like episodes (n)", 2, False),
    "n_severe_low_pH_episodes": ("Severe low-pH episodes (n)", 2, False),
    "physiological_stability": ("Physiological stability score", 3, True),
    "mean_recovery_time_min": ("Recovery time (min)", 1, False),
    "total_intervention_cost": ("Total intervention cost", 2, False),
    "n_interventions": ("Interventions (n)", 2, False),
    "n_alerts": ("Alerts (n)", 2, False),
    "mean_milk_yield_proxy": ("Milk yield proxy (kg/d)", 2, True),
    "cumulative_reward": ("Cumulative reward", 2, True),
    "success": ("Success rate (%)", 1, True),
    "final_health_risk": ("Final health-risk index", 3, False),
}
PRIMARY_METRICS = list(METRIC_SPEC)
CONTROLLER_ORDER = ["no_intervention", "expert_rules", "dqn", "ppo"]
CONTROLLER_LABEL = {
    "no_intervention": "No intervention",
    "expert_rules": "Expert rules",
    "dqn": "DQN",
    "ppo": "PPO",
}


# ---------------------------------------------------------------------------
def _save(df: pd.DataFrame, name: str, caption: str, float_fmt: str = "%.3f") -> None:
    ensure_dirs()
    csv = PATHS["tables"] / f"{name}.csv"
    md = PATHS["tables"] / f"{name}.md"
    df.to_csv(csv, index=False)
    with open(md, "w") as fh:
        fh.write(f"**{caption}**\n\n")
        fh.write(df.to_markdown(index=False, floatfmt=".3f"))
        fh.write(f"\n\n*{DISCLAIMER}*\n")
    print(f"  wrote {csv.name} / {md.name}   ({len(df)} rows)")


def _fmt_ci(row, dec: int) -> str:
    return f"{row['mean']:.{dec}f} ± {row['sd']:.{dec}f} [{row['ci_low']:.{dec}f}, {row['ci_high']:.{dec}f}]"


def _order_controllers(df: pd.DataFrame, col: str = "controller") -> pd.DataFrame:
    """Sort rows into the canonical controller order, if that column exists.

    Aggregations that are not grouped by controller (the ablation table groups
    by ablation/variant instead) are returned untouched.
    """
    if col not in df.columns:
        return df
    order = [c for c in CONTROLLER_ORDER if c in set(df[col])] + \
            [c for c in df[col].unique() if c not in CONTROLLER_ORDER]
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(col)


# ===========================================================================
# Descriptive tables (from config)
# ===========================================================================
def table1_state_variables(cfg) -> pd.DataFrame:
    rows = []
    for name, m in cfg["state_variables"].items():
        rows.append({
            "state_variable": name,
            "unit": m["unit"],
            "admissible_range": f"[{m['min']:g}, {m['max']:g}]",
            "plausible_range": f"[{m['plausible_low']:g}, {m['plausible_high']:g}]",
            "default_value": m["default"],
            "sensor_available": "yes" if m["observable"] else "no (latent)",
            "description": m["description"],
        })
    for name, desc in cfg["auxiliary_variables"].items():
        rows.append({"state_variable": f"{name} (auxiliary)", "unit": "-",
                     "admissible_range": "-", "plausible_range": "-",
                     "default_value": "-", "sensor_available": "no (latent)",
                     "description": desc})
    return pd.DataFrame(rows)


def table2_simulation_parameters(cfg) -> pd.DataFrame:
    rows: List[Dict] = []

    def add(group, k, v, note=""):
        rows.append({"group": group, "parameter": k, "value": v, "note": note})

    sim = cfg["simulation"]
    add("Simulation", "dt_minutes", sim["dt_minutes"], "discrete time step")
    add("Simulation", "episode_hours", sim["episode_hours"],
        f"{int(sim['episode_hours'] * 60 / sim['dt_minutes'])} steps per episode")
    add("Simulation", "steps_per_day", sim["steps_per_day"], "")
    pop = cfg["population"]
    add("Population", "n_cows", pop["n_cows"], "virtual herd size")
    add("Population", "split", str(pop["split"]), "train / validation / test (by individual)")
    for k, v in pop["hyper"].items():
        add("Population hyperparameters", k, f"N({v[0]}, {v[1]}) clipped to [{v[2]}, {v[3]}]", "")
    for k, v in pop["process_noise"].items():
        add("Within-cow process noise (SD per step)", k, v, "")
    for k, v in cfg["feeding"].items():
        add("Feeding plan", k, str(v), "")
    for k, v in cfg["physiology"].items():
        add("Physiology", k, v, "")
    for k, v in cfg["thresholds"].items():
        add("Thresholds", k, v, "")
    for sk, sv in cfg["sensors"].items():
        add("Sensors (nominal)", sk,
            f"noise SD {sv['noise_sd']}, bias {sv['bias']}, dropout {sv['dropout_prob']}, "
            f"delay {sv['delay_steps']}", "")
    est = cfg["estimator"]
    add("State estimator", "method", est["method"], "")
    add("State estimator", "kalman_process_var", est["kalman"]["process_var"], "")
    add("State estimator", "kalman_meas_var", est["kalman"]["meas_var"], "")
    ws = cfg["reward"]["weight_sets"][cfg["reward"]["active_weights"]]
    for k, v in ws.items():
        add("Reward weights (default set)", k, v, "")
    add("Reward", "scale", cfg["reward"]["scale"], "global scaling factor")
    for k, v in cfg["training"]["ppo"].items():
        add("PPO hyperparameters", k, str(v), "")
    for k, v in cfg["training"]["dqn"].items():
        add("DQN hyperparameters", k, str(v), "")
    return pd.DataFrame(rows)


def table4_scenarios(cfg) -> pd.DataFrame:
    rows = []
    train = set(cfg["training_scenarios"])
    for name, s in cfg["scenarios"].items():
        params = {k: v for k, v in s.items() if k not in ("label", "description", "family")}
        rows.append({
            "scenario": name,
            "label": s["label"],
            "family": s["family"],
            "description": s["description"],
            "seen_during_RL_training": "yes" if name in train else "NO (held out)",
            "parameters": "; ".join(f"{k}={v}" for k, v in params.items()) or "-",
        })
    return pd.DataFrame(rows)


# ===========================================================================
# Result tables (from simulation output)
# ===========================================================================
def _agg_block(df: pd.DataFrame, metrics: List[str], n_boot: int, seed: int,
               by: Optional[List[str]] = None) -> pd.DataFrame:
    """mean ± SD [95 % bootstrap CI] and median [IQR] for each metric/group."""
    by = by or ["controller"]
    out = []
    for keys, sub in df.groupby(by, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(by, keys))
        rec["n_episodes"] = int(len(sub))
        for m in metrics:
            vals = sub[m].astype(float) * (100.0 if m == "success" else 1.0)
            d = describe(vals)
            ci = bootstrap_ci(vals, np.mean, n_boot, 0.95, seed)
            dec = METRIC_SPEC[m][1]
            rec[f"{m}__mean"] = round(d["mean"], dec + 2)
            rec[f"{m}__sd"] = round(d["sd"], dec + 2)
            rec[f"{m}__median"] = round(d["median"], dec + 2)
            rec[f"{m}__iqr"] = round(d["iqr"], dec + 2)
            rec[f"{m}__ci_low"] = round(ci["ci_low"], dec + 2)
            rec[f"{m}__ci_high"] = round(ci["ci_high"], dec + 2)
            rec[f"{m}__formatted"] = (
                f"{d['mean']:.{dec}f} ± {d['sd']:.{dec}f} "
                f"[{ci['ci_low']:.{dec}f}, {ci['ci_high']:.{dec}f}]")
        out.append(rec)
    return _order_controllers(pd.DataFrame(out))


def table5_overall(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Table 5: pooled over every evaluation scenario."""
    agg = _agg_block(df, PRIMARY_METRICS, n_boot, seed)
    cols = ["controller", "n_episodes"] + [f"{m}__formatted" for m in PRIMARY_METRICS]
    pretty = agg[cols].rename(columns={"controller": "Controller", "n_episodes": "Episodes",
                                       **{f"{m}__formatted": METRIC_SPEC[m][0]
                                          for m in PRIMARY_METRICS}})
    pretty["Controller"] = pretty["Controller"].map(lambda c: CONTROLLER_LABEL.get(c, c))
    return pretty


def table5_full(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    return _agg_block(df, PRIMARY_METRICS, n_boot, seed)


def table6_by_scenario(df: pd.DataFrame, n_boot: int, seed: int,
                       metrics: Optional[List[str]] = None) -> pd.DataFrame:
    metrics = metrics or ["pct_time_below_low_pH", "n_sara_like_episodes",
                          "physiological_stability", "mean_recovery_time_min",
                          "total_intervention_cost", "mean_milk_yield_proxy",
                          "cumulative_reward", "success"]
    agg = _agg_block(df, metrics, n_boot, seed, by=["scenario", "controller"])
    labels = df.drop_duplicates("scenario").set_index("scenario")["scenario_label"].to_dict()
    cols = ["scenario", "controller", "n_episodes"] + [f"{m}__formatted" for m in metrics]
    out = agg[cols].copy()
    out.insert(1, "scenario_label", out["scenario"].map(labels))
    out = out.rename(columns={"controller": "Controller",
                              **{f"{m}__formatted": METRIC_SPEC[m][0] for m in metrics}})
    out["Controller"] = out["Controller"].map(lambda c: CONTROLLER_LABEL.get(c, c))
    return out.sort_values(["scenario", "Controller"])


def table7_robustness(df: pd.DataFrame, cfg, n_boot: int, seed: int) -> pd.DataFrame:
    """Table 7: degradation under sensor faults, paired against the clean
    reference scenario that the sensor scenarios are built on."""
    ref = cfg["scenarios"]["F_sensor_dropout"]["base_disturbance"]   # B_high_concentrate
    fault_scenarios = [s for s in df["scenario"].unique()
                       if cfg["scenarios"][s]["family"] in ("sensor", "mixed", "stress_test")]
    metrics = ["pct_time_below_low_pH", "physiological_stability", "cumulative_reward",
               "pct_steps_pH_unobserved"]
    rows = []
    for ctrl in [c for c in CONTROLLER_ORDER if c in set(df["controller"])]:
        sub = df[df["controller"] == ctrl]
        for scen in fault_scenarios:
            rec = {"controller": CONTROLLER_LABEL.get(ctrl, ctrl),
                   "reference_scenario": ref, "fault_scenario": scen,
                   "fault_label": cfg["scenarios"][scen]["label"]}
            for m in metrics:
                if m not in sub.columns:
                    continue
                hib = METRIC_SPEC.get(m, (None, 3, True))[2]
                cmp = compare_paired(sub, m, scen, ref, group_col="scenario",
                                     pair_keys=("eval_seed", "episode_index"),
                                     n_resamples=n_boot, seed=seed, higher_is_better=hib)
                rec[f"{m}__reference"] = round(cmp.mean_b, 3)
                rec[f"{m}__fault"] = round(cmp.mean_a, 3)
                rec[f"{m}__delta"] = round(cmp.mean_difference, 3)
                rec[f"{m}__ci"] = f"[{cmp.ci_low:.3f}, {cmp.ci_high:.3f}]"
                rec[f"{m}__hedges_g"] = round(cmp.hedges_g_paired, 3)
                if m == "cumulative_reward" and abs(cmp.mean_b) > 1e-9:
                    rec["robustness_degradation_pct"] = round(
                        -100.0 * cmp.mean_difference / abs(cmp.mean_b), 2)
            rows.append(rec)
    return pd.DataFrame(rows)


def table_paired_comparisons(df: pd.DataFrame, n_boot: int, seed: int,
                             reference: str = "expert_rules") -> pd.DataFrame:
    """Supplementary: every controller vs the expert baseline, per metric."""
    rows = []
    others = [c for c in CONTROLLER_ORDER if c in set(df["controller"]) and c != reference]
    for scope, sub in [("all_scenarios", df)] + [(s, g) for s, g in df.groupby("scenario")]:
        for ctrl in others:
            if ctrl not in set(sub["controller"]):
                continue
            for m in PRIMARY_METRICS:
                hib = METRIC_SPEC[m][2]
                try:
                    c = compare_paired(sub, m, ctrl, reference, n_resamples=n_boot,
                                       seed=seed, higher_is_better=hib)
                except ValueError:
                    continue
                d = c.as_dict()
                d["scope"] = scope
                d["higher_is_better"] = hib
                d["favours"] = (ctrl if ((c.mean_difference > 0) == hib) else reference)
                rows.append(d)
    cols = ["scope", "metric", "group_a", "group_b", "n_pairs", "mean_a", "mean_b",
            "mean_difference", "ci_low", "ci_high", "hedges_g_paired", "effect_magnitude",
            "matched_rank_biserial", "cliffs_delta", "wilcoxon_statistic", "wilcoxon_p",
            "pct_pairs_favouring_a", "pct_pairs_tied", "pct_pairs_favouring_b",
            "higher_is_better", "favours"]
    return pd.DataFrame(rows)[cols]


def table8_ablation(abl: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    metrics = ["pct_time_below_low_pH", "n_sara_like_episodes", "physiological_stability",
               "total_intervention_cost", "n_interventions", "mean_milk_yield_proxy",
               "cumulative_reward", "success"]
    agg = _agg_block(abl, metrics, n_boot, seed, by=["ablation", "variant", "scenario_group"])
    cols = ["ablation", "variant", "scenario_group", "n_episodes"] + \
           [f"{m}__formatted" for m in metrics]
    return agg[cols].rename(columns={f"{m}__formatted": METRIC_SPEC[m][0] for m in metrics})


# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all tables")
    ap.add_argument("--results", default="main_results")
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()

    ensure_dirs()
    cfg, sds = load_config(), load_seeds()
    n_boot = int(args.bootstrap or cfg["evaluation"]["bootstrap_resamples"])
    seed = int(sds["master_seed"]) % (2 ** 31)

    print(banner("RumenTwin-RL | generating tables"))

    _save(table1_state_variables(cfg), "table1_state_variables",
          "Table 1. State variables of the rumen digital twin.")
    _save(table2_simulation_parameters(cfg), "table2_simulation_parameters",
          "Table 2. Simulation parameters (all values from config.yaml).")
    _save(intervention_table(cfg), "table3_interventions",
          "Table 3. Simulated intervention definitions. These are decision-support "
          "actions inside a synthetic model, not a veterinary treatment protocol.")
    _save(table4_scenarios(cfg), "table4_scenarios",
          "Table 4. Experimental disturbance scenarios.")

    try:
        df = read_table(PATHS["results_raw"] / args.results)
    except FileNotFoundError:
        print("\n  No main results found - run scripts/run_experiments.py first.")
        return

    print(f"\n  loaded {len(df)} episode records "
          f"({df['controller'].nunique()} controllers x {df['scenario'].nunique()} scenarios)")

    _save(table5_overall(df, n_boot, seed), "table5_overall_performance",
          "Table 5. Overall controller performance pooled across all evaluation "
          "scenarios (mean ± SD [95 % bootstrap CI]).")
    table5_full(df, n_boot, seed).to_csv(PATHS["tables"] / "table5_overall_performance_full.csv",
                                         index=False)
    _save(table6_by_scenario(df, n_boot, seed), "table6_performance_by_scenario",
          "Table 6. Controller performance by disturbance scenario.")
    _save(table7_robustness(df, cfg, n_boot, seed), "table7_sensor_robustness",
          "Table 7. Sensor-failure robustness: paired change relative to the "
          "clean-sensor reference scenario.")
    _save(table_paired_comparisons(df, n_boot, seed), "tableS1_paired_comparisons",
          "Table S1. Paired comparisons against the expert-rule baseline "
          "(effect sizes and 95 % bootstrap CIs; p-values reported but not "
          "used as the primary evidence).")

    for name, fname, caption in [
        ("ablation_results", "table8_ablation_results",
         "Table 8. Ablation experiments."),
        ("sensitivity_results", "tableS2_reward_sensitivity",
         "Table S2. Reward-weight sensitivity analysis."),
    ]:
        try:
            extra = read_table(PATHS["results_raw"] / name)
        except FileNotFoundError:
            print(f"  (skipping {fname}: {name} not found)")
            continue
        if name == "ablation_results":
            _save(table8_ablation(extra, n_boot, seed), fname, caption)
        else:
            _save(_agg_block(extra, ["pct_time_below_low_pH", "n_sara_like_episodes",
                                     "physiological_stability", "total_intervention_cost",
                                     "n_interventions", "mean_milk_yield_proxy",
                                     "cumulative_reward", "success"],
                             n_boot, seed, by=["weight_set", "controller"]),
                  fname, caption)

    print("\nAll tables written to tables/")


if __name__ == "__main__":
    main()
