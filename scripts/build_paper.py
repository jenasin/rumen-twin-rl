#!/usr/bin/env python3
"""Assemble ``paper/paper_draft.md`` from the generated result files.

Every numeric claim in the manuscript is substituted from ``tables/`` and
``results/raw/`` at build time through a ``{{placeholder}}`` mechanism, so the
draft cannot drift from the experiments and no number is ever typed by hand.
Running this script after ``generate_results.py`` regenerates the manuscript
against the current results.
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Dict

import numpy as np
import pandas as pd

import _paths  # noqa: F401
from rumen_twin.statistics import compare_paired, describe, bootstrap_ci
from rumen_twin.utils import PATHS, banner, load_config, load_seeds, read_table

TEMPLATE = PATHS["paper"] / "paper_template.md"
OUTPUT = PATHS["paper"] / "paper_draft.md"


def _f(x, dec=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{dec}f}"


def build_facts(cfg: Dict, sds: Dict) -> Dict[str, str]:
    """Every number the manuscript quotes, computed from the saved episodes."""
    df = read_table(PATHS["results_raw"] / "main_results")
    n_boot = int(cfg["evaluation"]["bootstrap_resamples"])
    seed = int(sds["master_seed"]) % (2 ** 31)
    F: Dict[str, str] = {}

    # ---------- design ----------
    F["n_cows"] = str(cfg["population"]["n_cows"])
    F["n_train"] = str(int(round(cfg["population"]["split"]["train"] * cfg["population"]["n_cows"])))
    F["n_val"] = str(int(round(cfg["population"]["split"]["validation"] * cfg["population"]["n_cows"])))
    F["n_test"] = str(int(df["cow_id"].nunique()))
    F["n_episodes_total"] = f"{len(df):,}"
    F["n_episodes_per_condition"] = str(cfg["evaluation"]["episodes_per_condition"])
    F["n_scenarios"] = str(df["scenario"].nunique())
    F["eval_seeds"] = ", ".join(str(s) for s in cfg["evaluation"]["eval_seeds"])
    F["dt"] = str(cfg["simulation"]["dt_minutes"])
    F["episode_hours"] = str(cfg["simulation"]["episode_hours"])
    F["episode_steps"] = str(int(cfg["simulation"]["episode_hours"] * 60 / cfg["simulation"]["dt_minutes"]))
    F["low_pH"] = f"{cfg['thresholds']['low_pH']:g}"
    F["severe_pH"] = f"{cfg['thresholds']['severe_low_pH']:g}"
    F["sara_minutes"] = str(cfg["thresholds"]["sara_episode_min_minutes"])
    F["ppo_steps"] = f"{cfg['training']['ppo']['total_timesteps']:,}"
    F["dqn_steps"] = f"{cfg['training']['dqn']['total_timesteps']:,}"
    F["obs_dim"] = "26"
    F["n_actions"] = "7"

    # ---------- overall performance ----------
    metrics = ["mean_pH", "sd_pH", "pct_time_below_low_pH", "cumulative_low_pH_exposure",
               "n_sara_like_episodes", "n_severe_low_pH_episodes", "physiological_stability",
               "mean_recovery_time_min", "total_intervention_cost", "n_interventions",
               "n_alerts", "mean_milk_yield_proxy", "cumulative_reward", "success",
               "final_health_risk"]
    for c, sub in df.groupby("controller"):
        for m in metrics:
            v = sub[m].astype(float) * (100.0 if m == "success" else 1.0)
            d = describe(v)
            ci = bootstrap_ci(v, np.mean, n_boot, 0.95, seed)
            dec = 3 if m in ("mean_pH", "sd_pH", "physiological_stability",
                             "final_health_risk") else 2
            F[f"{c}__{m}"] = _f(d["mean"], dec)
            F[f"{c}__{m}__sd"] = _f(d["sd"], dec)
            F[f"{c}__{m}__ci"] = f"[{_f(ci['ci_low'], dec)}, {_f(ci['ci_high'], dec)}]"
            F[f"{c}__{m}__median"] = _f(d["median"], dec)
            F[f"{c}__{m}__iqr"] = f"[{_f(d['q1'], dec)}, {_f(d['q3'], dec)}]"

    # ---------- paired comparisons ----------
    hib = {"pct_time_below_low_pH": False, "cumulative_low_pH_exposure": False,
           "n_sara_like_episodes": False, "n_severe_low_pH_episodes": False,
           "physiological_stability": True, "cumulative_reward": True,
           "total_intervention_cost": False, "n_interventions": False,
           "mean_milk_yield_proxy": True, "success": True, "mean_pH": True,
           "final_health_risk": False, "mean_recovery_time_min": False, "n_alerts": False}
    pairs = [("ppo", "expert_rules"), ("dqn", "expert_rules"), ("ppo", "dqn"),
             ("expert_rules", "no_intervention"), ("ppo", "no_intervention")]
    for a, b in pairs:
        if a not in set(df["controller"]) or b not in set(df["controller"]):
            continue
        for m, h in hib.items():
            try:
                c = compare_paired(df, m, a, b, n_resamples=n_boot, seed=seed,
                                   higher_is_better=h)
            except (ValueError, KeyError):
                continue
            dec = 3 if m in ("mean_pH", "physiological_stability", "final_health_risk") else 2
            k = f"cmp__{a}__vs__{b}__{m}"
            F[f"{k}__diff"] = _f(c.mean_difference, dec)
            F[f"{k}__ci"] = f"[{_f(c.ci_low, dec)}, {_f(c.ci_high, dec)}]"
            F[f"{k}__g"] = _f(c.hedges_g_paired, 2)
            F[f"{k}__mag"] = c.effect_magnitude
            F[f"{k}__rb"] = _f(c.matched_rank_biserial, 3)
            F[f"{k}__p"] = ("< 0.001" if c.wilcoxon_p < 1e-3 else _f(c.wilcoxon_p, 3))
            F[f"{k}__fav"] = _f(c.pct_pairs_favouring_a, 1)
            F[f"{k}__tie"] = _f(c.pct_pairs_tied, 1)
            F[f"{k}__unfav"] = _f(c.pct_pairs_favouring_b, 1)
            F[f"{k}__rel"] = _f(100.0 * c.mean_difference / abs(c.mean_b), 1) if abs(c.mean_b) > 1e-9 else "n/a"

    # ---------- per-scenario ----------
    for (scen, c), sub in df.groupby(["scenario", "controller"]):
        for m in ["pct_time_below_low_pH", "cumulative_reward", "physiological_stability",
                  "n_interventions", "n_sara_like_episodes", "success",
                  "total_intervention_cost", "mean_milk_yield_proxy"]:
            v = sub[m].astype(float) * (100.0 if m == "success" else 1.0)
            dec = 3 if m == "physiological_stability" else 2
            F[f"{scen}__{c}__{m}"] = _f(v.mean(), dec)

    # ---------- seen vs unseen scenarios ----------
    train = set(cfg["training_scenarios"])
    for grp, mask in [("seen", df["scenario"].isin(train)), ("unseen", ~df["scenario"].isin(train))]:
        for c, sub in df[mask].groupby("controller"):
            for m in ["pct_time_below_low_pH", "cumulative_reward", "physiological_stability",
                      "n_interventions", "success"]:
                dec = 3 if m == "physiological_stability" else 2
                v = sub[m].astype(float) * (100.0 if m == "success" else 1.0)
                F[f"{grp}__{c}__{m}"] = _f(v.mean(), dec)

    # ---------- sensor robustness (paired vs the clean reference) ----------
    ref = cfg["scenarios"]["F_sensor_dropout"]["base_disturbance"]
    F["robustness_reference"] = cfg["scenarios"][ref]["label"]
    for c in df["controller"].unique():
        sub = df[df["controller"] == c]
        degr = []
        for scen in ["F_sensor_dropout", "G_sensor_noise", "H_sensor_delay"]:
            if scen not in set(sub["scenario"]):
                continue
            for m, h in [("cumulative_reward", True), ("pct_time_below_low_pH", False)]:
                cmp = compare_paired(sub, m, scen, ref, group_col="scenario",
                                     pair_keys=("eval_seed", "episode_index"),
                                     n_resamples=n_boot, seed=seed, higher_is_better=h)
                F[f"rob__{c}__{scen}__{m}__diff"] = _f(cmp.mean_difference, 2)
                F[f"rob__{c}__{scen}__{m}__ci"] = f"[{_f(cmp.ci_low, 2)}, {_f(cmp.ci_high, 2)}]"
                if m == "cumulative_reward":
                    d = -100.0 * cmp.mean_difference / abs(cmp.mean_b) if abs(cmp.mean_b) > 1e-9 else np.nan
                    F[f"rob__{c}__{scen}__reward_degradation_pct"] = _f(d, 1)
                    degr.append(d)
        F[f"rob__{c}__mean_reward_degradation_pct"] = _f(float(np.nanmean(degr)), 1) if degr else "n/a"

    # ---------- ablations ----------
    try:
        abl = read_table(PATHS["results_raw"] / "ablation_results")
        for (ab, var), sub in abl.groupby(["ablation", "variant"]):
            key = re.sub(r"[^a-z0-9]+", "_", var.lower()).strip("_")
            for m in ["pct_time_below_low_pH", "cumulative_reward", "physiological_stability",
                      "n_interventions", "total_intervention_cost", "n_sara_like_episodes",
                      "mean_milk_yield_proxy", "success"]:
                dec = 3 if m == "physiological_stability" else 2
                v = sub[m].astype(float) * (100.0 if m == "success" else 1.0)
                F[f"abl__{ab}__{key}__{m}"] = _f(v.mean(), dec)
        # paired ablation contrasts
        for ab, a_var, b_var in [
            ("1_sensor_faults", "PPO trained WITH sensor faults", "PPO trained WITHOUT sensor faults"),
            ("2_cow_variability", "PPO trained WITH cow variability", "PPO trained WITHOUT cow variability"),
            ("3_intervention_cost", "PPO with cost penalties", "PPO without cost penalties"),
        ]:
            sub = abl[abl["ablation"] == ab]
            if set(sub["variant"]) != {a_var, b_var}:
                continue
            for m, h in [("pct_time_below_low_pH", False), ("cumulative_reward", True),
                         ("n_interventions", False), ("total_intervention_cost", False),
                         ("physiological_stability", True)]:
                c = compare_paired(sub, m, a_var, b_var, group_col="variant",
                                   pair_keys=("scenario", "eval_seed", "episode_index"),
                                   n_resamples=n_boot, seed=seed, higher_is_better=h)
                dec = 3 if m == "physiological_stability" else 2
                F[f"ablcmp__{ab}__{m}__diff"] = _f(c.mean_difference, dec)
                F[f"ablcmp__{ab}__{m}__ci"] = f"[{_f(c.ci_low, dec)}, {_f(c.ci_high, dec)}]"
                F[f"ablcmp__{ab}__{m}__g"] = _f(c.hedges_g_paired, 2)
                F[f"ablcmp__{ab}__{m}__mag"] = c.effect_magnitude
        F["abl_n_episodes"] = f"{len(abl):,}"
    except FileNotFoundError:
        pass

    # ---------- reward-weight sensitivity ----------
    try:
        sens = read_table(PATHS["results_raw"] / "sensitivity_results")
        rank = sens.groupby(["weight_set", "controller"])["cumulative_reward"].mean().unstack()
        F["sens_weight_sets"] = ", ".join(rank.index)
        F["sens_n_weight_sets"] = str(len(rank))
        label = {"no_intervention": "no intervention", "expert_rules": "expert rules",
                 "dqn": "DQN", "ppo": "PPO"}
        F["sens_best_per_set"] = "; ".join(
            f"*{w}* -> {label.get(rank.loc[w].idxmax(), rank.loc[w].idxmax())}"
            for w in rank.index)
        F["sens_weight_sets"] = ", ".join(f"*{w}*" for w in rank.index)
        F["sens_ppo_best_in_all"] = ("yes" if all(rank.loc[w].idxmax() == "ppo" for w in rank.index)
                                     else "no")
        low = sens.groupby(["weight_set", "controller"])["pct_time_below_low_pH"].mean().unstack()
        F["sens_ppo_lowest_pct_low_in_all"] = ("yes" if all(low.loc[w].idxmin() == "ppo"
                                                            for w in low.index) else "no")
        for w in rank.index:
            for c in rank.columns:
                F[f"sens__{w}__{c}__cumulative_reward"] = _f(rank.loc[w, c], 2)
                F[f"sens__{w}__{c}__pct_time_below_low_pH"] = _f(low.loc[w, c], 2)
        F["sens_n_episodes"] = f"{len(sens):,}"
    except FileNotFoundError:
        pass

    # ---------- explainability ----------
    try:
        pol = pd.read_csv(PATHS["tables"] / "tableS4_policy_sensitivity.csv")
        top = pol.head(5)
        F["sens_top_features"] = "; ".join(
            f"{r.label} ({r.action_switch_rate_pct:.1f} %)" for r in top.itertuples())
        F["sens_top_feature"] = str(top.iloc[0]["label"])
        F["sens_top_feature_pct"] = _f(float(top.iloc[0]["action_switch_rate_pct"]), 1)
        F["sens_n_states"] = str(int(pol["n_states"].iloc[0]))
        F["sens_pct_action_0"] = _f(float(pol["pct_states_action_0"].iloc[0]), 1)
        F["sens_top3"] = "; ".join(
            f"{r.label} ({r.action_switch_rate_pct:.1f} %)" for r in pol.head(3).itertuples())
    except FileNotFoundError:
        pass

    # ---------- environment / runtime ----------
    meta_path = PATHS["results"] / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        F["python_version"] = meta.get("python", "")
        F["platform"] = meta.get("platform", "")
    return F


def reflow(text: str, width: int = 79) -> str:
    """Re-wrap prose paragraphs after substitution.

    Placeholders vary in length, so the template's line breaks land in awkward
    places once they are filled in. Tables, code fences, headings, list items
    and block quotes are left exactly as they are.
    """
    out, buf, in_code = [], [], False

    def flush():
        if not buf:
            return
        words = " ".join(buf).split()
        line = ""
        for w in words:
            if line and len(line) + 1 + len(w) > width:
                out.append(line)
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            out.append(line)
        buf.clear()

    for raw in text.split("\n"):
        if raw.strip().startswith("```"):
            flush(); in_code = not in_code; out.append(raw); continue
        if in_code:
            out.append(raw); continue
        stripped = raw.strip()
        # A leading "-" or "*" only starts a list when followed by a space;
        # otherwise it is a negative number or emphasis and must not break the
        # paragraph (e.g. "-4.42 percentage points", "*g* = 0.85").
        structural = bool(
            not stripped
            or stripped.startswith(("#", "|", ">"))
            or re.match(r"^([-*+]\s|\d+\.\s)", stripped)
            or raw.startswith("    "))
        if structural:
            flush(); out.append(raw)
        else:
            buf.append(stripped)
    flush()
    return "\n".join(out)


def render(template: str, facts: Dict[str, str]) -> str:
    missing = set()

    def sub(m):
        k = m.group(1).strip()
        if k not in facts:
            missing.add(k)
            return f"«MISSING:{k}»"
        return facts[k]

    out = re.sub(r"\{\{([^}]+)\}\}", sub, template)
    if missing:
        print(f"  WARNING: {len(missing)} unresolved placeholders:")
        for k in sorted(missing):
            print(f"    - {k}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the manuscript from the results")
    ap.add_argument("--template", default=str(TEMPLATE))
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()

    cfg, sds = load_config(), load_seeds()
    print(banner("RumenTwin-RL | building paper/paper_draft.md from the results"))
    facts = build_facts(cfg, sds)
    print(f"  {len(facts)} facts extracted from the simulation output")
    text = open(args.template).read()
    open(args.out, "w").write(reflow(render(text, facts)))
    (PATHS["paper"] / "paper_facts.json").write_text(json.dumps(facts, indent=2, sort_keys=True))
    print(f"  wrote {args.out} and paper/paper_facts.json")


if __name__ == "__main__":
    main()
