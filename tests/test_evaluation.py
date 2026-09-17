"""Metric definitions and the statistical helpers."""
import numpy as np
import pandas as pd
import pytest

from rumen_twin.environment import RumenTwinEnv
from rumen_twin.evaluation import _recovery_times, _runs_below, run_episode
from rumen_twin.expert_rules import ExpertRuleController, NoInterventionController
from rumen_twin.statistics import (bootstrap_ci, cliffs_delta, compare_paired, describe,
                                   hedges_g_paired, matched_rank_biserial)

REQUIRED_METRICS = [
    "mean_pH", "sd_pH", "pct_time_below_low_pH", "cumulative_low_pH_exposure",
    "n_sara_like_episodes", "n_severe_low_pH_episodes", "physiological_stability",
    "mean_recovery_time_min", "total_intervention_cost", "n_interventions", "n_alerts",
    "mean_milk_yield_proxy", "cumulative_reward", "success",
]


def test_runs_below_detects_maximal_runs():
    x = np.array([6.4, 5.7, 5.6, 6.3, 6.4, 5.5, 5.4, 5.3])
    assert _runs_below(x, 5.8) == [(1, 2), (5, 3)]
    assert _runs_below(np.array([6.4, 6.5]), 5.8) == []
    assert _runs_below(np.array([5.0, 5.0]), 5.8) == [(0, 2)]


def test_recovery_time_requires_a_confirmed_return():
    # dips at index 1, returns to >=6.2 from index 4 and stays for 4 steps
    pH = np.array([6.4, 5.7, 5.9, 6.0, 6.25, 6.3, 6.3, 6.3])
    assert _recovery_times(pH, 5.8, 6.2, 4, 15.0) == pytest.approx(3 * 15.0)
    # never recovers -> censored at the episode end (conservative)
    never = np.array([6.4, 5.7, 5.6, 5.6, 5.6])
    assert _recovery_times(never, 5.8, 6.2, 4, 15.0) == pytest.approx(4 * 15.0)
    assert np.isnan(_recovery_times(np.full(5, 6.4), 5.8, 6.2, 4, 15.0))


def test_all_required_metrics_are_produced(cfg, seeds, population):
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=population, split="test")
    m = run_episode(env, NoInterventionController(cfg), cow_id=3,
                    scenario="E_gradual_sara", seed=1).metrics
    for k in REQUIRED_METRICS:
        assert k in m, f"missing metric '{k}'"
        assert m[k] is not None
    assert m["n_steps"] == 192
    assert 0.0 <= m["pct_time_below_low_pH"] <= 100.0
    assert 0.0 <= m["physiological_stability"] <= 1.0
    assert m["cumulative_low_pH_exposure"] >= 0.0
    assert isinstance(m["success"], bool)


def test_no_intervention_records_no_interventions(cfg, seeds, population):
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=population, split="test")
    m = run_episode(env, NoInterventionController(cfg), cow_id=3,
                    scenario="B_high_concentrate", seed=2).metrics
    assert m["n_interventions"] == 0
    assert m["n_alerts"] == 0
    assert m["total_intervention_cost"] == 0.0


def test_expert_rules_improve_on_doing_nothing(cfg, seeds, population):
    """Sanity check on the baseline: the rules must actually control pH."""
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=population, split="test")
    none_, rules = [], []
    for s in range(12):
        none_.append(run_episode(env, NoInterventionController(cfg), cow_id=s,
                                 scenario="B_high_concentrate", seed=400 + s).metrics)
        rules.append(run_episode(env, ExpertRuleController(cfg), cow_id=s,
                                 scenario="B_high_concentrate", seed=400 + s).metrics)
    a = pd.DataFrame(none_)["pct_time_below_low_pH"].mean()
    b = pd.DataFrame(rules)["pct_time_below_low_pH"].mean()
    assert b < a, "the expert baseline should reduce low-pH exposure"


def test_trajectory_recording(cfg, seeds, population):
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=population, split="test")
    res = run_episode(env, ExpertRuleController(cfg), cow_id=4, scenario="A_normal",
                      seed=3, record=True)
    tr = res.trajectory
    assert len(tr) == 192
    for c in ("true_rumen_pH", "observed_pH", "estimated_pH", "action", "reward"):
        assert c in tr.columns
    assert tr["true_rumen_pH"].notna().all()


# --------------------------------------------------------------------------
def test_describe_reports_the_expected_summary():
    d = describe([1, 2, 3, 4, 5])
    assert d["n"] == 5 and d["mean"] == 3 and d["median"] == 3
    assert d["iqr"] == pytest.approx(2.0)
    assert np.isnan(describe([])["mean"])


def test_bootstrap_ci_brackets_the_estimate():
    x = np.random.default_rng(0).normal(10, 2, 400)
    ci = bootstrap_ci(x, np.mean, 2000, 0.95, 7)
    assert ci["ci_low"] < ci["estimate"] < ci["ci_high"]
    assert abs(ci["estimate"] - 10) < 0.4


def test_effect_sizes_have_the_expected_sign_and_range():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 300)
    better = base + 1.0
    assert hedges_g_paired(better, base) > 0.8
    assert hedges_g_paired(base, better) < -0.8
    assert hedges_g_paired(base, base) == 0.0
    rb = matched_rank_biserial(better, base)
    assert 0.9 < rb <= 1.0
    assert -1.0 <= cliffs_delta(better, base) <= 1.0


def test_compare_paired_matches_on_the_pair_keys():
    rng = np.random.default_rng(2)
    n = 80
    base = rng.normal(5, 1, n)
    effect = 0.8 + rng.normal(0, 0.15, n)      # a realistic, non-degenerate effect
    df = pd.concat([
        pd.DataFrame({"controller": "a", "scenario": "S", "eval_seed": 1,
                      "episode_index": range(n), "m": base}),
        pd.DataFrame({"controller": "b", "scenario": "S", "eval_seed": 1,
                      "episode_index": range(n), "m": base + effect}),
    ])
    c = compare_paired(df, "m", "b", "a")
    assert c.n_pairs == n
    assert c.mean_difference == pytest.approx(effect.mean(), abs=1e-9)
    assert c.ci_low < c.mean_difference < c.ci_high
    assert c.pct_pairs_favouring_a > 99.0
    assert c.effect_magnitude == "large"
    # rows must be matched on the pair keys, not just concatenated
    shuffled = pd.concat([df[df.controller == "a"],
                          df[df.controller == "b"].sample(frac=1.0, random_state=0)])
    assert compare_paired(shuffled, "m", "b", "a").mean_difference == \
        pytest.approx(c.mean_difference)


def test_compare_paired_raises_without_matching_rows():
    df = pd.DataFrame({"controller": ["a"], "scenario": ["S"], "eval_seed": [1],
                       "episode_index": [0], "m": [1.0]})
    with pytest.raises(ValueError):
        compare_paired(df, "m", "a", "b")
