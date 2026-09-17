"""Reproducibility: identical seeds must give bit-identical results."""
import numpy as np
import pandas as pd
import pytest

from rumen_twin.cow import CowPopulation
from rumen_twin.environment import RumenTwinEnv
from rumen_twin.evaluation import make_episode_plan, run_episode
from rumen_twin.expert_rules import ExpertRuleController, NoInterventionController
from rumen_twin.utils import episode_seed, stable_hash


def test_stable_hash_is_process_independent():
    """Python's built-in hash() is salted per process; ours must not be."""
    assert stable_hash("A_normal", 11, 3) == stable_hash("A_normal", 11, 3)
    # golden value: must not change between processes or machines
    assert stable_hash("A_normal", 11, 3) == 2174174160
    assert stable_hash("cow_order") == 3830099319
    assert stable_hash("B", 1) != stable_hash("B", 2)


def test_episode_seed_is_deterministic():
    a = episode_seed(500000, "E_gradual_sara", 22, 7, 314)
    b = episode_seed(500000, "E_gradual_sara", 22, 7, 314)
    assert a == b and 0 <= a < 2 ** 31 - 1


def test_population_is_identical_across_instantiations(cfg, seeds):
    a = CowPopulation(config=cfg, seeds=seeds).to_frame()
    b = CowPopulation(config=cfg, seeds=seeds).to_frame()
    pd.testing.assert_frame_equal(a, b)


def test_split_is_stable_and_disjoint(population, cfg):
    tr = {c.cow_id for c in population.subset("train")}
    va = {c.cow_id for c in population.subset("validation")}
    te = {c.cow_id for c in population.subset("test")}
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == cfg["population"]["n_cows"]
    frac = cfg["population"]["split"]
    assert len(tr) == round(frac["train"] * cfg["population"]["n_cows"])
    assert len(te) == cfg["population"]["n_cows"] - len(tr) - len(va)


def test_test_cows_have_different_parameters_from_training_cows(population):
    """Generalisation claim: no test cow is a copy of a training cow."""
    tr = np.array([c.vector() for c in population.subset("train")])
    te = np.array([c.vector() for c in population.subset("test")])
    for v in te[:40]:
        assert np.abs(tr - v).sum(axis=1).min() > 1e-8


def test_episode_plan_is_deterministic(population, cfg, seeds):
    ids = [c.cow_id for c in population.subset("test")]
    kw = dict(scenarios=["A_normal", "I_mixed"], eval_seeds=cfg["evaluation"]["eval_seeds"],
              episodes_per_condition=60, seed_offset=seeds["evaluation"]["episode_seed_offset"])
    pd.testing.assert_frame_equal(make_episode_plan(ids, **kw), make_episode_plan(ids, **kw))


def test_episode_plan_only_uses_the_requested_split(population, cfg, seeds):
    ids = [c.cow_id for c in population.subset("test")]
    plan = make_episode_plan(ids, ["A_normal"], [11, 22, 33], 180,
                             seeds["evaluation"]["episode_seed_offset"])
    assert set(plan["cow_id"]).issubset(set(ids))


def test_episode_plan_covers_every_test_cow(population, cfg, seeds):
    ids = [c.cow_id for c in population.subset("test")]
    plan = make_episode_plan(ids, ["A_normal"], cfg["evaluation"]["eval_seeds"],
                             cfg["evaluation"]["episodes_per_condition"],
                             seeds["evaluation"]["episode_seed_offset"])
    assert set(plan["cow_id"]) == set(ids)


def test_episode_plan_matches_cows_across_scenarios(population, cfg, seeds):
    ids = [c.cow_id for c in population.subset("test")]
    plan = make_episode_plan(ids, ["A_normal", "B_high_concentrate", "J_stress_test"],
                             [11, 22, 33], 90, seeds["evaluation"]["episode_seed_offset"])
    piv = plan.pivot_table(index=["eval_seed", "episode_index"], columns="scenario",
                           values="cow_id")
    assert (piv.nunique(axis=1) == 1).all(), "cows must be matched across scenarios"
    assert plan.groupby(["eval_seed", "episode_index"])["seed"].nunique().eq(3).all(), \
        "each scenario must still get its own disturbance realisation"


@pytest.mark.parametrize("controller_cls", [NoInterventionController, ExpertRuleController])
def test_metrics_are_reproducible(cfg, seeds, population, controller_cls):
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=population, split="test")
    ctrl = controller_cls(cfg)
    a = run_episode(env, ctrl, cow_id=11, scenario="E_gradual_sara", seed=99).metrics
    b = run_episode(env, ctrl, cow_id=11, scenario="E_gradual_sara", seed=99).metrics
    assert a == b


def test_rerunning_other_episodes_does_not_change_a_result(cfg, seeds, population):
    """No hidden global state: results must not depend on evaluation order."""
    env = RumenTwinEnv(config=cfg, seeds=seeds, population=population, split="test")
    ctrl = ExpertRuleController(cfg)
    first = run_episode(env, ctrl, cow_id=7, scenario="B_high_concentrate", seed=5).metrics
    for s in range(3):
        run_episode(env, ctrl, cow_id=20 + s, scenario="D_heat_stress", seed=100 + s)
    again = run_episode(env, ctrl, cow_id=7, scenario="B_high_concentrate", seed=5).metrics
    assert first == again
