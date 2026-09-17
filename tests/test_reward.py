"""Reward function: components, weights, and resistance to degenerate policies."""
import numpy as np
import pytest

from rumen_twin.interventions import ActionResult, InterventionManager
from rumen_twin.physiology import TwinState
from rumen_twin.reward import RewardFunction


def _state(pH=6.40, run=0, milk=31.0, rum=5.2, temp=38.9):
    return TwinState(pH, temp, rum, 0.24, 0.42, 0.02, milk, 0.88, 0.10, 0.05, low_pH_run=run)


NOOP = ActionResult(0, "no_intervention", True, 0.0, False)


@pytest.fixture
def rf(cfg):
    return RewardFunction(cfg)


@pytest.fixture
def mgr(cfg):
    return InterventionManager(cfg, 192)


def test_reward_is_always_finite(rf, mgr, cfg):
    for pH in np.linspace(cfg["state_variables"]["rumen_pH"]["min"],
                          cfg["state_variables"]["rumen_pH"]["max"], 40):
        for run in (0, 12, 48, 192):
            c = rf(_state(pH, run), NOOP, mgr, 10)
            assert np.isfinite(c.total)


def test_stability_peaks_at_the_target_state(rf, cfg):
    target = cfg["thresholds"]["target_pH"]
    best = rf.stability(_state(target))
    assert best > rf.stability(_state(target - 0.5))
    assert best > rf.stability(_state(target + 0.5))
    assert 0.0 <= best <= 1.0


def test_lower_pH_gives_lower_reward(rf, mgr):
    r = [rf(_state(pH), NOOP, mgr, 5).total for pH in (6.4, 6.1, 5.9, 5.7, 5.4, 5.2)]
    assert all(np.diff(r) < 0), f"reward not monotone in pH: {r}"


def test_sustained_low_pH_is_penalised_more_than_a_brief_dip(rf, mgr):
    """The w4b term must specifically punish *duration*, not just depth."""
    brief = rf(_state(5.70, run=1), NOOP, mgr, 5)
    long = rf(_state(5.70, run=36), NOOP, mgr, 5)
    assert long.total < brief.total
    assert long.duration > brief.duration
    assert brief.low_pH == pytest.approx(long.low_pH)      # same depth


def test_severe_penalty_only_fires_below_the_severe_threshold(rf, mgr, cfg):
    sev = cfg["thresholds"]["severe_low_pH"]
    assert rf(_state(sev + 0.05), NOOP, mgr, 5).severe_low_pH == 0.0
    assert rf(_state(sev - 0.15), NOOP, mgr, 5).severe_low_pH > 0.0


def test_duration_penalty_is_capped(rf, mgr, cfg):
    assert rf(_state(5.6, run=10 ** 6), NOOP, mgr, 5).duration == \
        pytest.approx(cfg["reward"]["duration_penalty"]["cap"])


def test_intervention_cost_reduces_reward(rf, mgr):
    free = rf(_state(), NOOP, mgr, 5).total
    paid = rf(_state(), ActionResult(3, "administer_buffer", True, 1.5, True), mgr, 5).total
    assert paid < free


def test_excess_penalty_is_quadratic_beyond_the_allowance(cfg, rf):
    allow = int(cfg["reward"]["excessive_intervention"]["free_allowance"])
    vals = []
    for n in (allow, allow + 2, allow + 4, allow + 6):
        m = InterventionManager(cfg, 192)
        for t in range(n):
            m.apply(1, t)
        vals.append(rf(_state(), NOOP, m, n - 1).excess)
    assert vals[0] == 0.0
    d1, d2, d3 = np.diff(vals)
    assert d3 > d2 > d1 > 0, "excess penalty should grow faster than linearly"


def test_alert_is_penalised_even_when_rejected(rf, mgr):
    """Closing the loophole: a call rejected on cooldown is still a false alarm."""
    accepted = ActionResult(6, "trigger_alert", True, 3.0, True)
    rejected = ActionResult(6, "trigger_alert", False, 0.25, True)
    assert rf(_state(), accepted, mgr, 5).alert == 1.0
    assert rf(_state(), rejected, mgr, 5).alert == 1.0


def test_buffer_spamming_on_a_healthy_cow_is_never_worthwhile(cfg, rf):
    """The reward must not admit the trivial 'always buffer' policy."""
    healthy = _state(6.40)
    idle = InterventionManager(cfg, 192)
    spam = InterventionManager(cfg, 192)
    r_idle = r_spam = 0.0
    for t in range(24):
        r_idle += rf(healthy, idle.apply(0, t), idle, t).total
        r_spam += rf(healthy, spam.apply(3, t), spam, t).total
    assert r_spam < r_idle
    assert r_spam < 0.2 * r_idle, "buffer spamming is not discouraged strongly enough"


def test_alert_spamming_is_never_worthwhile(cfg, rf):
    healthy = _state(6.40)
    idle = InterventionManager(cfg, 192)
    spam = InterventionManager(cfg, 192)
    r_idle = r_spam = 0.0
    for t in range(48):
        r_idle += rf(healthy, idle.apply(0, t), idle, t).total
        r_spam += rf(healthy, spam.apply(6, t), spam, t).total
    assert r_spam < r_idle


def test_a_justified_intervention_can_pay_for_itself(cfg, rf):
    """Cost must not be so high that correcting real acidosis is irrational."""
    m = InterventionManager(cfg, 192)
    acidotic = rf(_state(5.55, run=20), NOOP, m, 5).total
    corrected = rf(_state(6.25, run=0),
                   ActionResult(3, "administer_buffer", True,
                                cfg["interventions"][3]["cost"], True), m, 5).total
    assert corrected > acidotic


def test_weight_sets_change_the_balance(cfg):
    s = _state(6.40)
    m = InterventionManager(cfg, 192)
    buffer_call = ActionResult(3, "administer_buffer", True, 1.5, True)
    cheap = RewardFunction(cfg, "default")(s, buffer_call, m, 5).total
    averse = RewardFunction(cfg, "cost_averse")(s, buffer_call, m, 5).total
    free = RewardFunction(cfg, "cost_free")(s, buffer_call, m, 5).total
    assert averse < cheap < free


def test_cost_free_weight_set_ignores_all_cost_terms(cfg):
    rf = RewardFunction(cfg, "cost_free")
    m = InterventionManager(cfg, 192)
    for t in range(20):
        m.apply(3, t)
    c = rf(_state(), ActionResult(6, "trigger_alert", True, 3.0, True), m, 19)
    w = rf.w
    assert w["w5_intervention_cost"] == 0.0
    assert w["w6_excessive_intervention"] == 0.0
    assert w["w7_alert"] == 0.0
    # components are still *reported*, they simply carry zero weight
    assert c.cost > 0 and c.excess > 0 and c.alert == 1.0


def test_unknown_weight_set_raises(cfg):
    with pytest.raises(KeyError):
        RewardFunction(cfg, "no_such_weight_set")


def test_components_reconstruct_the_total(cfg, rf, mgr):
    c = rf(_state(5.65, run=18), ActionResult(3, "administer_buffer", True, 1.5, True), mgr, 7)
    w, k = rf.w, rf.scale
    expected = k * (w["w1_stability"] * c.stability + w["w2_milk"] * c.milk
                    - w["w3_low_pH"] * c.low_pH - w["w4_severe_low_pH"] * c.severe_low_pH
                    - w["w4b_duration"] * c.duration - w["w5_intervention_cost"] * c.cost
                    - w["w6_excessive_intervention"] * c.excess - w["w7_alert"] * c.alert)
    assert c.total == pytest.approx(expected)
